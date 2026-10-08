"""Document writes: upload, new content, ACL changes and deletes, for tenant admins only
(ADR 0008).

Every route gets its writer through `UploadWriter` or `DocumentWriter`, and both
resolve the admin check first: it reads the caller's own memberships on the
request's app_rw connection (`TenantConn`), and only a caller who passes it
reaches `get_ingest_service` and the app_ingest engine behind it.

Status codes:
- `POST /documents` from a non-admin is 403: the route exists for everyone.
- Routes with a document id answer a non-admin, another tenant's id, an unknown
  id and a deleted id with the same 404 as `GET /documents/{id}`, so a caller
  can't learn that an id exists.
- 409 when new content is byte-identical to another live document of the tenant.
- 413 too large, 415 not a supported document, 422 bad form or ACL, 503 when
  the embedding service is down.

Error bodies are fixed strings: no filename, no file content, no reason. The
tenant comes only from the Principal; a `tenant_id` anywhere in the request is
ignored. Embeddings are never returned.
"""

import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Annotated, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from pydantic import BaseModel

from ragmt.adapters.llm import EmbeddingError
from ragmt.api.documents import NOT_FOUND
from ragmt.api.uploads import (
    MalformedUploadError,
    NotMultipartError,
    UploadForm,
    UploadTooLargeError,
    read_upload_form,
)
from ragmt.auth.dependencies import get_principal
from ragmt.domain import DocumentTooLargeError, IngestError, Principal, UnsupportedDocumentError
from ragmt.ingest.service import (
    DocumentNotFoundError,
    DuplicateDocumentError,
    IngestResult,
    InvalidAclError,
)
from ragmt.tenancy import TenantConn
from ragmt.tenancy.admin import is_tenant_admin
from ragmt.tenancy.writer import IngestServiceDep, TenantWriter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])

FORBIDDEN = "Forbidden"
TOO_LARGE = "Document too large"
UNSUPPORTED = "Unsupported document type"
NOT_MULTIPART = "Expected multipart/form-data"
INVALID_UPLOAD = "Invalid upload"
INVALID_ACL = "Invalid ACL"
DUPLICATE = "Another document has the same content"
EMBEDDINGS_UNAVAILABLE = "Embedding service unavailable"


def _error(status_code: int, detail: str) -> HTTPException:
    return HTTPException(status_code=status_code, detail=detail)


# --- admin check, then the writer ----------------------------------------------------


async def require_tenant_admin(
    principal: Annotated[Principal, Depends(get_principal)], conn: TenantConn
) -> Principal:
    """The caller, if they are a tenant admin; 403 otherwise."""
    if not await is_tenant_admin(conn, principal.sub):
        raise _error(status.HTTP_403_FORBIDDEN, FORBIDDEN)
    return principal


async def require_tenant_admin_for_document(
    principal: Annotated[Principal, Depends(get_principal)], conn: TenantConn
) -> Principal:
    """Like `require_tenant_admin`, but a non-admin gets the 404 of a missing document."""
    if not await is_tenant_admin(conn, principal.sub):
        raise _error(status.HTTP_404_NOT_FOUND, NOT_FOUND)
    return principal


# The admin parameter comes first: FastAPI resolves sub-dependencies in order,
# and one that raises stops the rest.
async def get_upload_writer(
    admin: Annotated[Principal, Depends(require_tenant_admin)], service: IngestServiceDep
) -> TenantWriter:
    return TenantWriter(service, admin)


async def get_document_writer(
    admin: Annotated[Principal, Depends(require_tenant_admin_for_document)],
    service: IngestServiceDep,
) -> TenantWriter:
    return TenantWriter(service, admin)


UploadWriter = Annotated[TenantWriter, Depends(get_upload_writer)]
DocumentWriter = Annotated[TenantWriter, Depends(get_document_writer)]


# --- models ----------------------------------------------------------------------------


class UploadResult(BaseModel):
    id: UUID
    title: str
    chunks: int
    # True when the same bytes were already stored: nothing was written, and the
    # fields describe the existing document (its ACL is left as it was).
    unchanged: bool


class AclUpdate(BaseModel):
    principals: list[str]


class DocumentAcl(BaseModel):
    id: UUID
    principals: list[str]


# FastAPI can't describe a body the route reads itself, so Swagger gets it here.
def _multipart_body(**fields: dict[str, str]) -> dict[str, Any]:
    properties = {"file": {"type": "string", "format": "binary"}, **fields}
    schema = {"type": "object", "required": ["file"], "properties": properties}
    return {
        "requestBody": {"required": True, "content": {"multipart/form-data": {"schema": schema}}}
    }


_UPLOAD_BODY = _multipart_body(
    acl={
        "type": "string",
        "description": 'JSON list of principals, e.g. ["group:hr"]. '
        "Without it, only the uploader can read the document.",
    }
)
_CONTENT_BODY = _multipart_body()


# --- routes ------------------------------------------------------------------------------


@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    openapi_extra=_UPLOAD_BODY,
    responses={
        status.HTTP_403_FORBIDDEN: {"description": FORBIDDEN},
        status.HTTP_413_CONTENT_TOO_LARGE: {"description": TOO_LARGE},
        status.HTTP_415_UNSUPPORTED_MEDIA_TYPE: {"description": UNSUPPORTED},
    },
)
async def upload_document(request: Request, writer: UploadWriter) -> UploadResult:
    form = await _read_form(request, writer.max_bytes)
    acl = _parse_acl(form.acl)
    with _write_errors():
        result = await writer.ingest(form.data, form.filename, acl)
    return _upload_result(result)


@router.put(
    "/{document_id}/content",
    openapi_extra=_CONTENT_BODY,
    responses={
        status.HTTP_404_NOT_FOUND: {"description": NOT_FOUND},
        status.HTTP_409_CONFLICT: {"description": DUPLICATE},
        status.HTTP_413_CONTENT_TOO_LARGE: {"description": TOO_LARGE},
        status.HTTP_415_UNSUPPORTED_MEDIA_TYPE: {"description": UNSUPPORTED},
    },
)
async def replace_document_content(
    document_id: UUID, request: Request, writer: DocumentWriter
) -> UploadResult:
    """Give the document new content. Its chunks and title are rebuilt from the
    file; its ACL stays as it is (change it with `PUT /documents/{id}/acl`).

    The same bytes as now are 200 with `unchanged: true` and write nothing.
    """
    form = await _read_form(request, writer.max_bytes)
    if form.acl is not None:
        # Replacing content never touches the ACL; refuse rather than ignore it.
        raise _error(status.HTTP_422_UNPROCESSABLE_CONTENT, INVALID_UPLOAD)
    with _write_errors():
        result = await writer.replace(document_id, form.data, form.filename)
    return _upload_result(result)


@router.put(
    "/{document_id}/acl",
    responses={status.HTTP_404_NOT_FOUND: {"description": NOT_FOUND}},
)
async def set_document_acl(
    document_id: UUID, body: AclUpdate, writer: DocumentWriter
) -> DocumentAcl:
    """Replace the document's ACL. Its chunks follow in the same transaction."""
    try:
        stored = await writer.set_acl(document_id, body.principals)
    except DocumentNotFoundError:
        raise _error(status.HTTP_404_NOT_FOUND, NOT_FOUND) from None
    except InvalidAclError:
        raise _error(status.HTTP_422_UNPROCESSABLE_CONTENT, INVALID_ACL) from None
    return DocumentAcl(id=document_id, principals=list(stored))


@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={status.HTTP_404_NOT_FOUND: {"description": NOT_FOUND}},
)
async def delete_document(
    document_id: UUID,
    writer: DocumentWriter,
    purge: Annotated[bool, Query(description="Remove the rows instead of hiding them")] = False,
) -> Response:
    """Soft delete by default: the document disappears for every reader, its rows
    stay. `purge=true` removes the document and its chunks, soft-deleted or not."""
    try:
        if purge:
            await writer.hard_delete(document_id)
        else:
            await writer.soft_delete(document_id)
    except DocumentNotFoundError:
        raise _error(status.HTTP_404_NOT_FOUND, NOT_FOUND) from None
    return Response(status_code=status.HTTP_204_NO_CONTENT)


async def _read_form(request: Request, max_bytes: int) -> UploadForm:
    """The multipart body, read from the stream with the caps of `read_upload_form`."""
    try:
        return await read_upload_form(
            request.headers.get("content-type"),
            request.headers.get("content-length"),
            request.stream(),
            max_bytes,
        )
    except NotMultipartError:
        raise _error(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, NOT_MULTIPART) from None
    except UploadTooLargeError as exc:
        logger.info("upload rejected: %s", exc)
        raise _error(status.HTTP_413_CONTENT_TOO_LARGE, TOO_LARGE) from None
    except MalformedUploadError as exc:
        logger.info("upload rejected: %s", exc)
        raise _error(status.HTTP_422_UNPROCESSABLE_CONTENT, INVALID_UPLOAD) from None


@contextmanager
def _write_errors() -> Iterator[None]:
    """Map the service's errors on an upload or a replace to fixed HTTP errors."""
    try:
        yield
    except DocumentNotFoundError:
        raise _error(status.HTTP_404_NOT_FOUND, NOT_FOUND) from None
    except DuplicateDocumentError:
        raise _error(status.HTTP_409_CONFLICT, DUPLICATE) from None
    except DocumentTooLargeError:
        raise _error(status.HTTP_413_CONTENT_TOO_LARGE, TOO_LARGE) from None
    except UnsupportedDocumentError as exc:
        # `reason` is ours; the filename is the caller's and stays out of the log.
        logger.info("upload rejected: %s", exc.reason)
        raise _error(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, UNSUPPORTED) from None
    except InvalidAclError:
        raise _error(status.HTTP_422_UNPROCESSABLE_CONTENT, INVALID_ACL) from None
    except EmbeddingError as exc:
        logger.warning("embedding failed during upload: %s", type(exc).__name__)
        raise _error(status.HTTP_503_SERVICE_UNAVAILABLE, EMBEDDINGS_UNAVAILABLE) from None
    except IngestError as exc:
        logger.info("upload rejected: %s", type(exc).__name__)
        raise _error(status.HTTP_422_UNPROCESSABLE_CONTENT, INVALID_UPLOAD) from None


def _upload_result(result: IngestResult) -> UploadResult:
    return UploadResult(
        id=result.document_id, title=result.title, chunks=result.chunks, unchanged=result.unchanged
    )


def _parse_acl(raw: bytes | None) -> list[str] | None:
    """The `acl` form field: absent, or a JSON list of strings. Checked further by the service."""
    if raw is None:
        return None
    try:
        value = json.loads(raw)
    except ValueError:
        raise _error(status.HTTP_422_UNPROCESSABLE_CONTENT, INVALID_ACL) from None
    if not isinstance(value, list) or not all(isinstance(p, str) for p in value):
        raise _error(status.HTTP_422_UNPROCESSABLE_CONTENT, INVALID_ACL)
    return value
