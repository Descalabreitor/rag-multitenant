"""The leak oracle: who may read what, worked out in plain Python.

The property test (`test_properties.py`) applies random operations through the
real code and replays each one here. The oracle answers from its own model of
the rules, written from the ADRs, not from the SQL:

- A user's principals in a tenant are `user:<sub>`, `tenant:*`, and `group:<g>`
  for each of their memberships in that tenant (ADR 0002). Being in `admins`
  adds nothing beyond `group:admins` (ADR 0008).
- A document is readable when it is live and its ACL shares a principal with the
  user's. Its chunks are readable exactly when it is (ADR 0003).
- An upload without an ACL gets `user:<uploader>`, and an explicit ACL replaces
  that default. The same bytes as a live document of the same tenant give back
  that document, unchanged (ADR 0008).
- Replacing content keeps the ACL. A soft delete removes the ACL, so a deleted
  document stays unreadable. A purge removes the document.
- Every write names a tenant and only finds documents of that tenant.

Nothing here imports `ragmt`: an oracle that shared code with the system under
test could share its bugs.
"""

from dataclasses import dataclass, field
from typing import Literal
from uuid import UUID

TENANT_WIDE = "tenant:*"

type State = Literal["live", "deleted", "purged"]
# (document id, ordinal, content): a chunk as the database stores it.
type Chunk = tuple[UUID, int, str]


@dataclass
class Document:
    id: UUID
    tenant: str
    title: str
    data: bytes
    paragraphs: tuple[str, ...]
    acl: frozenset[str]
    state: State = "live"

    @property
    def chunks(self) -> frozenset[Chunk]:
        """One chunk per paragraph (the test's chunker), numbered from 0."""
        return frozenset((self.id, i, p) for i, p in enumerate(self.paragraphs))

    @property
    def texts(self) -> frozenset[str]:
        """Everything a reader of this document may be shown: title and paragraphs."""
        return frozenset({self.title, *self.paragraphs})


@dataclass
class Oracle:
    """Tenants are keys chosen by the test ("t0", "t1", ...), not database ids."""

    memberships: dict[str, frozenset[tuple[str, str]]] = field(default_factory=dict)
    # Every document ever created, purged ones included (state "purged").
    documents: dict[UUID, Document] = field(default_factory=dict)

    def add_tenant(self, tenant: str) -> None:
        self.memberships[tenant] = frozenset()

    def set_memberships(self, tenant: str, memberships: frozenset[tuple[str, str]]) -> None:
        self.memberships[tenant] = memberships

    # --- writes -------------------------------------------------------------------
    # Each returns whether the write finds its document. A write that doesn't find
    # it changes nothing.

    def live_duplicate(self, tenant: str, data: bytes) -> Document | None:
        """The live document of `tenant` with these exact bytes, if any."""
        return next(
            (
                d
                for d in self.documents.values()
                if d.tenant == tenant and d.state == "live" and d.data == data
            ),
            None,
        )

    def upload(
        self,
        document_id: UUID,
        tenant: str,
        uploader: str,
        title: str,
        data: bytes,
        paragraphs: tuple[str, ...],
        acl: list[str] | None,
    ) -> Document:
        """A new document (the caller has checked `live_duplicate` first)."""
        assert document_id not in self.documents
        doc = Document(
            id=document_id,
            tenant=tenant,
            title=title,
            data=data,
            paragraphs=paragraphs,
            acl=frozenset([f"user:{uploader}"] if acl is None else acl),
        )
        self.documents[document_id] = doc
        return doc

    def replace(
        self,
        tenant: str,
        document_id: UUID,
        title: str,
        data: bytes,
        paragraphs: tuple[str, ...],
    ) -> bool:
        doc = self._find(tenant, document_id, live_only=True)
        if doc is None:
            return False
        doc.title, doc.data, doc.paragraphs = title, data, paragraphs
        return True

    def set_acl(self, tenant: str, document_id: UUID, acl: list[str]) -> bool:
        doc = self._find(tenant, document_id, live_only=True)
        if doc is None:
            return False
        doc.acl = frozenset(acl)
        return True

    def soft_delete(self, tenant: str, document_id: UUID) -> bool:
        doc = self._find(tenant, document_id, live_only=True)
        if doc is None:
            return False
        doc.state, doc.acl = "deleted", frozenset()
        return True

    def purge(self, tenant: str, document_id: UUID) -> bool:
        doc = self._find(tenant, document_id, live_only=False)
        if doc is None:
            return False
        doc.state, doc.acl = "purged", frozenset()
        return True

    def _find(self, tenant: str, document_id: UUID, *, live_only: bool) -> Document | None:
        doc = self.documents.get(document_id)
        if doc is None or doc.tenant != tenant or doc.state == "purged":
            return None
        if live_only and doc.state != "live":
            return None
        return doc

    # --- reads --------------------------------------------------------------------

    def principals(self, tenant: str, user: str) -> frozenset[str]:
        groups = {f"group:{g}" for sub, g in self.memberships[tenant] if sub == user}
        return frozenset({f"user:{user}", TENANT_WIDE, *groups})

    def readable(self, tenant: str, user: str) -> list[Document]:
        """The documents `user` may read in `tenant`, in creation order."""
        principals = self.principals(tenant, user)
        return [
            d
            for d in self.documents.values()
            if d.tenant == tenant and d.state == "live" and d.acl & principals
        ]

    def readable_chunks(self, tenant: str, user: str) -> frozenset[Chunk]:
        return frozenset(c for d in self.readable(tenant, user) for c in d.chunks)

    def readable_texts(self, tenant: str, user: str) -> frozenset[str]:
        return frozenset(t for d in self.readable(tenant, user) for t in d.texts)
