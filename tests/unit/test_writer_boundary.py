"""Only the modules that own a writer engine may read INGEST_DATABASE_URL (ADR 0004, 0008).

app_ingest sees the whole tenant, so a read route that got hold of it would skip
the ACL filter. Two checks:

- A static one over src/: the setting (the attribute, not the variable's name in
  docstrings) may appear only in the modules listed here.
- One over the app's routes: no GET route depends, directly or through other
  dependencies, on `get_ingest_service`, the only way to the writer engine; and
  every route that does also depends on an admin check that comes before it.
"""

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from fastapi.dependencies.models import Dependant
from fastapi.routing import RouteContext, iter_route_contexts

from ragmt.api.app import create_app
from ragmt.api.writes import require_tenant_admin, require_tenant_admin_for_document
from ragmt.tenancy.writer import get_ingest_service
from tests.unit.test_api import SETTINGS

SRC = Path(__file__).resolve().parents[2] / "src" / "ragmt"

ALLOWED = {
    "settings.py",
    # The API's only writer engine; it runs the admin check before any write.
    "tenancy/writer.py",
    # A separate process that writes memberships.
    "permsync/__main__.py",
}

ADMIN_CHECKS = {require_tenant_admin, require_tenant_admin_for_document}


def test_ingest_database_url_is_read_only_by_the_writer_modules() -> None:
    users = {
        path.relative_to(SRC).as_posix()
        for path in SRC.rglob("*.py")
        if "ingest_database_url" in path.read_text(encoding="utf-8")
    }
    assert users <= ALLOWED, sorted(users - ALLOWED)


def _calls(dependant: Dependant) -> Iterator[Callable[..., Any] | None]:
    """Every dependency callable under `dependant`, depth first, in resolution order."""
    for sub in dependant.dependencies:
        yield from _calls(sub)
        yield sub.call


def _routes() -> list[RouteContext]:
    """Every API route, with the dependencies its routers add (FastAPI nests
    included routers, so `app.routes` alone doesn't list them)."""
    routes = [
        r
        for r in iter_route_contexts(create_app(SETTINGS).routes)
        if isinstance(getattr(r, "dependant", None), Dependant)
    ]
    assert routes
    return routes


def test_no_get_route_depends_on_the_writer_engine() -> None:
    readers = [r for r in _routes() if "GET" in (r.methods or ())]
    assert {r.path for r in readers} >= {"/documents", "/documents/{document_id}"}
    for route in readers:
        assert get_ingest_service not in set(_calls(route.dependant)), route.path


def test_every_route_on_the_writer_engine_checks_the_admin_first() -> None:
    writers = [r for r in _routes() if get_ingest_service in set(_calls(r.dependant))]
    # The walk does find the write routes, so the GET check above is not vacuous.
    assert {(r.path, m) for r in writers for m in r.methods or ()} == {
        ("/documents", "POST"),
        ("/documents/{document_id}/acl", "PUT"),
        ("/documents/{document_id}", "DELETE"),
    }
    for route in writers:
        calls = list(_calls(route.dependant))
        checks = [calls.index(check) for check in ADMIN_CHECKS if check in calls]
        assert checks, route.path
        assert min(checks) < calls.index(get_ingest_service), route.path
