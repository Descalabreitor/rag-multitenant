"""The fictional world `make seed` loads: two tenants, five users, seven documents.

Everything here is invented. Tenant ids and user subs are fixed, and
keycloak/realm-export.json uses the same ones: tenant ids are Organization ids,
user subs are Keycloak user ids, and memberships mirror the realm's groups
(tests/unit/test_realm_export.py checks that they agree). Document ids are not
fixed: the ingestion pipeline assigns them, and they stay the same for as long as
a file's bytes do (seed/load.py).

The documents are files in seed/files/, a mix of Markdown, HTML and .docx. A
.docx is built at load time from `<name>.docx.md` next to where it would be
(seed/docx.py), so the repo holds no binaries.

Every document carries its canary, a string that appears nowhere else, in each
of its sections. If a canary shows up in a response for someone outside the
document's ACL, that is a leak. Each user belongs to one tenant only, and group
names repeat across tenants on purpose (both have `finance`).

Each tenant has one admin (the `admins` group, ADR 0008): bob in Acme, carol in
Umbra. Neither is in finance, so being an admin visibly grants no read access:
admins upload, change ACLs and delete, but read only what the ACLs allow. The
seed uploads each tenant's documents as its admin.
"""

from dataclasses import dataclass
from uuid import UUID

from ragmt.domain import TENANT_ADMIN_GROUP

ACME = UUID("7f66d4f3-54bf-47da-9604-0551ebdb756b")
UMBRA = UUID("17033638-c3cc-4d1e-b1fb-539941854a6e")

ALICE = "4cb054f8-3faa-468f-a410-3c061363d034"
BOB = "36e84a7f-cac9-4d9b-b39f-480902553eef"
CAROL = "38425a66-cdc9-4f96-8d71-707edc2ee3b3"
DAVE = "19fc9100-084e-4582-a5bd-8e5df1ed928c"
ERIN = "ee6caa5b-fb1c-4234-b847-f5ada9b20d49"


@dataclass(frozen=True)
class Document:
    # Under seed/files/, POSIX-style. Its name is the filename the upload carries.
    path: str
    # The title the converter must find (the first H1); tests and e2e rely on it.
    title: str
    canary: str
    acl: tuple[str, ...]


@dataclass(frozen=True)
class Tenant:
    id: UUID
    name: str
    # (user_sub, group_name). Users without groups don't appear here: being in the
    # tenant at all comes from the JWT, not from this table (ADR 0002).
    memberships: tuple[tuple[str, str], ...]
    documents: tuple[Document, ...]

    @property
    def uploader(self) -> str:
        """The tenant admin the seed uploads as (only admins upload, ADR 0008)."""
        return next(sub for sub, group in self.memberships if group == TENANT_ADMIN_GROUP)


TENANTS = (
    Tenant(
        id=ACME,
        name="Acme Logistics",
        memberships=((ALICE, "finance"), (BOB, "engineering"), (BOB, "admins")),
        documents=(
            Document(
                path="acme/employee-handbook.md",
                title="Employee handbook",
                canary="CANARY-ACME-HANDBOOK",
                acl=("tenant:*",),
            ),
            Document(
                path="acme/q3-budget.html",
                title="Q3 budget",
                canary="CANARY-ACME-BUDGET",
                acl=("group:finance",),
            ),
            Document(
                path="acme/routing-runbook.md",
                title="Routing service runbook",
                canary="CANARY-ACME-RUNBOOK",
                acl=("group:engineering",),
            ),
            Document(
                path="acme/alice-review.docx",
                title="Performance review: Alice",
                canary="CANARY-ACME-ALICE-REVIEW",
                acl=(f"user:{ALICE}",),
            ),
        ),
    ),
    Tenant(
        id=UMBRA,
        name="Umbra Biotech",
        memberships=((CAROL, "research"), (CAROL, "admins"), (DAVE, "finance")),
        documents=(
            Document(
                path="umbra/lab-safety.html",
                title="Lab safety policy",
                canary="CANARY-UMBRA-SAFETY",
                acl=("tenant:*",),
            ),
            Document(
                path="umbra/trial-results.docx",
                title="Compound UB-7 trial results",
                canary="CANARY-UMBRA-TRIAL",
                acl=("group:research",),
            ),
            Document(
                path="umbra/annual-budget.md",
                title="Annual budget",
                canary="CANARY-UMBRA-BUDGET",
                acl=("group:finance",),
            ),
        ),
    ),
)
