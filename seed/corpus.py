"""The fictional world `make seed` loads: two tenants, five users, seven documents.

Everything here is invented. Ids are fixed so the seed is repeatable, and
keycloak/realm-export.json uses the same ones: tenant ids are Organization ids,
user subs are Keycloak user ids, and memberships mirror the realm's groups
(tests/unit/test_realm_export.py checks that they agree).

Every chunk ends with its document's canary, a string that appears nowhere else.
If a canary shows up in a response for someone outside the document's ACL, that
is a leak. Each user belongs to one tenant only, and group names repeat across
tenants on purpose (both have `finance`).
"""

from dataclasses import dataclass
from uuid import UUID

ACME = UUID("7f66d4f3-54bf-47da-9604-0551ebdb756b")
UMBRA = UUID("17033638-c3cc-4d1e-b1fb-539941854a6e")

ALICE = "4cb054f8-3faa-468f-a410-3c061363d034"
BOB = "36e84a7f-cac9-4d9b-b39f-480902553eef"
CAROL = "38425a66-cdc9-4f96-8d71-707edc2ee3b3"
DAVE = "19fc9100-084e-4582-a5bd-8e5df1ed928c"
ERIN = "ee6caa5b-fb1c-4234-b847-f5ada9b20d49"


@dataclass(frozen=True)
class Document:
    id: UUID
    title: str
    canary: str
    acl: tuple[str, ...]
    # (heading, body) pairs; each becomes one chunk.
    sections: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class Tenant:
    id: UUID
    name: str
    # (user_sub, group_name). Users without groups don't appear here: being in the
    # tenant at all comes from the JWT, not from this table (ADR 0002).
    memberships: tuple[tuple[str, str], ...]
    documents: tuple[Document, ...]


TENANTS = (
    Tenant(
        id=ACME,
        name="Acme Logistics",
        memberships=((ALICE, "finance"), (BOB, "engineering")),
        documents=(
            Document(
                id=UUID("e72cb086-b518-4880-9779-0ab1c8814abb"),
                title="Employee handbook",
                canary="CANARY-ACME-HANDBOOK",
                acl=("tenant:*",),
                sections=(
                    (
                        "Working hours",
                        "Core hours are 10:00 to 15:00. Outside them, people organise their "
                        "own time and record it in the timesheet by Friday.",
                    ),
                    (
                        "Expenses",
                        "Travel is booked through the internal portal. Receipts above 40 "
                        "credits must be attached within two weeks of the trip.",
                    ),
                ),
            ),
            Document(
                id=UUID("0e982739-b267-40bd-b180-2c57485c2521"),
                title="Q3 budget",
                canary="CANARY-ACME-BUDGET",
                acl=("group:finance",),
                sections=(
                    (
                        "Fleet",
                        "The fleet budget for Q3 is 1.2 million credits, of which 300,000 "
                        "go to replacing the oldest delivery drones.",
                    ),
                    (
                        "Hiring",
                        "Two planner roles are approved for Q3. A third depends on the "
                        "northern depot opening on schedule.",
                    ),
                ),
            ),
            Document(
                id=UUID("1065e2b9-7acc-4b4f-8461-08b49b2015b6"),
                title="Routing service runbook",
                canary="CANARY-ACME-RUNBOOK",
                acl=("group:engineering",),
                sections=(
                    (
                        "Restarting the router",
                        "Drain the queue first, then restart one replica at a time. The "
                        "health check needs about 90 seconds to turn green.",
                    ),
                ),
            ),
            Document(
                id=UUID("13f86cc8-c073-4529-85ce-bb9f580c6bdc"),
                title="Performance review: Alice",
                canary="CANARY-ACME-ALICE-REVIEW",
                acl=(f"user:{ALICE}",),
                sections=(
                    (
                        "Summary",
                        "Alice led the quarterly close two days ahead of plan and mentored "
                        "a new analyst. Next goal: own the depot cost model.",
                    ),
                ),
            ),
        ),
    ),
    Tenant(
        id=UMBRA,
        name="Umbra Biotech",
        memberships=((CAROL, "research"), (DAVE, "finance")),
        documents=(
            Document(
                id=UUID("34ba74ce-c217-4ef4-95b1-9583a7056270"),
                title="Lab safety policy",
                canary="CANARY-UMBRA-SAFETY",
                acl=("tenant:*",),
                sections=(
                    (
                        "Protective equipment",
                        "Goggles and gloves are required past the yellow line. Lab coats "
                        "stay inside the lab and are collected on Thursdays.",
                    ),
                ),
            ),
            Document(
                id=UUID("bb77923c-8d24-4868-be8c-0b8ff47116ef"),
                title="Compound UB-7 trial results",
                canary="CANARY-UMBRA-TRIAL",
                acl=("group:research",),
                sections=(
                    (
                        "Results",
                        "Compound UB-7 cut growth in the culture by 34% at the middle dose. "
                        "The high dose showed no further benefit.",
                    ),
                    (
                        "Next steps",
                        "Repeat the middle dose with a larger sample before any external "
                        "presentation.",
                    ),
                ),
            ),
            Document(
                id=UUID("193fedfc-f1be-42ea-90b3-d2553593f377"),
                title="Annual budget",
                canary="CANARY-UMBRA-BUDGET",
                acl=("group:finance",),
                sections=(
                    (
                        "Research spend",
                        "Research receives 60% of the annual budget. Equipment purchases "
                        "above 50,000 credits need board approval.",
                    ),
                ),
            ),
        ),
    ),
)
