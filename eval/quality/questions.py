"""The questions of the answer-quality baseline, over the seed corpus (seed/corpus.py).

Each question names the seed documents that answer it on their own (by title)
and a short reference answer, and the users who ask it. What a user should get
is not written by hand: it follows from the corpus ACLs and memberships, the
same rules RLS applies. A question is answerable for a user when they may read
at least one of its documents; otherwise the right answer is "I don't know".
So the same question is answerable for alice and unanswerable for erin, and a
question about Acme's budget asked by Umbra's finance user tests that a shared
group name doesn't cross tenants.

Questions with no documents are outside the corpus: nobody can answer them from
the sources, and a model that answers from its own knowledge is wrong here.
"""

from dataclasses import dataclass
from enum import StrEnum

from seed.corpus import ACME, ALICE, BOB, CAROL, DAVE, ERIN, TENANTS, UMBRA, Document, Tenant


@dataclass(frozen=True)
class User:
    username: str
    sub: str
    tenant: Tenant

    @property
    def groups(self) -> frozenset[str]:
        return frozenset(g for sub, g in self.tenant.memberships if sub == self.sub)

    @property
    def admin(self) -> "User":
        """The tenant's admin: the one who can read this user's audit rows."""
        return next(u for u in USERS.values() if u.sub == self.tenant.uploader)

    def can_read(self, document: Document) -> bool:
        """Whether RLS lets this user read `document` (ADR 0002, 0003)."""
        if document not in self.tenant.documents:
            return False
        principals = {"tenant:*", f"user:{self.sub}"} | {f"group:{g}" for g in self.groups}
        return not principals.isdisjoint(document.acl)


_TENANTS = {t.id: t for t in TENANTS}
# Erin is in Acme with no groups: the realm says so, and the corpus has no row for her.
USERS = {
    u.username: u
    for u in (
        User("alice", ALICE, _TENANTS[ACME]),
        User("bob", BOB, _TENANTS[ACME]),
        User("erin", ERIN, _TENANTS[ACME]),
        User("carol", CAROL, _TENANTS[UMBRA]),
        User("dave", DAVE, _TENANTS[UMBRA]),
    )
}
DOCUMENTS = {d.title: d for t in TENANTS for d in t.documents}


@dataclass(frozen=True)
class Question:
    id: str
    text: str
    # Titles of the documents that answer it, each on its own. Empty: none does.
    sources: tuple[str, ...]
    # A short reference answer; empty when no document answers it.
    reference: str
    askers: tuple[str, ...]


class Kind(StrEnum):
    """Why a case is answerable or not, for the breakdown in the results."""

    ANSWERABLE = "answerable"
    # A document answers it in the user's tenant, but its ACL excludes the user.
    DENIED_BY_ACL = "denied by ACL"
    # Only another tenant's documents answer it.
    OTHER_TENANT = "other tenant"
    OUT_OF_CORPUS = "out of corpus"


@dataclass(frozen=True)
class Case:
    """One question asked by one user."""

    question: Question
    user: User

    @property
    def id(self) -> str:
        return f"{self.question.id}/{self.user.username}"

    @property
    def expected(self) -> frozenset[str]:
        """The titles of the question's documents this user may read."""
        return frozenset(t for t in self.question.sources if self.user.can_read(DOCUMENTS[t]))

    @property
    def answerable(self) -> bool:
        return bool(self.expected)

    @property
    def kind(self) -> Kind:
        if self.expected:
            return Kind.ANSWERABLE
        if not self.question.sources:
            return Kind.OUT_OF_CORPUS
        if any(DOCUMENTS[t] in self.user.tenant.documents for t in self.question.sources):
            return Kind.DENIED_BY_ACL
        return Kind.OTHER_TENANT


HANDBOOK = "Employee handbook"
Q3_BUDGET = "Q3 budget"
RUNBOOK = "Routing service runbook"
REVIEW = "Performance review: Alice"
SAFETY = "Lab safety policy"
TRIAL = "Compound UB-7 trial results"
ANNUAL_BUDGET = "Annual budget"

QUESTIONS = (
    # Acme, employee handbook (tenant:*)
    Question(
        "core-hours",
        "What are the core working hours?",
        (HANDBOOK,),
        "Core hours are 10:00 to 15:00.",
        ("erin", "bob", "dave"),
    ),
    Question(
        "timesheet",
        "By when do I have to record my time in the timesheet?",
        (HANDBOOK,),
        "By Friday.",
        ("erin",),
    ),
    Question(
        "receipts",
        "Which travel receipts do I have to attach, and by when?",
        (HANDBOOK,),
        "Receipts above 40 credits, within two weeks of the trip.",
        ("alice", "carol"),
    ),
    Question(
        "travel-booking",
        "How do I book travel?",
        (HANDBOOK,),
        "Through the internal portal.",
        ("erin",),
    ),
    # Acme, Q3 budget (group:finance)
    Question(
        "fleet-budget",
        "What is the fleet budget for Q3?",
        (Q3_BUDGET,),
        "1.2 million credits.",
        ("alice", "erin", "bob", "dave"),
    ),
    Question(
        "drones",
        "How much of the Q3 fleet budget goes to replacing delivery drones?",
        (Q3_BUDGET,),
        "300,000 credits, for replacing the oldest delivery drones.",
        ("alice",),
    ),
    Question(
        "planners",
        "How many planner roles are approved for Q3?",
        (Q3_BUDGET,),
        "Two; a third depends on the northern depot opening on schedule.",
        ("alice", "erin"),
    ),
    # Acme, routing runbook (group:engineering)
    Question(
        "restart-router",
        "How do I restart the routing service?",
        (RUNBOOK,),
        "Drain the queue first, restart one replica at a time, and wait about 90 seconds "
        "for the health check to turn green.",
        ("bob", "alice"),
    ),
    Question(
        "health-check",
        "How long does the router's health check take to turn green after a restart?",
        (RUNBOOK,),
        "About 90 seconds.",
        ("bob",),
    ),
    # Acme, Alice's review (user:alice)
    Question(
        "quarterly-close",
        "What does Alice's performance review say about the quarterly close?",
        (REVIEW,),
        "She led it two days ahead of plan.",
        ("alice", "bob", "erin"),
    ),
    Question(
        "alice-goal",
        "What is Alice's next goal?",
        (REVIEW,),
        "To own the depot cost model.",
        ("alice",),
    ),
    # Umbra, lab safety (tenant:*)
    Question(
        "yellow-line",
        "What protective equipment is required past the yellow line?",
        (SAFETY,),
        "Goggles and gloves.",
        ("carol", "dave", "erin"),
    ),
    Question(
        "lab-coats",
        "When are lab coats collected?",
        (SAFETY,),
        "On Thursdays (they stay inside the lab).",
        ("dave",),
    ),
    # Umbra, trial results (group:research)
    Question(
        "ub7-effect",
        "By how much did compound UB-7 reduce growth at the middle dose?",
        (TRIAL,),
        "By 34%.",
        ("carol", "dave"),
    ),
    Question(
        "ub7-next",
        "What are the next steps for the UB-7 trial?",
        (TRIAL,),
        "Repeat the middle dose with a larger sample before any external presentation.",
        ("carol",),
    ),
    Question(
        "ub7-high-dose",
        "Did the high dose of UB-7 work better than the middle dose?",
        (TRIAL,),
        "No: the high dose showed no further benefit.",
        ("carol",),
    ),
    # Umbra, annual budget (group:finance)
    Question(
        "research-share",
        "What share of the annual budget goes to research?",
        (ANNUAL_BUDGET,),
        "60%.",
        ("dave", "carol", "alice"),
    ),
    Question(
        "board-approval",
        "Which equipment purchases need board approval?",
        (ANNUAL_BUDGET,),
        "Purchases above 50,000 credits.",
        ("dave",),
    ),
    # Outside the corpus
    Question(
        "parental-leave",
        "What is the parental leave policy?",
        (),
        "",
        ("erin", "carol"),
    ),
    Question("ceo", "Who is the CEO of Acme Logistics?", (), "", ("alice",)),
    Question("wifi", "What is the office Wi-Fi password?", (), "", ("bob",)),
    Question("capital", "What is the capital of France?", (), "", ("dave",)),
    Question("q4-hiring", "How many roles are approved for hiring in Q4?", (), "", ("alice",)),
)


def cases(users: frozenset[str] | None = None) -> list[Case]:
    """Every (question, asker) pair, in order; only `users`' if given."""
    return [
        Case(question, USERS[name])
        for question in QUESTIONS
        for name in question.askers
        if users is None or name in users
    ]
