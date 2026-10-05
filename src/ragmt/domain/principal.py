"""The authenticated caller, as the rest of the application sees it."""

from dataclasses import dataclass
from uuid import UUID


@dataclass(frozen=True, slots=True)
class Principal:
    """Identity from a verified token: who the user is and which tenant they act in.

    Groups are deliberately absent. They are resolved from `memberships` inside the
    database on every request (ADR 0002), never taken from the token.
    """

    sub: str
    tenant_id: UUID


# Members of the Keycloak group /<org alias>/admins, which permsync stores as this
# group name, are the tenant's admins: only they upload, change ACLs and delete
# (ADR 0008). Like every group, it is read from `memberships`, not from the token.
TENANT_ADMIN_GROUP = "admins"
