"""FastAPI dependency that turns the request's bearer token into a Principal.

Placeholder: JWT validation is not implemented yet, so every request is
rejected with 401 (fail closed). The auth module will replace the body of
`get_principal` and keep its return type; routes and the tenancy dependency
only rely on that.
"""

from fastapi import HTTPException, status

from ragmt.domain import Principal


async def get_principal() -> Principal:
    """Return the verified caller. Until JWT validation exists, always 401."""
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Not authenticated",
        headers={"WWW-Authenticate": "Bearer"},
    )
