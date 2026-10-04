"""tenant_session rejects bad context before it touches the database."""

from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import create_async_engine

from ragmt.tenancy import tenant_session

# Creating an engine doesn't connect; these tests never get that far.
ENGINE = create_async_engine("postgresql+asyncpg://nobody@127.0.0.1:1/none")


async def test_tenant_id_must_be_a_uuid() -> None:
    with pytest.raises(TypeError):
        async with tenant_session(ENGINE, str(uuid4())):  # type: ignore[arg-type]
            pass


async def test_user_sub_must_not_be_empty() -> None:
    with pytest.raises(ValueError, match="user_sub"):
        async with tenant_session(ENGINE, uuid4(), ""):
            pass
