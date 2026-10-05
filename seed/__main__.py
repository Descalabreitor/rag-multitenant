"""`python -m seed` (or `make seed`): load the fictional corpus as app_ingest."""

import asyncio

from sqlalchemy.ext.asyncio import create_async_engine

from ragmt.settings import get_settings
from seed.load import load


async def main() -> None:
    settings = get_settings()
    engine = create_async_engine(settings.ingest_database_url.get_secret_value())
    try:
        for result in await load(engine, settings.embedding_dim):
            print(
                f"{result.tenant}: {result.memberships} memberships, "
                f"{result.documents} documents, {result.chunks} chunks"
            )
    finally:
        await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
