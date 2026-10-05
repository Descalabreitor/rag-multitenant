"""`python -m seed` (or `make seed`): load the fictional corpus as app_ingest.

The embedder is the one LLM_PROVIDER names. "fake" (CI, tests) gives
hash-derived vectors: fine for checking who sees what, meaningless for ranking.
"""

import asyncio
import sys

from sqlalchemy.ext.asyncio import create_async_engine

from ragmt.settings import get_settings
from seed.embedder import SeedError, open_embedder
from seed.load import load


async def main() -> None:
    settings = get_settings()
    embedder = await open_embedder(settings)
    if settings.llm_provider == "fake":
        print(
            "LLM_PROVIDER=fake: embeddings are hash-derived, meaningless for ranking.",
            file=sys.stderr,
        )
    engine = create_async_engine(settings.ingest_database_url.get_secret_value())
    try:
        for result in await load(engine, embedder, settings):
            print(
                f"{result.tenant}: {result.memberships} memberships, "
                f"{len(result.documents)} documents, {result.chunks} chunks, "
                f"{result.changes} changes"
            )
    finally:
        await engine.dispose()
        await embedder.aclose()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except SeedError as exc:
        sys.exit(f"seed: {exc}")
