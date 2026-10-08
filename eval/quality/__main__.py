"""`python -m eval.quality` (or `make quality`): run the questions through POST /ask
with the real models and score the answers.

Needs the real stack, as `make e2e` does: Keycloak with KC_DEV_PASSWORD_CLIENT=true,
PostgreSQL migrated, and Ollama with OLLAMA_EMBED_MODEL and OLLAMA_CHAT_MODEL
pulled (and the judge model, if it is another one). Not run in CI.

1. **Re-embed the seed with Ollama.** The seed deduplicates by file hash, so a
   seed loaded with fake embeddings (the e2e tests, CI, `make seed` with
   LLM_PROVIDER=fake) would keep its meaningless vectors. Every document of the
   seed tenants is hard-deleted as app_ingest, then the seed is loaded again with
   OLLAMA_EMBED_MODEL, then permsync runs once. `--no-reseed` skips this.
2. **Ask.** The API at EVAL_API_URL, or one started here with LLM_PROVIDER and
   CHAT_PROVIDER set to ollama; other settings (RETRIEVAL_K...) come from the
   environment or .env. Each case gets a fresh token for its user from the dev
   password grant.
3. **Find what was retrieved.** The ask row in the audit trail, read as the
   tenant's admin, holds the retrieved chunk ids; their text is read as
   app_ingest (the API never returns chunk text). That text is the judge's
   context for faithfulness.
4. **Judge** (eval/quality/judge.py, configured by QUALITY_JUDGE_*), and score
   (eval/quality/scoring.py).

Raw results go to docs/results/raw/quality-<UTC stamp>/ (git-ignored): runs.json
and summary.md. docs/results/quality.md is written from them by hand.
"""

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import httpx
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from eval.quality.judge import Judge, JudgeConfig, JudgeError, plain_abstention
from eval.quality.questions import Case, cases
from eval.quality.scoring import CaseResult, summary
from eval.revocation import RAW_DIR, ROOT, Keycloak, api_server, env, load_dotenv
from ragmt.audit.events import question_sha256
from ragmt.domain import NO_CONTEXT_ANSWER
from ragmt.ingest.chunking import chunk
from ragmt.ingest.convert import UploadConverter
from ragmt.ingest.service import IngestService
from ragmt.permsync.__main__ import run as run_permsync
from ragmt.settings import Settings
from ragmt.tenancy import tenant_session
from seed.corpus import TENANTS
from seed.embedder import SeedError, open_embedder
from seed.load import load as load_seed

# A completion from an 8B model on a CPU can pass the 120 s default.
CHAT_TIMEOUT_SECONDS = 300
API_OVERRIDES = {
    "LLM_PROVIDER": "ollama",
    "CHAT_PROVIDER": "ollama",
    "CHAT_TIMEOUT_SECONDS": str(CHAT_TIMEOUT_SECONDS),
}
# The API embeds a probe at startup, which loads the model into Ollama first.
STARTUP_SECONDS = 180


# --- setup ----------------------------------------------------------------------------


def require_models(settings: Settings, judge: JudgeConfig) -> None:
    wanted = {settings.ollama_embed_model, settings.ollama_chat_model}
    if judge.provider == "ollama":
        wanted.add(judge.model)
    try:
        response = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=10)
        response.raise_for_status()
    except httpx.HTTPError:
        sys.exit(f"Ollama is not reachable at {settings.ollama_base_url}: `make quality` starts it")
    pulled = {m["name"] for m in response.json().get("models", [])}
    missing = [m for m in sorted(wanted) if m not in pulled and f"{m}:latest" not in pulled]
    if missing:
        sys.exit(f"not pulled in Ollama: {', '.join(missing)} (`docker compose up ollama-pull`)")


async def reseed(settings: Settings) -> None:
    """Every seed document deleted, then the seed loaded with Ollama embeddings."""
    try:
        embedder = await open_embedder(settings)
    except SeedError as exc:
        sys.exit(f"seed: {exc}")
    engine = create_async_engine(env("INGEST_DATABASE_URL"))
    try:
        service = IngestService(
            engine, UploadConverter(settings.ingest_max_bytes), chunk, embedder, settings
        )
        for tenant in TENANTS:
            async with tenant_session(engine, tenant.id) as conn:
                ids: list[UUID] = list(
                    (
                        await conn.execute(
                            text("SELECT id FROM documents WHERE tenant_id = :t"),
                            {"t": tenant.id},
                        )
                    ).scalars()
                )
            for document_id in ids:
                await service.hard_delete(tenant.id, tenant.uploader, document_id)
        for loaded in await load_seed(engine, embedder, settings):
            print(f"  {loaded.tenant}: {len(loaded.documents)} documents, {loaded.chunks} chunks")
    finally:
        await engine.dispose()
        await embedder.aclose()
    if await run_permsync(settings, once=True) != 0:
        sys.exit("permsync cycle failed")


# --- one case -------------------------------------------------------------------------


def bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


async def ask_row(http: httpx.AsyncClient, keycloak: Keycloak, case: Case) -> dict[str, Any]:
    """The newest ask row for this user and question, as the tenant's admin sees it."""
    token = await asyncio.to_thread(keycloak.user_token, case.user.admin.username)
    response = await http.get("/audit", headers=bearer(token), params={"limit": 50})
    response.raise_for_status()
    digest = question_sha256(case.question.text)
    for event in response.json()["events"]:
        if (
            event["action"] == "ask"
            and event["actor_sub"] == case.user.sub
            and event["details"].get("question_sha256") == digest
        ):
            row: dict[str, Any] = event
            return row
    raise LookupError(f"{case.id}: no audit row")


async def retrieved_chunks(
    engine: AsyncEngine, tenant_id: UUID, chunk_ids: list[str]
) -> list[tuple[str, str]]:
    """(title, text) of each retrieved chunk, in audit order, read as app_ingest."""
    if not chunk_ids:
        return []
    async with tenant_session(engine, tenant_id) as conn:
        rows = await conn.execute(
            text(
                "SELECT c.id, d.title, c.heading, c.content FROM chunks c"
                " JOIN documents d ON d.id = c.document_id WHERE c.id = ANY(:ids)"
            ),
            {"ids": [UUID(i) for i in chunk_ids]},
        )
        by_id = {str(r.id): r for r in rows}
    found = []
    for chunk_id in chunk_ids:
        row = by_id[chunk_id]
        heading = f" > {row.heading}" if row.heading else ""
        found.append((row.title, f"{row.title}{heading}\n{row.content}"))
    return found


async def run_case(
    case: Case,
    http: httpx.AsyncClient,
    keycloak: Keycloak,
    engine: AsyncEngine,
    judge: Judge,
) -> CaseResult:
    token = await asyncio.to_thread(keycloak.user_token, case.user.username)
    started = time.monotonic()
    response = await http.post(
        "/ask",
        headers=bearer(token),
        json={"question": case.question.text},
        timeout=CHAT_TIMEOUT_SECONDS + 30,
    )
    latency = time.monotonic() - started
    response.raise_for_status()
    body = response.json()

    row = await ask_row(http, keycloak, case)
    chunks = await retrieved_chunks(engine, case.user.tenant.id, row["chunk_ids"])
    retrieved = list(dict.fromkeys(title for title, _ in chunks))
    result = CaseResult(
        case=case.id,
        question=case.question.text,
        user=case.user.username,
        kind=case.kind,
        expected=sorted(case.expected),
        answer=body["answer"],
        model=body["model"],
        latency_s=round(latency, 2),
        cited=list(dict.fromkeys(c["title"] for c in body["citations"])),
        retrieved=retrieved,
        chunk_ids=row["chunk_ids"],
    )
    try:
        if body["answer"] == NO_CONTEXT_ANSWER or plain_abstention(body["answer"]):
            verdict_abstains, verdict_correct = True, not case.answerable
        else:
            verdict = await judge.verdict(
                case.question.text, case.question.reference, body["answer"]
            )
            verdict_abstains = verdict.abstains
            verdict_correct = verdict.correct if case.answerable else verdict.abstains
        result.abstained = verdict_abstains
        result.correct = verdict_correct
        if not verdict_abstains:
            faith = await judge.faithfulness(body["answer"], [t for _, t in chunks])
            result.supported_claims, result.claims = faith.supported, faith.claims
            result.unsupported = list(faith.unsupported)
    except JudgeError as exc:
        result.judge_error = str(exc)
    return result


# --- main -----------------------------------------------------------------------------


async def evaluate(args: argparse.Namespace, raw: Path) -> list[CaseResult]:
    judge_config = JudgeConfig.from_env()
    settings = Settings(llm_provider="ollama", chat_provider="ollama")
    require_models(settings, judge_config)
    selected = cases(frozenset(args.users.split(",")) if args.users else None)

    if not args.no_reseed:
        print("re-embedding the seed with Ollama, then one permsync cycle", flush=True)
        await reseed(settings)

    keycloak = Keycloak(env("OIDC_ISSUER"))
    judge = Judge(judge_config.build())
    engine = create_async_engine(env("INGEST_DATABASE_URL"))
    results: list[CaseResult] = []
    try:
        with api_server(raw / "api.log", API_OVERRIDES, STARTUP_SECONDS) as url:
            async with httpx.AsyncClient(base_url=url, timeout=30) as http:
                for n, case in enumerate(selected, 1):
                    result = await run_case(case, http, keycloak, engine, judge)
                    results.append(result)
                    print(
                        f"  [{n:>2}/{len(selected)}] {case.id:<28} {result.latency_s:6.1f} s  "
                        f"cited={result.cited} idk={result.abstained} "
                        f"correct={result.correct} "
                        f"faithful={result.supported_claims}/{result.claims}",
                        flush=True,
                    )
    finally:
        keycloak.close()
        await judge.aclose()
        await engine.dispose()
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval.quality", description=__doc__)
    parser.add_argument("--users", default="", help="only these askers, comma-separated")
    parser.add_argument(
        "--no-reseed", action="store_true", help="keep the stored seed embeddings as they are"
    )
    args = parser.parse_args(argv)

    load_dotenv(ROOT / ".env")
    os.chdir(ROOT)  # Settings reads .env from the working directory
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    raw = RAW_DIR / f"quality-{stamp}"
    raw.mkdir(parents=True, exist_ok=True)
    print(f"raw output in {raw}", flush=True)

    results = asyncio.run(evaluate(args, raw))

    settings = Settings()
    judge = JudgeConfig.from_env()
    setup = {
        "started": stamp,
        "chat_model": settings.ollama_chat_model,
        "embed_model": settings.ollama_embed_model,
        "retrieval_k": settings.retrieval_k,
        "judge": judge.describe(),
        "api": os.environ.get("EVAL_API_URL") or "started here",
    }
    (raw / "runs.json").write_text(
        json.dumps({"setup": setup, "cases": [asdict(r) for r in results]}, indent=2),
        encoding="utf-8",
    )
    header = ", ".join(f"{k}: {v}" for k, v in setup.items())
    text_summary = f"# Quality baseline run\n\n{header}\n\n{summary(results)}"
    (raw / "summary.md").write_text(text_summary, encoding="utf-8")
    print("\n" + text_summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
