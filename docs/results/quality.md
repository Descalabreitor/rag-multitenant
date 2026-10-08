# Answer-quality baseline

> **This is a baseline, not a claim of quality.** It is the reference point for project 2 (semantic caching and model routing): a cache or a cheaper model should leave these numbers where they are. The corpus has 7 short, made-up documents (17 chunks) and the questions were written by the same person who wrote the documents, so most of them are easy lookups. A small local model also grades its own answers. Treat the numbers as a regression check on this setup, not as evidence that the system answers well in general.

Run `20261007T222210Z` (raw output in `docs/results/raw/quality-20261007T222210Z/`, git-ignored). `make quality` reproduces it. Method: `eval/quality/`.

## Setup

- **Host:** AMD Ryzen 7 260, 16 logical CPUs, 31.3 GiB RAM, Windows 11; Docker Desktop. Ollama 0.35.1 in the compose container, **CPU only** (run without the `compose.gpu.yaml` override, so the laptop's RTX 5060 was not used).
- **Service:** the API started by the runner with `LLM_PROVIDER=ollama`, `CHAT_PROVIDER=ollama`, `CHAT_TIMEOUT_SECONDS=300`; everything else from `.env` at its defaults: `RETRIEVAL_K=5`, `HNSW_EF_SEARCH=40`, `HNSW_ITERATIVE_SCAN=relaxed_order`, `ASK_MAX_CONTEXT_CHARS=12000`. The prompt is the one in `ragmt.generation` (ADR 0009), temperature 0.
- **Models:** embeddings `nomic-embed-text` (768 dims), chat `llama3.1:8b` (Q4_K_M).
- **Corpus:** the seed (`seed/corpus.py`): Acme (4 documents, 10 chunks) and Umbra (3 documents, 7 chunks). The runner hard-deletes the seed tenants' documents and loads them again with Ollama embeddings first. The seed deduplicates by file hash, so a seed that was loaded with fake vectors (CI, e2e) would otherwise keep them.
- **Questions:** 23 questions (`eval/quality/questions.py`), each with its supporting documents and a short reference answer, asked by 1 to 4 of the five seed users: **39 cases**. Whether a case is answerable is not written by hand. It follows from the seed ACLs and memberships, by the same rules RLS applies: answerable if the user may read at least one supporting document. Unanswerable cases are split by why:
  - *denied by ACL*: a document in the user's tenant answers it, but its ACL excludes the user (e.g. bob, the Acme admin, asking about the finance-only Q3 budget);
  - *other tenant*: only the other tenant's documents answer it (e.g. dave, Umbra finance, asking about Acme's Q3 budget: same group name, other tenant);
  - *out of corpus*: no document answers it (parental leave, the CEO, the capital of France).
- **Judge:** `llama3.1:8b` through Ollama, the same model that answers (`QUALITY_JUDGE_PROVIDER`, `QUALITY_JUDGE_MODEL`, `QUALITY_JUDGE_BASE_URL`, `QUALITY_JUDGE_API_KEY` switch it to another model or to an OpenAI-compatible API). RAGAS isn't used: it would add a large dependency tree for two prompts that are short enough to own, and it isn't built for an offline 8B judge.

## How each number is measured

- **Citations** are the documents in the `citations` of the `POST /ask` response, compared with the case's expected documents by title. *Precision*: the share of cited documents that are expected (only for answers that cite something). *Recall*: the share of expected documents that are cited.
- **Retrieved:** the runner finds the case's row in `GET /audit`, read as the tenant's admin, by actor and question SHA-256. It reads the text of those chunk ids as app_ingest, because the API never returns chunk text. *Expected document retrieved* is the recall of the expected documents among the retrieved chunks. It separates "not retrieved" from "retrieved but not cited".
- **"I don't know":** a reply that is only one short "I don't know" sentence counts as abstaining without asking the judge. In a first trial, the 8B judge called "I don't know." a non-abstention and listed facts from the context as the answer's claims. Any other reply goes to the judge (verdict prompt: does it abstain, and does it give the reference answer's facts). An unanswerable case is right only if the answer abstains.
- **Faithfulness:** for answers that don't abstain, the judge lists the answer's claims and marks each one as supported or not by the retrieved chunk text, i.e. exactly what the model saw. Citation markers are removed first.

In this run, **21 of the 39 answers were exactly "I don't know."** and were scored by the rule. The judge decided the other **18** (correctness and faithfulness). I read those 18 answers by hand and agree with every verdict. At this size the judge confirms more than it measures.

## Results

### Answerable (20 cases)

| Metric | Value |
|---|---:|
| Expected document retrieved (doc recall in top-5) | 0.90 |
| Citation precision (mean over the 18 answers that cite) | 1.00 |
| Citation recall (mean) | 0.90 |
| Answers citing nothing | 2/20 |
| False "I don't know" | 2/20 |
| Correct vs reference (judge) | 18/20 |
| Faithfulness, mean supported-claim ratio (judge, 18 answers) | 1.00 |
| Fully faithful answers (judge) | 18/18 |

### Unanswerable (19 cases)

| Why unanswerable | Cases | Said "I don't know" | Cited a document |
|---|---:|---:|---:|
| denied by ACL | 8 | 8/8 | 0/8 |
| other tenant | 5 | 5/5 | 0/5 |
| out of corpus | 6 | 6/6 | 0/6 |
| **all** | 19 | 19/19 | 0/19 |

None of the 39 citations named a document outside the asking user's ACL. That is guaranteed by RLS and by citations being built from the retrieved set (ADR 0009), not by the model; the leak suite tests it, and here it is only observed.

### Latency

`POST /ask` over 39 calls, end to end (question embedding, retrieval, audit, one completion, CPU only): median **8.5 s**, p95 **18.5 s**, max 22.9 s. The first call includes loading the chat model. "I don't know" answers took 1.8 to 11.5 s; the 3-step restart procedure was the slowest (22.9 s). Latency here is mostly the CPU completion and says little about a GPU or a hosted model.

## What the misses are

Both wrong answers are **retrieval misses**, not generation errors:

- `drones/alice` ("How much of the Q3 fleet budget goes to replacing delivery drones?") and `planners/alice` ("How many planner roles are approved for Q3?") retrieved only chunks of the handbook and Alice's review: the Q3 budget, which alice may read, was not in her top 5. Given that context, "I don't know." was the correct reply, and the model did not guess.
- For alice, 5 of her 8 readable chunks are retrieved, and both questions lost to the handbook and the review. With 17 chunks, the gap is in ranking (short chunks with a heading, a sentence and a `Reference:` canary line), not in RLS overfiltering: `relaxed_order` returns k rows. Larger k, hybrid search (ADR 0009, alternatives) or chunk text without the canary line are the obvious things to measure next. None of them is in scope here.

Two smaller observations that the scores don't show:

- Two answers say "According to [doc:…], …", using the marker as a noun. The citation is valid, so the response is fine, but the sentence reads oddly once a client strips markers.
- The model never answered an out-of-corpus question from its own knowledge, the capital of France included.

## Limits

- **Small and easy.** 20 answerable cases over 7 documents. One flipped case moves a rate by 5 points. Every answerable question has a single supporting document, and none needs combining two documents or reasoning over the table in the Umbra budget beyond reading a cell.
- **Self-grading.** The judge is the answering model. A stronger or different judge (`QUALITY_JUDGE_PROVIDER=openai_compat`) is the first thing to change before reading anything into faithfulness. Here it only confirmed 18 short, verifiable answers.
- **One run, temperature 0.** No repeats. Ollama at temperature 0 is close to deterministic on the same hardware, but not guaranteed across versions or with a GPU.
- **Doc-level citations.** Precision and recall compare documents, not chunks or sentences: an answer that cites the right document for the wrong sentence still scores 1.0.
