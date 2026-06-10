# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

`rag-receipts` is an adaptive RAG engine: every query is routed between a fast path and a
deliberative agentic loop, and every pipeline component earns a **receipt** — its measured
contribution on labeled benchmark slices, shown next to the published anchor it is compared
against (direction-match only, never conflated). A FastAPI backend (`api/`) serves a Next.js
frontend (`web/`). See `README.md` for the product framing and the honesty notes.

## Commands

Python is **3.12 via uv**. The system interpreter is a different version and will break things —
**always** prefix Python/test/lint with `uv run`. Tests are fully offline ($0, zero keys).

```bash
# --- api (run from api/) ---
cd api && uv sync                       # install/lock deps
uv run pytest                           # full suite, offline, no keys (add -q for quiet)
uv run pytest tests/test_metrics.py::test_recall_single_gold_hit_in_top5 -v   # single test
uv run pytest -k graph                  # by keyword
uv run ruff check .                     # lint (E, F, I, UP; line-length 100)
uv run ruff format .                    # format

# run the API server — MUST be single-worker, MUST be `python -m uvicorn` (see invariants)
uv run python -m uvicorn ragreceipts.server.app:app --port 8000 --workers 1
TESTING=1 uv run python -m uvicorn ragreceipts.server.app:app --port 8000 --workers 1  # fake vendors

# --- web (run from web/) ---
cd web && pnpm install && pnpm dev      # localhost:3000 (Turbopack)
pnpm e2e                                # Playwright; boots the api in TESTING=1, fully offline
pnpm lint                               # eslint

# --- regenerate the typed client after changing any endpoint/model ---
cd api && uv run python -m ragreceipts.server.export_openapi > ../web/openapi.json
cd ../web && pnpm gen:api                # openapi-typescript -> src/lib/api/schema.d.ts

# --- full stack ---
docker compose up --build               # qdrant :6333, api :8000, web :3000 (.env -> api only)

# --- data / eval CLI (the `ragreceipts` entry point; ingest+eval need real keys) ---
uv run --project api python scripts/download_data.py --corpus all   # offline HF slices, no keys (repo root)
cd api && uv run ragreceipts ingest --corpus musique-dev-300         # build BM25 + dense indexes
uv run ragreceipts eval --corpus musique-dev-300 --slice smoke --presets bm25-only,dense-rrf,rerank --spend-cap-usd 2.50
uv run ragreceipts receipts promote <run_id>                         # strip text/answers -> receipts/
uv run python scripts/build_graph.py --corpus musique-dev-300        # keyed: build the graph artifact
```

## Architecture: two planes over one core

There are two execution planes, and they deliberately share the same retrieval core and the
same query entry point so a receipt measures exactly what the server runs:

- **Serving plane:** `server/app.py` → `agents/service.py::run_query` → the LangGraph state
  machine (`agents/graph.py`) → `RetrievalCore`.
- **Eval plane:** `eval/runner.py::AblationRunner` → the *same* `run_query` → the *same*
  `RetrievalCore`, looped over a preset × slice grid, emitting receipts.

### RetrievalCore + the Retriever protocol (`retrieval/`)
`RetrievalCore.retrieve(query)` is the **single retrieval entry point**. It is parameterized by a
`PipelineConfig` and composes up to four pieces, each implementing the one binding `Retriever`
protocol — `search(query, k) -> list[ScoredChunk]`, descending score, `chunk_id` tie-break:
- `SparseRetriever` (BM25 via `bm25s`), `DenseRetriever` (Qdrant **named vectors**:
  `contextual` vs `isolated`), `GraphRetriever` (graph mode, below), and a `RerankStage`.
- More than one enabled retriever → `rrf_fuse` (reciprocal rank fusion, `RRF_K=60`,
  `source="rrf"`); a single retriever passes through keeping its own source label.
- **Honest degradation is the headline behavior:** if a retriever raises `VendorUnavailable`,
  the core drops the most-fragile one first (graph, then dense), retries, and appends a
  `graph-skipped` / `dense-skipped` / `rerank-skipped` flag to the trace's `degraded` list.
  It only re-raises when nothing survives. Nothing degrades silently.

### Vendor seam (`vendors/`) and offline testing
Network access lives behind Protocols in `vendors/base.py` — `EmbedTransport`, `RerankTransport`,
`ClaudeTransport`, `OpenIETransport` — implemented by `{anthropic,voyage,cohere}_client.py`.
**No application code imports a vendor SDK outside its client module.** `retry.py::call_with_retry`
wraps all retryable failures and raises the single `VendorUnavailable` exception. Tests inject
fakes from `tests/fakes.py` (`FakeEmbed`/`FakeRerank`/`FakeClaude`/`FakeOpenIE`) plus
`QdrantClient(":memory:")`, so the whole suite runs with no keys and no network. Live vendor
scripts (`scripts/live_smoke_ingest.py`, `scripts/smoke_s2.py`, `scripts/build_graph.py`) are
**manual-only, never CI**.

### Config and the preset ladder (`config.py`, `constants.py`)
`PipelineConfig = IngestConfig + QueryConfig`. **Ingest flags** (`contextual`, `chunk_size`)
select which index variant is *built*; **query flags** (`bm25`/`dense`/`rerank`/`graph`/
`graph_recognition`/`route_mode`/`top_k_*`) gate code paths over the *same* index. `PRESETS` is the
binding ablation ladder, in order:
`bm25-only → dense-rrf → contextual → rerank → graph → graph-rrf → router-on`.
All model IDs, thresholds, and budgets are centralized in `constants.py` (e.g. `ROUTER_MODEL`,
`SYNTH_MODEL`, `ROUTE_CONFIDENCE_THRESHOLD=0.7`, `S2_MAX_HOPS=3`, `S2_TOKEN_CEILING=50_000`, and the
graph constants `PPR_DAMPING`/`GRAPH_BLEND`/`SYNONYM_THRESHOLD`/…) — receipts pin these values.

### System-1 / System-2 router (`agents/`)
A LangGraph state machine. The router (Haiku, temperature 0, `messages.parse → RouteDecision`)
classifies each query and reports a confidence. **Confidence is consumed, not decorative:** below
`ROUTE_CONFIDENCE_THRESHOLD` escalates to System-2 even when the route says "simple".
- **System-1:** retrieve → synthesize (one Sonnet pass).
- **System-2:** decompose → (retrieve-hop → CRAG-style grade → refine)* → synthesize, hard-bounded
  by `S2_MAX_HOPS` and a global `S2_TOKEN_CEILING` summed across every Claude call.
- **State-enforced disclosure:** `contradiction_flag` and `unresolved_subqueries` are OR'd/union'd
  from state, never trusted to the model alone; budget exhaustion is a state decision.
- Pydantic schemas in `agents/schemas.py` are **prompt-contract-binding** — field names appear in
  `agents/prompts.py`; change both in lockstep and bump `PROMPTS_VERSION` (recorded in receipts).
- `route_mode` (`AUTO`/`FORCE_S1`/`FORCE_S2`) is consumed by the agent graph, **not** by
  `RetrievalCore` (the core is route-agnostic). Tests inject `FakeCore` (a `.retrieve` double).

### Graph mode — HippoRAG-2 (`retrieval/graph.py`, `retrieval/graph_ppr.py`, `ingest/graph_index.py`, `agents/openie.py`)
A graph-retrieval variant behind the same `Retriever` protocol. Build (keyed, offline-tested with
`FakeOpenIE`): LLM OpenIE triples → a phrase+passage knowledge graph (relation / appears-in /
synonym edges, `cosine ≥ SYNONYM_THRESHOLD`) → a **byte-reproducible** artifact whose hash flows
into `manifest.index_hashes["graph"]`. Query: embed the query → cosine to phrase+passage nodes →
optional LLM "recognition memory" filter on phrase seeds (`recognition="llm"|"embedding"`) →
query-seeded **Personalized PageRank** (deterministic scipy power iteration) → blend PPR with each
passage's dense cosine. The retriever is self-contained — vectors live in the artifact, so no
Qdrant at query time. Node ordering is invariant: passages `[0, n_passage)` then phrases; all
stored vectors are L2-normalized so `vectors @ qvec` is true cosine. The graph "third route" in the
router and the two-sided "when do graphs help" receipt are the Phase-2 eval/web layer (Plan F).

### Eval, receipts, and honesty (`eval/`)
`AblationRunner` runs each `(preset, corpus)` cell through `run_query`, accumulating SQLite results
(`run_state.py`, resumable), then assembles a `Receipt`. Key invariants:
- **Cost discipline:** every run estimates cost first (`estimate_run_cost`), gates on a **hard
  `spend_cap_usd`** checked before each query, and is resumable by `run_id` (completed queries are
  never re-billed). RAGAS judge spend is heuristically estimated but excluded from the hard cap and
  disclosed per-query (`ragas_judge_usd_untracked`).
- **Anchors are direction-match only:** `ANCHOR_SPECS` defines published deltas vs a baseline
  preset; the runner compares *sign*, never magnitude, and every `PublishedAnchor.note` carries
  machine-readable comparability caveats (domain/technique/architecture mismatch). `nq-dev-300`
  auto-appends a corpus-scale note.
- **Pinning:** a receipt records the full config, the *subset* of `index_hashes` actually used,
  model IDs, `pricing_table_version`, `prompts_version`, and per-query metrics. Retrieval metrics
  (Recall@5, MRR@3 via `eval/alignment.py::is_hit`) are deterministic; answer metrics (EM/F1/RAGAS)
  carry a nondeterminism note. Multi-hop datasets gate: `router-on`/graph presets are **skipped
  with a disclosed reason** on single-hop corpora, never faked.
- **`strip_for_commit`:** committed `receipts/*.json` keep IDs + metrics only — never passage text
  or model answers (benchmark redistribution terms). Promote local runs with `receipts promote`.

### Server (`server/`)
- **Single-worker is a hard invariant.** Background ingest/eval jobs run on an in-process daemon
  thread keyed by SQLite rows (`server/jobs.py`); extra uvicorn workers would orphan queued jobs.
  Run with `--workers 1`, and as `python -m uvicorn` (not bare `uvicorn`) so `api/` is on `sys.path`
  for the `TESTING=1` seam to import the tests package.
- **Composition root:** `server/deps.py::build_deps` wires `AppDeps`. If `QDRANT_URL` or any vendor
  key is missing, the affected runner is left `None` and its endpoints return **503 naming the
  missing env var** — never a silent default, never a stack trace. `/health` reports per-vendor
  capability without needing any key.
- `TESTING=1` swaps the whole container for `tests/e2e_fixture.py::build_testing_deps` (in-memory
  Qdrant, fixture receipts, fake vendors) — this is how `pnpm e2e` stays offline.
- The `Real*` runners (`RealQueryRunner`/`RealEvalRunner`/`RealIngestSink`) are thin pins over the
  Plan A/B/C entry points with constructor seams for test injection.
- **Eval endpoint is estimate → confirm → run:** `POST /eval/runs` returns `needs_confirmation`
  with a cost estimate unless `confirm:true` and the estimate is within the cap.

### Web (`web/`)
Next.js 15 / React 19 client components, no Tailwind (theme in `globals.css`). The API contract is
**code-generated**: `web/src/lib/api/schema.d.ts` is produced from `openapi.json` by `pnpm gen:api`
and consumed via `openapi-fetch` — never hand-edit it; regenerate after endpoint/model changes.
Three pages: **Playground** (query → cited answer + hop-by-hop `TraceViewer` + degraded badges),
**Ablation Lab** (receipts vs published anchors with direction-match badges, recharts), **Corpora**
(manifests + BYO upload via `UploadForm`, which polls `/jobs/{id}`). Playwright (`playwright.config.ts`)
boots both servers — the api in `TESTING=1` — for hermetic e2e.

### Data layout
`data/` (gitignored): `corpora/<id>/` holds `raw/` (committed-elsewhere benchmark slices),
`chunks.jsonl` (canonical chunk order — sparse rows and Qdrant payloads index into it),
`sparse/`, `graph/`, and `manifest.json`; `receipts-local/` holds run envelopes until promoted.
Paths resolve from `RAGRECEIPTS_DATA_DIR` (default `../data` from `api/`).

## Conventions and invariants

- **`uv run` everything** Python-side; never invoke the system interpreter.
- **`RetrievalCore` is the only retrieval entry point; every Claude call goes through
  `ClaudeTransport`.** Keep all vendor SDK usage inside `vendors/`.
- **Offline/$0 tests are non-negotiable** — add a fake before adding a dependency on a real vendor;
  keyed work goes in a manual script + a runbook under `docs/runbooks/`, never CI.
- **Receipts must stay honest** — pin config/hashes/versions, keep anchors direction-match with
  caveats, and `strip_for_commit` before committing any receipt.
- **No `Co-Authored-By: Claude` trailer** on commits in this repo (it is a public portfolio repo;
  sole-author attribution). This overrides the default commit-message instruction.
- **Model IDs / pricing / SDK usage:** before writing or editing Anthropic integration code,
  consult the `claude-api` skill rather than memory; the current IDs live in `constants.py`.
- `docs/superpowers/` (design specs, implementation plans) is **gitignored** local planning
  scaffolding — it is the source of truth for in-progress work but is never committed.
```
