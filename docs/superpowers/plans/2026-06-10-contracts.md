# rag-receipts — Shared Contracts (binding for all plans)

Every implementation plan MUST use these exact names, signatures, paths, and model IDs.
If a plan needs something not defined here, it defines it in its own tasks — but it must
not redefine or rename anything below. Spec: `docs/superpowers/specs/2026-06-10-rag-receipts-design.md`.

## Tooling

- Python 3.12, managed with **uv** (`uv init`, `uv add`, `uv run pytest`). Package dir: `api/ragreceipts/`, tests: `api/tests/`.
- Lint/format: `ruff` (line length 100). Test: `pytest`.
- Web: **pnpm**, Next.js (App Router), TypeScript, recharts, openapi-typescript v7+ client generated from FastAPI's OpenAPI 3.1.
- All vendor network calls go through `ragreceipts/vendors/` behind Protocols (transport seam) — unit tests use fakes, zero keys in CI.

## Model & vendor constants — `api/ragreceipts/constants.py`

```python
ROUTER_MODEL = "claude-haiku-4-5-20251001"   # routing + CRAG grading, temperature=0
SYNTH_MODEL = "claude-sonnet-4-6"            # answer synthesis
JUDGE_MODEL = "claude-sonnet-4-6"            # RAGAS judge
EMBED_MODEL = "voyage-context-3"             # contextualized chunk embeddings
RERANK_MODEL = "rerank-v4.0-pro"             # Cohere Rerank v4.0 Pro (anchor variant)
RAGAS_EMBED_MODEL = "BAAI/bge-small-en-v1.5" # local sentence-transformers for RAGAS answer-relevancy
ROUTE_CONFIDENCE_THRESHOLD = 0.7             # below this, escalate to System-2
S2_MAX_HOPS = 3
S2_TOKEN_CEILING = 50_000                    # input+output summed across all Claude calls per query
```

## Anthropic SDK usage (binding — verified against claude-api skill 2026-06-10)

- Python SDK `anthropic`, client `anthropic.Anthropic()` (reads `ANTHROPIC_API_KEY`).
- Structured outputs for router/grader: `client.messages.parse(model=..., max_tokens=...,
  messages=[...], output_format=PydanticModel)` → `response.parsed_output`.
- `temperature=0` IS supported on Sonnet 4.6 / Haiku 4.5 (it is removed only on Opus 4.7+).
- Use typed exceptions (`anthropic.RateLimitError`, `anthropic.APIStatusError`); the SDK
  auto-retries 429/5xx with backoff (`max_retries` configurable). Honor `retry-after`.
- `max_tokens`: 1024 for routing/grading, 4096 for synthesis. No assistant prefills.
- Wrap the SDK behind `ClaudeTransport` (below) — application code never imports `anthropic`
  outside `vendors/`.

## Core types — `api/ragreceipts/types.py`

```python
from dataclasses import dataclass, field
from enum import Enum

@dataclass(frozen=True)
class Chunk:
    chunk_id: str          # f"{doc_id}:{position}"
    corpus_id: str
    doc_id: str
    passage_id: str        # parent passage ID for gold alignment (== doc_id when unsegmented)
    text: str
    position: int          # chunk index within document
    start_token: int       # whitespace-token offset within the parent passage (R3)
    end_token: int         # exclusive end offset; persisted to chunks.jsonl + Qdrant payload

@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    score: float
    source: str            # "bm25" | "dense" | "rrf" | "rerank"

class RouteMode(str, Enum):
    AUTO = "auto"
    FORCE_S1 = "force_s1"
    FORCE_S2 = "force_s2"
```

## Pipeline configuration — `api/ragreceipts/config.py`

```python
@dataclass(frozen=True)
class IngestConfig:               # ingest-time flags → index variants
    contextual: bool = True       # True: doc-grouped voyage-context-3; False: per-chunk isolated, same model
    chunk_size: int = 512         # tokens
    chunk_overlap: int = 64

@dataclass(frozen=True)
class QueryConfig:                # query-time flags → same index, different code path
    bm25: bool = True
    dense: bool = True
    rerank: bool = True
    route_mode: RouteMode = RouteMode.FORCE_S1
    top_k_fuse: int = 50          # candidates into RRF / rerank
    top_k_final: int = 5

@dataclass(frozen=True)
class PipelineConfig:
    name: str                     # preset name
    ingest: IngestConfig
    query: QueryConfig

PRESETS: dict[str, PipelineConfig]  # keys, in ladder order:
# "bm25-only"   bm25=T dense=F rerank=F contextual=F force_s1
# "dense-rrf"   bm25=T dense=T rerank=F contextual=F force_s1
# "contextual"  bm25=T dense=T rerank=F contextual=T force_s1
# "rerank"      bm25=T dense=T rerank=T contextual=T force_s1
# "router-on"   bm25=T dense=T rerank=T contextual=T route_mode=AUTO
```

Both dense vector sets are built at every ingest (Qdrant named vectors `"contextual"` and
`"isolated"` on the same points); `IngestConfig.contextual` selects the named vector at
query time and the matching manifest hash for receipts.

## Retrieval protocol — `api/ragreceipts/retrieval/base.py`

```python
from typing import Protocol

class Retriever(Protocol):
    def search(self, query: str, k: int) -> list[ScoredChunk]: ...
```

Implementations: `DenseRetriever` (Qdrant, named-vector selected by `IngestConfig.contextual`),
`SparseRetriever` (bm25s), `HybridRRF(retrievers: list[Retriever], rrf_k: int = 60)`.
Rerank is a stage, not a Retriever: `RerankStage.rerank(query: str, candidates: list[ScoredChunk], top_n: int) -> list[ScoredChunk]`.
RRF score for a chunk: `sum(1 / (rrf_k + rank_i))` over the rank lists containing it (rank is 1-based).
The single composed entry point used by S1, S2, and the eval harness:

```python
# api/ragreceipts/retrieval/core.py
class RetrievalCore:
    def __init__(self, config: PipelineConfig, dense: Retriever | None,
                 sparse: Retriever | None, rerank_stage: "RerankStage | None"): ...
    def retrieve(self, query: str) -> list[ScoredChunk]: ...
    # honors config.query flags; returns top_k_final chunks; emits TraceEvents via callback
```

## Vendor protocols — `api/ragreceipts/vendors/base.py`

```python
class EmbedTransport(Protocol):
    def embed_documents(self, documents: list[list[str]]) -> list[list[list[float]]]:
        """documents = list of docs, each a list of chunk texts (doc-grouped).
        Isolated mode is expressed by passing single-chunk documents."""
    def embed_query(self, query: str) -> list[float]: ...

class RerankTransport(Protocol):
    def rerank(self, query: str, texts: list[str], top_n: int) -> list[tuple[int, float]]:
        """returns (original_index, relevance_score) sorted desc."""

class ClaudeTransport(Protocol):
    def complete(self, *, model: str, system: str, user: str, max_tokens: int,
                 temperature: float = 0.0) -> "ClaudeResult": ...
    def parse(self, *, model: str, system: str, user: str, max_tokens: int,
              output_format: type, temperature: float = 0.0) -> "ParsedResult": ...

@dataclass(frozen=True)
class ClaudeResult:
    text: str
    input_tokens: int
    output_tokens: int

@dataclass(frozen=True)
class ParsedResult:
    parsed: object          # the validated Pydantic instance
    input_tokens: int
    output_tokens: int
```

Real impls: `VoyageClient`, `CohereClient`, `AnthropicClient` in `vendors/`; fakes in
`api/tests/fakes.py` (`FakeEmbed`, `FakeRerank`, `FakeClaude` with scripted responses).
Degraded behavior (spec): rerank failure → RRF order + `degraded:"rerank-skipped"`;
embed failure at query time → BM25-only + `degraded:"dense-skipped"`; Claude failure → raise.

## Traces — `api/ragreceipts/traces/models.py`

```python
@dataclass(frozen=True)
class TraceEvent:
    trace_id: str          # one per query
    seq: int               # ordering within trace
    node: str              # "route"|"s1_retrieve"|"s1_answer"|"decompose"|"retrieve_hop"|"grade"|"refine"|"synthesize"
    payload: dict          # JSON-serializable inputs/outputs/scores/flags
    model: str | None      # model ID if a Claude call happened
    input_tokens: int
    output_tokens: int
    duration_ms: float
```

Store: SQLite (WAL mode) via `traces/store.py` — `TraceStore.append(event)`, `TraceStore.get(trace_id) -> list[TraceEvent]`.

## Corpus manifest — emitted by every ingest, `data/corpora/{corpus_id}/manifest.json`

```json
{
  "corpus_id": "musique-dev-300",
  "dataset": {"name": "musique", "hf_id": "...", "split": "...", "revision": "..."},
  "chunking": {"chunk_size": 512, "chunk_overlap": 64},
  "embed_model": "voyage-context-3",
  "index_hashes": {"dense_contextual": "sha256:...", "dense_isolated": "sha256:...", "sparse": "sha256:..."},
  "tokenizer_artifact": "bm25s tokenizer vocab path, hashed into sparse hash",
  "n_docs": 0, "n_chunks": 0, "n_queries": 0,
  "created_at": "ISO8601"
}
```

## Metrics (binding definitions, from spec)

- A retrieved chunk is a **hit** for a gold passage iff `chunk.passage_id == gold.passage_id`;
  for span-format golds (NQ long answers), hit iff the chunk covers ≥50% of the gold span's tokens.
- `recall_at_5 = |golds with ≥1 hit in top-5| / |golds|` per query, averaged over queries.
- `mrr_at_3` = reciprocal rank of the first hit within top-3 (0 if none), averaged.
- `router-on` cell: primary metrics are answer-level EM/F1 vs gold answers + RAGAS; retrieval
  recall over the union of per-hop top-5 is a secondary diagnostic flagged `union_of_hops: true`.
- Abstentions (`abstained=true`) excluded from RAGAS, reported as `n_abstained`.
- Failures excluded from metrics, reported as `n_failed`. Never silently dropped.

## receipts.json schema — `api/ragreceipts/eval/receipts.py`

```python
@dataclass(frozen=True)
class PublishedAnchor:
    source: str             # e.g. "arXiv 2604.01733 Table I"
    published_value: float
    measured_value: float
    direction_match: bool
    note: str               # REQUIRED — comparability caveats (domain/technique mismatch)

@dataclass(frozen=True)
class Receipt:
    run_id: str
    corpus_id: str
    preset: str
    config: dict            # full PipelineConfig as dict
    index_hashes: dict      # the variant hashes actually used
    models: dict            # router/synth/judge/rerank/embed model IDs
    pricing_table_version: str
    n_total: int
    n_failed: int
    n_abstained: int
    metrics: dict           # recall_at_5, mrr_at_3, em, f1, ragas_faithfulness,
                            # ragas_answer_relevancy, latency_p50_ms, latency_p95_ms, usd_per_query
    per_query: list[dict]   # query_id, retrieved chunk_ids, answer, latency_ms, usd, flags
    anchors: list[PublishedAnchor]
```

Serialized via a versioned envelope `{"schema_version": 1, "receipt": {...}}`.
Committed headline runs live in `receipts/` (top-level, read-only at runtime); local runs in
SQLite + `data/receipts-local/`. Committed per-query records contain IDs + metrics, never passage text.

## Cost — `api/ragreceipts/eval/pricing.py`

`PRICING: dict[str, dict]` keyed by model ID with `usd_per_mtok_input` / `usd_per_mtok_output`
(+ per-call/per-1k pricing for voyage/cohere), plus `PRICING_VERSION = "2026-06-10"` recorded in
every receipt. usd_per_query computed from traced token counts × this table.

## Server (Plan C/D) — `api/ragreceipts/server/`

- FastAPI app `server/app.py`; **single-worker uvicorn**; SQLite WAL; ingest/eval jobs run in a
  dedicated worker thread keyed by SQLite job rows (`server/jobs.py`).
- Endpoints (paths binding): `GET /health` (per-vendor capability), `POST /query`
  (`{"query": str, "corpus_id": str, "preset": str}` → answer + trace_id + degraded flags),
  `GET /traces/{trace_id}`, `GET /corpora`, `POST /corpora/ingest` (BYO, job), `GET /jobs/{job_id}`,
  `POST /eval/runs` (with cost estimate + confirmation), `GET /eval/runs`, `GET /receipts`
  (committed + local).
- Frontend pages: `/` Playground, `/ablation` Ablation Lab, `/corpora` Corpora.

## LangGraph (Plan C)

State machine per spec §Query plane. Nodes named exactly: `route`, `s1_retrieve`, `s1_answer`,
`decompose`, `retrieve_hop`, `grade`, `refine`, `synthesize`. Budget enforced in graph state
(`hops_used`, `tokens_used`); bounded loops via conditional edges + `recursion_limit`.
Abstention surfaced as structured `abstained: bool` on the answer object, never prose-only.

## Commit conventions

TDD per task; conventional commits (`feat:`, `test:`, `docs:`); every commit ends with:
`Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`

---

# Seam Resolutions (2026-06-10, post-verification — BINDING, supersede conflicting plan text)

A three-lens verification pass over the five authored plans found cross-plan seam
mismatches. The following arbitrations are final; each plan has been (or must be)
amended to match.

## R1 — Corpus data layout: Spike 0's `raw/` layout is binding
`data/corpora/{corpus_id}/raw/{docs.jsonl, queries.jsonl, slice-full.json, slice-smoke.json, download_meta.json}`.
There is no `source/` dir and no `dataset.json`. Docs records: `{"doc_id","passage_id","title","text"}`.
Query records: `{"query_id","question","answer"/"answer_texts","answer_aliases","gold":{...typed...}}` with
gold = `{"type":"passage","passage_ids":[...]}` or `{"type":"span","doc_id","start_token","end_token"}`
(span records additionally carry a TOP-LEVEL `"gold_text"` field next to `gold`, per Spike 0's
download script — it is not nested inside the gold object).
Plan A's loaders read these paths/fields; the ingest manifest's `dataset` block (including a
`"name"` key, used by the runner's multi-hop gate) is constructed from `download_meta.json`.

## R2 — No intermediate eval queries file
Plan B's `load_queries` reads `raw/queries.jsonl` directly and normalizes in memory
(`gold_answers = [answer] + answer_aliases` for MuSiQue, `answer_texts` for NQ). Slices come
from `slice-smoke.json` / `slice-full.json` (query-id lists), never "first N lines".

## R3 — Span-hit mechanism: positional rule wins; Chunk carries token ranges
The eval-time span rule is Spike 0's positional token-range rule (validated by the human
hand-check), NOT text-overlap of a `span_text` string. To make it computable on retrieved
chunks, `Chunk` gains two fields (contracts §Core types amended):
`start_token: int` and `end_token: int` — whitespace-token offsets within the parent
passage's cleaned token sequence, persisted in `chunks.jsonl` and the Qdrant payload.
`eval/alignment.py` keeps Spike 0's shipped API — `GoldPassage(query_id, passage_id)`,
`GoldSpan(query_id, doc_id, start_token, end_token)` — and `is_hit`/`first_hit_rank` accept
any object with `passage_id`/`doc_id`/`start_token`/`end_token` (ChunkSpan or Chunk,
structurally). No `GoldPassage(passage_id, span_text)` anywhere.

## R4 — Chunker API: `chunk_passage` + `ChunkSpan` are the public API forever
Plan A reimplements the internals (sentence-window packing) but keeps
`chunk_passage(*, corpus_id, doc_id, passage_id, text, chunk_size=512, chunk_overlap=64) -> list[ChunkSpan]`
and `ChunkSpan` unchanged, and APPENDS to Spike 0's `test_chunker.py` (never replaces).
Any `chunk_document` helper is internal to `ingest/`.

## R5 — Fakes (api/tests/fakes.py): final constructors, defined once in Plan A
- `FakeClaude(script: list)` — ordered script consumed across both `complete()` and
  `parse()`: a `str` item → `ClaudeResult`, a Pydantic instance → `ParsedResult`; optional
  `(item, input_tokens, output_tokens)` tuples. Plan A authors it this way from the start;
  Plan C Task 7 only extends if needed (no constructor migration exists).
- `FakeRerank(script: dict[str, list[int]] | None = None, scores: dict[str, float] | None = None, fail: bool = False)`
  — Plan A's query-keyed ordering mode PLUS an additive text-keyed `scores` mode for Plan B's
  harness fixture.

## R6 — CLI ownership
Plan A CREATES `api/ragreceipts/cli.py` (+ `api/tests/test_cli.py`) with the `ingest`
subcommand and factory seams. Plan B MODIFIES both files, adding `eval` and `receipts`
subparsers, keeping Plan A's seams. Data dir resolution everywhere: `RAGRECEIPTS_DATA_DIR`
env var, default `../data` relative to `api/` (CLI defaults must agree with this);
`receipts promote` default `--receipts-dir ../receipts`.

## R7 — Qdrant versions & env semantics
qdrant-client pin `>=1.18,<2` is binding; the compose image is `qdrant/qdrant:v1.18.0`
(matching minor), not v1.13.x. Unset `QDRANT_URL` semantics: CLI paths fall back to local
file mode at `{data_dir}/qdrant-local`; the FastAPI server REQUIRES `QDRANT_URL` (compose
sets it) and fails its healthcheck with a named-env-var message when missing.

## R8 — Test import convention (repo-wide)
`api/tests/` IS a package: `api/tests/__init__.py` exists; `[tool.pytest.ini_options]
pythonpath = ["."]` in `api/pyproject.toml`; all test files import `from tests.fakes import ...`.
TESTING=1 server mode imports `tests.e2e_fixture` and is launched from `api/` via
`python -m uvicorn` (documented; guarded by a named RuntimeError).

## R9 — No discovery placeholders: pinned entry points
These names are now binding across plans (no grep-discovery steps): Plan B composition root
`cli.py::_build_core_real(config, corpus_id, data_dir)`; Plan B runner
`eval/runner.py::AblationRunner` (with `_run_preset`) and `eval/runner.py::estimate_run_cost`;
Plan C service `agents/service.py::run_query(query=, core=, claude=, store=, config=)` returning
`GraphResult`; Plan A ingest entry `ingest/pipeline.py::run_ingest(corpus_id=, data_dir=, ingest_config=, embed=, qdrant=)`;
RetrievalCore trace wiring is the constructor kwarg `on_trace` (Plan C passes it at
construction; no private-attribute assignment). Plan C Task 12/13 and Plan D Tasks 5/7/13
write complete adapter code against these names plus offline construction tests.

## R10 — Eval-plane guards that survive Plan C
The runner keeps TWO independent gates: the temporary "requires Plan C" skip (Plan C deletes
this one) and the permanent `MULTI_HOP_DATASETS` gate (router-on runs only on multi-hop
corpora — Plan C must keep and test it). After Plan C: `estimate_run_cost` gains a System-2
estimate for AUTO presets (hops × haiku grade/route cost + sonnet synthesis), and actual
per-query usd is computed from TraceStore events' `(model, input_tokens, output_tokens)`.

## R11 — Receipt additions
`Receipt` gains `prompts_version: str` ("n/a" in Plan B, populated from
`agents.prompts.PROMPTS_VERSION` by Plan C) and the envelope gains a fixed
`nondeterminism_note` string disclosing LLM nondeterminism. The literal is owned by
`eval/receipts.py::NONDETERMINISM_NOTE` (Plan B) and pinned here verbatim — fixture
JSONs duplicate it exactly; code imports it:
"LLM calls are nondeterministic even at temperature=0: answer-dependent metrics
(em, f1, ragas_*) can shift slightly between identical runs. Retrieval metrics
(recall_at_5, mrr_at_3) are deterministic for a fixed index. Treat small
answer-metric deltas as noise, not findings." Anchor notes for `nq-dev-300`
runs append the corpus-scale caveat from Spike 0's decisions doc (query-derived ~300-page
corpus; easier than open-corpus retrieval). Plan D fixture receipts use the committed
schema exactly (`retrieved_chunk_ids`, flags dict) and include a `contextual` preset fixture;
the Ablation Lab marks the contextual cell "cross-index" at the cell level.

## R12 — Spec deviation recorded
LlamaIndex is used for BYO document readers only (Plan D). Chunking and retrieval plumbing
are implemented directly because the spec itself mandates bypassing LlamaIndex's per-node
embedding path for contextualization. The spec decisions log is amended accordingly.
