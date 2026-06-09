# Plan A: Ingestion + Retrieval Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the offline-testable ingestion pipeline and flag-driven retrieval core (bm25s sparse + Qdrant dense named vectors + RRF fusion + Cohere rerank stage) that Plans B/C/D compose without modification, with a `PipelineConfig` preset ladder whose every query-time flag provably changes `RetrievalCore` behavior.

**Architecture:** All vendor network calls live behind `Protocol`s in `ragreceipts/vendors/` so the entire pipeline unit-tests offline with fakes from `api/tests/fakes.py`; Qdrant runs in-process via `QdrantClient(":memory:")` in tests. Ingest builds BOTH dense vector sets (contextualized + isolated, as Qdrant named vectors on the same points) plus a fully-rebuilt bm25s index with its tokenizer artifact, and emits a `manifest.json` with one content hash per index variant. `RetrievalCore` is the single composed entry point (S1, S2, and the eval harness all call it), honoring `QueryConfig` flags and emitting `TraceEvent`s through an injectable callback (the trace store itself arrives in Plan C).

**Tech Stack:** Python 3.12 + uv, pytest, ruff (line length 100), bm25s 0.3.9, qdrant-client 1.18 (local/in-memory mode), voyageai 0.4.0 (`voyage-context-3` contextualized embeddings), cohere 7.0.3 (`rerank-v4.0-pro`).

---

## Context

### What exists when this plan starts (Spike 0 deliverables, per plan ordering)

Spike 0 (gold-to-chunk alignment de-risking) has already produced, inside this repo:

- The `api/` uv project (`pyproject.toml` with hatchling build + dev group `pytest>=9.0`/`ruff>=0.15`, `requires-python = ">=3.12"`, package dir `api/ragreceipts/` with `__init__.py`), and `git init` has been run at the repo root `/Users/pratiksoni/PersonalProjects/rag-receipts/`.
- `api/ragreceipts/types.py` — the contracts core types. **This plan modifies it** (R3): `Chunk` gains `start_token`/`end_token`.
- `api/ragreceipts/eval/alignment.py` — the binding hit rules, hand-checked on 20 queries per corpus. Its real shipped shapes: `GoldPassage(query_id, passage_id)`, `GoldSpan(query_id, doc_id, start_token, end_token)`, `Gold = GoldPassage | GoldSpan`, `passage_hit(chunk, gold)` (exact `passage_id` match), `span_hit(span, gold)` (same `doc_id` required; ≥50% token overlap in integer form `2*overlap >= gold_len`), `is_hit(span, gold)`, `first_hit_rank(ranked, gold, k)`. Plan A does not touch this module — but it **imports `ChunkSpan` from `ragreceipts.ingest.chunker`**, which is the compatibility obligation behind R4: this plan rewrites the chunker's internals, yet `chunk_passage`'s exact keyword-only signature and the `ChunkSpan` dataclass MUST survive unchanged.
- `api/ragreceipts/ingest/chunker.py` — Spike 0's stub token-window chunker exposing the binding public API `chunk_passage(*, corpus_id, doc_id, passage_id, text, chunk_size=512, chunk_overlap=64) -> list[ChunkSpan]`, where `ChunkSpan(chunk, start_token, end_token)` carries the whitespace-token range (indices into `passage_text.split()`, start inclusive / end exclusive). Task 5 replaces the internals with sentence-window packing, keeping both names binding.
- `api/ragreceipts/ingest/musique.py` and `api/ragreceipts/ingest/nq.py` — dataset normalization (untouched here), plus `api/tests/{test_types,test_chunker,test_alignment,test_musique,test_nq}.py` (23 passing tests) and `scripts/{download_data,handcheck_alignment}.py`.
- The download script has materialized, per corpus, under the gitignored `data/` dir at the repo root — this `raw/` layout is **binding per R1** (there is no `source/` dir and no `dataset.json`):

```
data/corpora/{corpus_id}/raw/
├── docs.jsonl           # one JSON object per line: {"doc_id": str, "passage_id": str, "title": str, "text": str}
├── queries.jsonl        # one JSON object per line: {"query_id", "question",
│                        #   "answer" + "answer_aliases" (MuSiQue) or "answer_texts" (NQ),
│                        #   "gold": {"type": "passage", "passage_ids": [...]}
│                        #        or {"type": "span", "doc_id", "start_token", "end_token"}}
│                        #   (span records also carry a top-level "gold_text")
├── slice-full.json      # JSON array of 300 query_id strings (slice membership)
├── slice-smoke.json     # first 15 entries of slice-full.json
└── download_meta.json   # {"corpus_id", "dataset": {"hf_id", "config", "split", "revision"},
                         #  "selection_rule", "seed", "n_queries", "n_smoke",
                         #  "datasets_lib_version", "created_at", ...skip counts}
```

Corpus IDs in play: `nq-dev-300` (Natural Questions) and `musique-dev-300` (MuSiQue). Span-gold token indices are whitespace-token indices into the doc's `text` (indices into `text.split()`) — the same token space the chunker and the new `Chunk.start_token`/`end_token` fields use (R3). Plan A does **not** materialize any eval-queries file (R2): Plan B's `load_queries` reads `raw/queries.jsonl` directly and normalizes in memory; this plan only counts query records for the manifest's `n_queries`. The manifest's `dataset` block (including the `"name"` key consumed by Plan B's `MULTI_HOP_DATASETS` gate) is constructed from `download_meta.json` by `loaders.load_dataset_info`. All tests in this plan use a self-contained fixture corpus in this exact `raw/` format.

This plan creates everything else listed in its tasks. Nothing from Plans B/C/D exists yet.

### Binding contracts used by this plan (quoted from `docs/superpowers/plans/2026-06-10-contracts.md`)

`api/ragreceipts/constants.py` (this plan creates it with exactly these values):

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

Core types (`api/ragreceipts/types.py`): frozen dataclasses `Chunk(chunk_id, corpus_id, doc_id, passage_id, text, position, start_token, end_token)` with `chunk_id = f"{doc_id}:{position}"`, `passage_id == doc_id` when unsegmented, and `start_token`/`end_token` = whitespace-token offsets within the parent passage's token sequence (contracts as amended by R3 — persisted to `chunks.jsonl` and the Qdrant payload so span-gold hits stay computable on retrieved chunks); `ScoredChunk(chunk, score, source)` with `source ∈ {"bm25","dense","rrf","rerank"}`; `RouteMode(str, Enum)` with `AUTO/FORCE_S1/FORCE_S2`.

Config (`api/ragreceipts/config.py`): frozen dataclasses `IngestConfig(contextual=True, chunk_size=512, chunk_overlap=64)`, `QueryConfig(bm25=True, dense=True, rerank=True, route_mode=RouteMode.FORCE_S1, top_k_fuse=50, top_k_final=5)`, `PipelineConfig(name, ingest, query)`, and `PRESETS: dict[str, PipelineConfig]` with keys in ladder order `bm25-only → dense-rrf → contextual → rerank → router-on` (exact flag values in Task 2). Both dense vector sets are built at every ingest (Qdrant named vectors `"contextual"` and `"isolated"` on the same points); `IngestConfig.contextual` selects the named vector at query time and the matching manifest hash for receipts.

Retrieval protocol (`api/ragreceipts/retrieval/base.py`):

```python
class Retriever(Protocol):
    def search(self, query: str, k: int) -> list[ScoredChunk]: ...
```

Implementations: `DenseRetriever` (Qdrant, named vector selected by `IngestConfig.contextual`), `SparseRetriever` (bm25s), `HybridRRF(retrievers: list[Retriever], rrf_k: int = 60)`. Rerank is a stage, not a Retriever: `RerankStage.rerank(query: str, candidates: list[ScoredChunk], top_n: int) -> list[ScoredChunk]`. RRF score for a chunk: `sum(1 / (rrf_k + rank_i))` over the rank lists containing it (rank is **1-based**). The single composed entry point:

```python
# api/ragreceipts/retrieval/core.py
class RetrievalCore:
    def __init__(self, config: PipelineConfig, dense: Retriever | None,
                 sparse: Retriever | None, rerank_stage: "RerankStage | None"): ...
    def retrieve(self, query: str) -> list[ScoredChunk]: ...
    # honors config.query flags; returns top_k_final chunks; emits TraceEvents via callback
```

Vendor protocols (`api/ragreceipts/vendors/base.py`):

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

Real impls `VoyageClient`, `CohereClient` (this plan), `AnthropicClient` (Plan C); fakes in `api/tests/fakes.py` (`FakeEmbed`, `FakeRerank`, `FakeClaude`) — their constructors are FINAL per R5 and are authored once, here in Task 4: `FakeRerank(script: dict[str, list[int]] | None = None, scores: dict[str, float] | None = None, fail: bool = False)` (query-keyed ordering mode plus the text-keyed `scores` mode Plan B's harness fixture uses) and `FakeClaude(script: list)` (one ordered script consumed across both `complete()` and `parse()`; Plan C only extends if needed — no constructor migration exists). Degraded behavior: rerank failure → RRF order + `degraded:"rerank-skipped"`; embed failure at query time → BM25-only + `degraded:"dense-skipped"`; Claude failure → raise.

Trace model (`api/ragreceipts/traces/models.py`):

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

Corpus manifest (emitted by every ingest, `data/corpora/{corpus_id}/manifest.json`):

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

### Verified external API bindings (all verified 2026-06-10; behavior additionally confirmed by executing against the pinned versions)

| Library (pin) | Verified binding | Source |
|---|---|---|
| `voyageai>=0.4,<0.5` | `voyageai.Client(api_key=None, max_retries=0, timeout=None)`; `client.contextualized_embed(inputs: list[list[str]], model: str, input_type: "document"\|"query") -> ContextualizedEmbeddingsObject` with `.results: list` (each `.index: int`, `.embeddings: list[list[float]]`) and `.total_tokens`; `client.count_tokens(texts: list[str], model=None) -> int` (local tokenizer, downloads from HF hub on first call — never call in CI); query embedding = `contextualized_embed(inputs=[[query]], input_type="query").results[0].embeddings[0]`; limits per request: 1,000 docs / 16,000 chunks / 120,000 total tokens / 32K tokens per chunk; `voyage-context-3` default dimension 1024; exceptions `voyageai.error.{RateLimitError,ServerError,ServiceUnavailableError,APIConnectionError,...}` all carrying `.http_status` and `.headers` (constructor `VoyageError(message, http_body, http_status, json_body, headers, code)`); SDK does NOT auto-honor retry-after — manual retry recommended by vendor docs | https://docs.voyageai.com/docs/contextualized-chunk-embeddings , https://docs.voyageai.com/docs/rate-limits , empirically inspected voyageai 0.4.0 |
| `cohere>=7.0,<8` | `cohere.ClientV2(api_key=...)`; `co.rerank(model="rerank-v4.0-pro", query=..., documents=[...], top_n=...) -> V2RerankResponse` with `.results: list` (each `.index: int`, `.relevance_score: float`); base exception `cohere.core.api_error.ApiError(*, headers=None, status_code=None, body=None)` with subclasses incl. `TooManyRequestsError`; env var `CO_API_KEY` | https://docs.cohere.com/reference/rerank , empirically inspected cohere 7.0.3 |
| `bm25s>=0.3.9,<0.4` | `bm25s.tokenization.Tokenizer(stemmer=None, stopwords="en")`; `tokenizer.tokenize(texts, return_as="tuple", update_vocab=False, show_progress=False)`; `bm25s.BM25().index(corpus_tokens, show_progress=False)`; `retriever.retrieve(query_tokens, k=k, show_progress=False) -> (indices, scores)` arrays of shape `(n_queries, k)` — returns **int indices** when no corpus is attached; `retrieve` **raises ValueError if k > corpus size** (must clamp); all-stopword queries return zero scores (must filter `score <= 0`); persistence: `retriever.save(dir)`, `bm25s.BM25.load(dir)`, `tokenizer.save_vocab(save_dir=dir)` / `load_vocab(dir)` / `save_stopwords(save_dir=dir)` / `load_stopwords(dir)` writing `vocab.tokenizer.json` + `stopwords.tokenizer.json` | https://github.com/xhluca/bm25s , empirically executed against bm25s 0.3.9 |
| `qdrant-client>=1.18,<2` | `QdrantClient(":memory:")` in-process mode **supports named vectors** (empirically verified: `create_collection(vectors_config={"contextual": VectorParams(size=..., distance=Distance.COSINE), "isolated": ...})`, `upsert(points=[PointStruct(id=<uuid str>, vector={"contextual": [...], "isolated": [...]}, payload={...})])`, `query_points(collection_name, query=vec, using="contextual", limit=k, with_payload=True)`); `collection_exists`/`delete_collection` work in local mode; `QdrantClient(path=dir)` persists local mode to disk (round-trip verified) | https://qdrant.tech/documentation/concepts/vectors/ , https://github.com/qdrant/qdrant-client , empirically executed against qdrant-client 1.18.0 |

LlamaIndex is deliberately NOT used for chunking in this plan: the contextualizer must stay a direct Voyage SDK call (spec: LlamaIndex's per-node embedding path silently degrades doc-grouping), and a ~40-line fully-verified splitter beats an unverified framework dependency. LlamaIndex readers enter in Plan D for BYO documents only.

### Design decisions local to this plan

1. **Chunker token proxy:** `chunk_size`/`chunk_overlap` are counted in whitespace word tokens (deterministic, offline, zero deps). The Voyage per-request budget is NOT enforced with this proxy — `VoyageClient` enforces 120K/16K/32K limits using the SDK's real `count_tokens` at embed time. The manifest records the chunker's config; the proxy choice is documented in the chunker docstring.
2. **`VendorUnavailable(Exception)`** in `vendors/base.py` is the single "vendor down after retries" signal. Real clients raise it after retry exhaustion; fakes raise it when scripted to fail; `RetrievalCore` catches exactly it to degrade visibly.
3. **Qdrant point IDs:** Qdrant requires int/UUID ids, so point id = `uuid5(NAMESPACE_URL, "ragreceipts:" + chunk_id)` (deterministic). Full `Chunk` fields stored as payload; `DenseRetriever` reconstructs `Chunk` from payload.
4. **Named vector constants:** `VECTOR_CONTEXTUAL = "contextual"`, `VECTOR_ISOLATED = "isolated"` in `retrieval/dense.py`, plus `vector_name_for(contextual: bool)`.
5. **`RetrievalCore.retrieve` trace params:** the contracts fix `retrieve(self, query: str)`; this plan adds keyword-only optional params `trace_id: str | None = None, node: str = "s1_retrieve", seq_start: int = 0` (call shape `core.retrieve(query)` unchanged). The callback type is `TraceCallback = Callable[[TraceEvent], None]` defined in `traces/models.py`. Plan C passes `node="retrieve_hop"` for S2 hops and threads its own trace_id/seq.
6. **Single-retriever passthrough:** when only one of bm25/dense is enabled, `RetrievalCore` calls that retriever directly (results keep source `"bm25"`/`"dense"`); RRF fusion runs only with ≥2 lists. This keeps the `bm25-only` preset's receipts honest about their source.
7. **bm25s hygiene:** `k` clamped to corpus size; results with `score <= 0` dropped (zero-score padding from all-stopword queries); index rebuilt from scratch on every ingest (no incremental indexing, per spec non-goals); no stemmer (zero extra deps, deterministic — a stemming receipt is possible future work).
8. **Test imports (R8):** `api/tests/` IS a package — Task 1 creates `api/tests/__init__.py` and sets `pythonpath = ["."]` under `[tool.pytest.ini_options]` in `api/pyproject.toml`. All test files import shared helpers as `from tests.fakes import ...` / `from tests.corpus_fixtures import ...` — this is the repo-wide convention Plans B–D follow. All test commands run from `/Users/pratiksoni/PersonalProjects/rag-receipts/api`.
9. **CLI env contract:** `VOYAGE_API_KEY` (read by our code, passed to `voyageai.Client`), `COHERE_API_KEY` (passed to `cohere.ClientV2` — note the SDK's own default env var is `CO_API_KEY`, we standardize on `COHERE_API_KEY` in `.env`), `QDRANT_URL` (optional FOR THE CLI ONLY; when unset, CLI paths fall back to local file mode at `{data_dir}/qdrant-local` — per R7 this fallback is CLI-scoped: the FastAPI server (Plan D) REQUIRES `QDRANT_URL` (compose sets it) and fails its healthcheck with a named-env-var message when missing), `RAGRECEIPTS_DATA_DIR` (default `../data` relative to `api/`).

### File map created/modified by this plan ((m) = modifies an existing Spike 0 file)

```
api/ragreceipts/
├── constants.py  config.py  cli.py
├── types.py (m: Chunk gains start_token/end_token per R3)
├── traces/__init__.py  traces/models.py
├── vendors/__init__.py  vendors/base.py  vendors/retry.py
│   vendors/voyage_client.py  vendors/cohere_client.py
├── retrieval/__init__.py  retrieval/base.py  retrieval/sparse.py
│   retrieval/dense.py  retrieval/fusion.py  retrieval/rerank.py  retrieval/core.py
└── ingest/chunker.py (m: sentence-window internals behind the binding
    chunk_passage/ChunkSpan API per R4)  ingest/loaders.py  ingest/chunk_store.py
    ingest/contextualizer.py  ingest/indexer.py  ingest/hashing.py
    ingest/manifest.py  ingest/pipeline.py
api/tests/
├── __init__.py (new — tests/ IS a package per R8)
├── fakes.py  corpus_fixtures.py
├── test_types.py (m: Chunk constructions gain R3 fields)
├── test_alignment.py (m: only the _span helper gains R3 fields)
├── test_chunker.py (m: APPEND-only — Spike 0's tests stay, new tests added)
├── test_constants_types.py  test_config.py  test_traces_models.py  test_fakes.py
├── test_loaders.py  test_sparse.py  test_dense.py
├── test_fusion.py  test_rerank_stage.py  test_core.py
├── test_vendor_retry.py  test_voyage_client.py  test_cohere_client.py
├── test_ingest_pipeline.py  test_cli.py
api/scripts/live_smoke_ingest.py   # manual, never CI
```

---

### Task 1: Project wiring, constants.py, types.py (Chunk gains token ranges per R3)

**Files:**
- Modify: `api/pyproject.toml` (deps + pytest `pythonpath` config)
- Create: `api/ragreceipts/constants.py`, `api/tests/__init__.py`
- Modify: `api/ragreceipts/types.py` (Spike 0 file — `Chunk` gains `start_token`/`end_token` per R3)
- Modify: `api/ragreceipts/ingest/chunker.py` (Spike 0 stub — thread the token range into `Chunk`)
- Modify: `api/tests/test_types.py`, `api/tests/test_alignment.py` (Spike 0 tests construct `Chunk` directly)
- Test: `api/tests/test_constants_types.py`

- [ ] Confirm Spike 0 state and git repo. Run:
  ```bash
  ls /Users/pratiksoni/PersonalProjects/rag-receipts/api
  git -C /Users/pratiksoni/PersonalProjects/rag-receipts rev-parse --git-dir
  ```
  Expect `pyproject.toml` and `ragreceipts/` to exist and the git dir to resolve. If the git command fails, run `git -C /Users/pratiksoni/PersonalProjects/rag-receipts init`.
- [ ] Verify the package is importable in the uv venv: `cd /Users/pratiksoni/PersonalProjects/rag-receipts/api && uv run python -c "import ragreceipts; print('ok')"`. If this raises `ModuleNotFoundError` (Spike 0 used non-package `uv init`), add to `api/pyproject.toml`:
  ```toml
  [build-system]
  requires = ["hatchling"]
  build-backend = "hatchling.build"

  [tool.hatch.build.targets.wheel]
  packages = ["ragreceipts"]
  ```
  ensure `api/ragreceipts/__init__.py` exists (`touch ragreceipts/__init__.py`), then `uv sync` and re-run the import check.
- [ ] Add pinned dependencies (versions verified in Context). The dev group already exists from Spike 0 with stricter pins (`pytest>=9.0`, `ruff>=0.15`) — do NOT re-add it:
  ```bash
  cd /Users/pratiksoni/PersonalProjects/rag-receipts/api
  uv add "bm25s>=0.3.9,<0.4" "qdrant-client>=1.18,<2" "voyageai>=0.4,<0.5" "cohere>=7.0,<8"
  ```
- [ ] Ensure these blocks exist in `api/pyproject.toml` (Spike 0 already wrote `testpaths` and the ruff config — the NEW line is `pythonpath`, which makes `api/tests/` importable as the `tests` package per R8; do not duplicate blocks):
  ```toml
  [tool.pytest.ini_options]
  testpaths = ["tests"]
  pythonpath = ["."]

  [tool.ruff]
  line-length = 100
  ```
- [ ] Make the test dir a package (R8): `touch tests/__init__.py` (empty file). From here on, every test file imports shared helpers as `from tests.fakes import ...` / `from tests.corpus_fixtures import ...` — the repo-wide convention Plans B–D keep.
- [ ] Write the failing test `api/tests/test_constants_types.py` (complete file):
  ```python
  """Guards the binding contract values in constants.py and types.py against drift."""

  import dataclasses

  import pytest

  from ragreceipts import constants
  from ragreceipts.types import Chunk, RouteMode, ScoredChunk


  def test_model_constants_match_contracts():
      assert constants.ROUTER_MODEL == "claude-haiku-4-5-20251001"
      assert constants.SYNTH_MODEL == "claude-sonnet-4-6"
      assert constants.JUDGE_MODEL == "claude-sonnet-4-6"
      assert constants.EMBED_MODEL == "voyage-context-3"
      assert constants.RERANK_MODEL == "rerank-v4.0-pro"
      assert constants.RAGAS_EMBED_MODEL == "BAAI/bge-small-en-v1.5"
      assert constants.ROUTE_CONFIDENCE_THRESHOLD == 0.7
      assert constants.S2_MAX_HOPS == 3
      assert constants.S2_TOKEN_CEILING == 50_000


  def test_chunk_fields_and_id_convention():
      chunk = Chunk(chunk_id="doc7:2", corpus_id="tiny", doc_id="doc7",
                    passage_id="doc7-p1", text="some text", position=2,
                    start_token=10, end_token=12)
      assert chunk.chunk_id == f"{chunk.doc_id}:{chunk.position}"
      assert chunk.passage_id == "doc7-p1"
      assert (chunk.start_token, chunk.end_token) == (10, 12)   # R3 token range


  def test_chunk_is_frozen():
      chunk = Chunk(chunk_id="d:0", corpus_id="c", doc_id="d",
                    passage_id="d", text="t", position=0, start_token=0, end_token=1)
      with pytest.raises(dataclasses.FrozenInstanceError):
          chunk.text = "mutated"  # type: ignore[misc]


  def test_scored_chunk_carries_source():
      chunk = Chunk(chunk_id="d:0", corpus_id="c", doc_id="d",
                    passage_id="d", text="t", position=0, start_token=0, end_token=1)
      scored = ScoredChunk(chunk=chunk, score=1.5, source="bm25")
      assert scored.score == 1.5
      assert scored.source == "bm25"


  def test_route_mode_values():
      assert RouteMode.AUTO.value == "auto"
      assert RouteMode.FORCE_S1.value == "force_s1"
      assert RouteMode.FORCE_S2.value == "force_s2"
      assert RouteMode("auto") is RouteMode.AUTO
  ```
- [ ] Run it and watch it fail: `uv run pytest tests/test_constants_types.py -q` → expect `ModuleNotFoundError: No module named 'ragreceipts.constants'` (collection error).
- [ ] Create `api/ragreceipts/constants.py` (complete file — values are binding, verbatim from contracts):
  ```python
  """Model and vendor constants. Binding values from docs/superpowers/plans/2026-06-10-contracts.md."""

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
- [ ] Replace Spike 0's `api/ragreceipts/types.py` (complete file — contracts shapes plus the R3 token-range fields):
  ```python
  """Core value types. Binding shapes from docs/superpowers/plans/2026-06-10-contracts.md
  (Chunk token-range fields added by seam resolution R3)."""

  from dataclasses import dataclass
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
- [ ] Spike 0's stub chunker constructs `Chunk` directly and must now thread the window's token range (it already computes `start`/`end`; Task 5 replaces these internals wholesale — this minimal edit keeps the whole suite green until then). In `api/ragreceipts/ingest/chunker.py`, replace the `Chunk(...)` constructor call inside `chunk_passage` with:
  ```python
          chunk = Chunk(
              chunk_id=f"{doc_id}:{position}",
              corpus_id=corpus_id,
              doc_id=doc_id,
              passage_id=passage_id,
              text=" ".join(tokens[start:end]),
              position=position,
              start_token=start,
              end_token=end,
          )
  ```
- [ ] Replace Spike 0's `api/tests/test_types.py` (complete file — the `Chunk` constructions gain the R3 fields, assertions otherwise unchanged):
  ```python
  import dataclasses

  import pytest

  from ragreceipts.types import Chunk, RouteMode, ScoredChunk


  def test_chunk_is_frozen_and_carries_alignment_metadata():
      c = Chunk(chunk_id="d1:0", corpus_id="musique-dev-300", doc_id="d1",
                passage_id="p1", text="hello world", position=0,
                start_token=0, end_token=2)
      assert c.passage_id == "p1"
      assert c.chunk_id == f"{c.doc_id}:{c.position}"
      assert (c.start_token, c.end_token) == (0, 2)
      with pytest.raises(dataclasses.FrozenInstanceError):
          c.text = "nope"


  def test_scored_chunk_and_route_mode():
      c = Chunk(chunk_id="d1:0", corpus_id="c", doc_id="d1", passage_id="d1",
                text="t", position=0, start_token=0, end_token=1)
      s = ScoredChunk(chunk=c, score=1.5, source="bm25")
      assert s.source == "bm25"
      assert RouteMode.FORCE_S1.value == "force_s1"
      assert RouteMode.AUTO.value == "auto"
  ```
- [ ] In Spike 0's `api/tests/test_alignment.py`, update ONLY the `_span` helper to pass the new fields (the seven test functions stay byte-for-byte untouched — `is_hit`/`first_hit_rank` accept any object with `passage_id`/`doc_id`/`start_token`/`end_token`, structurally, per R3):
  ```python
  def _span(doc_id: str, passage_id: str, start: int, end: int, position: int = 0) -> ChunkSpan:
      chunk = Chunk(chunk_id=f"{doc_id}:{position}", corpus_id="c", doc_id=doc_id,
                    passage_id=passage_id, text="x " * (end - start), position=position,
                    start_token=start, end_token=end)
      return ChunkSpan(chunk=chunk, start_token=start, end_token=end)
  ```
- [ ] Run again: `uv run pytest tests/test_constants_types.py -q` → expect `5 passed`. Then prove the Spike 0 suite survives the type change: `uv run pytest -q` → expect `28 passed` (Spike 0's 23 + these 5).
- [ ] Lint and commit:
  ```bash
  uv run ruff format ragreceipts tests && uv run ruff check ragreceipts tests
  git add pyproject.toml uv.lock ragreceipts/constants.py ragreceipts/types.py ragreceipts/ingest/chunker.py tests/__init__.py tests/test_constants_types.py tests/test_types.py tests/test_alignment.py
  git commit -m "feat: add model constants; extend Chunk with R3 token-range fields" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
  ```

### Task 2: PipelineConfig + PRESETS

**Files:**
- Create: `api/ragreceipts/config.py`
- Test: `api/tests/test_config.py`

- [ ] Write the failing test `api/tests/test_config.py` (complete file):
  ```python
  """PRESETS ladder is binding: keys, order, and every flag value per contracts."""

  import dataclasses

  import pytest

  from ragreceipts.config import PRESETS, IngestConfig, PipelineConfig, QueryConfig
  from ragreceipts.types import RouteMode


  def test_preset_keys_in_ladder_order():
      assert list(PRESETS) == ["bm25-only", "dense-rrf", "contextual", "rerank", "router-on"]


  @pytest.mark.parametrize(
      ("key", "bm25", "dense", "rerank", "contextual", "route_mode"),
      [
          ("bm25-only", True, False, False, False, RouteMode.FORCE_S1),
          ("dense-rrf", True, True, False, False, RouteMode.FORCE_S1),
          ("contextual", True, True, False, True, RouteMode.FORCE_S1),
          ("rerank", True, True, True, True, RouteMode.FORCE_S1),
          ("router-on", True, True, True, True, RouteMode.AUTO),
      ],
  )
  def test_preset_flags_exact(key, bm25, dense, rerank, contextual, route_mode):
      preset = PRESETS[key]
      assert preset.name == key
      assert preset.query.bm25 is bm25
      assert preset.query.dense is dense
      assert preset.query.rerank is rerank
      assert preset.ingest.contextual is contextual
      assert preset.query.route_mode is route_mode


  def test_defaults_match_contracts():
      ingest = IngestConfig()
      assert (ingest.contextual, ingest.chunk_size, ingest.chunk_overlap) == (True, 512, 64)
      query = QueryConfig()
      assert (query.bm25, query.dense, query.rerank) == (True, True, True)
      assert query.route_mode is RouteMode.FORCE_S1
      assert (query.top_k_fuse, query.top_k_final) == (50, 5)


  def test_configs_are_frozen():
      with pytest.raises(dataclasses.FrozenInstanceError):
          QueryConfig().bm25 = False  # type: ignore[misc]
      with pytest.raises(dataclasses.FrozenInstanceError):
          PRESETS["rerank"].name = "x"  # type: ignore[misc]
  ```
- [ ] Run it and watch it fail: `uv run pytest tests/test_config.py -q` → expect `ModuleNotFoundError: No module named 'ragreceipts.config'`.
- [ ] Create `api/ragreceipts/config.py` (complete file):
  ```python
  """PipelineConfig: the single source of truth for pipeline behavior.

  Query-time flags (bm25/dense/rerank/route_mode) flip code paths on the same index.
  Ingest-time flags (contextual, chunking params) select among pre-built index variants —
  both dense vector sets are built at every ingest as Qdrant named vectors; `contextual`
  selects the named vector at query time and the matching manifest hash for receipts.
  Binding shapes from docs/superpowers/plans/2026-06-10-contracts.md.
  """

  from dataclasses import dataclass

  from ragreceipts.types import RouteMode


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


  PRESETS: dict[str, PipelineConfig] = {
      "bm25-only": PipelineConfig(
          name="bm25-only",
          ingest=IngestConfig(contextual=False),
          query=QueryConfig(bm25=True, dense=False, rerank=False, route_mode=RouteMode.FORCE_S1),
      ),
      "dense-rrf": PipelineConfig(
          name="dense-rrf",
          ingest=IngestConfig(contextual=False),
          query=QueryConfig(bm25=True, dense=True, rerank=False, route_mode=RouteMode.FORCE_S1),
      ),
      "contextual": PipelineConfig(
          name="contextual",
          ingest=IngestConfig(contextual=True),
          query=QueryConfig(bm25=True, dense=True, rerank=False, route_mode=RouteMode.FORCE_S1),
      ),
      "rerank": PipelineConfig(
          name="rerank",
          ingest=IngestConfig(contextual=True),
          query=QueryConfig(bm25=True, dense=True, rerank=True, route_mode=RouteMode.FORCE_S1),
      ),
      "router-on": PipelineConfig(
          name="router-on",
          ingest=IngestConfig(contextual=True),
          query=QueryConfig(bm25=True, dense=True, rerank=True, route_mode=RouteMode.AUTO),
      ),
  }
  ```
- [ ] Run again: `uv run pytest tests/test_config.py -q` → expect `8 passed`.
- [ ] Lint and commit:
  ```bash
  uv run ruff format ragreceipts tests && uv run ruff check ragreceipts tests
  git add ragreceipts/config.py tests/test_config.py
  git commit -m "feat: add PipelineConfig with ablation preset ladder" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
  ```

### Task 3: TraceEvent model + TraceCallback type

**Files:**
- Create: `api/ragreceipts/traces/__init__.py`, `api/ragreceipts/traces/models.py`
- Test: `api/tests/test_traces_models.py`

- [ ] Write the failing test `api/tests/test_traces_models.py` (complete file):
  ```python
  """TraceEvent shape is binding (contracts); TraceCallback is the Plan A → Plan C seam."""

  import dataclasses
  import json

  from ragreceipts.traces.models import TraceCallback, TraceEvent


  def _event() -> TraceEvent:
      return TraceEvent(
          trace_id="t1", seq=0, node="s1_retrieve",
          payload={"query": "q", "results": [], "degraded": []},
          model=None, input_tokens=0, output_tokens=0, duration_ms=1.5,
      )


  def test_trace_event_fields_and_frozen():
      event = _event()
      assert (event.trace_id, event.seq, event.node) == ("t1", 0, "s1_retrieve")
      assert dataclasses.fields(TraceEvent)[0].name == "trace_id"
      try:
          event.seq = 9  # type: ignore[misc]
          raised = False
      except dataclasses.FrozenInstanceError:
          raised = True
      assert raised


  def test_payload_is_json_serializable():
      json.dumps(dataclasses.asdict(_event()))


  def test_trace_callback_is_callable_alias():
      seen: list[TraceEvent] = []
      callback: TraceCallback = seen.append
      callback(_event())
      assert seen[0].trace_id == "t1"
  ```
- [ ] Run it and watch it fail: `uv run pytest tests/test_traces_models.py -q` → expect `ModuleNotFoundError: No module named 'ragreceipts.traces'`.
- [ ] Create the package: `touch api/ragreceipts/traces/__init__.py` (empty file; create the directory first with `mkdir -p api/ragreceipts/traces`).
- [ ] Create `api/ragreceipts/traces/models.py` (complete file):
  ```python
  """Trace model. Binding shape from contracts; the SQLite store arrives in Plan C.

  TraceCallback is the seam this plan exposes: RetrievalCore emits TraceEvents through it,
  and Plan C's TraceStore.append satisfies it without RetrievalCore changing.
  """

  from collections.abc import Callable
  from dataclasses import dataclass


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


  TraceCallback = Callable[[TraceEvent], None]
  ```
- [ ] Run again: `uv run pytest tests/test_traces_models.py -q` → expect `3 passed`.
- [ ] Lint and commit:
  ```bash
  uv run ruff format ragreceipts tests && uv run ruff check ragreceipts tests
  git add ragreceipts/traces tests/test_traces_models.py
  git commit -m "feat: add TraceEvent model and trace callback type" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
  ```

### Task 4: Vendor protocols + offline fakes

**Files:**
- Create: `api/ragreceipts/vendors/__init__.py`, `api/ragreceipts/vendors/base.py`, `api/tests/fakes.py`
- Test: `api/tests/test_fakes.py`

- [ ] Write the failing test `api/tests/test_fakes.py` (complete file):
  ```python
  """The fakes ARE the CI vendor layer — they must be deterministic and scriptable.
  Constructor shapes are FINAL per seam resolution R5 and are defined once, here."""

  import math

  import pytest

  from ragreceipts.vendors.base import ClaudeResult, VendorUnavailable
  from tests.fakes import FakeClaude, FakeEmbed, FakeRerank


  def _norm(vec: list[float]) -> float:
      return math.sqrt(sum(x * x for x in vec))


  class TestFakeEmbed:
      def test_deterministic_across_instances(self):
          a = FakeEmbed().embed_query("hello world")
          b = FakeEmbed().embed_query("hello world")
          assert a == b
          assert len(a) == 8
          assert _norm(a) == pytest.approx(1.0)

      def test_isolated_single_chunk_equals_query_embedding_of_same_text(self):
          fake = FakeEmbed()
          [[isolated]] = fake.embed_documents([["alpha beta"]])
          assert isolated == pytest.approx(fake.embed_query("alpha beta"))

      def test_doc_context_changes_chunk_vector(self):
          fake = FakeEmbed()
          [[isolated]] = fake.embed_documents([["alpha beta"]])
          [[contextual, _]] = [fake.embed_documents([["alpha beta", "gamma delta"]])[0]]
          assert contextual != pytest.approx(isolated)

      def test_query_aliases_redirect_query_vector(self):
          fake = FakeEmbed(query_aliases={"q": "target text"})
          assert fake.embed_query("q") == pytest.approx(fake.embed_query("target text"))

      def test_scripted_failures(self):
          with pytest.raises(VendorUnavailable):
              FakeEmbed(fail_query=True).embed_query("q")
          with pytest.raises(VendorUnavailable):
              FakeEmbed(fail_documents=True).embed_documents([["t"]])


  class TestFakeRerank:
      def test_default_reverses_order(self):
          got = FakeRerank().rerank("q", ["a", "b", "c"], top_n=3)
          assert [i for i, _ in got] == [2, 1, 0]
          scores = [s for _, s in got]
          assert scores == sorted(scores, reverse=True)

      def test_script_and_top_n(self):
          fake = FakeRerank(script={"q": [1, 0, 2]})
          assert [i for i, _ in fake.rerank("q", ["a", "b", "c"], top_n=2)] == [1, 0]

      def test_scores_mode_orders_by_candidate_text(self):
          # R5 additive mode — Plan B's harness fixture relies on exactly this shape
          fake = FakeRerank(scores={"high": 0.9, "low": 0.1})
          assert fake.rerank("any query", ["low", "high"], top_n=2) == [(1, 0.9), (0, 0.1)]

      def test_scores_mode_unknown_text_gets_zero(self):
          fake = FakeRerank(scores={"a": 0.5})
          assert fake.rerank("q", ["a", "mystery"], top_n=2) == [(0, 0.5), (1, 0.0)]

      def test_scripted_failure(self):
          with pytest.raises(VendorUnavailable):
              FakeRerank(fail=True).rerank("q", ["a"], top_n=1)


  class TestFakeClaude:
      def test_str_items_become_claude_results_in_order(self):
          fake = FakeClaude(script=["one", ("two", 30, 7)])
          first = fake.complete(model="m", system="s", user="u", max_tokens=64)
          assert first == ClaudeResult(text="one", input_tokens=10, output_tokens=5)
          second = fake.complete(model="m", system="s", user="u", max_tokens=64)
          assert (second.text, second.input_tokens, second.output_tokens) == ("two", 30, 7)
          assert fake.complete_calls[0]["model"] == "m"
          with pytest.raises(AssertionError):
              fake.complete(model="m", system="s", user="u", max_tokens=64)

      def test_object_items_become_parsed_results(self):
          class Routed:
              complexity = "simple"

          routed = Routed()
          fake = FakeClaude(script=[routed, (Routed(), 99, 3)])
          got = fake.parse(model="m", system="s", user="u", max_tokens=64, output_format=Routed)
          assert got.parsed is routed
          assert (got.input_tokens, got.output_tokens) == (10, 5)
          second = fake.parse(model="m", system="s", user="u", max_tokens=64,
                              output_format=Routed)
          assert (second.input_tokens, second.output_tokens) == (99, 3)
          assert fake.parse_calls[0]["output_format"] is Routed

      def test_one_script_consumed_across_complete_and_parse(self):
          class Graded:
              verdict = "good"

          fake = FakeClaude(script=["text answer", Graded()])
          assert fake.complete(model="m", system="s", user="u", max_tokens=8).text == "text answer"
          got = fake.parse(model="m", system="s", user="u", max_tokens=8, output_format=Graded)
          assert isinstance(got.parsed, Graded)

      def test_item_kind_mismatch_fails_loudly(self):
          class Routed:
              pass

          with pytest.raises(AssertionError):
              FakeClaude(script=[Routed()]).complete(model="m", system="s", user="u",
                                                     max_tokens=8)
          with pytest.raises(AssertionError):
              FakeClaude(script=["oops"]).parse(model="m", system="s", user="u",
                                                max_tokens=8, output_format=Routed)
  ```
- [ ] Run it and watch it fail: `uv run pytest tests/test_fakes.py -q` → expect `ModuleNotFoundError: No module named 'tests.fakes'`.
- [ ] Create the package: `mkdir -p api/ragreceipts/vendors && touch api/ragreceipts/vendors/__init__.py`.
- [ ] Create `api/ragreceipts/vendors/base.py` (complete file — protocol bodies verbatim from contracts, plus `VendorUnavailable`):
  ```python
  """Vendor transport seam. Binding shapes from docs/superpowers/plans/2026-06-10-contracts.md.

  Every network call in the system goes through one of these Protocols; application code
  never imports voyageai/cohere/anthropic outside ragreceipts/vendors/. Unit tests inject
  fakes from api/tests/fakes.py — zero keys, zero network in CI.
  """

  from dataclasses import dataclass
  from typing import Protocol


  class VendorUnavailable(Exception):
      """A vendor call failed after all retries (429/5xx/connection).

      Real clients raise this after retry exhaustion; fakes raise it when scripted to fail.
      RetrievalCore catches exactly this to degrade visibly (rerank-skipped / dense-skipped).
      """


  class EmbedTransport(Protocol):
      def embed_documents(self, documents: list[list[str]]) -> list[list[list[float]]]:
          """documents = list of docs, each a list of chunk texts (doc-grouped).
          Isolated mode is expressed by passing single-chunk documents."""
          ...

      def embed_query(self, query: str) -> list[float]: ...


  class RerankTransport(Protocol):
      def rerank(self, query: str, texts: list[str], top_n: int) -> list[tuple[int, float]]:
          """returns (original_index, relevance_score) sorted desc."""
          ...


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


  class ClaudeTransport(Protocol):
      def complete(self, *, model: str, system: str, user: str, max_tokens: int,
                   temperature: float = 0.0) -> ClaudeResult: ...

      def parse(self, *, model: str, system: str, user: str, max_tokens: int,
                output_format: type, temperature: float = 0.0) -> ParsedResult: ...
  ```
- [ ] Create `api/tests/fakes.py` (complete file):
  ```python
  """Offline vendor fakes (contracts: FakeEmbed, FakeRerank, FakeClaude).

  Constructor shapes are FINAL per seam resolution R5 — Plans B/C consume them as-is
  (no constructor migration exists anywhere).

  FakeEmbed vectors are sha256-derived unit vectors: deterministic everywhere, no model
  downloads. Doc-grouped embeddings mix in a document-level component (0.8*chunk + 0.2*doc,
  renormalized) so contextual vectors provably differ from isolated ones — which is what
  lets ingest tests assert dense_contextual != dense_isolated.
  """

  import hashlib
  import math

  from ragreceipts.vendors.base import ClaudeResult, ParsedResult, VendorUnavailable


  def _unit_vector(text: str, dim: int) -> list[float]:
      digest = hashlib.sha256(text.encode("utf-8")).digest()
      raw = [digest[i % len(digest)] / 255.0 - 0.5 for i in range(dim)]
      norm = math.sqrt(sum(x * x for x in raw)) or 1.0
      return [x / norm for x in raw]


  def _renormalize(vec: list[float]) -> list[float]:
      norm = math.sqrt(sum(x * x for x in vec)) or 1.0
      return [x / norm for x in vec]


  class FakeEmbed:
      """Deterministic EmbedTransport.

      query_aliases maps a query string to the text it should embed as — the test's lever
      for making dense retrieval favor a chunk with zero lexical overlap with the query.
      """

      def __init__(self, dim: int = 8, fail_query: bool = False, fail_documents: bool = False,
                   query_aliases: dict[str, str] | None = None):
          self.dim = dim
          self.fail_query = fail_query
          self.fail_documents = fail_documents
          self.query_aliases = query_aliases or {}
          self.document_calls: list[list[list[str]]] = []
          self.query_calls: list[str] = []

      def embed_documents(self, documents: list[list[str]]) -> list[list[list[float]]]:
          if self.fail_documents:
              raise VendorUnavailable("FakeEmbed scripted document failure")
          self.document_calls.append(documents)
          out: list[list[list[float]]] = []
          for doc in documents:
              doc_vec = _unit_vector("||".join(doc), self.dim)
              chunk_vecs = []
              for chunk in doc:
                  chunk_vec = _unit_vector(chunk, self.dim)
                  mixed = [0.8 * c + 0.2 * d for c, d in zip(chunk_vec, doc_vec)]
                  chunk_vecs.append(_renormalize(mixed))
              out.append(chunk_vecs)
          return out

      def embed_query(self, query: str) -> list[float]:
          if self.fail_query:
              raise VendorUnavailable("FakeEmbed scripted query failure")
          self.query_calls.append(query)
          return _unit_vector(self.query_aliases.get(query, query), self.dim)


  class FakeRerank:
      """Scripted RerankTransport (final R5 shape). Modes, in precedence order:

      - scores: dict keyed by candidate TEXT -> returns (original_index,
        scores.get(text, 0.0)) sorted desc, ties by index (Plan B's harness fixture mode);
      - script: dict keyed by QUERY -> explicit ordering of original indices, best first;
      - default: reversed candidate order (provably different from RRF order, which is
        what the rerank flag-flip test needs).
      """

      def __init__(self, script: dict[str, list[int]] | None = None,
                   scores: dict[str, float] | None = None, fail: bool = False):
          self.script = script or {}
          self.scores = scores
          self.fail = fail
          self.calls: list[tuple[str, list[str], int]] = []

      def rerank(self, query: str, texts: list[str], top_n: int) -> list[tuple[int, float]]:
          if self.fail:
              raise VendorUnavailable("FakeRerank scripted failure")
          self.calls.append((query, list(texts), top_n))
          if self.scores is not None:
              pairs = [(i, float(self.scores.get(text, 0.0))) for i, text in enumerate(texts)]
              pairs.sort(key=lambda p: (-p[1], p[0]))
              return pairs[:top_n]
          order = self.script.get(query, list(reversed(range(len(texts)))))
          return [(idx, 1.0 - 0.01 * pos) for pos, idx in enumerate(order)][:top_n]


  class FakeClaude:
      """Scripted ClaudeTransport (final R5 shape): ONE ordered script consumed across
      both complete() and parse(). Script items:

      - str                            -> ClaudeResult(text=item) from complete()
      - any other object (e.g. a Pydantic instance) -> ParsedResult(parsed=item) from parse()
      - (item, input_tokens, output_tokens) tuple   -> same, with explicit token counts

      AssertionError when the script runs dry or the popped item's kind does not match
      the method called (an under- or mis-scripted test must fail loudly, not hang).
      """

      DEFAULT_INPUT_TOKENS = 10
      DEFAULT_OUTPUT_TOKENS = 5

      def __init__(self, script: list | None = None):
          self.script = list(script or [])
          self.complete_calls: list[dict] = []
          self.parse_calls: list[dict] = []

      def _pop(self, caller: str) -> tuple[object, int, int]:
          if not self.script:
              raise AssertionError(f"FakeClaude.{caller} called with empty script")
          item = self.script.pop(0)
          if isinstance(item, tuple):
              payload, input_tokens, output_tokens = item
              return payload, input_tokens, output_tokens
          return item, self.DEFAULT_INPUT_TOKENS, self.DEFAULT_OUTPUT_TOKENS

      def complete(self, *, model: str, system: str, user: str, max_tokens: int,
                   temperature: float = 0.0) -> ClaudeResult:
          self.complete_calls.append({"model": model, "system": system, "user": user,
                                      "max_tokens": max_tokens, "temperature": temperature})
          payload, input_tokens, output_tokens = self._pop("complete")
          if not isinstance(payload, str):
              raise AssertionError(
                  f"FakeClaude.complete expected a str script item, got {type(payload).__name__}"
              )
          return ClaudeResult(text=payload, input_tokens=input_tokens,
                              output_tokens=output_tokens)

      def parse(self, *, model: str, system: str, user: str, max_tokens: int,
                output_format: type, temperature: float = 0.0) -> ParsedResult:
          self.parse_calls.append({"model": model, "system": system, "user": user,
                                   "max_tokens": max_tokens, "output_format": output_format,
                                   "temperature": temperature})
          payload, input_tokens, output_tokens = self._pop("parse")
          if isinstance(payload, str):
              raise AssertionError(
                  "FakeClaude.parse expected a parsed-object script item, got str"
              )
          return ParsedResult(parsed=payload, input_tokens=input_tokens,
                              output_tokens=output_tokens)
  ```
- [ ] Run again: `uv run pytest tests/test_fakes.py -q` → expect `14 passed`.
- [ ] Lint and commit:
  ```bash
  uv run ruff format ragreceipts tests && uv run ruff check ragreceipts tests
  git add ragreceipts/vendors tests/fakes.py tests/test_fakes.py
  git commit -m "feat: add vendor transport protocols and offline fakes" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
  ```

### Task 5: Sentence-window packing inside `chunk_passage` (R4 — binding API kept)

**Files:**
- Modify: `api/ragreceipts/ingest/chunker.py` (Spike 0 file — internals only)
- Modify (APPEND-only): `api/tests/test_chunker.py` (Spike 0 file — its 4 tests stay verbatim)

The chunker is implemented directly (no LlamaIndex — see Context). Per R4, Spike 0's public API is kept FOREVER: `chunk_passage(*, corpus_id, doc_id, passage_id, text, chunk_size=512, chunk_overlap=64) -> list[ChunkSpan]` and the `ChunkSpan` dataclass survive unchanged — `eval/alignment.py` and `scripts/handcheck_alignment.py` import them. Sentence-window packing is implemented INSIDE `chunk_passage`; the token ranges it computes are retained on every `ChunkSpan` AND on the `Chunk` itself (`start_token`/`end_token`, R3). The `chunk_document` helper is internal to `ingest/` (it renumbers positions across a document's passages). "Token" = whitespace word (indices into `text.split()` — the exact space span golds live in); the Voyage request budget is enforced separately with real token counts in Task 12. Token counts for the golden tests below were hand-counted: S1=6, S2=5, S3=6, S4=7, S5=5.

Spike 0's existing tests pin the oversized-sentence behavior: `test_sliding_windows_with_overlap` feeds 10 unpunctuated tokens (one giant "sentence") at `chunk_size=4, chunk_overlap=1` and expects windows `(0,4) (3,7) (6,10)` — so sentences longer than `chunk_size` fall back to a sliding token window with stride `chunk_size - chunk_overlap`, identical to the Spike 0 stub.

- [ ] APPEND the failing tests to Spike 0's `api/tests/test_chunker.py` — do NOT touch its existing four tests. First update its single import line from:
  ```python
  from ragreceipts.ingest.chunker import chunk_passage
  ```
  to:
  ```python
  from ragreceipts.ingest.chunker import chunk_document, chunk_passage
  ```
  then append this block at the end of the file (complete appended code):
  ```python
  # --- Plan A appends below: sentence-window packing + internal chunk_document (R4/R3) ---

  S1 = "The Eiffel Tower is in Paris."            # 6 tokens
  S2 = "It was completed in 1889."                # 5 tokens
  S3 = "It is made of wrought iron."              # 6 tokens
  S4 = "Millions of people visit it every year."  # 7 tokens
  S5 = "The tower is repainted regularly."        # 5 tokens
  SENT_TEXT = " ".join([S1, S2, S3, S4, S5])      # 29 tokens


  def test_sentence_packing_golden_windows():
      # size 12 / overlap 5: S1+S2=11 fits; S2 (5 <= 5) carried as overlap, +S3=11 fits;
      # S3 (6 > 5) NOT carried; S4+S5=12 fits exactly.
      spans = chunk_passage(corpus_id="c", doc_id="d", passage_id="p",
                            text=SENT_TEXT, chunk_size=12, chunk_overlap=5)
      assert [s.chunk.text for s in spans] == [f"{S1} {S2}", f"{S2} {S3}", f"{S4} {S5}"]
      assert [(s.start_token, s.end_token) for s in spans] == [(0, 11), (6, 17), (17, 29)]
      assert [s.chunk.position for s in spans] == [0, 1, 2]
      assert [s.chunk.chunk_id for s in spans] == ["d:0", "d:1", "d:2"]
      assert all(s.chunk.passage_id == "p" and s.chunk.corpus_id == "c" for s in spans)


  def test_chunk_carries_its_token_range():
      # R3: every Chunk mirrors its ChunkSpan range, and the text IS that token slice
      tokens = SENT_TEXT.split()
      for s in chunk_passage(corpus_id="c", doc_id="d", passage_id="p",
                             text=SENT_TEXT, chunk_size=12, chunk_overlap=5):
          assert (s.chunk.start_token, s.chunk.end_token) == (s.start_token, s.end_token)
          assert s.chunk.text == " ".join(tokens[s.start_token:s.end_token])


  def test_oversized_sentence_slides_with_stride():
      long_sentence = " ".join(f"w{i}" for i in range(30))   # one 30-token "sentence"
      spans = chunk_passage(corpus_id="c", doc_id="d", passage_id="p",
                            text=long_sentence, chunk_size=12, chunk_overlap=5)
      # stride = 12 - 5 = 7, same sliding rule Spike 0's stub used
      assert [(s.start_token, s.end_token) for s in spans] == [
          (0, 12), (7, 19), (14, 26), (21, 30),
      ]


  def test_chunk_document_positions_run_across_passages():
      chunks = chunk_document("tiny", "d", [("d-p0", f"{S1} {S2}"), ("d-p1", f"{S3} {S4}")],
                              chunk_size=100, chunk_overlap=10)
      assert [(c.passage_id, c.position, c.chunk_id) for c in chunks] == [
          ("d-p0", 0, "d:0"), ("d-p1", 1, "d:1"),
      ]
      # chunks never span passages; token ranges stay PASSAGE-relative (R3)
      assert [(c.start_token, c.end_token) for c in chunks] == [(0, 11), (0, 13)]
  ```
- [ ] Run it and watch it fail: `uv run pytest tests/test_chunker.py -q` → expect `ImportError: cannot import name 'chunk_document' from 'ragreceipts.ingest.chunker'` (collection error; the 4 Spike 0 tests cannot run until it resolves).
- [ ] Replace `api/ragreceipts/ingest/chunker.py` (complete file — `ChunkSpan` and `chunk_passage`'s signature are byte-compatible with Spike 0; only internals change):
  ```python
  """Sentence-window chunker (Plan A internals behind Spike 0's binding API).

  PUBLIC API — binding per Spike 0 and seam resolution R4:
  - chunk_passage(*, corpus_id, doc_id, passage_id, text, chunk_size=512,
    chunk_overlap=64) -> list[ChunkSpan], and the ChunkSpan dataclass, are kept
    forever — eval/alignment.py and scripts/handcheck_alignment.py import them.
  - chunk_document is an ingest-internal helper (positions run across passages).

  "Token" = whitespace word (indices into text.split()) — the same space Spike 0's
  span golds live in. Chunk text is always " ".join(tokens[start:end]), so the
  (start_token, end_token) range stored on every ChunkSpan AND Chunk (R3) is exact.
  Sentence boundaries are detected per token (a token ending in . ! or ? closes a
  sentence — abbreviations like "Dr." split; acceptable for packing, fully offline).
  Sentences are packed greedily up to chunk_size tokens, retaining trailing whole
  sentences up to chunk_overlap tokens as overlap; a sentence longer than chunk_size
  falls back to a sliding token window with stride chunk_size - chunk_overlap
  (identical to Spike 0's stub on unpunctuated text). Real Voyage token budgets are
  enforced at embed time by VoyageClient's batch planner (120K/16K/32K request caps).
  """

  from dataclasses import dataclass, replace

  from ragreceipts.types import Chunk

  _TERMINALS = (".", "!", "?")


  @dataclass(frozen=True)
  class ChunkSpan:
      """A chunk plus the whitespace-token range it covers in its parent passage text.

      start_token is inclusive, end_token exclusive; both index into passage_text.split().
      (Kept verbatim from Spike 0; the same range is also stored on the Chunk per R3.)
      """

      chunk: Chunk
      start_token: int
      end_token: int


  def _sentence_ranges(tokens: list[str]) -> list[tuple[int, int]]:
      """[start, end) token ranges of sentences; a token ending in .!? closes one."""
      ranges: list[tuple[int, int]] = []
      start = 0
      for i, token in enumerate(tokens):
          if token.endswith(_TERMINALS):
              ranges.append((start, i + 1))
              start = i + 1
      if start < len(tokens):
          ranges.append((start, len(tokens)))
      return ranges


  def _units(tokens: list[str], chunk_size: int, stride: int) -> list[tuple[int, int]]:
      """Sentence ranges, with oversized sentences hard-split into sliding windows."""
      units: list[tuple[int, int]] = []
      for start, end in _sentence_ranges(tokens):
          if end - start <= chunk_size:
              units.append((start, end))
              continue
          s = start
          while True:
              e = min(s + chunk_size, end)
              units.append((s, e))
              if e == end:
                  break
              s += stride
      return units


  def _pack(units: list[tuple[int, int]], chunk_size: int,
            chunk_overlap: int) -> list[tuple[int, int]]:
      """Greedy sentence packing with trailing-sentence overlap; returns window ranges."""
      windows: list[tuple[int, int]] = []
      window: list[tuple[int, int]] = []
      window_tokens = 0
      for unit in units:
          unit_tokens = unit[1] - unit[0]
          if window and window_tokens + unit_tokens > chunk_size:
              windows.append((window[0][0], window[-1][1]))
              kept: list[tuple[int, int]] = []
              kept_tokens = 0
              for prev in reversed(window):       # retain trailing sentences as overlap
                  prev_tokens = prev[1] - prev[0]
                  if kept_tokens + prev_tokens > chunk_overlap:
                      break
                  kept.insert(0, prev)
                  kept_tokens += prev_tokens
              if kept and (kept_tokens + unit_tokens > chunk_size
                           or kept[-1][1] != unit[0]):
                  kept, kept_tokens = [], 0       # overlap would overflow / isn't contiguous
              window, window_tokens = kept, kept_tokens
          window.append(unit)
          window_tokens += unit_tokens
      if window:
          windows.append((window[0][0], window[-1][1]))
      return windows


  def chunk_passage(
      *,
      corpus_id: str,
      doc_id: str,
      passage_id: str,
      text: str,
      chunk_size: int = 512,
      chunk_overlap: int = 64,
  ) -> list[ChunkSpan]:
      """Split `text` into sentence-packed windows of whitespace tokens.

      Signature and return type are binding (Spike 0 / R4). Empty or whitespace-only
      text yields [].
      """
      if chunk_size <= 0:
          raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
      if not 0 <= chunk_overlap < chunk_size:
          raise ValueError(f"chunk_overlap must be in [0, chunk_size), got {chunk_overlap}")
      tokens = text.split()
      if not tokens:
          return []
      stride = chunk_size - chunk_overlap
      spans: list[ChunkSpan] = []
      for position, (start, end) in enumerate(
          _pack(_units(tokens, chunk_size, stride), chunk_size, chunk_overlap)
      ):
          chunk = Chunk(
              chunk_id=f"{doc_id}:{position}",
              corpus_id=corpus_id,
              doc_id=doc_id,
              passage_id=passage_id,
              text=" ".join(tokens[start:end]),
              position=position,
              start_token=start,
              end_token=end,
          )
          spans.append(ChunkSpan(chunk=chunk, start_token=start, end_token=end))
      return spans


  def chunk_document(corpus_id: str, doc_id: str, passages: list[tuple[str, str]],
                     chunk_size: int, chunk_overlap: int) -> list[Chunk]:
      """Ingest-internal helper: chunk each (passage_id, text) of one document via
      chunk_passage, renumbering positions ACROSS the document so chunk_id stays
      unique per doc. start_token/end_token stay passage-relative (R3)."""
      chunks: list[Chunk] = []
      position = 0
      for passage_id, text in passages:
          for span in chunk_passage(corpus_id=corpus_id, doc_id=doc_id,
                                    passage_id=passage_id, text=text,
                                    chunk_size=chunk_size, chunk_overlap=chunk_overlap):
              chunks.append(replace(span.chunk, chunk_id=f"{doc_id}:{position}",
                                    position=position))
              position += 1
      return chunks
  ```
- [ ] Run again: `uv run pytest tests/test_chunker.py -q` → expect `8 passed` (Spike 0's 4 + the 4 appended — proving the rewrite is behavior-compatible with the stub on its golden cases).
- [ ] Run the whole suite to prove the alignment module and handcheck dependencies survive: `uv run pytest -q` → expect all tests passing (no failures in `test_alignment.py`).
- [ ] Lint and commit:
  ```bash
  uv run ruff format ragreceipts tests && uv run ruff check ragreceipts tests
  git add ragreceipts/ingest/chunker.py tests/test_chunker.py
  git commit -m "feat: sentence-window packing inside chunk_passage, ChunkSpan API kept (R4)" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
  ```

### Task 6: Corpus loaders, chunk store, shared test fixtures

**Files:**
- Create: `api/ragreceipts/ingest/loaders.py`, `api/ragreceipts/ingest/chunk_store.py`, `api/tests/corpus_fixtures.py`
- Test: `api/tests/test_loaders.py`

- [ ] Verify the downloaded Spike 0 corpora are present in the binding `raw/` layout (R1 — no adaptation step exists; the layout is fixed):
  ```bash
  ls /Users/pratiksoni/PersonalProjects/rag-receipts/data/corpora/ 2>/dev/null \
    && head -c 400 /Users/pratiksoni/PersonalProjects/rag-receipts/data/corpora/musique-dev-300/raw/docs.jsonl 2>/dev/null \
    || echo "no downloaded corpora present — fixture-only development is fine for this plan"
  ```
  Expected: either `musique-dev-300  nq-dev-300` plus a JSON line with `doc_id`/`passage_id`/`title`/`text` keys, or the fixture-only message.
- [ ] Create `api/tests/corpus_fixtures.py` (complete file — shared by loader, pipeline, and CLI tests; written in Spike 0's exact `raw/` format):
  ```python
  """Tiny fixture corpus in Spike 0's raw/ on-disk format, plus a Chunk factory for tests.

  The benchmark slices are unsegmented (doc_id == passage_id); the fixture gives d1 TWO
  passages on purpose — the format carries both ids, and the loader seam must handle
  segmented documents (contracts: "passage_id == doc_id when unsegmented").
  TINY_QUERIES mirrors both real gold shapes: q1 is a MuSiQue-style passage gold
  ("answer" + "answer_aliases"), q2 an NQ-style span gold ("answer_texts" + "gold_text";
  token indices into the doc text's .split()).
  """

  import json
  from pathlib import Path

  from ragreceipts.types import Chunk

  TINY_PASSAGES = [
      {"doc_id": "d1", "passage_id": "d1-p0", "title": "Eiffel Tower",
       "text": ("The Eiffel Tower is a wrought iron lattice tower in Paris. "
                "It was completed in 1889. Millions of visitors climb the tower every year.")},
      {"doc_id": "d1", "passage_id": "d1-p1", "title": "Eiffel Tower",
       "text": ("The tower is the tallest structure in Paris. "
                "Its height is about 330 metres.")},
      {"doc_id": "d2", "passage_id": "d2-p0", "title": "Cats",
       "text": ("Cats are small carnivorous mammals. Domestic cats often hunt mice and birds. "
                "A group of cats is called a clowder.")},
      {"doc_id": "d3", "passage_id": "d3-p0", "title": "Solar panels",
       "text": ("Solar panels convert sunlight into electricity. "
                "Photovoltaic cells are made of silicon. "
                "Panel efficiency has improved steadily.")},
  ]
  TINY_QUERIES = [
      {"query_id": "q1", "question": "How tall is the Eiffel Tower?",
       "answer": "330 metres", "answer_aliases": ["about 330 metres"],
       "gold": {"type": "passage", "passage_ids": ["d1-p1"]}},
      {"query_id": "q2", "question": "What do domestic cats hunt?",
       "answer_texts": ["mice and birds"],
       "gold": {"type": "span", "doc_id": "d2", "start_token": 9, "end_token": 12},
       "gold_text": "mice and birds."},
  ]
  TINY_DOWNLOAD_META = {
      "corpus_id": "tiny",
      "dataset": {"hf_id": "local/tiny-fixture", "config": "default",
                  "split": "test", "revision": "fixture-v1"},
      "selection_rule": "in-repo fixture",
      "seed": 0,
      "n_queries": 2,
      "n_smoke": 2,
  }


  def write_tiny_corpus(data_dir: Path, corpus_id: str = "tiny") -> Path:
      """Writes the fixture corpus under data_dir/corpora/{corpus_id}/raw; returns corpus dir."""
      corpus_dir = data_dir / "corpora" / corpus_id
      raw = corpus_dir / "raw"
      raw.mkdir(parents=True, exist_ok=True)
      with (raw / "docs.jsonl").open("w", encoding="utf-8") as fh:
          for row in TINY_PASSAGES:
              fh.write(json.dumps(row) + "\n")
      with (raw / "queries.jsonl").open("w", encoding="utf-8") as fh:
          for row in TINY_QUERIES:
              fh.write(json.dumps(row) + "\n")
      slice_full = [q["query_id"] for q in TINY_QUERIES]
      (raw / "slice-full.json").write_text(json.dumps(slice_full), encoding="utf-8")
      (raw / "slice-smoke.json").write_text(json.dumps(slice_full), encoding="utf-8")
      meta = {**TINY_DOWNLOAD_META, "corpus_id": corpus_id}
      (raw / "download_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
      return corpus_dir


  def make_chunk(chunk_id: str, text: str = "", corpus_id: str = "test",
                 passage_id: str | None = None, start_token: int = 0,
                 end_token: int | None = None) -> Chunk:
      doc_id, position = chunk_id.rsplit(":", 1)
      body = text or chunk_id
      return Chunk(chunk_id=chunk_id, corpus_id=corpus_id, doc_id=doc_id,
                   passage_id=passage_id or doc_id, text=body,
                   position=int(position), start_token=start_token,
                   end_token=end_token if end_token is not None
                   else start_token + len(body.split()))
  ```
- [ ] Write the failing test `api/tests/test_loaders.py` (complete file):
  ```python
  """Loaders read Spike 0's raw/ corpus layout (R1); chunk_store round-trips Chunk rows."""

  import pytest

  from ragreceipts.ingest.chunk_store import read_chunks, write_chunks
  from ragreceipts.ingest.loaders import (
      count_queries,
      dataset_name,
      group_documents,
      load_dataset_info,
      load_passages,
  )
  from tests.corpus_fixtures import TINY_PASSAGES, make_chunk, write_tiny_corpus


  def test_load_passages_preserves_order_and_fields(tmp_path):
      corpus_dir = write_tiny_corpus(tmp_path)
      passages = load_passages(corpus_dir)
      assert [p.passage_id for p in passages] == [row["passage_id"] for row in TINY_PASSAGES]
      assert passages[0].doc_id == "d1"
      assert passages[0].title == "Eiffel Tower"
      assert "wrought iron lattice" in passages[0].text


  def test_load_passages_missing_corpus_raises(tmp_path):
      with pytest.raises(FileNotFoundError):
          load_passages(tmp_path / "corpora" / "nope")


  def test_group_documents_by_doc_id_stable_order(tmp_path):
      passages = load_passages(write_tiny_corpus(tmp_path))
      docs = group_documents(passages)
      assert [doc[0].doc_id for doc in docs] == ["d1", "d2", "d3"]
      assert [p.passage_id for p in docs[0]] == ["d1-p0", "d1-p1"]


  def test_dataset_name_strips_dev_slice_suffix():
      assert dataset_name("musique-dev-300") == "musique"
      assert dataset_name("nq-dev-300") == "nq"
      assert dataset_name("tiny") == "tiny"


  def test_dataset_info_built_from_download_meta_and_query_count(tmp_path):
      corpus_dir = write_tiny_corpus(tmp_path)
      info = load_dataset_info(corpus_dir)
      # contracts manifest shape, incl. the "name" key Plan B's multi-hop gate reads (R1)
      assert info == {"name": "tiny", "hf_id": "local/tiny-fixture",
                      "split": "test", "revision": "fixture-v1"}
      assert count_queries(corpus_dir) == 2
      assert count_queries(tmp_path / "corpora" / "nope") == 0


  def test_chunk_store_round_trip(tmp_path):
      chunks = [make_chunk("d1:0", "alpha"), make_chunk("d1:1", "beta", passage_id="d1-p1")]
      path = tmp_path / "chunks.jsonl"
      write_chunks(path, chunks)
      assert read_chunks(path) == chunks
  ```
- [ ] Run it and watch it fail: `uv run pytest tests/test_loaders.py -q` → expect `ModuleNotFoundError: No module named 'ragreceipts.ingest.chunk_store'`.
- [ ] Create `api/ragreceipts/ingest/loaders.py` (complete file):
  ```python
  """Readers for Spike 0's raw benchmark-slice corpus layout (binding per R1):

      data/corpora/{corpus_id}/raw/{docs.jsonl, queries.jsonl,
                                    slice-full.json, slice-smoke.json, download_meta.json}

  docs.jsonl record: {"doc_id", "passage_id", "title", "text"}.
  queries.jsonl record: {"query_id", "question", "answer"/"answer_texts",
  "answer_aliases", "gold": {"type": "passage", "passage_ids": [...]} or
  {"type": "span", "doc_id", "start_token", "end_token"}} (+ top-level "gold_text").
  Plan A only COUNTS query records — per R2 no intermediate eval-queries file is ever
  materialized; Plan B's load_queries reads raw/queries.jsonl directly.
  SourcePassage is the seam everything downstream depends on.
  """

  import json
  import re
  from dataclasses import dataclass
  from pathlib import Path


  @dataclass(frozen=True)
  class SourcePassage:
      passage_id: str
      doc_id: str
      title: str
      text: str


  def load_passages(corpus_dir: Path) -> list[SourcePassage]:
      path = corpus_dir / "raw" / "docs.jsonl"
      passages: list[SourcePassage] = []
      with path.open(encoding="utf-8") as fh:
          for line in fh:
              line = line.strip()
              if not line:
                  continue
              row = json.loads(line)
              passages.append(SourcePassage(
                  passage_id=str(row["passage_id"]),
                  doc_id=str(row["doc_id"]),
                  title=str(row.get("title", "")),
                  text=str(row["text"]),
              ))
      return passages


  def dataset_name(corpus_id: str) -> str:
      """'musique-dev-300' -> 'musique', 'nq-dev-300' -> 'nq'; anything without the
      -dev-N suffix is its own name. Load-bearing: the manifest's dataset.name feeds
      Plan B's MULTI_HOP_DATASETS gate (router-on runs only on multi-hop corpora)."""
      return re.sub(r"-dev-\d+$", "", corpus_id)


  def load_dataset_info(corpus_dir: Path) -> dict:
      """Constructs the manifest's dataset block (contracts shape, incl. "name")
      from Spike 0's download_meta.json (R1)."""
      meta = json.loads(
          (corpus_dir / "raw" / "download_meta.json").read_text(encoding="utf-8")
      )
      dataset = meta["dataset"]
      return {
          "name": dataset_name(str(meta["corpus_id"])),
          "hf_id": dataset["hf_id"],
          "split": dataset["split"],
          "revision": dataset["revision"],
      }


  def count_queries(corpus_dir: Path) -> int:
      path = corpus_dir / "raw" / "queries.jsonl"
      if not path.exists():
          return 0
      with path.open(encoding="utf-8") as fh:
          return sum(1 for line in fh if line.strip())


  def group_documents(passages: list[SourcePassage]) -> list[list[SourcePassage]]:
      """Groups passages by doc_id, preserving first-seen document order and
      within-document passage order (dict preserves insertion order)."""
      by_doc: dict[str, list[SourcePassage]] = {}
      for passage in passages:
          by_doc.setdefault(passage.doc_id, []).append(passage)
      return list(by_doc.values())
  ```
- [ ] Create `api/ragreceipts/ingest/chunk_store.py` (complete file):
  ```python
  """chunks.jsonl persistence — the canonical chunk order shared by ALL index variants.

  SparseRetriever maps bm25s row indices into this list; Qdrant payloads duplicate the
  fields for dense lookups. Rows carry the FULL Chunk via asdict — including the R3
  start_token/end_token token-range fields. Row order is load-bearing: never reorder."""

  import json
  from dataclasses import asdict
  from pathlib import Path

  from ragreceipts.types import Chunk


  def write_chunks(path: Path, chunks: list[Chunk]) -> None:
      path.parent.mkdir(parents=True, exist_ok=True)
      with path.open("w", encoding="utf-8") as fh:
          for chunk in chunks:
              fh.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")


  def read_chunks(path: Path) -> list[Chunk]:
      chunks: list[Chunk] = []
      with path.open(encoding="utf-8") as fh:
          for line in fh:
              if line.strip():
                  chunks.append(Chunk(**json.loads(line)))
      return chunks
  ```
- [ ] Run again: `uv run pytest tests/test_loaders.py -q` → expect `6 passed`.
- [ ] Lint and commit:
  ```bash
  uv run ruff format ragreceipts tests && uv run ruff check ragreceipts tests
  git add ragreceipts/ingest/loaders.py ragreceipts/ingest/chunk_store.py tests/corpus_fixtures.py tests/test_loaders.py
  git commit -m "feat: add raw-layout corpus loaders and chunk store" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
  ```

### Task 7: SparseRetriever on bm25s (build / serialize / load)

**Files:**
- Create: `api/ragreceipts/retrieval/__init__.py`, `api/ragreceipts/retrieval/base.py`, `api/ragreceipts/retrieval/sparse.py`
- Test: `api/tests/test_sparse.py`

- [ ] Write the failing test `api/tests/test_sparse.py` (complete file):
  ```python
  """SparseRetriever: bm25s build/serialize/load with tokenizer artifact, plus edge guards."""

  import pytest

  from ragreceipts.ingest.chunk_store import read_chunks, write_chunks
  from ragreceipts.retrieval.sparse import SparseRetriever, build_sparse_index
  from tests.corpus_fixtures import make_chunk

  CHUNKS = [
      make_chunk("d1:0", "the eiffel tower is a lattice tower in paris", passage_id="d1-p0"),
      make_chunk("d2:0", "domestic cats often hunt mice and birds", passage_id="d2-p0"),
      make_chunk("d3:0", "solar panels convert sunlight into electricity", passage_id="d3-p0"),
  ]


  @pytest.fixture()
  def built(tmp_path):
      retriever = build_sparse_index(CHUNKS, tmp_path / "sparse")
      return retriever, tmp_path / "sparse"


  def test_search_returns_lexical_top_hit(built):
      retriever, _ = built
      results = retriever.search("eiffel tower paris", k=3)
      assert results[0].chunk.chunk_id == "d1:0"
      assert results[0].source == "bm25"
      assert results[0].score > 0


  def test_persisted_index_round_trips_with_tokenizer_artifact(built, tmp_path):
      retriever, index_dir = built
      assert (index_dir / "vocab.tokenizer.json").exists()      # the tokenizer artifact
      assert (index_dir / "stopwords.tokenizer.json").exists()
      chunks_path = tmp_path / "chunks.jsonl"
      write_chunks(chunks_path, CHUNKS)
      reloaded = SparseRetriever.load(index_dir, read_chunks(chunks_path))
      live = retriever.search("eiffel tower paris", k=3)
      loaded = reloaded.search("eiffel tower paris", k=3)
      assert [r.chunk.chunk_id for r in loaded] == [r.chunk.chunk_id for r in live]
      assert [r.score for r in loaded] == pytest.approx([r.score for r in live])


  def test_k_larger_than_corpus_is_clamped(built):
      retriever, _ = built
      results = retriever.search("eiffel tower paris", k=50)    # bm25s would raise unclamped
      assert len(results) <= len(CHUNKS)


  def test_zero_score_results_filtered(built):
      retriever, _ = built
      assert retriever.search("the of and", k=3) == []          # all stopwords -> 0.0 scores


  def test_empty_chunk_list_searches_empty(tmp_path):
      retriever = build_sparse_index(
          [make_chunk("d1:0", "only one document here")], tmp_path / "sparse"
      )
      assert retriever.search("unrelated query terms", k=5) == []
  ```
- [ ] Run it and watch it fail: `uv run pytest tests/test_sparse.py -q` → expect `ModuleNotFoundError: No module named 'ragreceipts.retrieval'`.
- [ ] Create the package: `mkdir -p api/ragreceipts/retrieval && touch api/ragreceipts/retrieval/__init__.py`.
- [ ] Create `api/ragreceipts/retrieval/base.py` (complete file — binding protocol):
  ```python
  """Retriever protocol. retrieval/ knows nothing about agents or HTTP.

  The Phase-2 graph retriever implements this same protocol; no graph flag, enum branch,
  or stub ships in v1 code (spec boundary rule)."""

  from typing import Protocol

  from ragreceipts.types import ScoredChunk


  class Retriever(Protocol):
      def search(self, query: str, k: int) -> list[ScoredChunk]: ...
  ```
- [ ] Create `api/ragreceipts/retrieval/sparse.py` (complete file; bm25s API verified — see Context table):
  ```python
  """BM25 sparse retrieval on bm25s, fully rebuilt on every ingest (no incremental indexing).

  Serialization: bm25s.BM25.save/load for the index matrices, plus the Tokenizer's
  vocab + stopwords artifacts (vocab.tokenizer.json / stopwords.tokenizer.json) saved
  beside them — the query-time tokenizer MUST use the build-time vocab or scores drift.
  Chunk row order comes from chunks.jsonl and is shared with the dense index.
  No stemmer: deterministic, zero extra deps (a stemming receipt is possible future work).
  """

  from pathlib import Path

  import bm25s
  from bm25s.tokenization import Tokenizer

  from ragreceipts.types import Chunk, ScoredChunk


  def _build_tokenizer(stopwords: str | list[str] = "en") -> Tokenizer:
      return Tokenizer(stemmer=None, stopwords=stopwords)


  def build_sparse_index(chunks: list[Chunk], index_dir: Path) -> "SparseRetriever":
      """Builds, persists, and returns a live SparseRetriever (full rebuild semantics)."""
      if not chunks:
          raise ValueError("cannot build a sparse index from zero chunks")
      index_dir.mkdir(parents=True, exist_ok=True)
      tokenizer = _build_tokenizer()
      corpus_tokens = tokenizer.tokenize(
          [c.text for c in chunks], return_as="tuple", show_progress=False
      )
      bm25 = bm25s.BM25()
      bm25.index(corpus_tokens, show_progress=False)
      bm25.save(str(index_dir))
      tokenizer.save_vocab(save_dir=str(index_dir))
      tokenizer.save_stopwords(save_dir=str(index_dir))
      return SparseRetriever(bm25, tokenizer, chunks)


  class SparseRetriever:
      def __init__(self, bm25: bm25s.BM25, tokenizer: Tokenizer, chunks: list[Chunk]):
          self._bm25 = bm25
          self._tokenizer = tokenizer
          self._chunks = chunks

      @classmethod
      def load(cls, index_dir: Path, chunks: list[Chunk]) -> "SparseRetriever":
          """chunks must be the same list (same order) the index was built from."""
          bm25 = bm25s.BM25.load(str(index_dir))
          tokenizer = _build_tokenizer(stopwords=[])
          tokenizer.load_vocab(str(index_dir))
          tokenizer.load_stopwords(str(index_dir))
          return cls(bm25, tokenizer, chunks)

      def search(self, query: str, k: int) -> list[ScoredChunk]:
          k = min(k, len(self._chunks))   # bm25s raises ValueError when k > corpus size
          if k <= 0:
              return []
          query_tokens = self._tokenizer.tokenize(
              [query], return_as="tuple", update_vocab=False, show_progress=False
          )
          indices, scores = self._bm25.retrieve(query_tokens, k=k, show_progress=False)
          results: list[ScoredChunk] = []
          for idx, score in zip(indices[0].tolist(), scores[0].tolist()):
              if score <= 0.0:            # zero-score padding (e.g. all-stopword queries)
                  continue
              results.append(ScoredChunk(chunk=self._chunks[idx], score=float(score),
                                         source="bm25"))
          return results
  ```
- [ ] Run again: `uv run pytest tests/test_sparse.py -q` → expect `5 passed`.
- [ ] Lint and commit:
  ```bash
  uv run ruff format ragreceipts tests && uv run ruff check ragreceipts tests
  git add ragreceipts/retrieval tests/test_sparse.py
  git commit -m "feat: add bm25s sparse retriever with persisted tokenizer" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
  ```

### Task 8: DenseRetriever on Qdrant named vectors + dense index writer

**Files:**
- Create: `api/ragreceipts/retrieval/dense.py`, `api/ragreceipts/ingest/indexer.py`
- Test: `api/tests/test_dense.py`

`QdrantClient(":memory:")` named-vector support was verified empirically (Context table). The index writer lives in `ingest/` (it is an ingest-time concern) but is created here because the dense tests need points to query.

- [ ] Write the failing test `api/tests/test_dense.py` (complete file):
  ```python
  """DenseRetriever over Qdrant named vectors ("contextual"/"isolated") in :memory: mode."""

  import pytest
  from qdrant_client import QdrantClient

  from ragreceipts.ingest.indexer import write_dense_index
  from ragreceipts.retrieval.dense import (
      VECTOR_CONTEXTUAL,
      VECTOR_ISOLATED,
      DenseRetriever,
      point_id_for_chunk,
      vector_name_for,
  )
  from ragreceipts.vendors.base import VendorUnavailable
  from tests.corpus_fixtures import make_chunk
  from tests.fakes import FakeEmbed

  # m1 is a two-chunk document; m2 repeats m1's first chunk text as a single-chunk doc.
  # Isolated vectors for the repeated text are identical; contextual vectors differ —
  # that asymmetry is what proves the named-vector selection actually changes behavior.
  C_M1A = make_chunk("m1:0", "alpha beta gamma", corpus_id="dense-test", passage_id="m1-p0")
  C_M1B = make_chunk("m1:1", "delta epsilon zeta", corpus_id="dense-test", passage_id="m1-p0")
  C_M2 = make_chunk("m2:0", "alpha beta gamma", corpus_id="dense-test", passage_id="m2-p0")
  CHUNKS = [C_M1A, C_M1B, C_M2]
  DOC_CHUNK_TEXTS = [[C_M1A.text, C_M1B.text], [C_M2.text]]


  @pytest.fixture()
  def indexed():
      fake = FakeEmbed()
      contextual = [vec for doc in fake.embed_documents(DOC_CHUNK_TEXTS) for vec in doc]
      isolated = [doc[0] for doc in
                  fake.embed_documents([[t] for doc in DOC_CHUNK_TEXTS for t in doc])]
      client = QdrantClient(":memory:")
      write_dense_index(client, "dense-test", CHUNKS, contextual, isolated)
      return client, fake


  def test_vector_name_for():
      assert vector_name_for(True) == VECTOR_CONTEXTUAL == "contextual"
      assert vector_name_for(False) == VECTOR_ISOLATED == "isolated"


  def test_point_ids_deterministic_and_distinct():
      assert point_id_for_chunk("m1:0") == point_id_for_chunk("m1:0")
      assert point_id_for_chunk("m1:0") != point_id_for_chunk("m2:0")


  def test_isolated_vector_ties_on_identical_text(indexed):
      client, fake = indexed
      retriever = DenseRetriever(client, "dense-test", VECTOR_ISOLATED, fake)
      results = retriever.search("alpha beta gamma", k=2)
      assert {r.chunk.chunk_id for r in results} == {"m1:0", "m2:0"}
      assert results[0].score == pytest.approx(1.0)
      assert results[1].score == pytest.approx(1.0)
      assert all(r.source == "dense" for r in results)


  def test_contextual_vector_separates_same_text_in_different_docs(indexed):
      client, fake = indexed
      retriever = DenseRetriever(client, "dense-test", VECTOR_CONTEXTUAL, fake)
      results = retriever.search("alpha beta gamma", k=2)
      assert results[0].chunk.chunk_id == "m2:0"        # single-chunk doc: context == chunk
      assert results[0].score == pytest.approx(1.0)
      assert results[1].score < 0.999                   # doc context shifted m1:0's vector


  def test_payload_round_trips_full_chunk(indexed):
      client, fake = indexed
      retriever = DenseRetriever(client, "dense-test", VECTOR_ISOLATED, fake)
      top = retriever.search("delta epsilon zeta", k=1)[0]
      assert top.chunk == C_M1B


  def test_k_larger_than_point_count(indexed):
      client, fake = indexed
      retriever = DenseRetriever(client, "dense-test", VECTOR_ISOLATED, fake)
      assert len(retriever.search("alpha beta gamma", k=10)) == 3


  def test_embed_failure_propagates_vendor_unavailable(indexed):
      client, _ = indexed
      retriever = DenseRetriever(client, "dense-test", VECTOR_ISOLATED,
                                 FakeEmbed(fail_query=True))
      with pytest.raises(VendorUnavailable):
          retriever.search("anything", k=1)


  def test_write_dense_index_rejects_empty_or_mismatched():
      client = QdrantClient(":memory:")
      with pytest.raises(ValueError):
          write_dense_index(client, "x", [], [], [])
      with pytest.raises(ValueError):
          write_dense_index(client, "x", CHUNKS, [[0.0] * 8] * 2, [[0.0] * 8] * 3)
  ```
- [ ] Run it and watch it fail: `uv run pytest tests/test_dense.py -q` → expect `ModuleNotFoundError: No module named 'ragreceipts.ingest.indexer'`.
- [ ] Create `api/ragreceipts/retrieval/dense.py` (complete file; qdrant API verified — see Context table):
  ```python
  """Dense retrieval over Qdrant named vectors.

  Both vector sets live on the same points (names below); IngestConfig.contextual selects
  which one to search via vector_name_for(). Point ids are uuid5 of the chunk_id (Qdrant
  requires int/UUID ids); the full Chunk — including the R3 start_token/end_token
  token-range fields — is stored as payload and reconstructed on read.
  """

  import uuid

  from qdrant_client import QdrantClient

  from ragreceipts.types import Chunk, ScoredChunk
  from ragreceipts.vendors.base import EmbedTransport

  VECTOR_CONTEXTUAL = "contextual"
  VECTOR_ISOLATED = "isolated"


  def vector_name_for(contextual: bool) -> str:
      return VECTOR_CONTEXTUAL if contextual else VECTOR_ISOLATED


  def point_id_for_chunk(chunk_id: str) -> str:
      return str(uuid.uuid5(uuid.NAMESPACE_URL, f"ragreceipts:{chunk_id}"))


  class DenseRetriever:
      def __init__(self, client: QdrantClient, collection: str, vector_name: str,
                   embed: EmbedTransport):
          self._client = client
          self._collection = collection
          self._vector_name = vector_name
          self._embed = embed

      def search(self, query: str, k: int) -> list[ScoredChunk]:
          if k <= 0:
              return []
          vector = self._embed.embed_query(query)   # may raise VendorUnavailable
          response = self._client.query_points(
              collection_name=self._collection, query=vector,
              using=self._vector_name, limit=k, with_payload=True,
          )
          results: list[ScoredChunk] = []
          for point in response.points:
              payload = point.payload or {}
              chunk = Chunk(
                  chunk_id=payload["chunk_id"], corpus_id=payload["corpus_id"],
                  doc_id=payload["doc_id"], passage_id=payload["passage_id"],
                  text=payload["text"], position=int(payload["position"]),
                  start_token=int(payload["start_token"]),
                  end_token=int(payload["end_token"]),
              )
              results.append(ScoredChunk(chunk=chunk, score=float(point.score), source="dense"))
          return results
  ```
- [ ] Create `api/ragreceipts/ingest/indexer.py` (complete file):
  ```python
  """Dense index writer: BOTH vector sets as named vectors on the same points, every ingest.

  payload = asdict(chunk), so the R3 start_token/end_token fields ride along
  automatically and DenseRetriever can reconstruct the full Chunk."""

  from dataclasses import asdict

  from qdrant_client import QdrantClient, models

  from ragreceipts.retrieval.dense import VECTOR_CONTEXTUAL, VECTOR_ISOLATED, point_id_for_chunk
  from ragreceipts.types import Chunk


  def write_dense_index(client: QdrantClient, collection: str, chunks: list[Chunk],
                        contextual_vectors: list[list[float]],
                        isolated_vectors: list[list[float]]) -> None:
      if not chunks:
          raise ValueError("cannot write a dense index from zero chunks")
      if not (len(chunks) == len(contextual_vectors) == len(isolated_vectors)):
          raise ValueError(
              f"chunk/vector count mismatch: {len(chunks)} chunks, "
              f"{len(contextual_vectors)} contextual, {len(isolated_vectors)} isolated"
          )
      dim = len(contextual_vectors[0])
      if client.collection_exists(collection):
          client.delete_collection(collection)     # full rebuild semantics, same as sparse
      client.create_collection(
          collection_name=collection,
          vectors_config={
              VECTOR_CONTEXTUAL: models.VectorParams(size=dim, distance=models.Distance.COSINE),
              VECTOR_ISOLATED: models.VectorParams(size=dim, distance=models.Distance.COSINE),
          },
      )
      points = [
          models.PointStruct(
              id=point_id_for_chunk(chunk.chunk_id),
              vector={VECTOR_CONTEXTUAL: ctx, VECTOR_ISOLATED: iso},
              payload=asdict(chunk),
          )
          for chunk, ctx, iso in zip(chunks, contextual_vectors, isolated_vectors, strict=True)
      ]
      client.upsert(collection_name=collection, points=points)
  ```
- [ ] Run again: `uv run pytest tests/test_dense.py -q` → expect `8 passed`.
- [ ] Lint and commit:
  ```bash
  uv run ruff format ragreceipts tests && uv run ruff check ragreceipts tests
  git add ragreceipts/retrieval/dense.py ragreceipts/ingest/indexer.py tests/test_dense.py
  git commit -m "feat: add qdrant dense retriever with named vectors" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
  ```

### Task 9: HybridRRF with hand-computed golden tests

**Files:**
- Create: `api/ragreceipts/retrieval/fusion.py`
- Test: `api/tests/test_fusion.py`

Golden math (rrf_k=60, 1-based ranks, per contracts). Lists: R1 = [A, B, C], R2 = [B, D].
A = 1/61 ≈ 0.016393 · B = 1/62 + 1/61 ≈ 0.032522 · C = 1/63 ≈ 0.015873 · D = 1/62 ≈ 0.016129.
Expected order: **B, A, D, C**.

- [ ] Write the failing test `api/tests/test_fusion.py` (complete file):
  ```python
  """Hand-computed golden tests for reciprocal rank fusion (rrf_k=60, 1-based ranks)."""

  import pytest

  from ragreceipts.retrieval.fusion import HybridRRF
  from ragreceipts.types import ScoredChunk
  from tests.corpus_fixtures import make_chunk

  A = make_chunk("a:0", "text a")
  B = make_chunk("b:0", "text b")
  C = make_chunk("c:0", "text c")
  D = make_chunk("d:0", "text d")


  class ListRetriever:
      """Test stand-in for the Retriever protocol: returns a fixed ranked list."""

      def __init__(self, ranked, source="bm25"):
          self._ranked = [ScoredChunk(chunk=ch, score=10.0 - i, source=source)
                          for i, ch in enumerate(ranked)]

      def search(self, query: str, k: int):
          return self._ranked[:k]


  def test_golden_rrf_scores_and_order():
      fused = HybridRRF([ListRetriever([A, B, C]), ListRetriever([B, D])]).search("q", k=4)
      assert [s.chunk.chunk_id for s in fused] == ["b:0", "a:0", "d:0", "c:0"]
      assert fused[0].score == pytest.approx(1 / 62 + 1 / 61)
      assert fused[1].score == pytest.approx(1 / 61)
      assert fused[2].score == pytest.approx(1 / 62)
      assert fused[3].score == pytest.approx(1 / 63)
      assert all(s.source == "rrf" for s in fused)


  def test_truncates_to_k():
      fused = HybridRRF([ListRetriever([A, B, C]), ListRetriever([B, D])]).search("q", k=2)
      assert [s.chunk.chunk_id for s in fused] == ["b:0", "a:0"]


  def test_each_retriever_consulted_with_k():
      calls = []

      class Spy(ListRetriever):
          def search(self, query, k):
              calls.append(k)
              return super().search(query, k)

      HybridRRF([Spy([A]), Spy([B])]).search("q", k=7)
      assert calls == [7, 7]


  def test_ties_break_deterministically_by_chunk_id():
      fused = HybridRRF([ListRetriever([B]), ListRetriever([A])]).search("q", k=2)
      assert [s.chunk.chunk_id for s in fused] == ["a:0", "b:0"]   # equal 1/61, id ascending
      assert fused[0].score == pytest.approx(fused[1].score)


  def test_custom_rrf_k():
      fused = HybridRRF([ListRetriever([A])], rrf_k=10).search("q", k=1)
      assert fused[0].score == pytest.approx(1 / 11)


  def test_requires_at_least_one_retriever():
      with pytest.raises(ValueError):
          HybridRRF([])
  ```
- [ ] Run it and watch it fail: `uv run pytest tests/test_fusion.py -q` → expect `ModuleNotFoundError: No module named 'ragreceipts.retrieval.fusion'`.
- [ ] Create `api/ragreceipts/retrieval/fusion.py` (complete file):
  ```python
  """Reciprocal Rank Fusion over any set of Retrievers.

  RRF score for a chunk = sum(1 / (rrf_k + rank_i)) over the rank lists containing it,
  rank 1-based (binding definition from contracts). Ties break by chunk_id ascending so
  fusion is fully deterministic — receipts must be reproducible.
  """

  from ragreceipts.retrieval.base import Retriever
  from ragreceipts.types import ScoredChunk


  class HybridRRF:
      def __init__(self, retrievers: list[Retriever], rrf_k: int = 60):
          if not retrievers:
              raise ValueError("HybridRRF needs at least one retriever")
          self._retrievers = retrievers
          self._rrf_k = rrf_k

      def search(self, query: str, k: int) -> list[ScoredChunk]:
          scores: dict[str, float] = {}
          chunk_by_id = {}
          for retriever in self._retrievers:
              for rank, scored in enumerate(retriever.search(query, k), start=1):
                  chunk_id = scored.chunk.chunk_id
                  chunk_by_id[chunk_id] = scored.chunk
                  scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (self._rrf_k + rank)
          ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
          return [ScoredChunk(chunk=chunk_by_id[chunk_id], score=score, source="rrf")
                  for chunk_id, score in ordered[:k]]
  ```
- [ ] Run again: `uv run pytest tests/test_fusion.py -q` → expect `6 passed`.
- [ ] Lint and commit:
  ```bash
  uv run ruff format ragreceipts tests && uv run ruff check ragreceipts tests
  git add ragreceipts/retrieval/fusion.py tests/test_fusion.py
  git commit -m "feat: add reciprocal rank fusion retriever" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
  ```

### Task 10: RerankStage

**Files:**
- Create: `api/ragreceipts/retrieval/rerank.py`
- Test: `api/tests/test_rerank_stage.py`

- [ ] Write the failing test `api/tests/test_rerank_stage.py` (complete file):
  ```python
  """RerankStage maps transport (index, score) pairs back onto candidate chunks."""

  import pytest

  from ragreceipts.retrieval.rerank import RerankStage
  from ragreceipts.types import ScoredChunk
  from ragreceipts.vendors.base import VendorUnavailable
  from tests.corpus_fixtures import make_chunk
  from tests.fakes import FakeRerank

  CANDIDATES = [
      ScoredChunk(chunk=make_chunk("a:0", "text a"), score=0.03, source="rrf"),
      ScoredChunk(chunk=make_chunk("b:0", "text b"), score=0.02, source="rrf"),
      ScoredChunk(chunk=make_chunk("c:0", "text c"), score=0.01, source="rrf"),
  ]


  def test_reorders_per_transport_and_relabels_source():
      stage = RerankStage(FakeRerank(script={"q": [2, 0, 1]}))
      got = stage.rerank("q", CANDIDATES, top_n=3)
      assert [s.chunk.chunk_id for s in got] == ["c:0", "a:0", "b:0"]
      assert [s.score for s in got] == pytest.approx([1.0, 0.99, 0.98])
      assert all(s.source == "rerank" for s in got)


  def test_passes_chunk_texts_and_top_n_to_transport():
      fake = FakeRerank()
      RerankStage(fake).rerank("q", CANDIDATES, top_n=2)
      query, texts, top_n = fake.calls[0]
      assert (query, texts, top_n) == ("q", ["text a", "text b", "text c"], 2)


  def test_truncates_to_top_n():
      got = RerankStage(FakeRerank()).rerank("q", CANDIDATES, top_n=2)
      assert len(got) == 2


  def test_empty_candidates_short_circuits():
      fake = FakeRerank()
      assert RerankStage(fake).rerank("q", [], top_n=5) == []
      assert fake.calls == []


  def test_transport_failure_propagates():
      with pytest.raises(VendorUnavailable):
          RerankStage(FakeRerank(fail=True)).rerank("q", CANDIDATES, top_n=2)
  ```
- [ ] Run it and watch it fail: `uv run pytest tests/test_rerank_stage.py -q` → expect `ModuleNotFoundError: No module named 'ragreceipts.retrieval.rerank'`.
- [ ] Create `api/ragreceipts/retrieval/rerank.py` (complete file):
  ```python
  """Rerank is a stage, not a Retriever (contracts). Degradation on VendorUnavailable is
  RetrievalCore's job — this stage propagates the exception."""

  from ragreceipts.types import ScoredChunk
  from ragreceipts.vendors.base import RerankTransport


  class RerankStage:
      def __init__(self, transport: RerankTransport):
          self._transport = transport

      def rerank(self, query: str, candidates: list[ScoredChunk],
                 top_n: int) -> list[ScoredChunk]:
          if not candidates:
              return []
          ranked = self._transport.rerank(query, [c.chunk.text for c in candidates], top_n)
          return [ScoredChunk(chunk=candidates[index].chunk, score=float(score),
                              source="rerank")
                  for index, score in ranked[:top_n]]
  ```
- [ ] Run again: `uv run pytest tests/test_rerank_stage.py -q` → expect `5 passed`.
- [ ] Lint and commit:
  ```bash
  uv run ruff format ragreceipts tests && uv run ruff check ragreceipts tests
  git add ragreceipts/retrieval/rerank.py tests/test_rerank_stage.py
  git commit -m "feat: add rerank stage over rerank transport" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
  ```

### Task 11: RetrievalCore — flags, degradation, trace emission

**Files:**
- Create: `api/ragreceipts/retrieval/core.py`
- Test: `api/tests/test_core.py`

The flag-flip fixture: query `"capital of france"`; `C_LEX` ("france capital city facts about the capital of france") wins BM25 (tf=2 for both content terms); `C_SEM` ("the eiffel tower attracts millions of visitors") has ZERO lexical overlap but wins dense via `FakeEmbed(query_aliases=...)`; `C_MID` ("paris is the capital of france") is the middle lexical hit. Guaranteed hybrid RRF order (regardless of how dense ranks the two lexical chunks at ranks 2/3): C_LEX ≥ 1/61+1/63 > C_MID ≤ 1/62+1/62 > C_SEM = 1/61. `route_mode` deliberately does NOT change `RetrievalCore` output — it is consumed by the Plan C router; the test pins that invariant explicitly.

- [ ] Write the failing test `api/tests/test_core.py` (complete file):
  ```python
  """Flag-flip tests: every QueryConfig flag provably changes RetrievalCore behavior
  (route_mode provably does NOT — it belongs to the Plan C router). Plus degraded paths
  and TraceEvent emission."""

  import pytest
  from qdrant_client import QdrantClient

  from ragreceipts.config import IngestConfig, PipelineConfig, QueryConfig
  from ragreceipts.ingest.indexer import write_dense_index
  from ragreceipts.retrieval.core import RetrievalCore
  from ragreceipts.retrieval.dense import VECTOR_ISOLATED, DenseRetriever
  from ragreceipts.retrieval.rerank import RerankStage
  from ragreceipts.retrieval.sparse import build_sparse_index
  from ragreceipts.traces.models import TraceEvent
  from ragreceipts.types import RouteMode
  from ragreceipts.vendors.base import VendorUnavailable
  from tests.corpus_fixtures import make_chunk
  from tests.fakes import FakeEmbed, FakeRerank

  QUERY = "capital of france"
  C_LEX = make_chunk("d1:0", "france capital city facts about the capital of france",
                     corpus_id="flagflip", passage_id="d1-p0")
  C_SEM = make_chunk("d2:0", "the eiffel tower attracts millions of visitors",
                     corpus_id="flagflip", passage_id="d2-p0")
  C_MID = make_chunk("d3:0", "paris is the capital of france",
                     corpus_id="flagflip", passage_id="d3-p0")
  CHUNKS = [C_LEX, C_SEM, C_MID]


  def qc(**overrides) -> QueryConfig:
      base = dict(bm25=True, dense=True, rerank=False, route_mode=RouteMode.FORCE_S1,
                  top_k_fuse=3, top_k_final=3)
      base.update(overrides)
      return QueryConfig(**base)


  @pytest.fixture()
  def stack(tmp_path):
      fake = FakeEmbed(query_aliases={QUERY: C_SEM.text})
      sparse = build_sparse_index(CHUNKS, tmp_path / "sparse")
      client = QdrantClient(":memory:")
      vectors = [doc[0] for doc in fake.embed_documents([[c.text] for c in CHUNKS])]
      write_dense_index(client, "flagflip", CHUNKS, vectors, vectors)
      dense = DenseRetriever(client, "flagflip", VECTOR_ISOLATED, fake)
      return {"sparse": sparse, "dense": dense, "client": client}


  def make_core(stack, query: QueryConfig, *, dense=None, rerank_fail=False,
                on_trace=None) -> RetrievalCore:
      config = PipelineConfig(name="test", ingest=IngestConfig(contextual=False), query=query)
      return RetrievalCore(config, dense or stack["dense"], stack["sparse"],
                           RerankStage(FakeRerank(fail=rerank_fail)), on_trace=on_trace)


  def ids(results):
      return [r.chunk.chunk_id for r in results]


  def test_bm25_only_flag(stack):
      results = make_core(stack, qc(dense=False)).retrieve(QUERY)
      assert ids(results) == ["d1:0", "d3:0"]
      assert all(r.source == "bm25" for r in results)
      assert "d2:0" not in ids(results)


  def test_dense_only_flag(stack):
      results = make_core(stack, qc(bm25=False)).retrieve(QUERY)
      assert ids(results)[0] == "d2:0"
      assert results[0].source == "dense"
      assert results[0].score > 0.99


  def test_hybrid_fuses_both_flags(stack):
      results = make_core(stack, qc()).retrieve(QUERY)
      assert ids(results) == ["d1:0", "d3:0", "d2:0"]
      assert all(r.source == "rrf" for r in results)


  def test_rerank_flag_reorders(stack):
      base = make_core(stack, qc()).retrieve(QUERY)
      reranked = make_core(stack, qc(rerank=True)).retrieve(QUERY)
      assert ids(reranked) == list(reversed(ids(base)))   # FakeRerank default reverses
      assert all(r.source == "rerank" for r in reranked)


  def test_top_k_final_flag(stack):
      assert len(make_core(stack, qc(top_k_final=2)).retrieve(QUERY)) == 2
      assert len(make_core(stack, qc(top_k_final=3)).retrieve(QUERY)) == 3


  def test_top_k_fuse_flag(stack):
      narrow = make_core(stack, qc(top_k_fuse=1)).retrieve(QUERY)
      wide = make_core(stack, qc(top_k_fuse=3)).retrieve(QUERY)
      assert ids(narrow) == ["d1:0", "d2:0"]   # one candidate per retriever, id tie-break
      assert "d3:0" not in ids(narrow)
      assert "d3:0" in ids(wide)


  def test_route_mode_does_not_change_retrieval_core(stack):
      s1 = make_core(stack, qc(route_mode=RouteMode.FORCE_S1)).retrieve(QUERY)
      auto = make_core(stack, qc(route_mode=RouteMode.AUTO)).retrieve(QUERY)
      assert ids(s1) == ids(auto)   # route_mode is consumed by the Plan C router only


  def test_dense_failure_degrades_to_bm25_with_flag(stack):
      events: list[TraceEvent] = []
      failing = DenseRetriever(stack["client"], "flagflip", VECTOR_ISOLATED,
                               FakeEmbed(fail_query=True))
      results = make_core(stack, qc(), dense=failing, on_trace=events.append).retrieve(QUERY)
      assert ids(results) == ["d1:0", "d3:0"]
      assert all(r.source == "bm25" for r in results)
      assert events[0].payload["degraded"] == ["dense-skipped"]


  def test_rerank_failure_degrades_to_rrf_order(stack):
      events: list[TraceEvent] = []
      core = make_core(stack, qc(rerank=True), rerank_fail=True, on_trace=events.append)
      results = core.retrieve(QUERY)
      assert ids(results) == ["d1:0", "d3:0", "d2:0"]    # RRF order preserved
      assert all(r.source == "rrf" for r in results)
      assert events[0].payload["degraded"] == ["rerank-skipped"]


  def test_dense_only_failure_raises(stack):
      failing = DenseRetriever(stack["client"], "flagflip", VECTOR_ISOLATED,
                               FakeEmbed(fail_query=True))
      core = make_core(stack, qc(bm25=False), dense=failing)
      with pytest.raises(VendorUnavailable):
          core.retrieve(QUERY)                            # no bm25 to fall back to


  def test_invalid_configs_rejected(stack):
      with pytest.raises(ValueError):
          make_core(stack, qc(bm25=False, dense=False))
      config = PipelineConfig(name="t", ingest=IngestConfig(), query=qc())
      with pytest.raises(ValueError):
          RetrievalCore(config, None, stack["sparse"], None)   # dense flag on, none given


  def test_trace_event_payload_and_threading(stack):
      events: list[TraceEvent] = []
      core = make_core(stack, qc(rerank=True), on_trace=events.append)
      results = core.retrieve(QUERY, trace_id="t-123", node="retrieve_hop", seq_start=7)
      assert len(events) == 1
      event = events[0]
      assert (event.trace_id, event.node, event.seq) == ("t-123", "retrieve_hop", 7)
      assert event.model is None and event.input_tokens == 0 and event.output_tokens == 0
      assert event.duration_ms >= 0.0
      assert event.payload["query"] == QUERY
      assert event.payload["config"]["rerank"] is True
      assert [r["chunk_id"] for r in event.payload["results"]] == ids(results)
      assert len(event.payload["candidates"]) == 3
      assert event.payload["degraded"] == []


  def test_trace_id_generated_when_absent(stack):
      events: list[TraceEvent] = []
      make_core(stack, qc(), on_trace=events.append).retrieve(QUERY)
      assert events[0].node == "s1_retrieve"
      assert len(events[0].trace_id) == 32                # uuid4().hex
  ```
- [ ] Run it and watch it fail: `uv run pytest tests/test_core.py -q` → expect `ModuleNotFoundError: No module named 'ragreceipts.retrieval.core'`.
- [ ] Create `api/ragreceipts/retrieval/core.py` (complete file):
  ```python
  """RetrievalCore: the single composed retrieval entry point (contracts).

  System-1, System-2 hops, and the eval harness all execute THIS code, parameterized only
  by PipelineConfig (shared-retrieval-core invariant). Honors config.query flags, returns
  top_k_final chunks, emits one TraceEvent per call through the injected TraceCallback.
  Degrade visibly, never silently: dense failure -> BM25-only + "dense-skipped";
  rerank failure -> RRF order + "rerank-skipped"; no fallback available -> raise.
  route_mode is deliberately ignored here — it is consumed by the Plan C router.
  """

  import time
  import uuid

  from ragreceipts.config import PipelineConfig
  from ragreceipts.retrieval.base import Retriever
  from ragreceipts.retrieval.fusion import HybridRRF
  from ragreceipts.retrieval.rerank import RerankStage
  from ragreceipts.traces.models import TraceCallback, TraceEvent
  from ragreceipts.types import ScoredChunk
  from ragreceipts.vendors.base import VendorUnavailable


  class RetrievalCore:
      def __init__(self, config: PipelineConfig, dense: Retriever | None,
                   sparse: Retriever | None, rerank_stage: RerankStage | None,
                   on_trace: TraceCallback | None = None):
          query = config.query
          if not query.bm25 and not query.dense:
              raise ValueError("config must enable at least one of bm25/dense")
          if query.bm25 and sparse is None:
              raise ValueError("config enables bm25 but no sparse retriever was provided")
          if query.dense and dense is None:
              raise ValueError("config enables dense but no dense retriever was provided")
          if query.rerank and rerank_stage is None:
              raise ValueError("config enables rerank but no rerank stage was provided")
          self._config = config
          self._dense = dense
          self._sparse = sparse
          self._rerank_stage = rerank_stage
          self._on_trace = on_trace

      def retrieve(self, query: str, *, trace_id: str | None = None,
                   node: str = "s1_retrieve", seq_start: int = 0) -> list[ScoredChunk]:
          q = self._config.query
          started = time.perf_counter()
          degraded: list[str] = []

          retrievers: list[Retriever] = []
          if q.bm25:
              retrievers.append(self._sparse)
          if q.dense:
              retrievers.append(self._dense)

          try:
              candidates = self._fused_search(retrievers, query, q.top_k_fuse)
          except VendorUnavailable:
              if not q.bm25:
                  raise                       # nothing to fall back to: fail visibly
              degraded.append("dense-skipped")
              candidates = self._sparse.search(query, q.top_k_fuse)

          if q.rerank:
              try:
                  final = self._rerank_stage.rerank(query, candidates, q.top_k_final)
              except VendorUnavailable:
                  degraded.append("rerank-skipped")
                  final = candidates[: q.top_k_final]
          else:
              final = candidates[: q.top_k_final]

          self._emit(query, candidates, final, degraded,
                     trace_id=trace_id or uuid.uuid4().hex, node=node, seq=seq_start,
                     duration_ms=(time.perf_counter() - started) * 1000.0)
          return final

      @staticmethod
      def _fused_search(retrievers: list[Retriever], query: str, k: int) -> list[ScoredChunk]:
          if len(retrievers) == 1:            # passthrough keeps source labels honest
              return retrievers[0].search(query, k)
          return HybridRRF(retrievers, rrf_k=60).search(query, k)

      def _emit(self, query: str, candidates: list[ScoredChunk], final: list[ScoredChunk],
                degraded: list[str], *, trace_id: str, node: str, seq: int,
                duration_ms: float) -> None:
          if self._on_trace is None:
              return
          q = self._config.query
          self._on_trace(TraceEvent(
              trace_id=trace_id,
              seq=seq,
              node=node,
              payload={
                  "query": query,
                  "config": {"bm25": q.bm25, "dense": q.dense, "rerank": q.rerank,
                             "route_mode": q.route_mode.value,
                             "top_k_fuse": q.top_k_fuse, "top_k_final": q.top_k_final},
                  "candidates": [{"chunk_id": c.chunk.chunk_id, "score": c.score,
                                  "source": c.source} for c in candidates],
                  "results": [{"chunk_id": c.chunk.chunk_id, "score": c.score,
                               "source": c.source} for c in final],
                  "degraded": degraded,
              },
              model=None,
              input_tokens=0,
              output_tokens=0,
              duration_ms=duration_ms,
          ))
  ```
- [ ] Run again: `uv run pytest tests/test_core.py -q` → expect `13 passed`.
- [ ] Run the whole suite to catch regressions: `uv run pytest -q` → expect all tests from Tasks 1–11 passing.
- [ ] Lint and commit:
  ```bash
  uv run ruff format ragreceipts tests && uv run ruff check ragreceipts tests
  git add ragreceipts/retrieval/core.py tests/test_core.py
  git commit -m "feat: add RetrievalCore honoring query-time flags with trace emission" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
  ```

### Task 12: Retry helper + VoyageClient (contextualized embeddings, token-aware batching)

**Files:**
- Create: `api/ragreceipts/vendors/retry.py`, `api/ragreceipts/vendors/voyage_client.py`
- Test: `api/tests/test_vendor_retry.py`, `api/tests/test_voyage_client.py`

All signatures verified against voyageai 0.4.0 and docs.voyageai.com/docs/contextualized-chunk-embeddings (Context table). Tests inject a stub SDK object — the `voyageai` package is imported (it's a dependency) but never makes a network call; `voyageai.error` exceptions are constructed locally.

- [ ] Write the failing test `api/tests/test_vendor_retry.py` (complete file):
  ```python
  """call_with_retry: backoff, retry-after, predicate, exhaustion -> VendorUnavailable."""

  import pytest

  from ragreceipts.vendors.base import VendorUnavailable
  from ragreceipts.vendors.retry import call_with_retry


  class Flaky(Exception):
      def __init__(self, message="boom", headers=None, status_code=None):
          super().__init__(message)
          self.headers = headers or {}
          self.status_code = status_code


  def test_returns_on_first_success():
      assert call_with_retry(lambda: 42, retryable=(Flaky,), max_attempts=3,
                             sleep=lambda s: None, label="x") == 42


  def test_exponential_backoff_without_retry_after():
      sleeps, attempts = [], []

      def fn():
          attempts.append(1)
          if len(attempts) < 3:
              raise Flaky()
          return "ok"

      assert call_with_retry(fn, retryable=(Flaky,), max_attempts=5,
                             sleep=sleeps.append, label="x") == "ok"
      assert sleeps == [1.0, 2.0]


  def test_retry_after_header_honored():
      sleeps, script = [], [Flaky(headers={"retry-after": "7"}), "ok"]

      def fn():
          item = script.pop(0)
          if isinstance(item, Exception):
              raise item
          return item

      assert call_with_retry(fn, retryable=(Flaky,), max_attempts=3,
                             sleep=sleeps.append, label="x") == "ok"
      assert sleeps == [7.0]


  def test_exhaustion_raises_vendor_unavailable_with_cause():
      def fn():
          raise Flaky("always")

      with pytest.raises(VendorUnavailable) as exc_info:
          call_with_retry(fn, retryable=(Flaky,), max_attempts=2,
                          sleep=lambda s: None, label="voyage")
      assert "voyage" in str(exc_info.value)
      assert isinstance(exc_info.value.__cause__, Flaky)


  def test_should_retry_predicate_reraises_non_retryable():
      def fn():
          raise Flaky("bad request", status_code=400)

      with pytest.raises(Flaky):
          call_with_retry(fn, retryable=(Flaky,), max_attempts=3, sleep=lambda s: None,
                          label="x", should_retry=lambda e: e.status_code == 429)


  def test_unlisted_exceptions_propagate():
      def fn():
          raise KeyError("not a vendor error")

      with pytest.raises(KeyError):
          call_with_retry(fn, retryable=(Flaky,), max_attempts=3,
                          sleep=lambda s: None, label="x")
  ```
- [ ] Write the failing test `api/tests/test_voyage_client.py` (complete file):
  ```python
  """VoyageClient: batch planning against the 120K/16K/1K caps, per-chunk 32K cap,
  retry honoring retry-after, doc-grouped vs query embedding calls."""

  from types import SimpleNamespace

  import pytest
  import voyageai.error

  from ragreceipts.vendors.base import VendorUnavailable
  from ragreceipts.vendors.voyage_client import (
      MAX_TOKENS_PER_CHUNK,
      MAX_TOKENS_PER_REQUEST,
      VoyageClient,
      plan_batches,
  )


  def embed_response(per_doc_chunk_counts: list[int], dim: int = 4) -> SimpleNamespace:
      results = [
          SimpleNamespace(index=i,
                          embeddings=[[float(i), float(j)] + [0.0] * (dim - 2)
                                      for j in range(n)])
          for i, n in enumerate(per_doc_chunk_counts)
      ]
      return SimpleNamespace(results=results, total_tokens=0)


  class StubSdk:
      """Scripted voyageai.Client stand-in: pops responses; raises Exception items."""

      def __init__(self, script):
          self.script = list(script)
          self.calls: list[dict] = []

      def contextualized_embed(self, *, inputs, model, input_type):
          self.calls.append({"inputs": inputs, "model": model, "input_type": input_type})
          item = self.script.pop(0)
          if isinstance(item, Exception):
              raise item
          return item


  def words(texts: list[str]) -> int:
      return sum(len(t.split()) for t in texts)


  class TestPlanBatches:
      def test_splits_on_token_budget(self):
          docs = [["a"], ["b"], ["c"]]
          assert plan_batches(docs, [50, 60, 30], max_tokens=100) == [(0, 1), (1, 3)]

      def test_splits_on_chunk_cap(self):
          docs = [["x"] * 3, ["y"] * 3]
          assert plan_batches(docs, [1, 1], max_chunks=4) == [(0, 1), (1, 2)]

      def test_splits_on_doc_cap(self):
          docs = [["a"], ["b"], ["c"]]
          assert plan_batches(docs, [1, 1, 1], max_docs=2) == [(0, 2), (2, 3)]

      def test_single_doc_over_budget_raises(self):
          with pytest.raises(ValueError):
              plan_batches([["a"]], [MAX_TOKENS_PER_REQUEST + 1])

      def test_empty(self):
          assert plan_batches([], []) == []


  class TestEmbedDocuments:
      def test_batches_and_reassembles_in_order(self):
          # two 10-chunk docs at 7K tokens/chunk = 70K tokens/doc -> two requests
          docs = [[f"d{d}c{c}" for c in range(10)] for d in range(2)]
          sdk = StubSdk([embed_response([10]), embed_response([10])])
          client = VoyageClient(sdk=sdk, count_tokens=lambda texts: 7_000 * len(texts))
          out = client.embed_documents(docs)
          assert len(sdk.calls) == 2
          assert sdk.calls[0]["inputs"] == docs[:1]
          assert sdk.calls[0]["model"] == "voyage-context-3"
          assert sdk.calls[0]["input_type"] == "document"
          assert len(out) == 2 and [len(d) for d in out] == [10, 10]

      def test_per_chunk_token_cap_enforced(self):
          sdk = StubSdk([])
          client = VoyageClient(sdk=sdk,
                                count_tokens=lambda texts: MAX_TOKENS_PER_CHUNK + 1)
          with pytest.raises(ValueError):
              client.embed_documents([["one oversized chunk"]])
          assert sdk.calls == []

      def test_empty_input(self):
          assert VoyageClient(sdk=StubSdk([]), count_tokens=words).embed_documents([]) == []

      def test_retry_honors_retry_after_then_succeeds(self):
          sleeps = []
          sdk = StubSdk([
              voyageai.error.RateLimitError("rate limited", http_status=429,
                                            headers={"retry-after": "2"}),
              embed_response([1]),
          ])
          client = VoyageClient(sdk=sdk, count_tokens=words, sleep=sleeps.append)
          out = client.embed_documents([["tiny doc"]])
          assert sleeps == [2.0]
          assert len(out[0][0]) == 4

      def test_exhaustion_raises_vendor_unavailable(self):
          errors = [voyageai.error.ServerError("boom", http_status=500)] * 2
          client = VoyageClient(sdk=StubSdk(errors), count_tokens=words,
                                sleep=lambda s: None, max_attempts=2)
          with pytest.raises(VendorUnavailable):
              client.embed_documents([["tiny doc"]])


  class TestEmbedQuery:
      def test_query_call_shape_and_result(self):
          sdk = StubSdk([embed_response([1])])
          got = VoyageClient(sdk=sdk, count_tokens=words).embed_query("what is rrf?")
          assert sdk.calls[0]["inputs"] == [["what is rrf?"]]
          assert sdk.calls[0]["input_type"] == "query"
          assert got == [0.0, 0.0, 0.0, 0.0]
  ```
- [ ] Run both and watch them fail: `uv run pytest tests/test_vendor_retry.py tests/test_voyage_client.py -q` → expect `ModuleNotFoundError: No module named 'ragreceipts.vendors.retry'`.
- [ ] Create `api/ragreceipts/vendors/retry.py` (complete file):
  ```python
  """Shared vendor retry loop: exponential backoff (1s doubling, 30s cap) that honors a
  retry-after header when the exception carries one. After max_attempts -> VendorUnavailable
  (the signal RetrievalCore degrades on). Exceptions outside `retryable`, or rejected by
  `should_retry`, propagate untouched — programming errors must stay loud."""

  from collections.abc import Callable

  from ragreceipts.vendors.base import VendorUnavailable


  def retry_after_seconds(err: BaseException) -> float | None:
      headers = getattr(err, "headers", None) or {}
      raw = headers.get("retry-after") or headers.get("Retry-After")
      if raw is None:
          return None
      try:
          return float(raw)
      except (TypeError, ValueError):
          return None


  def call_with_retry(fn: Callable[[], object], *,
                      retryable: tuple[type[BaseException], ...],
                      max_attempts: int,
                      sleep: Callable[[float], None],
                      label: str,
                      should_retry: Callable[[BaseException], bool] | None = None):
      delay = 1.0
      last: BaseException | None = None
      for attempt in range(max_attempts):
          try:
              return fn()
          except retryable as err:
              if should_retry is not None and not should_retry(err):
                  raise
              last = err
              if attempt == max_attempts - 1:
                  break
              wait = retry_after_seconds(err)
              sleep(wait if wait is not None else delay)
              delay = min(delay * 2.0, 30.0)
      raise VendorUnavailable(f"{label} failed after {max_attempts} attempts: {last!r}") from last
  ```
- [ ] Create `api/ragreceipts/vendors/voyage_client.py` (complete file):
  ```python
  """EmbedTransport over the official voyageai SDK — contextualized chunk embeddings.

  Binding verified 2026-06-10 against
  https://docs.voyageai.com/docs/contextualized-chunk-embeddings and voyageai 0.4.0:
  client.contextualized_embed(inputs=list[list[str]], model=..., input_type=...) returns
  .results (each .index/.embeddings); query embedding = inputs=[[query]], input_type="query".
  Per-request caps: 1,000 docs / 16,000 chunks / 120,000 total tokens / 32K tokens-per-chunk.
  The 120K context window doubles as the per-request budget (spec ingestion plane).
  Isolated mode is the CALLER passing single-chunk documents (vendors/base.py contract).
  SDK auto-retry is disabled (max_retries=0); retry.call_with_retry owns backoff so the
  retry-after header is honored (the SDK does not honor it itself).
  """

  import time
  from collections.abc import Callable

  import voyageai
  import voyageai.error

  from ragreceipts.constants import EMBED_MODEL
  from ragreceipts.vendors.retry import call_with_retry

  MAX_TOKENS_PER_REQUEST = 120_000
  MAX_CHUNKS_PER_REQUEST = 16_000
  MAX_DOCS_PER_REQUEST = 1_000
  MAX_TOKENS_PER_CHUNK = 32_000

  _RETRYABLE = (
      voyageai.error.RateLimitError,
      voyageai.error.ServerError,
      voyageai.error.ServiceUnavailableError,
      voyageai.error.APIConnectionError,
  )


  def plan_batches(documents: list[list[str]], doc_token_counts: list[int],
                   max_tokens: int = MAX_TOKENS_PER_REQUEST,
                   max_chunks: int = MAX_CHUNKS_PER_REQUEST,
                   max_docs: int = MAX_DOCS_PER_REQUEST) -> list[tuple[int, int]]:
      """Greedy contiguous batching -> [start, end) doc-index pairs. A document never
      spans requests (Voyage contextualizes within a single request). Documents larger
      than the whole budget must be split upstream (spec: BYO docs >120K tokens are split
      into logical documents at ingest — Plan D concern, disclosed in the manifest)."""
      batches: list[tuple[int, int]] = []
      start = 0
      tokens = 0
      chunks = 0
      for i, doc in enumerate(documents):
          doc_tokens = doc_token_counts[i]
          if doc_tokens > max_tokens:
              raise ValueError(
                  f"document {i} has {doc_tokens} tokens > {max_tokens} per-request budget; "
                  "split it into logical documents upstream"
              )
          over = i > start and (tokens + doc_tokens > max_tokens
                                or chunks + len(doc) > max_chunks
                                or i - start >= max_docs)
          if over:
              batches.append((start, i))
              start, tokens, chunks = i, 0, 0
          tokens += doc_tokens
          chunks += len(doc)
      if start < len(documents):
          batches.append((start, len(documents)))
      return batches


  class VoyageClient:
      """count_tokens defaults to the SDK's local tokenizer (downloads its vocab from the
      HF hub on FIRST use — fine for live runs, never reached in CI because tests always
      inject count_tokens)."""

      def __init__(self, api_key: str | None = None, model: str = EMBED_MODEL,
                   max_attempts: int = 5,
                   sleep: Callable[[float], None] = time.sleep,
                   sdk: object | None = None,
                   count_tokens: Callable[[list[str]], int] | None = None):
          self._sdk = sdk if sdk is not None else voyageai.Client(api_key=api_key,
                                                                  max_retries=0)
          self._model = model
          self._max_attempts = max_attempts
          self._sleep = sleep
          self._count_tokens = count_tokens or (
              lambda texts: self._sdk.count_tokens(texts, model=self._model)
          )

      def _call(self, fn):
          return call_with_retry(fn, retryable=_RETRYABLE, max_attempts=self._max_attempts,
                                 sleep=self._sleep, label="voyage")

      def embed_documents(self, documents: list[list[str]]) -> list[list[list[float]]]:
          if not documents:
              return []
          doc_token_counts = [self._count_tokens(doc) if doc else 0 for doc in documents]
          for d, (doc, total) in enumerate(zip(documents, doc_token_counts)):
              if total <= MAX_TOKENS_PER_CHUNK:
                  continue                      # no single chunk can exceed the cap
              for c, chunk in enumerate(doc):
                  if self._count_tokens([chunk]) > MAX_TOKENS_PER_CHUNK:
                      raise ValueError(
                          f"doc {d} chunk {c} exceeds the {MAX_TOKENS_PER_CHUNK}-token "
                          "per-chunk cap; re-chunk with a smaller chunk_size"
                      )
          out: list[list[list[float]]] = []
          for start, end in plan_batches(documents, doc_token_counts):
              batch = documents[start:end]
              result = self._call(lambda b=batch: self._sdk.contextualized_embed(
                  inputs=b, model=self._model, input_type="document"))
              ordered = sorted(result.results, key=lambda r: r.index)  # index is per-request
              out.extend([list(emb) for emb in r.embeddings] for r in ordered)
          return out

      def embed_query(self, query: str) -> list[float]:
          result = self._call(lambda: self._sdk.contextualized_embed(
              inputs=[[query]], model=self._model, input_type="query"))
          return list(result.results[0].embeddings[0])
  ```
- [ ] Run again: `uv run pytest tests/test_vendor_retry.py tests/test_voyage_client.py -q` → expect `17 passed`.
- [ ] Lint and commit:
  ```bash
  uv run ruff format ragreceipts tests && uv run ruff check ragreceipts tests
  git add ragreceipts/vendors/retry.py ragreceipts/vendors/voyage_client.py tests/test_vendor_retry.py tests/test_voyage_client.py
  git commit -m "feat: add voyage contextualized embedding client with token-aware batching" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
  ```

### Task 13: CohereClient (rerank-v4.0-pro)

**Files:**
- Create: `api/ragreceipts/vendors/cohere_client.py`
- Test: `api/tests/test_cohere_client.py`

Binding verified against cohere 7.0.3 and https://docs.cohere.com/reference/rerank (Context table). Retryability is decided by `ApiError.status_code` (predicate) rather than by exception subclass, because the fern-generated subclasses' constructors are not part of the verified surface while `ApiError(*, headers=None, status_code=None, body=None)` is.

- [ ] Write the failing test `api/tests/test_cohere_client.py` (complete file):
  ```python
  """CohereClient: rerank call shape, desc-sorted results, status-code-based retry."""

  from types import SimpleNamespace

  import pytest
  from cohere.core.api_error import ApiError

  from ragreceipts.vendors.base import VendorUnavailable
  from ragreceipts.vendors.cohere_client import CohereClient


  def rerank_response(pairs: list[tuple[int, float]]) -> SimpleNamespace:
      return SimpleNamespace(results=[
          SimpleNamespace(index=i, relevance_score=s) for i, s in pairs
      ])


  class StubSdk:
      def __init__(self, script):
          self.script = list(script)
          self.calls: list[dict] = []

      def rerank(self, *, model, query, documents, top_n):
          self.calls.append({"model": model, "query": query,
                             "documents": list(documents), "top_n": top_n})
          item = self.script.pop(0)
          if isinstance(item, Exception):
              raise item
          return item


  def test_call_shape_and_mapping():
      sdk = StubSdk([rerank_response([(2, 0.9), (0, 0.5)])])
      got = CohereClient(sdk=sdk).rerank("q", ["a", "b", "c"], top_n=2)
      assert got == [(2, 0.9), (0, 0.5)]
      assert sdk.calls[0] == {"model": "rerank-v4.0-pro", "query": "q",
                              "documents": ["a", "b", "c"], "top_n": 2}


  def test_results_sorted_desc_even_if_api_misorders():
      sdk = StubSdk([rerank_response([(0, 0.1), (1, 0.8)])])
      assert CohereClient(sdk=sdk).rerank("q", ["a", "b"], top_n=2) == [(1, 0.8), (0, 0.1)]


  def test_retries_429_honoring_retry_after():
      sleeps = []
      sdk = StubSdk([
          ApiError(status_code=429, headers={"retry-after": "3"}, body="slow down"),
          rerank_response([(0, 0.7)]),
      ])
      got = CohereClient(sdk=sdk, sleep=sleeps.append).rerank("q", ["a"], top_n=1)
      assert got == [(0, 0.7)]
      assert sleeps == [3.0]


  def test_5xx_retries_then_vendor_unavailable():
      errors = [ApiError(status_code=503, body="down")] * 2
      client = CohereClient(sdk=StubSdk(errors), sleep=lambda s: None, max_attempts=2)
      with pytest.raises(VendorUnavailable):
          client.rerank("q", ["a"], top_n=1)


  def test_4xx_other_than_429_propagates_unretried():
      sdk = StubSdk([ApiError(status_code=400, body="bad request")])
      with pytest.raises(ApiError):
          CohereClient(sdk=sdk).rerank("q", ["a"], top_n=1)
      assert len(sdk.calls) == 1
  ```
- [ ] Run it and watch it fail: `uv run pytest tests/test_cohere_client.py -q` → expect `ModuleNotFoundError: No module named 'ragreceipts.vendors.cohere_client'`.
- [ ] Create `api/ragreceipts/vendors/cohere_client.py` (complete file):
  ```python
  """RerankTransport over the official cohere SDK v2 (rerank-v4.0-pro, the anchor variant
  benchmarked in arXiv 2604.01733; rerank-v4.0-fast available via the model param).

  Binding verified 2026-06-10 against https://docs.cohere.com/reference/rerank and
  cohere 7.0.3: cohere.ClientV2(api_key=...).rerank(model=, query=, documents=, top_n=)
  -> response.results with .index/.relevance_score. Base error:
  cohere.core.api_error.ApiError(*, headers=None, status_code=None, body=None).
  """

  import time
  from collections.abc import Callable

  import cohere
  from cohere.core.api_error import ApiError

  from ragreceipts.constants import RERANK_MODEL
  from ragreceipts.vendors.retry import call_with_retry

  _RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


  def _should_retry(err: BaseException) -> bool:
      status = getattr(err, "status_code", None)
      return status is None or status in _RETRYABLE_STATUS


  class CohereClient:
      def __init__(self, api_key: str | None = None, model: str = RERANK_MODEL,
                   max_attempts: int = 5,
                   sleep: Callable[[float], None] = time.sleep,
                   sdk: object | None = None):
          self._sdk = sdk if sdk is not None else cohere.ClientV2(api_key=api_key)
          self._model = model
          self._max_attempts = max_attempts
          self._sleep = sleep

      def rerank(self, query: str, texts: list[str], top_n: int) -> list[tuple[int, float]]:
          response = call_with_retry(
              lambda: self._sdk.rerank(model=self._model, query=query,
                                       documents=list(texts), top_n=top_n),
              retryable=(ApiError,), max_attempts=self._max_attempts,
              sleep=self._sleep, label="cohere rerank", should_retry=_should_retry,
          )
          pairs = [(r.index, float(r.relevance_score)) for r in response.results]
          return sorted(pairs, key=lambda p: -p[1])   # enforce desc, per protocol contract
  ```
- [ ] Run again: `uv run pytest tests/test_cohere_client.py -q` → expect `5 passed`.
- [ ] Lint and commit:
  ```bash
  uv run ruff format ragreceipts tests && uv run ruff check ragreceipts tests
  git add ragreceipts/vendors/cohere_client.py tests/test_cohere_client.py
  git commit -m "feat: add cohere rerank client with retry-after handling" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
  ```

### Task 14: Ingest pipeline — contextualizer, hashing, manifest, run_ingest

**Files:**
- Create: `api/ragreceipts/ingest/contextualizer.py`, `api/ragreceipts/ingest/hashing.py`, `api/ragreceipts/ingest/manifest.py`, `api/ragreceipts/ingest/pipeline.py`
- Test: `api/tests/test_ingest_pipeline.py`

- [ ] Write the failing test `api/tests/test_ingest_pipeline.py` (complete file):
  ```python
  """End-to-end ingest (offline): both vector sets, sparse rebuild, manifest with hashes."""

  import json

  import pytest
  from qdrant_client import QdrantClient

  from ragreceipts.config import IngestConfig
  from ragreceipts.ingest.chunk_store import read_chunks
  from ragreceipts.ingest.contextualizer import embed_corpus
  from ragreceipts.ingest.hashing import hash_files, hash_vectors
  from ragreceipts.ingest.pipeline import run_ingest
  from ragreceipts.retrieval.dense import VECTOR_CONTEXTUAL, VECTOR_ISOLATED, DenseRetriever
  from ragreceipts.retrieval.sparse import SparseRetriever
  from tests.corpus_fixtures import TINY_PASSAGES, write_tiny_corpus
  from tests.fakes import FakeEmbed

  CONFIG = IngestConfig(chunk_size=40, chunk_overlap=10)


  def ingest(tmp_path, qdrant=None):
      write_tiny_corpus(tmp_path)
      return run_ingest(corpus_id="tiny", data_dir=tmp_path, ingest_config=CONFIG,
                        embed=FakeEmbed(), qdrant=qdrant or QdrantClient(":memory:"))


  class TestEmbedCorpus:
      def test_both_vector_sets_chunk_aligned(self):
          docs = [["a b c", "d e f"], ["g h i"]]
          contextual, isolated = embed_corpus(docs, FakeEmbed())
          assert len(contextual) == len(isolated) == 3
          assert contextual[0] != pytest.approx(isolated[0])   # multi-chunk doc: context shifts
          assert contextual[2] == pytest.approx(isolated[2])   # single-chunk doc: identical


  class TestHashing:
      def test_hash_vectors_deterministic_and_order_sensitive(self):
          a, b = [1.0, 2.0], [3.0, 4.0]
          assert hash_vectors([a, b]) == hash_vectors([a, b])
          assert hash_vectors([a, b]) != hash_vectors([b, a])
          assert hash_vectors([a]).startswith("sha256:")

      def test_hash_files_content_sensitive(self, tmp_path):
          (tmp_path / "x.json").write_text("one")
          first = hash_files([tmp_path / "x.json"])
          (tmp_path / "x.json").write_text("two")
          assert first != hash_files([tmp_path / "x.json"])


  class TestRunIngest:
      def test_manifest_schema_and_counts(self, tmp_path):
          manifest = ingest(tmp_path)
          assert set(manifest) == {"corpus_id", "dataset", "chunking", "embed_model",
                                   "index_hashes", "tokenizer_artifact",
                                   "n_docs", "n_chunks", "n_queries", "created_at"}
          assert manifest["corpus_id"] == "tiny"
          # dataset block constructed from raw/download_meta.json, incl. "name" (R1)
          assert manifest["dataset"] == {"name": "tiny", "hf_id": "local/tiny-fixture",
                                         "split": "test", "revision": "fixture-v1"}
          assert manifest["chunking"] == {"chunk_size": 40, "chunk_overlap": 10}
          assert manifest["embed_model"] == "voyage-context-3"
          assert manifest["n_docs"] == 3
          assert manifest["n_queries"] == 2
          assert manifest["tokenizer_artifact"] == "sparse/vocab.tokenizer.json"
          on_disk = json.loads((tmp_path / "corpora" / "tiny" / "manifest.json").read_text())
          assert on_disk == manifest

      def test_index_hashes_present_and_distinct(self, tmp_path):
          hashes = ingest(tmp_path)["index_hashes"]
          assert set(hashes) == {"dense_contextual", "dense_isolated", "sparse"}
          assert all(v.startswith("sha256:") for v in hashes.values())
          # d1 has two passages -> a multi-chunk doc -> contextual must differ from isolated
          assert hashes["dense_contextual"] != hashes["dense_isolated"]

      def test_hashes_reproducible_across_runs(self, tmp_path):
          first = ingest(tmp_path / "run1")["index_hashes"]
          second = ingest(tmp_path / "run2")["index_hashes"]
          assert first == second

      def test_chunks_jsonl_written_with_alignment_metadata(self, tmp_path):
          manifest = ingest(tmp_path)
          chunks = read_chunks(tmp_path / "corpora" / "tiny" / "chunks.jsonl")
          assert len(chunks) == manifest["n_chunks"] > 0
          assert {c.passage_id for c in chunks} == {"d1-p0", "d1-p1", "d2-p0", "d3-p0"}
          assert all(c.chunk_id == f"{c.doc_id}:{c.position}" for c in chunks)
          # R3: persisted token ranges are exact slices of the parent passage's tokens
          passage_text = {row["passage_id"]: row["text"] for row in TINY_PASSAGES}
          for c in chunks:
              tokens = passage_text[c.passage_id].split()
              assert 0 <= c.start_token < c.end_token <= len(tokens)
              assert c.text == " ".join(tokens[c.start_token:c.end_token])

      def test_both_named_vectors_queryable(self, tmp_path):
          client = QdrantClient(":memory:")
          ingest(tmp_path, qdrant=client)
          fake = FakeEmbed()
          for name in (VECTOR_CONTEXTUAL, VECTOR_ISOLATED):
              hits = DenseRetriever(client, "tiny", name, fake).search("anything", k=3)
              assert len(hits) == 3

      def test_sparse_index_loadable_and_searches(self, tmp_path):
          ingest(tmp_path)
          corpus_dir = tmp_path / "corpora" / "tiny"
          retriever = SparseRetriever.load(corpus_dir / "sparse",
                                           read_chunks(corpus_dir / "chunks.jsonl"))
          top = retriever.search("eiffel tower paris", k=3)[0]
          assert top.chunk.doc_id == "d1"

      def test_missing_corpus_raises(self, tmp_path):
          with pytest.raises(FileNotFoundError):
              run_ingest(corpus_id="nope", data_dir=tmp_path, ingest_config=CONFIG,
                         embed=FakeEmbed(), qdrant=QdrantClient(":memory:"))
  ```
- [ ] Run it and watch it fail: `uv run pytest tests/test_ingest_pipeline.py -q` → expect `ModuleNotFoundError: No module named 'ragreceipts.ingest.contextualizer'`.
- [ ] Create `api/ragreceipts/ingest/contextualizer.py` (complete file):
  ```python
  """Contextualizer: builds BOTH dense vector sets on every ingest (decision #8).

  Direct EmbedTransport calls — NOT LlamaIndex's per-node embedding path, which would
  silently degrade doc-grouping to single-chunk documents (spec ingestion plane).
  Contextual = doc-grouped call; isolated = the same chunks as single-chunk documents
  (vendors/base.py contract). Outputs are flattened to global chunk order."""

  from ragreceipts.vendors.base import EmbedTransport


  def embed_corpus(doc_chunk_texts: list[list[str]],
                   embed: EmbedTransport) -> tuple[list[list[float]], list[list[float]]]:
      contextual_nested = embed.embed_documents(doc_chunk_texts)
      isolated_nested = embed.embed_documents(
          [[text] for doc in doc_chunk_texts for text in doc]
      )
      contextual = [vec for doc in contextual_nested for vec in doc]
      isolated = [doc[0] for doc in isolated_nested]
      if len(contextual) != len(isolated):
          raise RuntimeError(
              f"contextual/isolated vector counts diverged: {len(contextual)} vs {len(isolated)}"
          )
      return contextual, isolated
  ```
- [ ] Create `api/ragreceipts/ingest/hashing.py` (complete file):
  ```python
  """Content hashes for manifest index_hashes — receipts must be traceable to exact
  corpus state. Vectors are hashed as little-endian float64 in chunk order; file hashes
  cover name + bytes, sorted by path for determinism."""

  import hashlib
  import struct
  from pathlib import Path


  def hash_vectors(vectors: list[list[float]]) -> str:
      digest = hashlib.sha256()
      for vector in vectors:
          for value in vector:
              digest.update(struct.pack("<d", value))
      return f"sha256:{digest.hexdigest()}"


  def hash_files(paths: list[Path]) -> str:
      digest = hashlib.sha256()
      for path in sorted(paths):
          digest.update(path.name.encode("utf-8"))
          digest.update(path.read_bytes())
      return f"sha256:{digest.hexdigest()}"
  ```
- [ ] Create `api/ragreceipts/ingest/manifest.py` (complete file):
  ```python
  """Corpus manifest (binding JSON shape from contracts). tokenizer_artifact is stored as a
  path RELATIVE to the corpus dir (machine-independent); its bytes are hashed into the
  sparse hash because hash_files covers every file in sparse/."""

  import json
  from datetime import datetime, timezone
  from pathlib import Path


  def build_manifest(*, corpus_id: str, dataset: dict, chunking: dict, embed_model: str,
                     index_hashes: dict, tokenizer_artifact: str,
                     n_docs: int, n_chunks: int, n_queries: int) -> dict:
      return {
          "corpus_id": corpus_id,
          "dataset": dataset,
          "chunking": chunking,
          "embed_model": embed_model,
          "index_hashes": index_hashes,
          "tokenizer_artifact": tokenizer_artifact,
          "n_docs": n_docs,
          "n_chunks": n_chunks,
          "n_queries": n_queries,
          "created_at": datetime.now(timezone.utc).isoformat(),
      }


  def write_manifest(corpus_dir: Path, manifest: dict) -> Path:
      path = corpus_dir / "manifest.json"
      path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
      return path


  def read_manifest(corpus_dir: Path) -> dict:
      return json.loads((corpus_dir / "manifest.json").read_text(encoding="utf-8"))
  ```
- [ ] Create `api/ragreceipts/ingest/pipeline.py` (complete file):
  ```python
  """run_ingest: loaders -> chunker -> contextualizer (BOTH vector sets) -> index writers
  -> manifest.json. Full rebuild of every index variant on every ingest.

  Reads Spike 0's raw/ layout (R1); n_queries is counted straight off raw/queries.jsonl —
  per R2 no eval-queries file is materialized here (Plan B reads raw/queries.jsonl
  directly). The manifest's dataset block (incl. "name") comes from download_meta.json."""

  from pathlib import Path

  from qdrant_client import QdrantClient

  from ragreceipts.config import IngestConfig
  from ragreceipts.constants import EMBED_MODEL
  from ragreceipts.ingest.chunk_store import write_chunks
  from ragreceipts.ingest.chunker import chunk_document
  from ragreceipts.ingest.contextualizer import embed_corpus
  from ragreceipts.ingest.hashing import hash_files, hash_vectors
  from ragreceipts.ingest.indexer import write_dense_index
  from ragreceipts.ingest.loaders import (
      count_queries,
      group_documents,
      load_dataset_info,
      load_passages,
  )
  from ragreceipts.ingest.manifest import build_manifest, write_manifest
  from ragreceipts.retrieval.sparse import build_sparse_index
  from ragreceipts.types import Chunk
  from ragreceipts.vendors.base import EmbedTransport


  def run_ingest(*, corpus_id: str, data_dir: Path, ingest_config: IngestConfig,
                 embed: EmbedTransport, qdrant: QdrantClient,
                 embed_model: str = EMBED_MODEL) -> dict:
      corpus_dir = data_dir / "corpora" / corpus_id
      passages = load_passages(corpus_dir)
      dataset = load_dataset_info(corpus_dir)
      documents = group_documents(passages)

      chunks: list[Chunk] = []
      for doc in documents:
          chunks.extend(chunk_document(
              corpus_id, doc[0].doc_id, [(p.passage_id, p.text) for p in doc],
              ingest_config.chunk_size, ingest_config.chunk_overlap,
          ))
      if not chunks:
          raise ValueError(f"corpus {corpus_id} produced no chunks")
      write_chunks(corpus_dir / "chunks.jsonl", chunks)

      # Regroup chunk texts per document (chunks were generated doc-by-doc, so a doc_id
      # transition marks a new document). Docs that produced zero chunks simply don't appear.
      doc_chunk_texts: list[list[str]] = []
      current_doc: str | None = None
      for chunk in chunks:
          if chunk.doc_id != current_doc:
              doc_chunk_texts.append([])
              current_doc = chunk.doc_id
          doc_chunk_texts[-1].append(chunk.text)

      contextual, isolated = embed_corpus(doc_chunk_texts, embed)
      write_dense_index(qdrant, corpus_id, chunks, contextual, isolated)

      sparse_dir = corpus_dir / "sparse"
      build_sparse_index(chunks, sparse_dir)
      sparse_files = sorted(p for p in sparse_dir.iterdir() if p.is_file())

      manifest = build_manifest(
          corpus_id=corpus_id,
          dataset=dataset,
          chunking={"chunk_size": ingest_config.chunk_size,
                    "chunk_overlap": ingest_config.chunk_overlap},
          embed_model=embed_model,
          index_hashes={
              "dense_contextual": hash_vectors(contextual),
              "dense_isolated": hash_vectors(isolated),
              "sparse": hash_files(sparse_files),
          },
          tokenizer_artifact="sparse/vocab.tokenizer.json",
          n_docs=len(documents),
          n_chunks=len(chunks),
          n_queries=count_queries(corpus_dir),
      )
      write_manifest(corpus_dir, manifest)
      return manifest
  ```
- [ ] Run again: `uv run pytest tests/test_ingest_pipeline.py -q` → expect `10 passed`.
- [ ] Lint and commit:
  ```bash
  uv run ruff format ragreceipts tests && uv run ruff check ragreceipts tests
  git add ragreceipts/ingest tests/test_ingest_pipeline.py
  git commit -m "feat: add ingest pipeline emitting corpus manifest with index hashes" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
  ```

### Task 15: CLI entry point `ragreceipts ingest --corpus <id>`

**Files:**
- Create: `api/ragreceipts/cli.py`
- Modify: `api/pyproject.toml` (`[project.scripts]`)
- Test: `api/tests/test_cli.py`

Ownership note (R6): Plan A CREATES `cli.py` and `test_cli.py`; Plan B will **MODIFY both files in place — never recreate them** — adding `eval` and `receipts` subparsers plus the composition root `_build_core_real(config, corpus_id, data_dir)` (name pinned by R9). The seams Plan B keeps unchanged: `main(argv: list[str] | None) -> int`, the `ingest` subparser and its flags, the module-level factories `build_embed_transport()`/`build_qdrant(data_dir)` (monkeypatch targets for tests, reused by Plan D's server), and data-dir resolution via `RAGRECEIPTS_DATA_DIR` with default `../data` relative to `api/`.

- [ ] Write the failing test `api/tests/test_cli.py` (complete file):
  ```python
  """CLI wiring: factories are monkeypatched so the test stays offline and keyless."""

  import json

  from qdrant_client import QdrantClient

  import ragreceipts.cli as cli
  from tests.corpus_fixtures import write_tiny_corpus
  from tests.fakes import FakeEmbed


  def test_ingest_command_writes_manifest_and_prints_it(tmp_path, monkeypatch, capsys):
      write_tiny_corpus(tmp_path)
      monkeypatch.setattr(cli, "build_embed_transport", lambda: FakeEmbed())
      monkeypatch.setattr(cli, "build_qdrant", lambda data_dir: QdrantClient(":memory:"))
      code = cli.main(["ingest", "--corpus", "tiny", "--data-dir", str(tmp_path),
                       "--chunk-size", "40", "--chunk-overlap", "10"])
      assert code == 0
      assert (tmp_path / "corpora" / "tiny" / "manifest.json").exists()
      printed = json.loads(capsys.readouterr().out)
      assert printed["corpus_id"] == "tiny"
      assert printed["chunking"] == {"chunk_size": 40, "chunk_overlap": 10}


  def test_missing_corpus_exits_nonzero_with_named_message(tmp_path, monkeypatch, capsys):
      monkeypatch.setattr(cli, "build_embed_transport", lambda: FakeEmbed())
      monkeypatch.setattr(cli, "build_qdrant", lambda data_dir: QdrantClient(":memory:"))
      code = cli.main(["ingest", "--corpus", "nope", "--data-dir", str(tmp_path)])
      assert code == 1
      assert "nope" in capsys.readouterr().err


  def test_missing_voyage_key_is_a_named_error(monkeypatch):
      monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
      try:
          cli.build_embed_transport()
          raised = False
      except SystemExit as err:
          raised = "VOYAGE_API_KEY" in str(err)
      assert raised
  ```
- [ ] Run it and watch it fail: `uv run pytest tests/test_cli.py -q` → expect `ModuleNotFoundError: No module named 'ragreceipts.cli'`.
- [ ] Create `api/ragreceipts/cli.py` (complete file):
  ```python
  """ragreceipts CLI. `ragreceipts ingest --corpus <id>` rebuilds every index variant for a
  Spike 0 corpus. Missing keys produce a named env-var message, never a stack trace.

  Factories build_embed_transport/build_qdrant are module-level seams: tests monkeypatch
  them; Plan D's server reuses them. Plan B MODIFIES this file in place (R6), adding
  `eval` and `receipts` subparsers plus _build_core_real(config, corpus_id, data_dir),
  and keeps these seams and main(argv) unchanged."""

  import argparse
  import json
  import os
  import sys
  from pathlib import Path

  from qdrant_client import QdrantClient

  from ragreceipts.config import IngestConfig
  from ragreceipts.ingest.pipeline import run_ingest
  from ragreceipts.vendors.voyage_client import VoyageClient


  def build_embed_transport() -> VoyageClient:
      api_key = os.environ.get("VOYAGE_API_KEY")
      if not api_key:
          raise SystemExit(
              "VOYAGE_API_KEY is not set — ingest needs Voyage embeddings (set it in .env)"
          )
      return VoyageClient(api_key=api_key)


  def build_qdrant(data_dir: Path) -> QdrantClient:
      url = os.environ.get("QDRANT_URL")
      if url:
          return QdrantClient(url=url)
      # CLI-scoped fallback ONLY (R7): local file mode, no server. The FastAPI server
      # (Plan D) REQUIRES QDRANT_URL and fails its healthcheck with a named-env-var
      # message when it is missing.
      return QdrantClient(path=str(data_dir / "qdrant-local"))


  def main(argv: list[str] | None = None) -> int:
      parser = argparse.ArgumentParser(prog="ragreceipts")
      subparsers = parser.add_subparsers(dest="command", required=True)
      ingest = subparsers.add_parser("ingest",
                                     help="(re)build all index variants for a corpus")
      ingest.add_argument("--corpus", required=True, help="corpus id, e.g. nq-dev-300")
      ingest.add_argument("--data-dir", type=Path,
                          default=Path(os.environ.get("RAGRECEIPTS_DATA_DIR", "../data")),
                          help="data dir holding corpora/ (default ../data, run from api/)")
      ingest.add_argument("--chunk-size", type=int, default=IngestConfig().chunk_size)
      ingest.add_argument("--chunk-overlap", type=int, default=IngestConfig().chunk_overlap)
      args = parser.parse_args(argv)

      if args.command == "ingest":
          try:
              manifest = run_ingest(
                  corpus_id=args.corpus,
                  data_dir=args.data_dir,
                  ingest_config=IngestConfig(chunk_size=args.chunk_size,
                                             chunk_overlap=args.chunk_overlap),
                  embed=build_embed_transport(),
                  qdrant=build_qdrant(args.data_dir),
              )
          except FileNotFoundError:
              print(f"error: corpus '{args.corpus}' not found under "
                    f"{args.data_dir / 'corpora'} — run the Spike 0 download script first",
                    file=sys.stderr)
              return 1
          print(json.dumps(manifest, indent=2))
          return 0
      return 2
  ```
- [ ] Add the console script to `api/pyproject.toml`, then re-sync:
  ```toml
  [project.scripts]
  ragreceipts = "ragreceipts.cli:main"
  ```
  Run `uv sync`, then verify: `uv run ragreceipts ingest --help` → prints usage with `--corpus`.
- [ ] Run again: `uv run pytest tests/test_cli.py -q` → expect `3 passed`.
- [ ] Lint and commit:
  ```bash
  uv run ruff format ragreceipts tests && uv run ruff check ragreceipts tests
  git add pyproject.toml uv.lock ragreceipts/cli.py tests/test_cli.py
  git commit -m "feat: add ragreceipts ingest CLI entry point" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
  ```

### Task 16: Live smoke script (manual, never CI) + whole-plan verification

**Files:**
- Create: `api/scripts/live_smoke_ingest.py`

- [ ] Create the directory: `mkdir -p api/scripts`.
- [ ] Create `api/scripts/live_smoke_ingest.py` (complete file — manual script, real keys, real money; intentionally NOT a test and NOT imported by any test):
  ```python
  """Manual live smoke: ingest the first N documents of a real corpus with REAL vendor keys.

  NEVER wired into CI (spec: 5-query live smoke is manual/nightly only). Costs cents.
  Run from api/:
      VOYAGE_API_KEY=... uv run python scripts/live_smoke_ingest.py --corpus nq-dev-300
  Optionally set COHERE_API_KEY to also smoke the rerank stage.
  Artifacts land in a temp dir printed at the end; nothing under data/ is touched.
  """

  import argparse
  import json
  import os
  import shutil
  import sys
  import tempfile
  from pathlib import Path

  from qdrant_client import QdrantClient

  from ragreceipts.config import IngestConfig
  from ragreceipts.ingest.chunk_store import read_chunks
  from ragreceipts.ingest.loaders import load_passages
  from ragreceipts.ingest.pipeline import run_ingest
  from ragreceipts.retrieval.dense import VECTOR_CONTEXTUAL, DenseRetriever
  from ragreceipts.retrieval.rerank import RerankStage
  from ragreceipts.retrieval.sparse import SparseRetriever
  from ragreceipts.vendors.cohere_client import CohereClient
  from ragreceipts.vendors.voyage_client import VoyageClient


  def main() -> int:
      parser = argparse.ArgumentParser(description=__doc__)
      parser.add_argument("--corpus", required=True, help="e.g. nq-dev-300")
      parser.add_argument("--data-dir", type=Path, default=Path("../data"))
      parser.add_argument("--n-docs", type=int, default=5)
      args = parser.parse_args()

      if not os.environ.get("VOYAGE_API_KEY"):
          sys.exit("VOYAGE_API_KEY is required for the live smoke")

      source_corpus_dir = args.data_dir / "corpora" / args.corpus
      passages = load_passages(source_corpus_dir)
      keep_doc_ids: list[str] = []
      for passage in passages:
          if passage.doc_id not in keep_doc_ids:
              keep_doc_ids.append(passage.doc_id)
          if len(keep_doc_ids) >= args.n_docs:
              break
      subset = [p for p in passages if p.doc_id in keep_doc_ids]

      workdir = Path(tempfile.mkdtemp(prefix="ragreceipts-smoke-"))
      smoke_id = f"{args.corpus}-smoke{args.n_docs}"
      smoke_raw = workdir / "corpora" / smoke_id / "raw"
      smoke_raw.mkdir(parents=True)
      with (smoke_raw / "docs.jsonl").open("w", encoding="utf-8") as fh:
          for p in subset:
              fh.write(json.dumps({"doc_id": p.doc_id, "passage_id": p.passage_id,
                                   "title": p.title, "text": p.text}) + "\n")
      # carry the dataset pins; no queries.jsonl in the doc subset -> n_queries == 0
      shutil.copy(source_corpus_dir / "raw" / "download_meta.json",
                  smoke_raw / "download_meta.json")

      embed = VoyageClient(api_key=os.environ["VOYAGE_API_KEY"])
      qdrant = QdrantClient(path=str(workdir / "qdrant"))
      manifest = run_ingest(corpus_id=smoke_id, data_dir=workdir,
                            ingest_config=IngestConfig(), embed=embed, qdrant=qdrant)
      print(json.dumps(manifest, indent=2))

      corpus_dir = workdir / "corpora" / smoke_id
      chunks = read_chunks(corpus_dir / "chunks.jsonl")
      query = subset[0].title or subset[0].text.split(".")[0]
      dense_hits = DenseRetriever(qdrant, smoke_id, VECTOR_CONTEXTUAL, embed).search(query, 3)
      sparse_hits = SparseRetriever.load(corpus_dir / "sparse", chunks).search(query, 3)
      print("query:", query)
      print("dense top-3:", [(s.chunk.chunk_id, round(s.score, 4)) for s in dense_hits])
      print("sparse top-3:", [(s.chunk.chunk_id, round(s.score, 4)) for s in sparse_hits])

      cohere_key = os.environ.get("COHERE_API_KEY")
      if cohere_key and dense_hits:
          stage = RerankStage(CohereClient(api_key=cohere_key))
          reranked = stage.rerank(query, dense_hits + sparse_hits, top_n=3)
          print("rerank top-3:", [(s.chunk.chunk_id, round(s.score, 4)) for s in reranked])
      else:
          print("rerank smoke skipped (COHERE_API_KEY not set)")
      print("smoke artifacts in:", workdir)
      return 0


  if __name__ == "__main__":
      raise SystemExit(main())
  ```
- [ ] Sanity-check the script parses and shows help without keys: `uv run python scripts/live_smoke_ingest.py --help` → prints usage (no network, no keys needed for `--help`).
- [ ] Whole-plan verification — run the FULL suite with vendor keys explicitly stripped from the environment to prove the offline guarantee:
  ```bash
  cd /Users/pratiksoni/PersonalProjects/rag-receipts/api
  env -u VOYAGE_API_KEY -u COHERE_API_KEY -u ANTHROPIC_API_KEY -u CO_API_KEY uv run pytest -q
  ```
  Expect: all tests pass (135 tests if counts above were followed exactly — Spike 0's 23, of which 3 files were touched per R3/R4, plus this plan's 112), zero network access, zero keys.
- [ ] Lint everything one final time: `uv run ruff format ragreceipts tests scripts && uv run ruff check ragreceipts tests scripts` → no findings.
- [ ] (Optional, manual, costs cents — only if Spike 0 corpora are downloaded and you have real keys) run the live smoke:
  ```bash
  VOYAGE_API_KEY=... COHERE_API_KEY=... uv run python scripts/live_smoke_ingest.py --corpus nq-dev-300
  ```
  Expect: a printed manifest with three `sha256:` hashes, dense/sparse top-3 hits, rerank top-3.
- [ ] Commit:
  ```bash
  git add scripts/live_smoke_ingest.py
  git commit -m "chore: add live smoke ingest script and full-suite verification" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
  ```

---

## Done means

- `env -u VOYAGE_API_KEY -u COHERE_API_KEY -u ANTHROPIC_API_KEY -u CO_API_KEY uv run pytest -q` passes from `api/` — the entire pipeline is proven offline through the transport seam.
- `PRESETS` matches the contracts ladder exactly, and every `QueryConfig` flag (`bm25`, `dense`, `rerank`, `top_k_fuse`, `top_k_final`) has a test proving it changes `RetrievalCore` output; `route_mode` has a test proving it does NOT (Plan C's concern).
- `ragreceipts ingest --corpus <id>` rebuilds chunks.jsonl + both Qdrant named-vector sets + the bm25s index with its tokenizer artifact, and emits a contracts-shaped `manifest.json` with reproducible `sha256:` hashes per index variant.
- Degradation is visible: `dense-skipped` / `rerank-skipped` appear in `TraceEvent.payload["degraded"]` and results fall back exactly as the spec dictates; with no fallback available the error raises.
- Plan B can now build the eval CLI on `RetrievalCore` + `PRESETS` + `read_manifest`; Plan C can thread `trace_id`/`node="retrieve_hop"`/`seq_start` and a `TraceStore.append`-backed `TraceCallback` without touching retrieval code.

