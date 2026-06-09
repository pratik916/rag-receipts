# rag-receipts Plan B: Eval Plane CLI & Receipts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the project's differentiator — an offline-testable ablation runner CLI (`ragreceipts eval`) that measures each retrieval component's contribution on labeled corpora and emits honest, anchor-annotated `receipts.json` files, plus a `receipts promote` command that commits redistribution-safe headline receipts.

**Architecture:** Plan B builds the entire eval plane under `api/ragreceipts/eval/` on top of Plan A's retrieval core and Spike 0's alignment rules: binding metric implementations (Recall@5 / MRR@3 as thin wrappers over Spike 0's `eval/alignment.py`, SQuAD-style EM/F1), a versioned pricing table, the `Receipt`/`PublishedAnchor` schema with machine-readable comparability caveats, a RAGAS v0.4 adapter behind a Protocol, and a resumable SQLite-backed ablation runner with a pre-run cost estimate and a hard spend cap. A harness self-test on a tiny in-repo labeled corpus proves the receipts can fail (rerank flip changes Recall@5; misaligned golds score zero) and is CI-enforced.

**Tech Stack:** Python 3.12 + uv, pytest, stdlib `sqlite3`/`argparse`/`dataclasses`, `ragas>=0.4` (collections API, lazy-imported), `sentence-transformers` (local `BAAI/bge-small-en-v1.5` for RAGAS answer-relevancy), Anthropic SDK only via Plan A's `ClaudeTransport` seam.

---

## Context

### Where this plan starts

Spike 0 and Plan A are complete. **Spike 0** shipped `eval/alignment.py` (the binding hit rules, kept verbatim per R3) and the raw benchmark slices under `data/corpora/{corpus_id}/raw/` (R1). **Plan A** shipped ingestion (`chunks.jsonl` + both index variants + `manifest.json`), the retrieval core, `PipelineConfig`, vendor transports + fakes, and the `ragreceipts ingest` CLI with its factory seams. Plan B adds files under `api/ragreceipts/eval/`, **modifies** `api/ragreceipts/cli.py` and `api/tests/test_cli.py` to add the `eval`/`receipts` subcommands (R6), adds one vendors helper, and tests. Plan C (LangGraph) does **not** exist yet — every preset with `route_mode != FORCE_S1` must be *skipped with disclosure*, never faked.

All commands in this plan run from **`api/`** unless stated otherwise. Tests run offline with **zero API keys** — vendor calls only through the contracts' transport Protocols with fakes. `api/tests/` is a package (R8): `api/tests/__init__.py` exists and every test file imports shared fixtures via `from tests.fakes import ...` / `from tests.harness_fixtures import ...`.

### Binding contracts already in force (quoted from `docs/superpowers/plans/2026-06-10-contracts.md`)

Constants — `api/ragreceipts/constants.py`:

```python
ROUTER_MODEL = "claude-haiku-4-5-20251001"
SYNTH_MODEL = "claude-sonnet-4-6"
JUDGE_MODEL = "claude-sonnet-4-6"
EMBED_MODEL = "voyage-context-3"
RERANK_MODEL = "rerank-v4.0-pro"
RAGAS_EMBED_MODEL = "BAAI/bge-small-en-v1.5"
```

Core types — `api/ragreceipts/types.py`:

```python
@dataclass(frozen=True)
class Chunk:
    chunk_id: str          # f"{doc_id}:{position}"
    corpus_id: str
    doc_id: str
    passage_id: str        # parent passage ID for gold alignment (== doc_id when unsegmented)
    text: str
    position: int
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

Config — `api/ragreceipts/config.py`: `IngestConfig(contextual, chunk_size, chunk_overlap)`, `QueryConfig(bm25, dense, rerank, route_mode, top_k_fuse=50, top_k_final=5)`, `PipelineConfig(name, ingest, query)`, and `PRESETS: dict[str, PipelineConfig]` with keys, in ladder order: `"bm25-only"`, `"dense-rrf"`, `"contextual"`, `"rerank"`, `"router-on"` (only `router-on` has `route_mode=AUTO`; all others `FORCE_S1`).

Retrieval — `api/ragreceipts/retrieval/base.py` and `core.py`:

```python
class Retriever(Protocol):
    def search(self, query: str, k: int) -> list[ScoredChunk]: ...

# api/ragreceipts/retrieval/core.py
class RetrievalCore:
    def __init__(self, config: PipelineConfig, dense: Retriever | None,
                 sparse: Retriever | None, rerank_stage: "RerankStage | None"): ...
    def retrieve(self, query: str) -> list[ScoredChunk]: ...
    # honors config.query flags; returns top_k_final chunks
```

Vendor seam — `api/ragreceipts/vendors/base.py`:

```python
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

Fakes live in `api/tests/fakes.py` (`FakeEmbed`, `FakeRerank`, `FakeClaude`). `FakeRerank`'s final constructor (R5) is `FakeRerank(script: dict[str, list[int]] | None = None, scores: dict[str, float] | None = None, fail: bool = False)` — `script` is Plan A's query-keyed ordering mode; `scores` is the additive text-keyed mode Plan B's harness fixture uses.

Metrics (binding definitions): a retrieved chunk is a **hit** for a gold passage iff `chunk.passage_id == gold.passage_id`; for span-format golds (NQ long answers), hit iff the chunk covers ≥50% of the gold span's tokens (Spike 0's positional token-range rule, R3 — span golds are token ranges, never `span_text` strings). `recall_at_5` = |golds with ≥1 hit in top-5| / |golds| per query, averaged over queries. `mrr_at_3` = reciprocal rank of the first hit within top-3 (0 if none), averaged. Abstentions excluded from RAGAS, reported as `n_abstained`. Failures excluded from metrics, reported as `n_failed`. Never silently dropped.

Receipt schema (binding — `api/ragreceipts/eval/receipts.py`): `PublishedAnchor(source, published_value, measured_value, direction_match, note)` with `note` REQUIRED; `Receipt(run_id, corpus_id, preset, config, index_hashes, models, pricing_table_version, prompts_version, n_total, n_failed, n_abstained, metrics, per_query, anchors)` — `prompts_version` is `"n/a"` in Plan B and populated from `agents.prompts.PROMPTS_VERSION` by Plan C (R11); serialized via a versioned envelope `{"schema_version": 1, "nondeterminism_note": "...", "receipt": {...}}` whose `nondeterminism_note` is a fixed string disclosing LLM nondeterminism (R11). Metrics dict keys: `recall_at_5, mrr_at_3, em, f1, ragas_faithfulness, ragas_answer_relevancy, latency_p50_ms, latency_p95_ms, usd_per_query`. Committed per-query records contain IDs + metrics, never passage text. Anchor notes on `nq-dev-300` runs additionally carry the corpus-scale caveat from Spike 0's decisions doc (R11).

Corpus manifest — `data/corpora/{corpus_id}/manifest.json` with `dataset.name`, `index_hashes.{dense_contextual,dense_isolated,sparse}`, `n_queries`, etc. (full JSON shape in the contracts).

### Cross-plan seams (pinned upstream surfaces)

These surfaces are owned by Spike 0 / Plan A and **pinned** by the seam resolutions (R1–R3, R5, R6, R9) plus Plan A's authored code — there are no assumed shapes and no discovery steps. **Task 1 still CI-enforces every one of them** (`api/tests/test_plan_a_seams.py`): if a seam test fails, the upstream code drifted from its binding resolution — STOP and reconcile the upstream code (or the call sites listed in the third column), never fork a second definition.

| Upstream surface (owner) | Pinned shape | Plan B call sites |
|---|---|---|
| `ragreceipts/eval/alignment.py` (**Spike 0**, kept verbatim per R3) | `GoldPassage(query_id: str, passage_id: str)`; `GoldSpan(query_id: str, doc_id: str, start_token: int, end_token: int)`; `Gold = GoldPassage \| GoldSpan`; `is_hit(chunk_or_span, gold) -> bool` and `first_hit_rank(ranked, gold, k) -> int \| None` accept any object carrying `passage_id`/`doc_id`/`start_token`/`end_token` — structurally satisfied by `Chunk`, which carries token offsets per R3. Span golds are token ranges; **no `span_text` anywhere** | `eval/metrics.py`, `eval/queries.py`, tests |
| `tests/fakes.py::FakeRerank` (Plan A, final constructor per R5) | `FakeRerank(script: dict[str, list[int]] \| None = None, scores: dict[str, float] \| None = None, fail: bool = False)`; `scores` maps candidate **text** → relevance score | `tests/test_plan_a_seams.py`, `tests/test_harness_selftest.py` |
| `retrieval/rerank.py::RerankStage` (Plan A) | `RerankStage(transport: RerankTransport)`; `.rerank(query, candidates, top_n) -> list[ScoredChunk]` | `cli.py::_build_core_real`, harness self-test |
| `retrieval/sparse.py::SparseRetriever` (Plan A) | `SparseRetriever.load(index_dir: Path, chunks: list[Chunk])` with `index_dir = data/corpora/{corpus_id}/sparse` and `chunks` read via `ingest/chunk_store.read_chunks(.../chunks.jsonl)` (same order the index was built from) | `cli.py::_build_core_real` ONLY |
| `retrieval/dense.py::DenseRetriever` (Plan A) | `DenseRetriever(client: QdrantClient, collection: str, vector_name: str, embed: EmbedTransport)`; collection name == corpus_id; named vector via `vector_name_for(contextual: bool)` | `cli.py::_build_core_real` ONLY |
| `data/corpora/{corpus_id}/raw/` (**Spike 0** download script, R1/R2) | `queries.jsonl`: one JSON object per line — MuSiQue `{"query_id","question","answer","answer_aliases","gold":{"type":"passage","passage_ids":[...]}}`; NQ `{"query_id","question","answer_texts","gold":{"type":"span","doc_id","start_token","end_token"},"gold_text"}`. Slices `slice-smoke.json`/`slice-full.json` are JSON arrays of query ids | `eval/queries.py` ONLY |
| `cli.py` + `tests/test_cli.py` (Plan A creates with `ingest`; **Plan B modifies**, R6) | `main(argv) -> int`; module-level factory seams `build_embed_transport()` / `build_qdrant(data_dir)`; data dir default `RAGRECEIPTS_DATA_DIR` env → `../data` relative to `api/` | Task 9 |

Composition-root names are R9-pinned and binding: `cli.py::_build_core_real(config, corpus_id, data_dir)`, `eval/runner.py::AblationRunner` (with `_run_preset`), `eval/runner.py::estimate_run_cost`.

Note on span tests: Spike 0's `span_hit` gates on the document — a span hit requires `chunk.doc_id == gold.doc_id` AND ≥50% token overlap with the gold span (integer form `2*overlap >= gold_len`). Every span-gold test in this plan exercises both the 50% boundary and the same-document requirement.

### External APIs verified for this plan (2026-06-10)

| API | What was verified | Source |
|---|---|---|
| RAGAS v0.4 collections API | `from ragas.metrics.collections import Faithfulness, AnswerRelevancy`; `Faithfulness(llm=...)`, `AnswerRelevancy(llm=..., embeddings=...)`; sync `.score(user_input=, response=, retrieved_contexts=)` → result with `.value` (async `.ascore` also exists). v0.2 `evaluate()`/`SingleTurnSample` API is obsolete — do not use it. | https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/ · https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/answer_relevance/ · https://docs.ragas.io/en/stable/howtos/migrations/migrate_from_v03_to_v04/ |
| RAGAS `llm_factory` direct Anthropic provider | `from ragas.llms import llm_factory; llm_factory("<model>", provider="anthropic", client=anthropic.Anthropic())` — uses the Instructor adapter automatically, no LangChain wrapper | https://docs.ragas.io/en/stable/howtos/llm-adapters/ |
| RAGAS local embeddings | `from ragas.embeddings import HuggingFaceEmbeddings; HuggingFaceEmbeddings(model="BAAI/bge-small-en-v1.5")` (local sentence-transformers path; `embedding_factory` is deprecated for this) | https://docs.ragas.io/en/stable/references/embeddings/ |
| Anthropic pricing + model IDs | `claude-haiku-4-5-20251001` $1.00/$5.00 per MTok; `claude-sonnet-4-6` $3.00/$15.00 per MTok; IDs confirmed valid | claude-api skill model table (cached 2026-05-26) / https://platform.claude.com/docs/en/pricing.md |
| Voyage pricing | `voyage-context-3` $0.18 per 1M tokens (200M free tokens) | https://docs.voyageai.com/docs/pricing |
| Cohere rerank pricing | 1 search unit = 1 query + up to 100 docs; docs >500 tokens auto-chunk and each chunk counts (cohere.com/pricing). Rerank 4 Pro per-search price is **not** on cohere.com/pricing (routes to sales); $0.0025/search ($2.50 per 1K) corroborated by two resellers. **Re-verify on the Cohere dashboard during the first keyed run (Task 10).** | https://cohere.com/pricing · https://openrouter.ai/cohere/rerank-4-pro · https://vercel.com/ai-gateway/models/rerank-v4-pro |
| Qdrant in tests | `QdrantClient(":memory:")` local mode exists and serves the same client API (github.com/qdrant/qdrant-client README). **Plan B needs no Qdrant at all**: the runner depends only on the `Retriever` Protocol, and all tests use in-test `ListRetriever` fakes. Named-vector parity in `:memory:` mode is Plan A's concern, not re-verified here. | https://github.com/qdrant/qdrant-client |

### What Plan B deliberately does NOT do

- No LangGraph, no System-2, no router — `router-on` is skipped with a disclosed reason. The S1 synthesize-with-citations call defined in Task 7 is the **temporary** generation path; Plan C replaces that one call site with the graph.
- No RAGAS judge token metering: RAGAS v0.4 does not expose per-call token usage uniformly through the collections API, so judge cost is not added to `usd_per_query` and is **not counted against the hard spend cap** in Plan B. The pre-run estimate DOES include a per-ok-query judge heuristic when `--ragas` is set (so the confirmation gate is honest), and the runner records `ragas_judge_usd_untracked: true` in each receipt's per-query flags when RAGAS ran — the omission is disclosed, not silent.
- No web UI, no FastAPI endpoints (Plan C/D).

---

### Task 1: Dependencies + CI-enforced Plan A seam tests

**Files:**
- Modify: `api/pyproject.toml` (via `uv add`)
- Test: `api/tests/test_plan_a_seams.py`

- [ ] Add Plan B dependencies (pinned to the verified APIs):

  ```bash
  uv add "ragas>=0.4,<0.5" "sentence-transformers>=3.0"
  ```

  Expected: both resolve and install (sentence-transformers pulls torch — this is accepted; RAGAS imports stay lazy so the test suite does not import torch unless a RAGAS test runs).

- [ ] Write the seam tests. Create `api/tests/test_plan_a_seams.py`:

  ```python
  """CI-enforced verification of the upstream surfaces Plan B binds to.

  These tests pin the cross-plan seams listed in the Plan B Context table:
  Spike 0's alignment API (kept verbatim per R3) and Plan A's FakeRerank
  final constructor (R5). If one fails, the upstream code drifted from its
  binding resolution - reconcile the upstream code or the call sites named
  in the Context seam table; do NOT fork a second definition of these names.
  """
  from ragreceipts.eval.alignment import GoldPassage, GoldSpan, first_hit_rank, is_hit
  from ragreceipts.types import Chunk
  from tests.fakes import FakeRerank


  def make_chunk(passage_id: str, *, start_token: int = 0, end_token: int = 1,
                 text: str = "x", position: int = 0) -> Chunk:
      return Chunk(
          chunk_id=f"{passage_id}:{position}",
          corpus_id="seam",
          doc_id=passage_id,
          passage_id=passage_id,
          text=text,
          position=position,
          start_token=start_token,
          end_token=end_token,
      )


  def test_alignment_passage_seam() -> None:
      gold = GoldPassage(query_id="q", passage_id="p1")
      assert is_hit(make_chunk("p1"), gold) is True
      assert is_hit(make_chunk("p2"), gold) is False


  def test_alignment_span_seam_is_positional_50pct() -> None:
      # Spike 0's positional rule (R3): gold [10, 20) = 10 tokens; a chunk
      # covering [0, 15) overlaps 5/10 = 50% -> hit; [0, 14) is 40% -> miss.
      # is_hit works structurally on Chunk because Chunk carries token offsets.
      gold = GoldSpan(query_id="q", doc_id="d1", start_token=10, end_token=20)
      assert is_hit(make_chunk("d1", start_token=0, end_token=15), gold) is True
      assert is_hit(make_chunk("d1", start_token=0, end_token=14), gold) is False
      # same-document requirement: full overlap in the wrong doc is a miss
      assert is_hit(make_chunk("d2", start_token=0, end_token=20), gold) is False


  def test_first_hit_rank_seam_is_one_based_and_k_bounded() -> None:
      gold = GoldPassage(query_id="q", passage_id="p1")
      ranked = [make_chunk("x"), make_chunk("p1", position=1), make_chunk("p1", position=2)]
      assert first_hit_rank(ranked, gold, k=3) == 2
      assert first_hit_rank(ranked, gold, k=1) is None


  def test_fake_rerank_scores_seam() -> None:
      # R5 final constructor: FakeRerank(script=None, scores=None, fail=False);
      # the text-keyed scores mode exists for Plan B's harness fixture.
      fake = FakeRerank(scores={"high": 0.9, "low": 0.1})
      out = fake.rerank("any query", ["low", "high"], top_n=2)
      assert out == [(1, 0.9), (0, 0.1)]
  ```

- [ ] Run the seam tests:

  ```bash
  uv run pytest tests/test_plan_a_seams.py -q
  ```

  Expected: **PASS**. If any test fails with `ImportError`/`TypeError`, the upstream code drifted from its binding R-resolution: fix Spike 0's `eval/alignment.py` / Plan A's `tests/fakes.py` back to the pinned shapes (the resolutions supersede any plan text), re-run until green. Do not proceed to Task 2 with red seams.

- [ ] Commit:

  ```bash
  git add pyproject.toml uv.lock tests/test_plan_a_seams.py
  git commit -m "test(eval): pin Plan A seams + add ragas/sentence-transformers deps" \
    -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
  ```

---

### Task 2: `eval/metrics.py` — Recall@5, MRR@3, EM, F1 with golden tests

**Files:**
- Create: `api/ragreceipts/eval/metrics.py`
- Test: `api/tests/test_metrics.py`

- [ ] Write the failing golden tests first. Create `api/tests/test_metrics.py`:

  ```python
  """Golden tests for the binding metric definitions (contracts §Metrics).

  Every expected value below is hand-computed in the inline comments.
  Span golds are positional token ranges (Spike 0's rule, R3) - never
  span_text strings.
  """
  import pytest

  from ragreceipts.eval.alignment import GoldPassage, GoldSpan
  from ragreceipts.eval.metrics import (
      exact_match,
      f1,
      mrr_at_k,
      normalize_answer,
      recall_at_k,
  )
  from ragreceipts.types import Chunk


  def make_chunk(passage_id: str, text: str = "", position: int = 0,
                 start_token: int = 0, end_token: int = 1) -> Chunk:
      return Chunk(
          chunk_id=f"{passage_id}:{position}",
          corpus_id="t",
          doc_id=passage_id,
          passage_id=passage_id,
          text=text,
          position=position,
          start_token=start_token,
          end_token=end_token,
      )


  # ---------- recall_at_k ----------

  def test_recall_single_gold_hit_in_top5() -> None:
      retrieved = [make_chunk(p) for p in ["a", "b", "gold", "c", "d", "e"]]
      golds = [GoldPassage(query_id="q", passage_id="gold")]
      # gold at rank 3 (within top-5): 1 of 1 golds hit -> 1.0
      assert recall_at_k(retrieved, golds, k=5) == 1.0


  def test_recall_single_gold_outside_top5() -> None:
      retrieved = [make_chunk(p) for p in ["a", "b", "c", "d", "e", "gold"]]
      golds = [GoldPassage(query_id="q", passage_id="gold")]
      # gold at rank 6 (outside top-5): 0 of 1 -> 0.0
      assert recall_at_k(retrieved, golds, k=5) == 0.0


  def test_recall_multi_gold_partial() -> None:
      retrieved = [make_chunk(p) for p in ["g1", "x", "y", "z", "w"]]
      golds = [GoldPassage(query_id="q", passage_id="g1"),
               GoldPassage(query_id="q", passage_id="g2")]
      # g1 hit, g2 not retrieved: 1 of 2 golds -> 0.5
      assert recall_at_k(retrieved, golds, k=5) == 0.5


  def test_recall_span_gold_50pct_boundary_is_hit() -> None:
      # Gold span [10, 20) = 10 tokens; chunk covers tokens [0, 15) of the
      # same doc -> overlap 5/10 = 50% -> hit (binding rule: chunk covers
      # >=50% of the gold span's tokens; integer form 2*overlap >= gold_len).
      gold = GoldSpan(query_id="q", doc_id="d1", start_token=10, end_token=20)
      chunk = make_chunk("d1", start_token=0, end_token=15)
      assert recall_at_k([chunk], [gold], k=5) == 1.0


  def test_recall_span_gold_below_50pct_is_miss() -> None:
      # Chunk covers tokens [0, 14): overlap 4/10 = 40% -> miss
      gold = GoldSpan(query_id="q", doc_id="d1", start_token=10, end_token=20)
      chunk = make_chunk("d1", start_token=0, end_token=14)
      assert recall_at_k([chunk], [gold], k=5) == 0.0


  def test_recall_span_gold_requires_same_document() -> None:
      # Full token overlap but a different doc_id -> miss (Spike 0's span_hit)
      gold = GoldSpan(query_id="q", doc_id="d1", start_token=0, end_token=10)
      chunk = make_chunk("d2", start_token=0, end_token=10)
      assert recall_at_k([chunk], [gold], k=5) == 0.0


  def test_recall_no_golds_raises() -> None:
      with pytest.raises(ValueError):
          recall_at_k([make_chunk("a")], [], k=5)


  # ---------- mrr_at_k ----------

  def test_mrr_first_hit_rank1() -> None:
      retrieved = [make_chunk(p) for p in ["gold", "a", "b"]]
      golds = [GoldPassage(query_id="q", passage_id="gold")]
      assert mrr_at_k(retrieved, golds, k=3) == 1.0


  def test_mrr_first_hit_rank3() -> None:
      retrieved = [make_chunk(p) for p in ["a", "b", "gold"]]
      golds = [GoldPassage(query_id="q", passage_id="gold")]
      # 1/3
      assert mrr_at_k(retrieved, golds, k=3) == pytest.approx(1 / 3)


  def test_mrr_hit_at_rank4_is_zero_within_top3() -> None:
      retrieved = [make_chunk(p) for p in ["a", "b", "c", "gold"]]
      golds = [GoldPassage(query_id="q", passage_id="gold")]
      assert mrr_at_k(retrieved, golds, k=3) == 0.0


  def test_mrr_multi_gold_uses_first_hit_of_any_gold() -> None:
      retrieved = [make_chunk(p) for p in ["a", "g2", "g1"]]
      golds = [GoldPassage(query_id="q", passage_id="g1"),
               GoldPassage(query_id="q", passage_id="g2")]
      # first chunk hitting ANY gold is rank 2 (g2) -> 1/2
      assert mrr_at_k(retrieved, golds, k=3) == 0.5


  def test_mrr_span_gold_uses_first_hit_rank() -> None:
      # rank 1 misses (other doc), rank 2 covers the whole span -> 1/2
      gold = GoldSpan(query_id="q", doc_id="d1", start_token=10, end_token=20)
      retrieved = [make_chunk("d2", start_token=0, end_token=30),
                   make_chunk("d1", start_token=8, end_token=22, position=1)]
      assert mrr_at_k(retrieved, [gold], k=3) == 0.5


  # ---------- normalization / EM / F1 (SQuAD-style) ----------

  def test_normalize_answer() -> None:
      assert normalize_answer("The  Eiffel Tower!") == "eiffel tower"
      assert normalize_answer("A dog, an apple, the end.") == "dog apple end"


  def test_exact_match_normalized() -> None:
      assert exact_match("The Eiffel Tower", ["eiffel tower"]) == 1.0
      assert exact_match("Eiffel", ["eiffel tower"]) == 0.0


  def test_exact_match_multi_gold() -> None:
      assert exact_match("Paris", ["London", "paris"]) == 1.0


  def test_f1_hand_computed_partial_overlap() -> None:
      # pred tokens: {paris, france} (2); gold tokens: {paris} (1); overlap 1
      # precision = 1/2, recall = 1/1, F1 = 2*(0.5*1)/(0.5+1) = 2/3
      assert f1("Paris France", ["Paris"]) == pytest.approx(2 / 3)


  def test_f1_takes_max_over_golds() -> None:
      # vs "Paris": F1 = 2/3 (above). vs "Paris France": F1 = 1.0. max -> 1.0
      assert f1("Paris France", ["Paris", "paris france"]) == 1.0


  def test_f1_zero_overlap() -> None:
      assert f1("London", ["Paris"]) == 0.0


  def test_f1_empty_prediction_vs_nonempty_gold() -> None:
      assert f1("", ["Paris"]) == 0.0
  ```

- [ ] Run them — expected failure mode: `ModuleNotFoundError: No module named 'ragreceipts.eval.metrics'`:

  ```bash
  uv run pytest tests/test_metrics.py -q
  ```

- [ ] Implement. Create `api/ragreceipts/eval/metrics.py`:

  ```python
  """Retrieval + answer metrics.

  Binding definitions: docs/superpowers/plans/2026-06-10-contracts.md §Metrics.
  Gold-to-chunk alignment is owned by Spike 0's eval/alignment.py (kept
  verbatim per R3): GoldPassage(query_id, passage_id) exact-ID match and
  GoldSpan(query_id, doc_id, start_token, end_token) positional >=50%
  token-overlap. This module never reimplements the hit rule - recall/MRR are
  thin wrappers over is_hit / first_hit_rank, which work structurally on
  Chunk because Chunk carries start_token/end_token (R3).
  EM/F1 use the standard SQuAD normalization (lowercase, strip punctuation,
  drop articles a/an/the, collapse whitespace).
  """
  from __future__ import annotations

  import re
  import string
  from collections import Counter

  from ragreceipts.eval.alignment import Gold, first_hit_rank, is_hit
  from ragreceipts.types import Chunk

  _ARTICLES = re.compile(r"\b(a|an|the)\b")


  def recall_at_k(retrieved: list[Chunk], golds: list[Gold], k: int = 5) -> float:
      """Fraction of golds with >=1 hit in the top-k retrieved chunks (per query)."""
      if not golds:
          raise ValueError("recall_at_k requires at least one gold")
      top = retrieved[:k]
      hits = sum(1 for gold in golds if any(is_hit(chunk, gold) for chunk in top))
      return hits / len(golds)


  def mrr_at_k(retrieved: list[Chunk], golds: list[Gold], k: int = 3) -> float:
      """Reciprocal rank (1-based) of the first chunk in the top-k hitting ANY gold; 0 if none."""
      ranks = [
          rank
          for rank in (first_hit_rank(retrieved, gold, k) for gold in golds)
          if rank is not None
      ]
      return 1.0 / min(ranks) if ranks else 0.0


  def normalize_answer(s: str) -> str:
      """SQuAD-style: lowercase, strip punctuation, drop articles, collapse whitespace."""
      s = s.lower()
      s = "".join(ch for ch in s if ch not in string.punctuation)
      s = _ARTICLES.sub(" ", s)
      return " ".join(s.split())


  def exact_match(prediction: str, gold_answers: list[str]) -> float:
      """1.0 iff the normalized prediction equals any normalized gold answer."""
      pred = normalize_answer(prediction)
      return 1.0 if any(pred == normalize_answer(g) for g in gold_answers) else 0.0


  def f1(prediction: str, gold_answers: list[str]) -> float:
      """Max token-overlap F1 over gold answers (SQuAD definition)."""
      pred_tokens = normalize_answer(prediction).split()
      best = 0.0
      for gold in gold_answers:
          gold_tokens = normalize_answer(gold).split()
          if not pred_tokens or not gold_tokens:
              best = max(best, float(pred_tokens == gold_tokens))
              continue
          common = Counter(pred_tokens) & Counter(gold_tokens)
          overlap = sum(common.values())
          if overlap == 0:
              continue
          precision = overlap / len(pred_tokens)
          recall = overlap / len(gold_tokens)
          best = max(best, 2 * precision * recall / (precision + recall))
      return best
  ```

- [ ] Run again — expected: **all tests in `tests/test_metrics.py` PASS**:

  ```bash
  uv run pytest tests/test_metrics.py -q
  ```

- [ ] Lint and commit:

  ```bash
  uv run ruff check ragreceipts/eval/metrics.py tests/test_metrics.py
  git add ragreceipts/eval/metrics.py tests/test_metrics.py
  git commit -m "feat(eval): binding Recall@5/MRR@3/EM/F1 metrics with hand-computed golden tests" \
    -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
  ```

---

### Task 3: `eval/pricing.py` — versioned pricing table

**Files:**
- Create: `api/ragreceipts/eval/pricing.py`
- Test: `api/tests/test_pricing.py`

- [ ] Write the failing tests. Create `api/tests/test_pricing.py`:

  ```python
  """Golden tests for the versioned pricing table. All values hand-computed."""
  import pytest

  from ragreceipts.constants import EMBED_MODEL, RERANK_MODEL, ROUTER_MODEL, SYNTH_MODEL
  from ragreceipts.eval.pricing import (
      PRICING,
      PRICING_VERSION,
      usd_for_rerank,
      usd_for_tokens,
  )


  def test_pricing_version_is_dated() -> None:
      assert PRICING_VERSION == "2026-06-10"


  def test_all_contract_models_priced() -> None:
      for model in (ROUTER_MODEL, SYNTH_MODEL, EMBED_MODEL, RERANK_MODEL):
          assert model in PRICING


  def test_haiku_one_mtok_input() -> None:
      # 1,000,000 input tokens x $1.00/MTok = $1.00
      assert usd_for_tokens(ROUTER_MODEL, 1_000_000, 0) == pytest.approx(1.00)


  def test_sonnet_mixed() -> None:
      # 100k in x $3/MTok = $0.30; 10k out x $15/MTok = $0.15; total $0.45
      assert usd_for_tokens(SYNTH_MODEL, 100_000, 10_000) == pytest.approx(0.45)


  def test_voyage_embed_input_only() -> None:
      # 1M tokens x $0.18/MTok = $0.18; output side is 0 for embeddings
      assert usd_for_tokens(EMBED_MODEL, 1_000_000, 0) == pytest.approx(0.18)


  def test_rerank_per_search_unit() -> None:
      # 1,000 search units x $0.0025 = $2.50
      assert usd_for_rerank(1_000) == pytest.approx(2.50)


  def test_unpriced_model_raises_keyerror() -> None:
      with pytest.raises(KeyError):
          usd_for_tokens("gpt-4o", 1, 1)
  ```

- [ ] Run — expected `ModuleNotFoundError: No module named 'ragreceipts.eval.pricing'`:

  ```bash
  uv run pytest tests/test_pricing.py -q
  ```

- [ ] Implement. Create `api/ragreceipts/eval/pricing.py`:

  ```python
  """Versioned pricing table. PRICING_VERSION is recorded in every receipt.

  Prices verified 2026-06-10:
  - claude-haiku-4-5-20251001 $1.00/$5.00 per MTok and claude-sonnet-4-6
    $3.00/$15.00 per MTok: claude-api skill model table /
    https://platform.claude.com/docs/en/pricing.md
  - voyage-context-3 $0.18 per 1M tokens: https://docs.voyageai.com/docs/pricing
  - rerank-v4.0-pro $0.0025 per search unit (1 query + up to 100 docs; docs
    >500 tokens auto-chunk, each chunk counts): search-unit definition from
    https://cohere.com/pricing; the per-search price is not published there
    (sales-gated) and is corroborated by https://openrouter.ai/cohere/rerank-4-pro
    and https://vercel.com/ai-gateway/models/rerank-v4-pro - RE-VERIFY on the
    Cohere billing dashboard during the first keyed run (see
    docs/runbooks/first-keyed-run.md) and bump PRICING_VERSION if it differs.

  Lookups raise KeyError for unknown models: an unpriced call must never be
  silently billed at $0.
  """
  from __future__ import annotations

  from ragreceipts.constants import EMBED_MODEL, RERANK_MODEL, ROUTER_MODEL, SYNTH_MODEL

  PRICING_VERSION = "2026-06-10"

  PRICING: dict[str, dict] = {
      ROUTER_MODEL: {"usd_per_mtok_input": 1.00, "usd_per_mtok_output": 5.00},
      SYNTH_MODEL: {"usd_per_mtok_input": 3.00, "usd_per_mtok_output": 15.00},
      # JUDGE_MODEL == SYNTH_MODEL ("claude-sonnet-4-6"): same key, priced once.
      EMBED_MODEL: {"usd_per_mtok_input": 0.18, "usd_per_mtok_output": 0.0},
      RERANK_MODEL: {"usd_per_search_unit": 0.0025},
  }


  def usd_for_tokens(model: str, input_tokens: int, output_tokens: int) -> float:
      """Cost in USD for a token-billed model. KeyError if the model is unpriced."""
      entry = PRICING[model]
      return (
          input_tokens * entry["usd_per_mtok_input"]
          + output_tokens * entry["usd_per_mtok_output"]
      ) / 1_000_000


  def usd_for_rerank(n_search_units: int, model: str = RERANK_MODEL) -> float:
      """Cost in USD for search-unit-billed rerank calls."""
      return n_search_units * PRICING[model]["usd_per_search_unit"]
  ```

- [ ] Run again — expected: **all tests in `tests/test_pricing.py` PASS**:

  ```bash
  uv run pytest tests/test_pricing.py -q
  ```

- [ ] Commit:

  ```bash
  git add ragreceipts/eval/pricing.py tests/test_pricing.py
  git commit -m "feat(eval): versioned pricing table for haiku/sonnet/voyage/rerank" \
    -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
  ```

---

### Task 4: `eval/receipts.py` — Receipt schema, envelope, anchors, commit-stripping

**Files:**
- Create: `api/ragreceipts/eval/receipts.py`
- Test: `api/tests/test_receipts.py`

- [ ] Write the failing tests. Create `api/tests/test_receipts.py`:

  ```python
  """Receipt envelope round-trip + anchor-content golden tests."""
  import json

  import pytest

  from ragreceipts.eval.receipts import (
      ANCHOR_SPECS,
      NONDETERMINISM_NOTE,
      SCHEMA_VERSION,
      PublishedAnchor,
      Receipt,
      build_anchor,
      from_envelope,
      make_run_doc,
      strip_for_commit,
      to_envelope,
  )


  def sample_receipt() -> Receipt:
      return Receipt(
          run_id="r1",
          corpus_id="musique-dev-300",
          preset="rerank",
          config={"name": "rerank", "ingest": {"contextual": True}, "query": {"rerank": True}},
          index_hashes={"sparse": "sha256:aa", "dense_contextual": "sha256:bb"},
          models={
              "router": "claude-haiku-4-5-20251001",
              "synth": "claude-sonnet-4-6",
              "judge": "claude-sonnet-4-6",
              "rerank": "rerank-v4.0-pro",
              "embed": "voyage-context-3",
          },
          pricing_table_version="2026-06-10",
          prompts_version="n/a",
          n_total=15,
          n_failed=1,
          n_abstained=2,
          metrics={
              "recall_at_5": 0.8,
              "mrr_at_3": 0.6,
              "em": 0.5,
              "f1": 0.55,
              "ragas_faithfulness": None,
              "ragas_answer_relevancy": None,
              "latency_p50_ms": 900.0,
              "latency_p95_ms": 2100.0,
              "usd_per_query": 0.012,
          },
          per_query=[
              {
                  "query_id": "q1",
                  "retrieved_chunk_ids": ["d1:0", "d2:3"],
                  "answer": "Paris [1]",
                  "latency_ms": 900.0,
                  "usd": 0.011,
                  "flags": {"status": "ok", "em": 1.0, "f1": 1.0},
              }
          ],
          anchors=[
              PublishedAnchor(
                  source="arXiv 2604.01733 Table I (Cohere Rerank v4.0 Pro on T2-RAGBench)",
                  published_value=0.121,
                  measured_value=0.09,
                  direction_match=True,
                  note="financial-domain anchor; direction-match only",
              )
          ],
      )


  def test_envelope_round_trip() -> None:
      receipt = sample_receipt()
      env = to_envelope(receipt)
      assert env["schema_version"] == SCHEMA_VERSION == 1
      # must survive a real JSON wire trip, not just dict identity
      restored = from_envelope(json.loads(json.dumps(env)))
      assert restored == receipt


  def test_envelope_rejects_unknown_schema_version() -> None:
      env = to_envelope(sample_receipt())
      env["schema_version"] = 999
      with pytest.raises(ValueError):
          from_envelope(env)


  def test_envelope_carries_fixed_nondeterminism_note() -> None:
      # R11: every envelope discloses LLM nondeterminism via one fixed string.
      env = to_envelope(sample_receipt())
      assert env["nondeterminism_note"] == NONDETERMINISM_NOTE
      assert "nondeterministic" in env["nondeterminism_note"]


  def test_receipt_prompts_version_is_na_in_plan_b() -> None:
      # R11: "n/a" until Plan C populates agents.prompts.PROMPTS_VERSION.
      receipt = sample_receipt()
      assert receipt.prompts_version == "n/a"
      assert to_envelope(receipt)["receipt"]["prompts_version"] == "n/a"


  def test_nq_anchor_notes_append_corpus_scale_caveat() -> None:
      # R11: nq-dev-300 anchors carry Spike 0's corpus-scale caveat (D1) -
      # query-derived ~300-page corpus, easier than open-corpus retrieval.
      spec = ANCHOR_SPECS["rerank"][0]
      nq = build_anchor(spec, measured_delta=0.04, corpus_id="nq-dev-300")
      other = build_anchor(spec, measured_delta=0.04, corpus_id="musique-dev-300")
      assert "query-derived" in nq.note and "~300" in nq.note
      assert "easier than" in nq.note
      assert "query-derived" not in other.note
      assert other.note == spec.note


  def test_build_anchor_direction_match() -> None:
      spec = ANCHOR_SPECS["rerank"][0]
      assert spec.metric == "recall_at_5"
      assert spec.baseline_preset == "contextual"
      up = build_anchor(spec, measured_delta=0.04)
      down = build_anchor(spec, measured_delta=-0.02)
      assert up.direction_match is True
      assert down.direction_match is False
      assert up.published_value == pytest.approx(0.121)
      assert up.measured_value == pytest.approx(0.04)


  def test_rerank_anchor_notes_carry_required_caveats() -> None:
      notes = " ".join(spec.note for spec in ANCHOR_SPECS["rerank"])
      assert "financial" in notes
      assert "direction-match only" in notes
      assert "2604.01733" in ANCHOR_SPECS["rerank"][0].source
      # both Recall@5 (+12.1pp) and MRR@3 (+17.2pp) anchors exist
      assert {s.metric for s in ANCHOR_SPECS["rerank"]} == {"recall_at_5", "mrr_at_3"}
      assert ANCHOR_SPECS["rerank"][1].published_value == pytest.approx(0.172)


  def test_contextual_anchor_notes_technique_mismatch() -> None:
      note = ANCHOR_SPECS["contextual"][0].note
      assert "LLM-prefix" in note
      assert "voyage-context-3" in note
      assert "self-benchmark" in note
      assert "cross-index" in note


  def test_router_on_anchor_notes_architecture_mismatch() -> None:
      note = ANCHOR_SPECS["router-on"][0].note
      assert "CRAG" in ANCHOR_SPECS["router-on"][0].source
      assert "union_of_hops" in note
      assert "answer-level" in note


  def test_bm25_only_has_no_anchor() -> None:
      assert ANCHOR_SPECS["bm25-only"] == []


  def test_strip_for_commit_removes_text_keeps_ids_and_metrics() -> None:
      doc = make_run_doc(
          run_id="r1",
          corpus_id="musique-dev-300",
          slice_name="smoke",
          receipts=[sample_receipt()],
          skipped=[],
      )
      stripped = strip_for_commit(doc)
      pq = stripped["receipts"][0]["receipt"]["per_query"][0]
      assert "answer" not in pq  # no model/passage text in committed receipts
      assert pq["retrieved_chunk_ids"] == ["d1:0", "d2:3"]
      assert pq["flags"]["em"] == 1.0
      # original doc untouched (deep copy)
      assert "answer" in doc["receipts"][0]["receipt"]["per_query"][0]
  ```

- [ ] Run — expected `ModuleNotFoundError: No module named 'ragreceipts.eval.receipts'`:

  ```bash
  uv run pytest tests/test_receipts.py -q
  ```

- [ ] Implement. Create `api/ragreceipts/eval/receipts.py`:

  ```python
  """receipts.json schema (binding: contracts §receipts.json schema + R11).

  - Receipt/PublishedAnchor dataclasses exactly as in the contracts; Receipt
    carries prompts_version ("n/a" until Plan C populates it from
    agents.prompts.PROMPTS_VERSION - R11).
  - Versioned envelope {"schema_version": 1, "nondeterminism_note": ...,
    "receipt": {...}} per receipt; the note is one fixed string (R11).
  - ANCHOR_SPECS: the published anchors for each preset cell, with REQUIRED
    machine-readable comparability caveats in `note` (spec §Eval plane and
    §Research grounding). Cross-domain anchors claim direction-match only.
    nq-dev-300 runs append the corpus-scale caveat at build_anchor time (R11).
  - strip_for_commit: committed per-query records are IDs + metrics only -
    never passage text, never model answers (benchmark redistribution terms).
  """
  from __future__ import annotations

  import dataclasses
  import json
  from dataclasses import dataclass
  from datetime import datetime, timezone
  from pathlib import Path

  SCHEMA_VERSION = 1

  # R11: fixed disclosure attached to every envelope.
  NONDETERMINISM_NOTE = (
      "LLM calls are nondeterministic even at temperature=0: answer-dependent "
      "metrics (em, f1, ragas_*) can shift slightly between identical runs. "
      "Retrieval metrics (recall_at_5, mrr_at_3) are deterministic for a fixed "
      "index. Treat small answer-metric deltas as noise, not findings."
  )

  # R11: corpus-scale caveat from Spike 0's decisions doc (D1), appended to every
  # anchor note on nq-dev-300 runs.
  NQ_CORPUS_SCALE_NOTE = (
      " | nq-dev-300 corpus-scale caveat (Spike 0 decisions D1): the corpus is "
      "query-derived (~300 content-deduped Wikipedia pages), so retrieval "
      "difficulty is 'find the right chunk among ~300 pages' - easier than the "
      "open-corpus retrieval behind published numbers."
  )


  @dataclass(frozen=True)
  class PublishedAnchor:
      source: str             # e.g. "arXiv 2604.01733 Table I"
      published_value: float
      measured_value: float
      direction_match: bool
      note: str               # REQUIRED - comparability caveats (domain/technique mismatch)


  @dataclass(frozen=True)
  class Receipt:
      run_id: str
      corpus_id: str
      preset: str
      config: dict            # full PipelineConfig as dict
      index_hashes: dict      # the variant hashes actually used
      models: dict            # router/synth/judge/rerank/embed model IDs
      pricing_table_version: str
      prompts_version: str    # "n/a" in Plan B; agents.prompts.PROMPTS_VERSION in Plan C (R11)
      n_total: int
      n_failed: int
      n_abstained: int
      metrics: dict           # recall_at_5, mrr_at_3, em, f1, ragas_faithfulness,
                              # ragas_answer_relevancy, latency_p50_ms, latency_p95_ms,
                              # usd_per_query
      per_query: list[dict]   # query_id, retrieved chunk_ids, answer, latency_ms, usd, flags
      anchors: list[PublishedAnchor]


  def to_envelope(receipt: Receipt) -> dict:
      return {
          "schema_version": SCHEMA_VERSION,
          "nondeterminism_note": NONDETERMINISM_NOTE,
          "receipt": dataclasses.asdict(receipt),
      }


  def from_envelope(data: dict) -> Receipt:
      if data.get("schema_version") != SCHEMA_VERSION:
          raise ValueError(
              f"unsupported receipt schema_version: {data.get('schema_version')!r} "
              f"(this build reads version {SCHEMA_VERSION})"
          )
      raw = dict(data["receipt"])
      raw["anchors"] = [PublishedAnchor(**a) for a in raw["anchors"]]
      return Receipt(**raw)


  # ---------------------------------------------------------------------------
  # Published anchors per preset cell.
  # ---------------------------------------------------------------------------

  @dataclass(frozen=True)
  class AnchorSpec:
      """An anchor template: published delta on `metric` vs `baseline_preset`.

      measured_value for the receipt is computed by the runner as
      metrics[metric](preset) - metrics[metric](baseline_preset); direction_match
      compares the SIGN of that delta against the sign of published_value.
      """

      source: str
      published_value: float
      metric: str
      baseline_preset: str | None
      note: str


  ANCHOR_SPECS: dict[str, list[AnchorSpec]] = {
      # Ladder base: nothing to compare against; honest empty list.
      "bm25-only": [],
      "dense-rrf": [
          AnchorSpec(
              source="arXiv 2604.01733 (T2-RAGBench: BM25 0.644 vs dense 0.587 R@5)",
              published_value=0.057,
              metric="recall_at_5",
              baseline_preset="bm25-only",
              note=(
                  "Comparison mismatch disclosed: the published figure is BM25-alone beating "
                  "SOTA-dense-alone by +5.7pp Recall@5 on a financial-domain benchmark (single "
                  "non-peer-reviewed Apr 2026 preprint); this cell measures "
                  "hybrid(BM25+dense+RRF) minus BM25-alone on NQ/MuSiQue - an adjacent but "
                  "distinct comparison, included to ground the design rule that sparse "
                  "retrieval must be kept. Cross-domain: direction-match only, never magnitude."
              ),
          )
      ],
      "contextual": [
          AnchorSpec(
              source=(
                  "arXiv 2604.01733 (independent eval: +2-3pp for LLM-prefix contextual "
                  "retrieval) vs anthropic.com/news/contextual-retrieval"
              ),
              published_value=0.025,  # midpoint of the independent +2-3pp range
              metric="recall_at_5",
              baseline_preset="dense-rrf",
              note=(
                  "Technique mismatch (REQUIRED caveat): the independent +2-3pp figure is for "
                  "LLM-prefix-style contextual retrieval, while this cell uses voyage-context-3 "
                  "whole-document contextualized embeddings - a different technique. Voyage's "
                  "own deltas and Anthropic's 35-67% failure-reduction headline are vendor "
                  "self-benchmarks, verified-as-stated but not independently replicated. This "
                  "cell is also a cross-index comparison (contextual vs isolated named vectors, "
                  "different manifest hashes), not a query-time flag flip. Direction-match only."
              ),
          )
      ],
      "rerank": [
          AnchorSpec(
              source="arXiv 2604.01733 Table I (Cohere Rerank v4.0 Pro on T2-RAGBench)",
              published_value=0.121,
              metric="recall_at_5",
              baseline_preset="contextual",
              note=(
                  "+12.1pp Recall@5 over hybrid RRF was measured on T2-RAGBench, a "
                  "financial-domain benchmark, in a single non-peer-reviewed Apr 2026 preprint; "
                  "our corpora are NQ/MuSiQue. Cross-domain: direction-match only, never "
                  "magnitude reproduction. The rerank model matches the anchor variant exactly "
                  "(rerank-v4.0-pro). Domain transfer is itself a finding, not a failure."
              ),
          ),
          AnchorSpec(
              source="arXiv 2604.01733 Table I (Cohere Rerank v4.0 Pro on T2-RAGBench)",
              published_value=0.172,
              metric="mrr_at_3",
              baseline_preset="contextual",
              note=(
                  "+17.2pp MRR@3 over hybrid RRF; same financial-domain, single-preprint "
                  "caveat as the Recall@5 anchor - direction-match only, never magnitude."
              ),
          ),
      ],
      "router-on": [
          AnchorSpec(
              source="arXiv 2604.01733 (CRAG 0.658 < plain hybrid fusion 0.695)",
              published_value=-0.037,
              metric="f1",
              baseline_preset="rerank",
              note=(
                  "Architecture mismatch disclosed: the published number shows agentic "
                  "rewriting REPLACING strong retrieval (CRAG) underperforming plain hybrid "
                  "fusion, while router-on layers a System-2 loop ON TOP OF the same retrieval "
                  "core - the closest independent agentic-vs-static datapoint, not the same "
                  "design. Primary metrics for this cell are answer-level EM/F1 + RAGAS; "
                  "retrieval recall over the union of per-hop top-5 is a secondary diagnostic "
                  "flagged union_of_hops:true. Runs on the multi-hop corpus only."
              ),
          )
      ],
  }


  def build_anchor(
      spec: AnchorSpec, measured_delta: float, *, corpus_id: str = ""
  ) -> PublishedAnchor:
      note = spec.note
      if corpus_id == "nq-dev-300":
          note += NQ_CORPUS_SCALE_NOTE  # R11: corpus-scale caveat, machine-appended
      return PublishedAnchor(
          source=spec.source,
          published_value=spec.published_value,
          measured_value=round(measured_delta, 4),
          direction_match=(measured_delta > 0) == (spec.published_value > 0),
          note=note,
      )


  # ---------------------------------------------------------------------------
  # Run documents: data/receipts-local/<run_id>.json
  # ---------------------------------------------------------------------------

  def make_run_doc(
      *,
      run_id: str,
      corpus_id: str,
      slice_name: str,
      receipts: list[Receipt],
      skipped: list,  # list[SkippedCell] - kept untyped to avoid a runner import cycle
  ) -> dict:
      return {
          "schema_version": SCHEMA_VERSION,
          "run_id": run_id,
          "corpus_id": corpus_id,
          "slice": slice_name,
          "created_at": datetime.now(timezone.utc).isoformat(),
          "receipts": [to_envelope(r) for r in receipts],
          "skipped": [{"preset": s.preset, "reason": s.reason} for s in skipped],
      }


  def write_run_doc(doc: dict, data_dir: Path) -> Path:
      path = data_dir / "receipts-local" / f"{doc['run_id']}.json"
      path.parent.mkdir(parents=True, exist_ok=True)
      path.write_text(json.dumps(doc, indent=2) + "\n")
      return path


  def read_run_doc(path: Path) -> dict:
      doc = json.loads(path.read_text())
      if doc.get("schema_version") != SCHEMA_VERSION:
          raise ValueError(f"unsupported run-doc schema_version in {path}")
      return doc


  # Committed receipts: IDs + metrics ONLY (no passage text, no model answers).
  _COMMITTED_PER_QUERY_KEYS = {
      "query_id",
      "retrieved_chunk_ids",
      "latency_ms",
      "usd",
      "flags",
  }


  def strip_for_commit(run_doc: dict) -> dict:
      """Deep-copy a run doc and strip per-query fields not safe to commit."""
      out = json.loads(json.dumps(run_doc))
      for env in out["receipts"]:
          env["receipt"]["per_query"] = [
              {k: v for k, v in pq.items() if k in _COMMITTED_PER_QUERY_KEYS}
              for pq in env["receipt"]["per_query"]
          ]
      return out
  ```

- [ ] Run again — expected: **all tests in `tests/test_receipts.py` PASS**:

  ```bash
  uv run pytest tests/test_receipts.py -q
  ```

- [ ] Commit:

  ```bash
  git add ragreceipts/eval/receipts.py tests/test_receipts.py
  git commit -m "feat(eval): Receipt/PublishedAnchor schema, anchors with required caveats, commit-stripping" \
    -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
  ```

---

### Task 5: `eval/ragas_adapter.py` — RAGAS v0.4 behind a Protocol, fake for CI

**Files:**
- Create: `api/ragreceipts/eval/ragas_adapter.py`
- Create: `api/ragreceipts/vendors/ragas_clients.py`
- Modify: `api/tests/fakes.py` (append `FakeRagas`)
- Test: `api/tests/test_ragas_adapter.py`

- [ ] Write the failing tests. Create `api/tests/test_ragas_adapter.py`:

  ```python
  """The RAGAS seam: Protocol + fake. The real judge is keyed and NEVER runs in CI."""
  from ragreceipts.eval.ragas_adapter import RagasJudge, RagasScores, RagasV04Judge
  from tests.fakes import FakeRagas  # tests/ is a package (R8)


  def test_fake_ragas_conforms_to_protocol_and_scripts_scores() -> None:
      fake = FakeRagas(scores=[RagasScores(faithfulness=0.9, answer_relevancy=0.8)])
      judge: RagasJudge = fake  # structural typing check
      out = judge.score(question="q?", answer="a", contexts=["c1", "c2"])
      assert out == RagasScores(faithfulness=0.9, answer_relevancy=0.8)
      assert fake.calls == [{"question": "q?", "answer": "a", "contexts": ["c1", "c2"]}]


  def test_real_judge_class_exists_but_is_not_constructed_offline() -> None:
      # Construction would import ragas + download a sentence-transformers model;
      # CI only asserts the class is importable and documents the keyed path.
      assert RagasV04Judge.__init__ is not object.__init__
  ```

- [ ] Run — expected `ModuleNotFoundError: No module named 'ragreceipts.eval.ragas_adapter'`:

  ```bash
  uv run pytest tests/test_ragas_adapter.py -q
  ```

- [ ] Implement the adapter. Create `api/ragreceipts/eval/ragas_adapter.py`:

  ```python
  """RAGAS v0.4 adapter behind a Protocol (transport-seam rule: CI uses FakeRagas).

  Verified against docs.ragas.io (stable, fetched 2026-06-10):
  - collections metrics: from ragas.metrics.collections import Faithfulness, AnswerRelevancy
      https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/
      https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/answer_relevance/
  - direct Anthropic provider (Instructor adapter, no LangChain wrapper):
      llm_factory("<model>", provider="anthropic", client=...)
      https://docs.ragas.io/en/stable/howtos/llm-adapters/
  - local sentence-transformers embeddings (no API key, works offline):
      HuggingFaceEmbeddings(model="BAAI/bge-small-en-v1.5")
      https://docs.ragas.io/en/stable/references/embeddings/
  WARNING: most online examples show the obsolete v0.2 evaluate()/SingleTurnSample
  API - do not use it. The v0.3->v0.4 migration guide confirms .ascore()/.score()
  returning MetricResult with .value:
      https://docs.ragas.io/en/stable/howtos/migrations/migrate_from_v03_to_v04/
  """
  from __future__ import annotations

  from dataclasses import dataclass
  from typing import Protocol

  from ragreceipts.constants import JUDGE_MODEL, RAGAS_EMBED_MODEL


  @dataclass(frozen=True)
  class RagasScores:
      faithfulness: float
      answer_relevancy: float


  class RagasJudge(Protocol):
      def score(self, *, question: str, answer: str, contexts: list[str]) -> RagasScores: ...


  class RagasV04Judge:
      """Real RAGAS v0.4 judge. Requires ANTHROPIC_API_KEY; never constructed in CI.

      All third-party imports are lazy so `import ragreceipts.eval.ragas_adapter`
      stays cheap and offline-safe.
      """

      def __init__(
          self,
          anthropic_client: object,
          judge_model: str = JUDGE_MODEL,
          embed_model: str = RAGAS_EMBED_MODEL,
      ) -> None:
          from ragas.embeddings import HuggingFaceEmbeddings
          from ragas.llms import llm_factory
          from ragas.metrics.collections import AnswerRelevancy, Faithfulness

          llm = llm_factory(judge_model, provider="anthropic", client=anthropic_client)
          embeddings = HuggingFaceEmbeddings(model=embed_model)  # local, zero keys
          self._faithfulness = Faithfulness(llm=llm)
          self._answer_relevancy = AnswerRelevancy(llm=llm, embeddings=embeddings)

      def score(self, *, question: str, answer: str, contexts: list[str]) -> RagasScores:
          faith = self._faithfulness.score(
              user_input=question, response=answer, retrieved_contexts=contexts
          )
          rel = self._answer_relevancy.score(user_input=question, response=answer)
          return RagasScores(
              faithfulness=float(faith.value), answer_relevancy=float(rel.value)
          )
  ```

- [ ] Implement the vendor-side client constructor (the eval plane never imports `anthropic`). Create `api/ragreceipts/vendors/ragas_clients.py`:

  ```python
  """Raw vendor client constructor for the RAGAS judge (transport seam).

  Contracts rule: application code never imports `anthropic` outside vendors/.
  ragas's llm_factory needs the raw SDK client (not our ClaudeTransport), so the
  one place that constructs it lives here. anthropic.Anthropic() reads
  ANTHROPIC_API_KEY from the environment.
  """
  from __future__ import annotations


  def make_anthropic_client() -> object:
      import anthropic

      return anthropic.Anthropic()
  ```

- [ ] Add `from ragreceipts.eval.ragas_adapter import RagasScores` to the import block at the **top** of `api/tests/fakes.py` (keeping it at module top avoids a ruff E402), then append `FakeRagas` at the end of the file (do not touch Plan A's existing fakes):

  ```python
  # --- Plan B: RAGAS judge fake -------------------------------------------------


  class FakeRagas:
      """Scripted RagasJudge for CI: returns queued scores in call order."""

      def __init__(self, scores: list[RagasScores]) -> None:
          self._scores = list(scores)
          self.calls: list[dict] = []

      def score(self, *, question: str, answer: str, contexts: list[str]) -> RagasScores:
          self.calls.append(
              {"question": question, "answer": answer, "contexts": contexts}
          )
          return self._scores.pop(0)
  ```

- [ ] Run again — expected: **all tests in `tests/test_ragas_adapter.py` PASS** (and the rest of the suite still green):

  ```bash
  uv run pytest tests/test_ragas_adapter.py -q && uv run pytest -q
  ```

- [ ] Commit:

  ```bash
  git add ragreceipts/eval/ragas_adapter.py ragreceipts/vendors/ragas_clients.py \
    tests/fakes.py tests/test_ragas_adapter.py
  git commit -m "feat(eval): RAGAS v0.4 collections adapter behind Protocol with CI fake" \
    -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
  ```

---

### Task 6: `eval/queries.py` + `eval/run_state.py` — query loading, slices, resumable run state

**Files:**
- Create: `api/ragreceipts/eval/queries.py`
- Create: `api/ragreceipts/eval/run_state.py`
- Test: `api/tests/test_queries.py`
- Test: `api/tests/test_run_state.py`

- [ ] Write the failing query-loading tests. Create `api/tests/test_queries.py`:

  ```python
  """load_queries reads Spike 0's raw layout directly and normalizes in memory
  (R1/R2): no intermediate eval queries file exists anywhere."""
  import json
  from pathlib import Path

  import pytest

  from ragreceipts.eval.alignment import GoldPassage, GoldSpan
  from ragreceipts.eval.queries import (
      QueryRecord,
      load_queries,
      slice_queries,
      slice_query_ids,
  )


  def musique_record(i: int) -> dict:
      """Spike 0 raw queries.jsonl record shape (MuSiQue, typed passage gold)."""
      return {
          "query_id": f"mq{i}",
          "question": f"question {i}?",
          "answer": f"answer {i}",
          "answer_aliases": [f"alias {i}"],
          "gold": {"type": "passage", "passage_ids": [f"mu-p{i}a", f"mu-p{i}b"]},
      }


  def nq_record(i: int) -> dict:
      """Spike 0 raw queries.jsonl record shape (NQ, typed span gold)."""
      return {
          "query_id": f"nqq-{i}",
          "question": f"who did thing {i}?",
          "answer_texts": [f"short answer {i}"],
          "gold": {"type": "span", "doc_id": f"nq-d{i}",
                   "start_token": 4, "end_token": 24},
          "gold_text": "twenty tokens of gold text",
      }


  def write_raw_corpus(tmp_path: Path, corpus_id: str, records: list[dict]) -> Path:
      raw = tmp_path / "corpora" / corpus_id / "raw"
      raw.mkdir(parents=True)
      (raw / "queries.jsonl").write_text(
          "\n".join(json.dumps(r) for r in records) + "\n"
      )
      full = [r["query_id"] for r in records]
      (raw / "slice-full.json").write_text(json.dumps(full))
      (raw / "slice-smoke.json").write_text(json.dumps(full[:15]))
      return tmp_path


  def test_load_queries_normalizes_musique_golds(tmp_path: Path) -> None:
      data_dir = write_raw_corpus(tmp_path, "musique-dev-300",
                                  [musique_record(i) for i in range(3)])
      queries = load_queries(data_dir, "musique-dev-300")
      assert len(queries) == 3
      assert isinstance(queries[0], QueryRecord)
      q0 = queries[0]
      assert q0.query_id == "mq0"
      assert q0.question == "question 0?"
      # R2: gold_answers = [answer] + answer_aliases
      assert q0.gold_answers == ["answer 0", "alias 0"]
      assert q0.golds == [
          GoldPassage(query_id="mq0", passage_id="mu-p0a"),
          GoldPassage(query_id="mq0", passage_id="mu-p0b"),
      ]


  def test_load_queries_normalizes_nq_span_golds(tmp_path: Path) -> None:
      data_dir = write_raw_corpus(tmp_path, "nq-dev-300",
                                  [nq_record(i) for i in range(2)])
      queries = load_queries(data_dir, "nq-dev-300")
      q0 = queries[0]
      # R2: gold_answers = answer_texts for NQ
      assert q0.gold_answers == ["short answer 0"]
      # R3: span golds are positional token ranges, never span_text strings
      assert q0.golds == [
          GoldSpan(query_id="nqq-0", doc_id="nq-d0", start_token=4, end_token=24)
      ]


  def test_slices_come_from_slice_files(tmp_path: Path) -> None:
      data_dir = write_raw_corpus(tmp_path, "c",
                                  [musique_record(i) for i in range(20)])
      queries = load_queries(data_dir, "c")
      smoke_ids = slice_query_ids(data_dir, "c", "smoke")
      assert smoke_ids == [f"mq{i}" for i in range(15)]
      smoke = slice_queries(queries, smoke_ids)
      assert [q.query_id for q in smoke] == smoke_ids
      assert len(slice_queries(queries, slice_query_ids(data_dir, "c", "full"))) == 20


  def test_slice_order_is_the_files_order_not_line_order(tmp_path: Path) -> None:
      data_dir = write_raw_corpus(tmp_path, "c",
                                  [musique_record(i) for i in range(3)])
      raw = data_dir / "corpora" / "c" / "raw"
      (raw / "slice-smoke.json").write_text(json.dumps(["mq2", "mq0"]))
      queries = load_queries(data_dir, "c")
      smoke = slice_queries(queries, slice_query_ids(data_dir, "c", "smoke"))
      assert [q.query_id for q in smoke] == ["mq2", "mq0"]


  def test_slice_referencing_unknown_query_id_raises(tmp_path: Path) -> None:
      data_dir = write_raw_corpus(tmp_path, "c", [musique_record(0)])
      with pytest.raises(ValueError) as exc:
          slice_queries(load_queries(data_dir, "c"), ["mq0", "ghost-id"])
      assert "ghost-id" in str(exc.value)


  def test_unknown_slice_raises(tmp_path: Path) -> None:
      data_dir = write_raw_corpus(tmp_path, "c", [musique_record(0)])
      with pytest.raises(ValueError):
          slice_query_ids(data_dir, "c", "medium")


  def test_missing_queries_file_names_the_download_script(tmp_path: Path) -> None:
      with pytest.raises(FileNotFoundError) as exc:
          load_queries(tmp_path, "nope")
      assert "queries.jsonl" in str(exc.value)
      assert "scripts/download_data.py" in str(exc.value)  # the real producer (Spike 0)
  ```

- [ ] Write the failing run-state tests. Create `api/tests/test_run_state.py`:

  ```python
  from pathlib import Path

  from ragreceipts.eval.run_state import RunStore


  def test_start_run_is_idempotent(tmp_path: Path) -> None:
      store = RunStore(tmp_path / "runs.db")
      store.start_run(run_id="r1", corpus_id="c", slice_name="smoke",
                      presets=["bm25-only"], spend_cap_usd=5.0)
      store.start_run(run_id="r1", corpus_id="c", slice_name="smoke",
                      presets=["bm25-only"], spend_cap_usd=5.0)  # no error, no dup


  def test_record_and_resume_skips_completed(tmp_path: Path) -> None:
      store = RunStore(tmp_path / "runs.db")
      store.start_run(run_id="r1", corpus_id="c", slice_name="smoke",
                      presets=["bm25-only"], spend_cap_usd=5.0)
      store.record_result(
          run_id="r1", preset="bm25-only", query_id="q1", status="ok",
          retrieved=[{"chunk_id": "d1:0", "passage_id": "p1",
                      "start_token": 0, "end_token": 1, "text": "t"}],
          answer="Paris [1]", latency_ms=12.5, usd=0.01,
          input_tokens=100, output_tokens=20, error=None,
      )
      assert store.completed_query_ids("r1", "bm25-only") == {"q1"}
      assert store.completed_query_ids("r1", "rerank") == set()


  def test_spent_usd_sums_across_presets(tmp_path: Path) -> None:
      store = RunStore(tmp_path / "runs.db")
      store.start_run(run_id="r1", corpus_id="c", slice_name="smoke",
                      presets=["a", "b"], spend_cap_usd=5.0)
      for preset, usd in (("a", 0.01), ("b", 0.02)):
          store.record_result(
              run_id="r1", preset=preset, query_id="q1", status="ok",
              retrieved=[], answer="x", latency_ms=1.0, usd=usd,
              input_tokens=1, output_tokens=1, error=None,
          )
      assert abs(store.spent_usd("r1") - 0.03) < 1e-9


  def test_results_for_round_trips_rows(tmp_path: Path) -> None:
      store = RunStore(tmp_path / "runs.db")
      store.start_run(run_id="r1", corpus_id="c", slice_name="smoke",
                      presets=["a"], spend_cap_usd=5.0)
      store.record_result(
          run_id="r1", preset="a", query_id="q9", status="failed",
          retrieved=[], answer=None, latency_ms=3.0, usd=0.0,
          input_tokens=0, output_tokens=0, error="RuntimeError('boom')",
      )
      rows = store.results_for("r1", "a")
      assert len(rows) == 1
      assert rows[0]["status"] == "failed"
      assert rows[0]["error"] == "RuntimeError('boom')"
      assert rows[0]["retrieved"] == []
  ```

- [ ] Run — expected `ModuleNotFoundError` for both new modules:

  ```bash
  uv run pytest tests/test_queries.py tests/test_run_state.py -q
  ```

- [ ] Implement query loading. Create `api/ragreceipts/eval/queries.py`:

  ```python
  """Query/gold loading + slices.

  Reads Spike 0's raw slice layout DIRECTLY and normalizes in memory (R1/R2 -
  there is no intermediate eval queries file):

    data/corpora/{corpus_id}/raw/queries.jsonl - one JSON object per line:
      MuSiQue: {"query_id","question","answer","answer_aliases",
                "gold":{"type":"passage","passage_ids":[...]}}
      NQ:      {"query_id","question","answer_texts",
                "gold":{"type":"span","doc_id","start_token","end_token"},
                "gold_text"}
    data/corpora/{corpus_id}/raw/slice-{smoke,full}.json - JSON arrays of
      query_id strings (slice-smoke.json is the first 15 of slice-full.json,
      Spike 0 decisions D4 - the size lives in the data, not in this module).

  Normalization: gold_answers = [answer] + answer_aliases (MuSiQue) or
  answer_texts (NQ); golds become Spike 0's typed GoldPassage / GoldSpan
  from eval/alignment.py. Span golds are token ranges, never text.
  """
  from __future__ import annotations

  import json
  from dataclasses import dataclass
  from pathlib import Path

  from ragreceipts.eval.alignment import Gold, GoldPassage, GoldSpan


  @dataclass(frozen=True)
  class QueryRecord:
      query_id: str
      question: str
      gold_answers: list[str]
      golds: list[Gold]


  def _normalize(raw: dict) -> QueryRecord:
      gold = raw["gold"]
      if gold["type"] == "passage":
          golds: list[Gold] = [
              GoldPassage(query_id=raw["query_id"], passage_id=pid)
              for pid in gold["passage_ids"]
          ]
          gold_answers = [raw["answer"], *raw.get("answer_aliases", [])]
      elif gold["type"] == "span":
          golds = [
              GoldSpan(query_id=raw["query_id"], doc_id=gold["doc_id"],
                       start_token=gold["start_token"], end_token=gold["end_token"])
          ]
          gold_answers = list(raw["answer_texts"])
      else:
          raise ValueError(f"{raw['query_id']}: unknown gold type {gold['type']!r}")
      return QueryRecord(query_id=raw["query_id"], question=raw["question"],
                         gold_answers=gold_answers, golds=golds)


  def load_queries(data_dir: Path, corpus_id: str) -> list[QueryRecord]:
      path = data_dir / "corpora" / corpus_id / "raw" / "queries.jsonl"
      if not path.exists():
          raise FileNotFoundError(
              f"{path} not found - run Spike 0's download script first: "
              f"`uv run --project api python scripts/download_data.py --corpus all` "
              f"(from the repo root) materializes data/corpora/{corpus_id}/raw/"
          )
      with path.open() as f:
          return [_normalize(json.loads(line)) for line in f if line.strip()]


  def slice_query_ids(data_dir: Path, corpus_id: str, slice_name: str) -> list[str]:
      """Read Spike 0's slice file: a JSON array of query_id strings."""
      if slice_name not in ("smoke", "full"):
          raise ValueError(f"unknown slice {slice_name!r}; expected 'smoke' or 'full'")
      path = data_dir / "corpora" / corpus_id / "raw" / f"slice-{slice_name}.json"
      if not path.exists():
          raise FileNotFoundError(
              f"{path} not found - Spike 0's scripts/download_data.py writes the "
              f"slice files next to queries.jsonl"
          )
      return list(json.loads(path.read_text()))


  def slice_queries(queries: list[QueryRecord], slice_ids: list[str]) -> list[QueryRecord]:
      """Project queries onto a slice (query-id list), preserving the slice's order."""
      by_id = {q.query_id: q for q in queries}
      missing = [qid for qid in slice_ids if qid not in by_id]
      if missing:
          raise ValueError(
              f"slice references query_ids absent from queries.jsonl: {missing}"
          )
      return [by_id[qid] for qid in slice_ids]


  def load_manifest(data_dir: Path, corpus_id: str) -> dict:
      return json.loads((data_dir / "corpora" / corpus_id / "manifest.json").read_text())
  ```

- [ ] Implement run state. Create `api/ragreceipts/eval/run_state.py`:

  ```python
  """SQLite-backed run state: this is what makes eval runs resumable.

  WAL mode per spec §Server runtime constraints. Primary key
  (run_id, preset, query_id) means a resumed run skips completed queries and a
  re-recorded query replaces its row (INSERT OR REPLACE).
  """
  from __future__ import annotations

  import json
  import sqlite3
  from datetime import datetime, timezone
  from pathlib import Path

  _SCHEMA = """
  CREATE TABLE IF NOT EXISTS eval_runs (
    run_id TEXT PRIMARY KEY,
    corpus_id TEXT NOT NULL,
    slice TEXT NOT NULL,
    presets TEXT NOT NULL,
    spend_cap_usd REAL NOT NULL,
    created_at TEXT NOT NULL
  );
  CREATE TABLE IF NOT EXISTS eval_results (
    run_id TEXT NOT NULL,
    preset TEXT NOT NULL,
    query_id TEXT NOT NULL,
    status TEXT NOT NULL,              -- 'ok' | 'abstained' | 'failed'
    retrieved TEXT NOT NULL,           -- JSON [{chunk_id, passage_id, start_token, end_token, text}]
    answer TEXT,
    latency_ms REAL NOT NULL,
    usd REAL NOT NULL,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    error TEXT,
    PRIMARY KEY (run_id, preset, query_id)
  );
  """


  class RunStore:
      def __init__(self, db_path: Path) -> None:
          db_path.parent.mkdir(parents=True, exist_ok=True)
          self._conn = sqlite3.connect(db_path)
          self._conn.execute("PRAGMA journal_mode=WAL")
          self._conn.executescript(_SCHEMA)
          self._conn.commit()

      def start_run(self, *, run_id: str, corpus_id: str, slice_name: str,
                    presets: list[str], spend_cap_usd: float) -> None:
          self._conn.execute(
              "INSERT OR IGNORE INTO eval_runs VALUES (?, ?, ?, ?, ?, ?)",
              (run_id, corpus_id, slice_name, json.dumps(presets), spend_cap_usd,
               datetime.now(timezone.utc).isoformat()),
          )
          self._conn.commit()

      def record_result(self, *, run_id: str, preset: str, query_id: str, status: str,
                        retrieved: list[dict], answer: str | None, latency_ms: float,
                        usd: float, input_tokens: int, output_tokens: int,
                        error: str | None) -> None:
          self._conn.execute(
              "INSERT OR REPLACE INTO eval_results "
              "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
              (run_id, preset, query_id, status, json.dumps(retrieved), answer,
               latency_ms, usd, input_tokens, output_tokens, error),
          )
          self._conn.commit()

      def completed_query_ids(self, run_id: str, preset: str) -> set[str]:
          rows = self._conn.execute(
              "SELECT query_id FROM eval_results WHERE run_id = ? AND preset = ?",
              (run_id, preset),
          ).fetchall()
          return {r[0] for r in rows}

      def spent_usd(self, run_id: str) -> float:
          row = self._conn.execute(
              "SELECT COALESCE(SUM(usd), 0.0) FROM eval_results WHERE run_id = ?",
              (run_id,),
          ).fetchone()
          return float(row[0])

      def results_for(self, run_id: str, preset: str) -> list[dict]:
          rows = self._conn.execute(
              "SELECT query_id, status, retrieved, answer, latency_ms, usd, "
              "input_tokens, output_tokens, error "
              "FROM eval_results WHERE run_id = ? AND preset = ?",
              (run_id, preset),
          ).fetchall()
          return [
              {
                  "query_id": r[0],
                  "status": r[1],
                  "retrieved": json.loads(r[2]),
                  "answer": r[3],
                  "latency_ms": r[4],
                  "usd": r[5],
                  "input_tokens": r[6],
                  "output_tokens": r[7],
                  "error": r[8],
              }
              for r in rows
          ]
  ```

- [ ] Run again — expected: **all tests in both files PASS**:

  ```bash
  uv run pytest tests/test_queries.py tests/test_run_state.py -q
  ```

- [ ] Commit:

  ```bash
  git add ragreceipts/eval/queries.py ragreceipts/eval/run_state.py \
    tests/test_queries.py tests/test_run_state.py
  git commit -m "feat(eval): query/gold loading from Spike 0 raw slices + resumable SQLite run state" \
    -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
  ```

---

### Task 7: `eval/runner.py` — ablation runner with cost guard, skips, and receipt building

**Files:**
- Create: `api/ragreceipts/eval/runner.py`
- Test: `api/tests/test_runner.py`

- [ ] Write the failing tests. Create `api/tests/test_runner.py`:

  ```python
  """Runner unit tests: skips, cost guard, resume, disclosure, receipt building.

  All vendor traffic goes through local Protocol stubs (ClaudeTransport from the
  contracts) - zero keys, zero network. Corpus fixtures use Spike 0's raw
  layout (R1/R2): raw/queries.jsonl with typed golds + slice-file query-id
  lists.
  """
  import json
  from pathlib import Path

  import pytest

  from ragreceipts.eval.ragas_adapter import RagasScores
  from ragreceipts.eval.run_state import RunStore
  from ragreceipts.eval.runner import (
      AblationRunner,
      S1Answer,
      SpendCapExceeded,
      estimate_run_cost,
  )
  from ragreceipts.types import Chunk, ScoredChunk
  from ragreceipts.vendors.base import ParsedResult
  from tests.fakes import FakeRagas


  # ---------- local stubs (contracts Protocols) ----------

  class StubClaude:
      """ClaudeTransport stub; answers keyed by the question text."""

      def __init__(self, answers: dict[str, S1Answer]) -> None:
          self._answers = answers
          self.parse_calls = 0

      def complete(self, *, model, system, user, max_tokens, temperature=0.0):
          raise AssertionError("Plan B synthesis uses parse(), not complete()")

      def parse(self, *, model, system, user, max_tokens, output_format, temperature=0.0):
          self.parse_calls += 1
          question = user.rsplit("Question: ", 1)[1]
          return ParsedResult(
              parsed=self._answers[question], input_tokens=1000, output_tokens=100
          )


  class StubCore:
      """Duck-typed RetrievalCore: fixed results per question text."""

      def __init__(self, results: dict[str, list[ScoredChunk]],
                   fail_on: frozenset[str] = frozenset()) -> None:
          self._results = results
          self._fail_on = fail_on

      def retrieve(self, query: str) -> list[ScoredChunk]:
          if query in self._fail_on:
              raise RuntimeError("retrieval exploded")
          return self._results[query]


  def sc(passage_id: str, text: str = "some text") -> ScoredChunk:
      chunk = Chunk(chunk_id=f"{passage_id}:0", corpus_id="c", doc_id=passage_id,
                    passage_id=passage_id, text=text, position=0,
                    start_token=0, end_token=len(text.split()))
      return ScoredChunk(chunk=chunk, score=1.0, source="bm25")


  def write_eval_corpus(tmp_path: Path, corpus_id: str, dataset_name: str) -> Path:
      """Spike 0 raw layout (R1) + Plan A's ingest manifest."""
      raw = tmp_path / "corpora" / corpus_id / "raw"
      raw.mkdir(parents=True)
      lines = []
      for i in range(2):
          lines.append(json.dumps({
              "query_id": f"q{i}",
              "question": f"question {i}?",
              "answer": f"answer {i}",
              "answer_aliases": [],
              "gold": {"type": "passage", "passage_ids": [f"p{i}"]},
          }))
      (raw / "queries.jsonl").write_text("\n".join(lines) + "\n")
      (raw / "slice-full.json").write_text(json.dumps(["q0", "q1"]))
      (raw / "slice-smoke.json").write_text(json.dumps(["q0", "q1"]))
      (tmp_path / "corpora" / corpus_id / "manifest.json").write_text(json.dumps({
          "corpus_id": corpus_id,
          "dataset": {"name": dataset_name, "hf_id": "x", "split": "dev", "revision": "r"},
          "index_hashes": {
              "dense_contextual": "sha256:c", "dense_isolated": "sha256:i",
              "sparse": "sha256:s",
          },
          "n_queries": 2,
      }))
      return tmp_path


  def make_runner(tmp_path: Path, dataset: str = "musique", *,
                  fail_on: frozenset[str] = frozenset(),
                  ragas=None, abstain_q1: bool = True) -> AblationRunner:
      data_dir = write_eval_corpus(tmp_path, "c1", dataset)
      results = {
          "question 0?": [sc("p0", "gold passage zero"), sc("x1"), sc("x2")],
          "question 1?": [sc("y1"), sc("y2"), sc("y3")],  # gold p1 NOT retrieved
      }
      answers = {
          "question 0?": S1Answer(answer="Answer 0 [1]", abstained=False),
          "question 1?": S1Answer(
              answer="The passages do not contain this.", abstained=abstain_q1
          ),
      }
      return AblationRunner(
          core_factory=lambda cfg: StubCore(results, fail_on=fail_on),
          claude=StubClaude(answers),
          store=RunStore(tmp_path / "runs.db"),
          data_dir=data_dir,
          ragas=ragas,
      )


  # ---------- the two router-on gates (R10: explicitly separate) ----------

  def test_router_on_skipped_requires_plan_c(tmp_path: Path) -> None:
      # TEMPORARY gate: on a multi-hop corpus only the "requires Plan C" skip
      # fires. Plan C deletes exactly this skip - and ONLY this one.
      runner = make_runner(tmp_path, dataset="musique")
      doc = runner.run(run_id="r1", corpus_id="c1", slice_name="smoke",
                       presets=["bm25-only", "router-on"], spend_cap_usd=5.0)
      assert [e["receipt"]["preset"] for e in doc["receipts"]] == ["bm25-only"]
      assert doc["skipped"][0]["preset"] == "router-on"
      assert "requires Plan C" in doc["skipped"][0]["reason"]


  def test_router_on_skipped_on_simple_corpus(tmp_path: Path) -> None:
      # PERMANENT gate (MULTI_HOP_DATASETS): checked BEFORE the Plan C skip, so
      # a single-hop corpus is refused for THIS reason even after Plan C lands.
      # Plan C must keep and test this gate (R10).
      runner = make_runner(tmp_path, dataset="nq")
      doc = runner.run(run_id="r1", corpus_id="c1", slice_name="smoke",
                       presets=["router-on"], spend_cap_usd=5.0)
      assert doc["receipts"] == []
      assert "multi-hop" in doc["skipped"][0]["reason"]
      assert "requires Plan C" not in doc["skipped"][0]["reason"]


  # ---------- cost estimate (hand-computed) ----------

  def test_estimate_run_cost_hand_computed() -> None:
      # bm25-only/query: sonnet 3300 in x $3/M + 300 out x $15/M = 0.0099+0.0045 = 0.0144
      assert estimate_run_cost(["bm25-only"], 10) == pytest.approx(0.144)
      # rerank/query: 0.0144 + embed 40 x 0.18/1e6 (=0.0000072) + 1 search unit 0.0025
      assert estimate_run_cost(["rerank"], 1) == pytest.approx(0.0169072)
      # AUTO presets are skipped in Plan B -> zero estimated cost (temporary, R10)
      assert estimate_run_cost(["router-on"], 100) == 0.0


  def test_estimate_includes_ragas_judge_heuristic_when_enabled() -> None:
      # Per-ok-query judge heuristic (assumes every query is ok - conservative):
      # sonnet 4000 in x $3/M + 500 out x $15/M = 0.012 + 0.0075 = 0.0195.
      # bm25-only/query 0.0144 + 0.0195 = 0.0339 -> x10 = 0.339
      assert estimate_run_cost(["bm25-only"], 10, ragas=True) == pytest.approx(0.339)
      # AUTO presets still contribute zero even with ragas
      assert estimate_run_cost(["router-on"], 100, ragas=True) == 0.0


  # ---------- end-to-end receipt ----------

  def test_receipt_metrics_and_fields(tmp_path: Path) -> None:
      runner = make_runner(tmp_path)
      doc = runner.run(run_id="r1", corpus_id="c1", slice_name="smoke",
                       presets=["bm25-only"], spend_cap_usd=5.0)
      receipt = doc["receipts"][0]["receipt"]
      m = receipt["metrics"]
      # q0: gold p0 at rank 1 -> recall 1.0, mrr 1.0; q1: gold absent -> 0, 0
      assert m["recall_at_5"] == pytest.approx(0.5)
      assert m["mrr_at_3"] == pytest.approx(0.5)
      # q0 answer "Answer 0 [1]" normalizes to "answer 0 1" vs gold "answer 0":
      # EM 0, F1 = 2*(2/3 * 2/2)/(2/3 + 1) = 0.8; q1 abstains -> EM 0, F1 0
      assert m["em"] == pytest.approx(0.0)
      assert m["f1"] == pytest.approx(0.4)
      # usd/query: sonnet 1000 in + 100 out = (3000 + 1500)/1e6 = 0.0045 (no dense/rerank)
      assert m["usd_per_query"] == pytest.approx(0.0045)
      assert m["ragas_faithfulness"] is None  # no judge wired -> disclosed null
      assert receipt["n_total"] == 2
      assert receipt["n_abstained"] == 1
      assert receipt["n_failed"] == 0
      assert receipt["pricing_table_version"] == "2026-06-10"
      assert receipt["prompts_version"] == "n/a"  # R11: Plan C populates this
      assert receipt["index_hashes"] == {"sparse": "sha256:s"}  # bm25-only: sparse only
      assert receipt["models"]["rerank"] == "rerank-v4.0-pro"
      assert receipt["config"]["query"]["route_mode"] == "force_s1"
      # R11: every envelope carries the fixed nondeterminism disclosure
      assert "nondeterministic" in doc["receipts"][0]["nondeterminism_note"]
      # run doc landed in data/receipts-local/
      assert (tmp_path / "receipts-local" / "r1.json").exists()


  def test_index_hashes_select_isolated_variant_for_dense_rrf(tmp_path: Path) -> None:
      runner = make_runner(tmp_path)
      doc = runner.run(run_id="r1", corpus_id="c1", slice_name="smoke",
                       presets=["dense-rrf"], spend_cap_usd=5.0)
      hashes = doc["receipts"][0]["receipt"]["index_hashes"]
      assert hashes == {"sparse": "sha256:s", "dense_isolated": "sha256:i"}


  # ---------- failure disclosure ----------

  def test_failures_disclosed_and_excluded(tmp_path: Path) -> None:
      runner = make_runner(tmp_path, fail_on=frozenset({"question 1?"}))
      doc = runner.run(run_id="r1", corpus_id="c1", slice_name="smoke",
                       presets=["bm25-only"], spend_cap_usd=5.0)
      receipt = doc["receipts"][0]["receipt"]
      assert receipt["n_failed"] == 1
      assert receipt["n_total"] == 2
      # metrics computed over the surviving query only
      assert receipt["metrics"]["recall_at_5"] == pytest.approx(1.0)
      failed = [p for p in receipt["per_query"] if p["flags"]["status"] == "failed"]
      assert len(failed) == 1 and "RuntimeError" in failed[0]["flags"]["error"]


  # ---------- spend cap + resume ----------

  def test_spend_cap_aborts_midrun_then_resumes(tmp_path: Path) -> None:
      runner = make_runner(tmp_path)
      # per-query estimate for bm25-only is 0.0144; actual per query is 0.0045.
      # cap 0.016: q0 admitted (0 + 0.0144 <= 0.016), then before q1:
      # 0.0045 + 0.0144 = 0.0189 > 0.016 -> abort mid-run.
      with pytest.raises(SpendCapExceeded) as exc:
          runner.run(run_id="r1", corpus_id="c1", slice_name="smoke",
                     presets=["bm25-only"], spend_cap_usd=0.016)
      assert "r1" in str(exc.value)
      # resumable: same run_id, higher cap; q0 must not re-run
      claude_before = runner._claude.parse_calls
      doc = runner.run(run_id="r1", corpus_id="c1", slice_name="smoke",
                       presets=["bm25-only"], spend_cap_usd=5.0)
      assert runner._claude.parse_calls == claude_before + 1  # only q1 ran
      assert doc["receipts"][0]["receipt"]["n_total"] == 2


  # ---------- RAGAS exclusion of abstentions ----------

  def test_ragas_runs_on_ok_only_and_is_disclosed(tmp_path: Path) -> None:
      fake = FakeRagas(scores=[RagasScores(faithfulness=0.9, answer_relevancy=0.8)])
      runner = make_runner(tmp_path, ragas=fake)
      doc = runner.run(run_id="r1", corpus_id="c1", slice_name="smoke",
                       presets=["bm25-only"], spend_cap_usd=5.0)
      receipt = doc["receipts"][0]["receipt"]
      assert len(fake.calls) == 1  # q1 abstained -> excluded from RAGAS
      assert receipt["metrics"]["ragas_faithfulness"] == pytest.approx(0.9)
      assert receipt["metrics"]["ragas_answer_relevancy"] == pytest.approx(0.8)
      ok = [p for p in receipt["per_query"] if p["flags"]["status"] == "ok"][0]
      assert ok["flags"]["ragas_judge_usd_untracked"] is True
  ```

- [ ] Run — expected `ModuleNotFoundError: No module named 'ragreceipts.eval.runner'`:

  ```bash
  uv run pytest tests/test_runner.py -q
  ```

- [ ] Implement. Create `api/ragreceipts/eval/runner.py`:

  ```python
  """Ablation runner: preset ladder x corpus slice -> one Receipt per runnable cell.

  Degrade visibly, never silently:
  - presets with route_mode != FORCE_S1 face TWO INDEPENDENT gates (R10):
    GATE 1 (PERMANENT): router-on runs on multi-hop corpora only
    (MULTI_HOP_DATASETS); Plan C keeps and tests this gate.
    GATE 2 (TEMPORARY): System-2 does not exist until Plan C; Plan C deletes
    only this skip. Both produce a disclosed SkippedCell, never a fake run.
  - per-query failures -> status 'failed', excluded from metrics, counted in
    n_failed; abstentions -> 'abstained', excluded from RAGAS, counted in
    n_abstained; both visible in per_query flags.
  - hard spend cap: checked BEFORE every query against actual spent USD plus
    the per-query estimate; on breach the run aborts with saved state and the
    exact resume instructions.
  Cost notes: query-embedding cost uses the EST_QUERY_EMBED_TOKENS heuristic
  (EmbedTransport does not report usage); RAGAS judge cost is not metered in
  Plan B - the pre-run ESTIMATE includes a per-ok-query judge heuristic when
  requested (ragas=True), but actual judge spend is NOT counted against the
  hard cap and is disclosed via the ragas_judge_usd_untracked flag.
  """
  from __future__ import annotations

  import dataclasses
  import math
  import time
  import uuid
  from dataclasses import dataclass
  from datetime import datetime, timezone
  from pathlib import Path
  from typing import Callable

  from pydantic import BaseModel

  from ragreceipts.config import PRESETS, PipelineConfig
  from ragreceipts.constants import (
      EMBED_MODEL,
      JUDGE_MODEL,
      RERANK_MODEL,
      ROUTER_MODEL,
      SYNTH_MODEL,
  )
  from ragreceipts.eval.metrics import exact_match, f1, mrr_at_k, recall_at_k
  from ragreceipts.eval.pricing import PRICING_VERSION, usd_for_rerank, usd_for_tokens
  from ragreceipts.eval.queries import (
      QueryRecord,
      load_manifest,
      load_queries,
      slice_queries,
      slice_query_ids,
  )
  from ragreceipts.eval.ragas_adapter import RagasJudge
  from ragreceipts.eval.receipts import (
      ANCHOR_SPECS,
      Receipt,
      build_anchor,
      make_run_doc,
      write_run_doc,
  )
  from ragreceipts.eval.run_state import RunStore
  from ragreceipts.retrieval.core import RetrievalCore
  from ragreceipts.types import Chunk, RouteMode, ScoredChunk
  from ragreceipts.vendors.base import ClaudeTransport

  # PERMANENT gate data (R10): router-on cells run on multi-hop corpora only.
  # Plan C must keep and test this gate.
  MULTI_HOP_DATASETS = {"musique", "2wikimultihopqa"}

  # Pre-run estimate inputs: ~5 chunks x 512 tokens + ~740 tokens prompt/question.
  EST_SYNTH_INPUT_TOKENS = 3_300
  EST_SYNTH_OUTPUT_TOKENS = 300
  EST_QUERY_EMBED_TOKENS = 40
  # Per-ok-query RAGAS judge heuristic: faithfulness reads ~5 chunks of context,
  # answer-relevancy generates ~3 reverse questions -> ~4k in / ~500 out sonnet.
  EST_RAGAS_INPUT_TOKENS = 4_000
  EST_RAGAS_OUTPUT_TOKENS = 500

  S1_SYSTEM = (
      "You answer questions strictly from the numbered passages provided. "
      "Cite supporting passages inline as [n]. If the passages do not contain "
      "the information needed, set abstained=true and say so briefly in answer. "
      "Never use outside knowledge and never invent facts."
  )


  class S1Answer(BaseModel):
      answer: str
      abstained: bool


  @dataclass(frozen=True)
  class SkippedCell:
      preset: str
      reason: str


  class SpendCapExceeded(RuntimeError):
      pass


  def new_run_id(corpus_id: str, slice_name: str) -> str:
      stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
      return f"{corpus_id}-{slice_name}-{stamp}-{uuid.uuid4().hex[:6]}"


  def synthesize(
      claude: ClaudeTransport, query: str, chunks: list[ScoredChunk]
  ) -> tuple[S1Answer, int, int]:
      """Plan B's temporary S1 generation path (synthesize-with-citations).

      Plan C replaces THIS call site with the LangGraph s1_answer node; the
      prompt and the structured abstention field stay.
      """
      numbered = "\n\n".join(
          f"[{i}] {sc.chunk.text}" for i, sc in enumerate(chunks, start=1)
      )
      result = claude.parse(
          model=SYNTH_MODEL,
          system=S1_SYSTEM,
          user=f"Passages:\n{numbered}\n\nQuestion: {query}",
          max_tokens=4096,
          output_format=S1Answer,
          temperature=0.0,
      )
      parsed = result.parsed
      if not isinstance(parsed, S1Answer):
          raise TypeError(
              f"expected S1Answer from ClaudeTransport.parse, got {type(parsed)!r}"
          )
      return parsed, result.input_tokens, result.output_tokens


  def estimate_run_cost(
      preset_names: list[str], n_queries: int, *, ragas: bool = False
  ) -> float:
      """Pre-run cost estimate (spec: estimate + confirmation gate + hard cap).

      ragas=True adds a per-ok-query judge heuristic (assumes every query is
      'ok' - conservative). The HARD CAP still meters only tracked spend:
      actual RAGAS judge usage is untracked in Plan B (disclosed in the runbook
      and via the ragas_judge_usd_untracked flag).
      """
      total = 0.0
      for name in preset_names:
          cfg = PRESETS[name]
          if cfg.query.route_mode is not RouteMode.FORCE_S1:
              # TEMPORARY (R10): AUTO presets are skipped in Plan B -> no cost.
              # After Plan C this becomes a System-2 estimate (hops x haiku
              # route/grade + sonnet synthesis) instead of a skip.
              continue
          per_q = usd_for_tokens(
              SYNTH_MODEL, EST_SYNTH_INPUT_TOKENS, EST_SYNTH_OUTPUT_TOKENS
          )
          if cfg.query.dense:
              per_q += usd_for_tokens(EMBED_MODEL, EST_QUERY_EMBED_TOKENS, 0)
          if cfg.query.rerank:
              per_q += usd_for_rerank(1)
          if ragas:
              per_q += usd_for_tokens(
                  JUDGE_MODEL, EST_RAGAS_INPUT_TOKENS, EST_RAGAS_OUTPUT_TOKENS
              )
          total += per_q * n_queries
      return total


  def _mean(values: list[float]) -> float | None:
      return sum(values) / len(values) if values else None


  def _percentile(values: list[float], pct: float) -> float:
      vals = sorted(values)
      idx = max(0, math.ceil(pct * len(vals)) - 1)
      return vals[idx]


  def _chunk_from_stored(d: dict, corpus_id: str) -> Chunk:
      doc_id, _, position = d["chunk_id"].rpartition(":")  # chunk_id = f"{doc_id}:{position}"
      return Chunk(
          chunk_id=d["chunk_id"],
          corpus_id=corpus_id,
          doc_id=doc_id,
          passage_id=d["passage_id"],
          text=d["text"],
          position=int(position),
          start_token=int(d["start_token"]),  # R3: span hits need token offsets
          end_token=int(d["end_token"]),
      )


  def _used_index_hashes(cfg: PipelineConfig, manifest: dict) -> dict:
      """Record only the index-variant hashes this cell actually used."""
      hashes = manifest["index_hashes"]
      used: dict = {}
      if cfg.query.bm25:
          used["sparse"] = hashes["sparse"]
      if cfg.query.dense:
          key = "dense_contextual" if cfg.ingest.contextual else "dense_isolated"
          used[key] = hashes[key]
      return used


  def _config_as_dict(cfg: PipelineConfig) -> dict:
      query = dataclasses.asdict(cfg.query)
      query["route_mode"] = cfg.query.route_mode.value
      return {
          "name": cfg.name,
          "ingest": dataclasses.asdict(cfg.ingest),
          "query": query,
      }


  class AblationRunner:
      def __init__(
          self,
          *,
          core_factory: Callable[[PipelineConfig], RetrievalCore],
          claude: ClaudeTransport,
          store: RunStore,
          data_dir: Path,
          ragas: RagasJudge | None = None,
          clock: Callable[[], float] = time.perf_counter,
      ) -> None:
          self._core_factory = core_factory
          self._claude = claude
          self._store = store
          self._data_dir = data_dir
          self._ragas = ragas
          self._clock = clock

      def run(
          self,
          *,
          run_id: str,
          corpus_id: str,
          slice_name: str,
          presets: list[str],
          spend_cap_usd: float,
      ) -> dict:
          manifest = load_manifest(self._data_dir, corpus_id)
          queries = slice_queries(
              load_queries(self._data_dir, corpus_id),
              slice_query_ids(self._data_dir, corpus_id, slice_name),
          )
          self._store.start_run(
              run_id=run_id, corpus_id=corpus_id, slice_name=slice_name,
              presets=presets, spend_cap_usd=spend_cap_usd,
          )
          receipts: list[Receipt] = []
          skipped: list[SkippedCell] = []
          results_by_preset: dict[str, dict] = {}
          for name in presets:
              cfg = PRESETS[name]
              if cfg.query.route_mode is not RouteMode.FORCE_S1:
                  # GATE 1 - PERMANENT (R10): router-on runs on multi-hop corpora
                  # only. Checked FIRST. Plan C keeps and tests this gate.
                  dataset = manifest.get("dataset", {}).get("name", "")
                  if dataset not in MULTI_HOP_DATASETS:
                      skipped.append(SkippedCell(
                          preset=name,
                          reason=(
                              f"skipped: '{name}' runs on the multi-hop corpus only; "
                              f"corpus dataset is '{dataset}'"
                          ),
                      ))
                      continue
                  # GATE 2 - TEMPORARY (R10): System-2 does not exist until
                  # Plan C. Plan C deletes ONLY this block; the multi-hop gate
                  # above stays.
                  skipped.append(SkippedCell(
                      preset=name,
                      reason=(
                          "skipped: requires Plan C (LangGraph System-2 is not "
                          "built yet; only route_mode=force_s1 cells are runnable)"
                      ),
                  ))
                  continue
              self._run_preset(
                  run_id=run_id, cfg=cfg, queries=queries, spend_cap_usd=spend_cap_usd
              )
              receipt = self._build_receipt(
                  run_id=run_id, corpus_id=corpus_id, cfg=cfg, manifest=manifest,
                  queries=queries, results_by_preset=results_by_preset,
              )
              results_by_preset[name] = receipt.metrics
              receipts.append(receipt)
          doc = make_run_doc(
              run_id=run_id, corpus_id=corpus_id, slice_name=slice_name,
              receipts=receipts, skipped=skipped,
          )
          write_run_doc(doc, self._data_dir)
          return doc

      def _run_preset(
          self,
          *,
          run_id: str,
          cfg: PipelineConfig,
          queries: list[QueryRecord],
          spend_cap_usd: float,
      ) -> None:
          done = self._store.completed_query_ids(run_id, cfg.name)
          core = self._core_factory(cfg)
          per_query_estimate = estimate_run_cost([cfg.name], 1)
          for q in queries:
              if q.query_id in done:
                  continue  # resumable: completed queries are never re-billed
              spent = self._store.spent_usd(run_id)
              if spent + per_query_estimate > spend_cap_usd:
                  raise SpendCapExceeded(
                      f"hard spend cap hit mid-run: spent ${spent:.4f} of cap "
                      f"${spend_cap_usd:.2f} before query {q.query_id!r} in preset "
                      f"{cfg.name!r}. State is saved; resume with --run-id {run_id} "
                      f"and a higher --spend-cap-usd."
                  )
              t0 = self._clock()
              try:
                  scored_chunks = core.retrieve(q.question)
                  parsed, tin, tout = synthesize(self._claude, q.question, scored_chunks)
                  latency_ms = (self._clock() - t0) * 1000.0
                  usd = usd_for_tokens(SYNTH_MODEL, tin, tout)
                  if cfg.query.rerank:
                      usd += usd_for_rerank(1)
                  if cfg.query.dense:
                      usd += usd_for_tokens(EMBED_MODEL, EST_QUERY_EMBED_TOKENS, 0)
                  self._store.record_result(
                      run_id=run_id, preset=cfg.name, query_id=q.query_id,
                      status="abstained" if parsed.abstained else "ok",
                      retrieved=[
                          {
                              "chunk_id": sc.chunk.chunk_id,
                              "passage_id": sc.chunk.passage_id,
                              "start_token": sc.chunk.start_token,
                              "end_token": sc.chunk.end_token,
                              "text": sc.chunk.text,
                          }
                          for sc in scored_chunks
                      ],
                      answer=parsed.answer, latency_ms=latency_ms, usd=usd,
                      input_tokens=tin, output_tokens=tout, error=None,
                  )
              except Exception as exc:  # disclosed, never batch-fatal
                  latency_ms = (self._clock() - t0) * 1000.0
                  self._store.record_result(
                      run_id=run_id, preset=cfg.name, query_id=q.query_id,
                      status="failed", retrieved=[], answer=None,
                      latency_ms=latency_ms, usd=0.0, input_tokens=0,
                      output_tokens=0, error=repr(exc),
                  )

      def _build_receipt(
          self,
          *,
          run_id: str,
          corpus_id: str,
          cfg: PipelineConfig,
          manifest: dict,
          queries: list[QueryRecord],
          results_by_preset: dict[str, dict],
      ) -> Receipt:
          by_id = {q.query_id: q for q in queries}
          rows = [
              r for r in self._store.results_for(run_id, cfg.name)
              if r["query_id"] in by_id
          ]
          scored = [r for r in rows if r["status"] in ("ok", "abstained")]
          failed = [r for r in rows if r["status"] == "failed"]

          recalls: list[float] = []
          mrrs: list[float] = []
          ems: list[float] = []
          f1s: list[float] = []
          for r in scored:
              q = by_id[r["query_id"]]
              chunks = [_chunk_from_stored(d, corpus_id) for d in r["retrieved"]]
              recalls.append(recall_at_k(chunks, q.golds, k=5))
              mrrs.append(mrr_at_k(chunks, q.golds, k=3))
              ems.append(exact_match(r["answer"] or "", q.gold_answers))
              f1s.append(f1(r["answer"] or "", q.gold_answers))

          ragas_faith = ragas_rel = None
          ragas_flags: dict = {}
          if self._ragas is not None:
              ok_rows = [r for r in scored if r["status"] == "ok"]  # no abstentions
              faiths: list[float] = []
              rels: list[float] = []
              for r in ok_rows:
                  q = by_id[r["query_id"]]
                  s = self._ragas.score(
                      question=q.question,
                      answer=r["answer"] or "",
                      contexts=[d["text"] for d in r["retrieved"]],
                  )
                  faiths.append(s.faithfulness)
                  rels.append(s.answer_relevancy)
              ragas_faith = _mean(faiths)
              ragas_rel = _mean(rels)
              ragas_flags = {"ragas_judge_usd_untracked": True}

          latencies = [r["latency_ms"] for r in rows]
          metrics = {
              "recall_at_5": _mean(recalls),
              "mrr_at_3": _mean(mrrs),
              "em": _mean(ems),
              "f1": _mean(f1s),
              "ragas_faithfulness": ragas_faith,
              "ragas_answer_relevancy": ragas_rel,
              "latency_p50_ms": _percentile(latencies, 0.50) if latencies else None,
              "latency_p95_ms": _percentile(latencies, 0.95) if latencies else None,
              "usd_per_query": (
                  sum(r["usd"] for r in scored) / len(scored) if scored else None
              ),
          }

          anchors = []
          for spec in ANCHOR_SPECS.get(cfg.name, []):
              baseline = results_by_preset.get(spec.baseline_preset or "")
              if (
                  baseline is None
                  or metrics.get(spec.metric) is None
                  or baseline.get(spec.metric) is None
              ):
                  continue  # baseline cell absent from this run; CLI discloses
              anchors.append(
                  build_anchor(
                      spec,
                      metrics[spec.metric] - baseline[spec.metric],
                      corpus_id=corpus_id,  # R11: nq-dev-300 appends the scale caveat
                  )
              )

          per_query = []
          for r in rows:
              flags: dict = {"status": r["status"]}
              if r["status"] in ("ok", "abstained"):
                  q = by_id[r["query_id"]]
                  flags["em"] = exact_match(r["answer"] or "", q.gold_answers)
                  flags["f1"] = f1(r["answer"] or "", q.gold_answers)
              if r["error"]:
                  flags["error"] = r["error"]
              flags.update(ragas_flags)
              per_query.append({
                  "query_id": r["query_id"],
                  "retrieved_chunk_ids": [d["chunk_id"] for d in r["retrieved"]],
                  "answer": r["answer"],
                  "latency_ms": r["latency_ms"],
                  "usd": r["usd"],
                  "flags": flags,
              })

          return Receipt(
              run_id=run_id,
              corpus_id=corpus_id,
              preset=cfg.name,
              config=_config_as_dict(cfg),
              index_hashes=_used_index_hashes(cfg, manifest),
              models={
                  "router": ROUTER_MODEL,
                  "synth": SYNTH_MODEL,
                  "judge": JUDGE_MODEL,
                  "rerank": RERANK_MODEL,
                  "embed": EMBED_MODEL,
              },
              pricing_table_version=PRICING_VERSION,
              prompts_version="n/a",  # R11: Plan C populates agents.prompts.PROMPTS_VERSION
              n_total=len(rows),
              n_failed=len(failed),
              n_abstained=sum(1 for r in rows if r["status"] == "abstained"),
              metrics=metrics,
              per_query=per_query,
              anchors=anchors,
          )
  ```

- [ ] Run again — expected: **all tests in `tests/test_runner.py` PASS**:

  ```bash
  uv run pytest tests/test_runner.py -q
  ```

- [ ] Run the whole suite to catch regressions, then commit:

  ```bash
  uv run pytest -q
  git add ragreceipts/eval/runner.py tests/test_runner.py
  git commit -m "feat(eval): resumable ablation runner with spend cap, skip disclosure, receipt building" \
    -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
  ```

---

### Task 8: Harness self-test — receipts that can't fail aren't receipts (CI-enforced)

**Files:**
- Create: `api/tests/harness_fixtures.py`
- Test: `api/tests/test_harness_selftest.py`

This task exercises Plan A's **real** `RetrievalCore` + `RerankStage` + `FakeRerank` end-to-end through the runner, on a tiny in-repo labeled corpus engineered so the rerank flag provably moves Recall@5 — and so a deliberately misaligned gold mapping provably scores zero. Both run in CI on every push.

- [ ] Create the fixture corpus builder `api/tests/harness_fixtures.py`:

  ```python
  """Tiny in-repo labeled corpus for the harness self-test (spec §Testing).

  Engineered so flipping the rerank flag PROVABLY changes Recall@5:
  - q0's gold chunk sits at RRF rank 7 (outside top_k_final=5, inside
    top_k_fuse=50); the scripted reranker scores it 0.99 and lifts it to rank 1.
  - q1-q3 have their gold at rank 1 regardless.
  Therefore recall@5(rerank off) = 3/4 = 0.75 and recall@5(rerank on) = 4/4 = 1.0.
  If the ablation runner ever stops detecting that delta, CI fails:
  receipts that can't fail aren't receipts.

  Corpus files use Spike 0's raw layout (R1): raw/queries.jsonl with typed
  passage golds plus slice-full.json / slice-smoke.json query-id lists.
  """
  from __future__ import annotations

  import json
  from pathlib import Path

  from ragreceipts.types import Chunk, ScoredChunk

  N_QUERIES = 4


  def _chunk(passage_id: str, text: str) -> Chunk:
      return Chunk(chunk_id=f"{passage_id}:0", corpus_id="harness", doc_id=passage_id,
                   passage_id=passage_id, text=text, position=0,
                   start_token=0, end_token=len(text.split()))


  class ListRetriever:
      """Retriever-protocol fake returning a fixed ranking per question text."""

      def __init__(self, rankings: dict[str, list[Chunk]], source: str) -> None:
          self._rankings = rankings
          self._source = source

      def search(self, query: str, k: int) -> list[ScoredChunk]:
          chunks = self._rankings[query][:k]
          n = len(chunks)
          return [
              ScoredChunk(chunk=c, score=float(n - i), source=self._source)
              for i, c in enumerate(chunks)
          ]


  def build_harness_fixture(*, misaligned: bool = False) -> dict:
      """Rankings + scripted rerank scores + raw-layout query records.

      misaligned=True deliberately breaks every gold passage_id ("WRONG-..."):
      the self-test asserts this scores recall 0.0, proving the alignment rule
      is load-bearing (an is_hit that always matched would fail that test).
      """
      rankings: dict[str, list[Chunk]] = {}
      rerank_scores: dict[str, float] = {}
      queries: list[dict] = []
      for i in range(N_QUERIES):
          gold = _chunk(f"g{i}", f"gold passage text {i}")
          fillers = [_chunk(f"f{i}-{j}", f"filler text {i}-{j}") for j in range(6)]
          ranking = fillers + [gold] if i == 0 else [gold] + fillers
          question = f"harness question {i}?"
          rankings[question] = ranking
          rerank_scores[gold.text] = 0.99
          for j, filler in enumerate(fillers):
              rerank_scores[filler.text] = 0.5 - 0.01 * j
          gold_pid = f"WRONG-g{i}" if misaligned else f"g{i}"
          queries.append({
              "query_id": f"q{i}",
              "question": question,
              "answer": f"gold answer {i}",
              "answer_aliases": [],
              "gold": {"type": "passage", "passage_ids": [gold_pid]},
          })
      return {"rankings": rankings, "rerank_scores": rerank_scores, "queries": queries}


  def write_harness_corpus(data_dir: Path, corpus_id: str, queries: list[dict]) -> None:
      raw = data_dir / "corpora" / corpus_id / "raw"
      raw.mkdir(parents=True)
      (raw / "queries.jsonl").write_text(
          "\n".join(json.dumps(q) for q in queries) + "\n"
      )
      query_ids = [q["query_id"] for q in queries]
      (raw / "slice-full.json").write_text(json.dumps(query_ids))
      (raw / "slice-smoke.json").write_text(json.dumps(query_ids[:15]))
      (data_dir / "corpora" / corpus_id / "manifest.json").write_text(json.dumps({
          "corpus_id": corpus_id,
          "dataset": {"name": "musique", "hf_id": "in-repo-fixture",
                      "split": "fixture", "revision": "0"},
          "index_hashes": {"dense_contextual": "sha256:c",
                          "dense_isolated": "sha256:i", "sparse": "sha256:s"},
          "n_queries": len(queries),
      }))
  ```

- [ ] Write the self-test. Create `api/tests/test_harness_selftest.py`:

  ```python
  """Harness self-test (spec §Testing, 'the on-brand one'). CI-enforced.

  Runs Plan A's REAL RetrievalCore + RerankStage with FakeRerank through the
  Plan B runner on the in-repo fixture corpus. Two properties must hold forever:
    1. flipping the rerank flag changes Recall@5 (0.75 -> 1.0 here);
    2. a deliberately misaligned gold mapping scores 0.0 while the aligned
       mapping scores 1.0 - the alignment rule is load-bearing.
  """
  from pathlib import Path

  import pytest

  from ragreceipts.eval.run_state import RunStore
  from ragreceipts.eval.runner import AblationRunner, S1Answer
  from ragreceipts.retrieval.core import RetrievalCore
  from ragreceipts.retrieval.rerank import RerankStage
  from ragreceipts.vendors.base import ParsedResult
  from tests.fakes import FakeRerank  # tests/ is a package (R8)
  from tests.harness_fixtures import (
      ListRetriever,
      build_harness_fixture,
      write_harness_corpus,
  )


  class EchoClaude:
      """ClaudeTransport stub answering each fixture question with its gold answer."""

      def complete(self, *, model, system, user, max_tokens, temperature=0.0):
          raise AssertionError("self-test synthesis uses parse(), not complete()")

      def parse(self, *, model, system, user, max_tokens, output_format,
                temperature=0.0):
          question = user.rsplit("Question: ", 1)[1]  # "harness question {i}?"
          i = question.split()[-1].rstrip("?")
          return ParsedResult(
              parsed=S1Answer(answer=f"gold answer {i}", abstained=False),
              input_tokens=500, output_tokens=50,
          )


  def make_runner(tmp_path: Path, *, misaligned: bool = False) -> AblationRunner:
      fixture = build_harness_fixture(misaligned=misaligned)
      write_harness_corpus(tmp_path, "harness", fixture["queries"])

      def core_factory(cfg) -> RetrievalCore:
          sparse = ListRetriever(fixture["rankings"], source="bm25")
          dense = ListRetriever(fixture["rankings"], source="dense")
          # R5 final constructor: FakeRerank(script=None, scores=None, fail=False);
          # the text-keyed scores mode exists exactly for this fixture.
          stage = RerankStage(FakeRerank(scores=fixture["rerank_scores"]))
          return RetrievalCore(
              config=cfg,
              dense=dense if cfg.query.dense else None,
              sparse=sparse if cfg.query.bm25 else None,
              rerank_stage=stage if cfg.query.rerank else None,
          )

      return AblationRunner(
          core_factory=core_factory,
          claude=EchoClaude(),
          store=RunStore(tmp_path / "runs.db"),
          data_dir=tmp_path,
      )


  def metrics_for(doc: dict, preset: str) -> dict:
      for env in doc["receipts"]:
          if env["receipt"]["preset"] == preset:
              return env["receipt"]["metrics"]
      raise AssertionError(f"no receipt for preset {preset!r} in run doc")


  def test_rerank_flip_provably_changes_recall_at_5(tmp_path: Path) -> None:
      runner = make_runner(tmp_path)
      doc = runner.run(run_id="selftest", corpus_id="harness", slice_name="smoke",
                       presets=["contextual", "rerank"], spend_cap_usd=5.0)
      off = metrics_for(doc, "contextual")  # bm25+dense+RRF, rerank OFF
      on = metrics_for(doc, "rerank")       # same + rerank ON
      assert off["recall_at_5"] == pytest.approx(0.75)
      assert on["recall_at_5"] == pytest.approx(1.0)
      assert on["recall_at_5"] > off["recall_at_5"]  # the receipt CAN fail
      # MRR@3 moves too: q0 has no top-3 hit without rerank
      assert off["mrr_at_3"] == pytest.approx(0.75)
      assert on["mrr_at_3"] == pytest.approx(1.0)
      # and the rerank cell carries its anchors with direction computed
      rerank_receipt = doc["receipts"][1]["receipt"]
      assert rerank_receipt["preset"] == "rerank"
      assert len(rerank_receipt["anchors"]) == 2
      assert all(a["direction_match"] is True for a in rerank_receipt["anchors"])


  def test_full_ladder_runs_offline_with_disclosed_skip(tmp_path: Path) -> None:
      runner = make_runner(tmp_path)
      doc = runner.run(
          run_id="ladder", corpus_id="harness", slice_name="smoke",
          presets=["bm25-only", "dense-rrf", "contextual", "rerank", "router-on"],
          spend_cap_usd=5.0,
      )
      assert [e["receipt"]["preset"] for e in doc["receipts"]] == [
          "bm25-only", "dense-rrf", "contextual", "rerank",
      ]
      assert doc["skipped"] == [{
          "preset": "router-on",
          "reason": ("skipped: requires Plan C (LangGraph System-2 is not built "
                     "yet; only route_mode=force_s1 cells are runnable)"),
      }]
      assert (tmp_path / "receipts-local" / "ladder.json").exists()


  def test_misaligned_golds_provably_score_zero(tmp_path: Path) -> None:
      aligned = make_runner(tmp_path / "aligned")
      doc_ok = aligned.run(run_id="a", corpus_id="harness", slice_name="smoke",
                           presets=["rerank"], spend_cap_usd=5.0)
      misaligned = make_runner(tmp_path / "broken", misaligned=True)
      doc_bad = misaligned.run(run_id="b", corpus_id="harness", slice_name="smoke",
                               presets=["rerank"], spend_cap_usd=5.0)
      assert metrics_for(doc_ok, "rerank")["recall_at_5"] == pytest.approx(1.0)
      assert metrics_for(doc_bad, "rerank")["recall_at_5"] == pytest.approx(0.0)
      assert metrics_for(doc_bad, "rerank")["mrr_at_3"] == pytest.approx(0.0)
      # answer-level metrics are alignment-independent and stay perfect - the
      # zero comes from the alignment rule alone, not from a broken pipeline
      assert metrics_for(doc_bad, "rerank")["em"] == pytest.approx(1.0)
  ```

- [ ] Run — expected: **PASS** (this is the first test that integrates Plan A's `RetrievalCore` + `RerankStage` with Plan B's runner; if it fails on a constructor mismatch, that is a seam-table reconciliation per Task 1, fixed in `core_factory` here):

  ```bash
  uv run pytest tests/test_harness_selftest.py -q
  ```

- [ ] Sanity-check the self-test can actually fail (mutation check, manual one-off): temporarily change `rerank_scores[gold.text] = 0.99` to `0.01` in `harness_fixtures.py`, re-run, confirm `test_rerank_flip_provably_changes_recall_at_5` FAILS, then revert. Do not commit the mutation.

- [ ] Commit:

  ```bash
  git add tests/harness_fixtures.py tests/test_harness_selftest.py
  git commit -m "test(eval): harness self-test - rerank flip moves Recall@5, misaligned golds score zero" \
    -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
  ```

---

### Task 9: CLI — `ragreceipts eval` and `ragreceipts receipts promote`

Plan A CREATED `api/ragreceipts/cli.py` (with the `ingest` subcommand and the module-level factory seams `build_embed_transport`/`build_qdrant`) and `api/tests/test_cli.py`. Per R6, Plan B **MODIFIES** both: it adds the `eval` and `receipts` subparsers alongside `ingest`, keeps Plan A's seams, and APPENDS tests — never deletes Plan A's.

**Files:**
- Modify: `api/ragreceipts/cli.py` (add `eval` + `receipts` subparsers; keep `ingest` + factory seams)
- Modify: `api/tests/test_cli.py` (append Plan B's tests after Plan A's)

- [ ] Update the module docstring and import block at the **top** of `api/tests/test_cli.py` to exactly this (Plan A's imports kept, Plan B's added; all module-level imports stay at the top to satisfy ruff E402):

  ```python
  """CLI wiring tests.

  Plan A: ingest wiring with monkeypatched factories (offline, keyless).
  Plan B (appended): eval arg validation, named missing-key errors, the cost
  confirm gate, the offline composition-root construction test, and promote.
  The eval happy path against real vendors is the keyed manual step (Task 10);
  the offline end-to-end eval path is covered by the harness self-test.
  """

  import json
  from pathlib import Path

  import pytest
  from qdrant_client import QdrantClient

  import ragreceipts.cli as cli
  from ragreceipts.cli import main
  from ragreceipts.config import PRESETS
  from ragreceipts.eval.receipts import Receipt, make_run_doc, write_run_doc
  from ragreceipts.ingest.chunk_store import write_chunks
  from ragreceipts.retrieval.core import RetrievalCore
  from ragreceipts.retrieval.sparse import build_sparse_index
  from ragreceipts.types import Chunk
  from tests.corpus_fixtures import write_tiny_corpus
  from tests.fakes import FakeEmbed, FakeRerank
  ```

  (Plan A's existing test functions — `test_ingest_command_writes_manifest_and_prints_it`, `test_missing_corpus_exits_nonzero_with_named_message`, `test_missing_voyage_key_is_a_named_error` — stay untouched between this block and the appended tests below.)

- [ ] Append the Plan B tests at the end of `api/tests/test_cli.py`:

  ```python
  # =====================================================================
  # Plan B (R6): eval + receipts promote - appended after Plan A's tests
  # =====================================================================

  KEYS = ("ANTHROPIC_API_KEY", "VOYAGE_API_KEY", "COHERE_API_KEY")


  def minimal_receipt() -> Receipt:
      return Receipt(
          run_id="r1", corpus_id="c1", preset="bm25-only",
          config={"name": "bm25-only"}, index_hashes={"sparse": "sha256:s"},
          models={"router": "claude-haiku-4-5-20251001", "synth": "claude-sonnet-4-6",
                  "judge": "claude-sonnet-4-6", "rerank": "rerank-v4.0-pro",
                  "embed": "voyage-context-3"},
          pricing_table_version="2026-06-10", prompts_version="n/a",
          n_total=1, n_failed=0, n_abstained=0,
          metrics={"recall_at_5": 1.0, "mrr_at_3": 1.0, "em": 1.0, "f1": 1.0,
                   "ragas_faithfulness": None, "ragas_answer_relevancy": None,
                   "latency_p50_ms": 1.0, "latency_p95_ms": 1.0, "usd_per_query": 0.001},
          per_query=[{"query_id": "q0", "retrieved_chunk_ids": ["d:0"],
                      "answer": "secret model text", "latency_ms": 1.0, "usd": 0.001,
                      "flags": {"status": "ok", "em": 1.0, "f1": 1.0}}],
          anchors=[],
      )


  def write_min_corpus(data_dir: Path, corpus_id: str = "c1") -> None:
      """Spike 0 raw layout (R1) + Plan A manifest."""
      raw = data_dir / "corpora" / corpus_id / "raw"
      raw.mkdir(parents=True)
      (raw / "queries.jsonl").write_text(json.dumps({
          "query_id": "q0", "question": "q?", "answer": "a", "answer_aliases": [],
          "gold": {"type": "passage", "passage_ids": ["p0"]},
      }) + "\n")
      (raw / "slice-full.json").write_text(json.dumps(["q0"]))
      (raw / "slice-smoke.json").write_text(json.dumps(["q0"]))
      (data_dir / "corpora" / corpus_id / "manifest.json").write_text(json.dumps({
          "corpus_id": corpus_id,
          "dataset": {"name": "nq", "hf_id": "x", "split": "dev", "revision": "r"},
          "index_hashes": {"dense_contextual": "c", "dense_isolated": "i", "sparse": "s"},
          "n_queries": 1,
      }))


  def test_unknown_preset_rejected_with_valid_list(capsys) -> None:
      rc = main(["eval", "--corpus", "c1", "--presets", "bm25-only,nope", "--yes"])
      assert rc == 2
      err = capsys.readouterr().err
      assert "nope" in err and "bm25-only" in err


  def test_missing_keys_produce_named_env_var_errors(monkeypatch, capsys) -> None:
      for key in KEYS:
          monkeypatch.delenv(key, raising=False)
      rc = main(["eval", "--corpus", "c1", "--presets", "rerank", "--yes"])
      assert rc == 2
      err = capsys.readouterr().err
      assert "ANTHROPIC_API_KEY" in err
      assert "VOYAGE_API_KEY" in err
      assert "COHERE_API_KEY" in err


  def test_bm25_only_needs_no_voyage_or_cohere_key(monkeypatch, tmp_path) -> None:
      monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
      monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
      monkeypatch.delenv("COHERE_API_KEY", raising=False)
      # R6: --data-dir defaults to RAGRECEIPTS_DATA_DIR (hermetic here)
      monkeypatch.setenv("RAGRECEIPTS_DATA_DIR", str(tmp_path))
      # passes key validation, then fails on the missing corpus - proving the
      # voyage/cohere keys were not demanded for a sparse-only run
      with pytest.raises(FileNotFoundError):
          main(["eval", "--corpus", "missing-corpus", "--presets", "bm25-only", "--yes"])


  def test_data_dir_default_honors_env_var(monkeypatch, capsys, tmp_path) -> None:
      # R6: data dir resolution everywhere is RAGRECEIPTS_DATA_DIR env var,
      # default ../data relative to api/ - promote shares the same default.
      monkeypatch.setenv("RAGRECEIPTS_DATA_DIR", str(tmp_path))
      rc = main(["receipts", "promote", "ghost"])
      assert rc == 2
      assert str(tmp_path) in capsys.readouterr().err


  def test_confirm_gate_aborts_before_any_spend(monkeypatch, capsys, tmp_path) -> None:
      for key in KEYS:
          monkeypatch.setenv(key, "k")
      write_min_corpus(tmp_path)
      monkeypatch.setattr("builtins.input", lambda prompt: "n")
      rc = main(["eval", "--corpus", "c1", "--slice", "smoke",
                 "--presets", "rerank", "--data-dir", str(tmp_path)])
      assert rc == 1
      out = capsys.readouterr().out
      assert "Estimated cost" in out
      assert not (tmp_path / "receipts-local").exists()  # nothing ran, nothing spent


  def test_build_core_real_composes_offline_with_fakes(tmp_path, monkeypatch) -> None:
      """Offline construction test for the R9-pinned composition root
      cli._build_core_real(config, corpus_id, data_dir): Plan A's real
      SparseRetriever.load / DenseRetriever / RerankStage assembled with fakes
      monkeypatched into the module-level factory seams - zero keys, zero
      network (bm25s builds locally; Qdrant runs in :memory: mode)."""
      corpus_dir = tmp_path / "corpora" / "c1"
      corpus_dir.mkdir(parents=True)
      chunks = [Chunk(chunk_id="d1:0", corpus_id="c1", doc_id="d1", passage_id="d1",
                      text="alpha bravo charlie", position=0,
                      start_token=0, end_token=3)]
      write_chunks(corpus_dir / "chunks.jsonl", chunks)
      build_sparse_index(chunks, corpus_dir / "sparse")
      monkeypatch.setattr(cli, "build_embed_transport", lambda: FakeEmbed())
      monkeypatch.setattr(cli, "build_qdrant", lambda data_dir: QdrantClient(":memory:"))
      monkeypatch.setattr(cli, "build_rerank_transport", lambda: FakeRerank())
      for preset in ("bm25-only", "dense-rrf", "contextual", "rerank"):
          core = cli._build_core_real(PRESETS[preset], "c1", tmp_path)
          assert isinstance(core, RetrievalCore)


  def test_promote_strips_text_and_writes_to_receipts_dir(tmp_path, capsys) -> None:
      data_dir = tmp_path / "data"
      receipts_dir = tmp_path / "receipts"
      doc = make_run_doc(run_id="r1", corpus_id="c1", slice_name="smoke",
                         receipts=[minimal_receipt()], skipped=[])
      write_run_doc(doc, data_dir)
      rc = main(["receipts", "promote", "r1",
                 "--data-dir", str(data_dir), "--receipts-dir", str(receipts_dir)])
      assert rc == 0
      committed = json.loads((receipts_dir / "r1.json").read_text())
      pq = committed["receipts"][0]["receipt"]["per_query"][0]
      assert "answer" not in pq          # IDs + metrics only
      assert pq["retrieved_chunk_ids"] == ["d:0"]
      assert pq["flags"]["f1"] == 1.0


  def test_promote_missing_run_is_actionable(tmp_path, capsys) -> None:
      rc = main(["receipts", "promote", "ghost", "--data-dir", str(tmp_path)])
      assert rc == 2
      assert "ghost" in capsys.readouterr().err
  ```

- [ ] Run — expected: Plan A's three tests still PASS; every appended Plan B test FAILS (argparse exits with "invalid choice: 'eval'" / missing `build_rerank_transport` attribute — the subparsers don't exist yet):

  ```bash
  uv run pytest tests/test_cli.py -q
  ```

- [ ] Implement. Replace `api/ragreceipts/cli.py` with this complete modified file — Plan A's `ingest` subcommand and `build_embed_transport`/`build_qdrant` seams are preserved verbatim (the ingest body moves into `_cmd_ingest`, behavior unchanged); Plan B adds the `eval`/`receipts` subparsers, `build_rerank_transport`, the R9-pinned `_build_core_real`, `_make_claude`, `_cmd_eval`, and `_cmd_promote`:

  ```python
  """ragreceipts CLI.

    ragreceipts ingest --corpus <id>                  (Plan A) rebuild all index variants
    ragreceipts eval --corpus <id> --slice smoke|full \
        --presets bm25-only,dense-rrf,contextual,rerank,router-on \
        [--spend-cap-usd 5.0] [--run-id <resume-id>] [--ragas] [--yes]
    ragreceipts receipts promote <run_id>

  Plan A created this module with the ingest subcommand; Plan B (R6) adds the
  eval/receipts subparsers. Factories build_embed_transport/build_qdrant
  (Plan A) and build_rerank_transport (Plan B) are module-level seams: tests
  monkeypatch them; Plan D's server reuses them. Missing keys produce a named
  env-var message, never a stack trace.
  Data dir resolution (R6): RAGRECEIPTS_DATA_DIR env var, default ../data
  relative to api/; `receipts promote` defaults --receipts-dir to ../receipts.

  eval writes {data_dir}/receipts-local/<run_id>.json; promote copies a run to
  receipts/<run_id>.json with passage text and answers stripped (IDs + metrics
  only - benchmark redistribution terms).
  """
  from __future__ import annotations

  import argparse
  import json
  import os
  import sys
  from pathlib import Path

  from qdrant_client import QdrantClient

  from ragreceipts.config import IngestConfig, PRESETS, PipelineConfig
  from ragreceipts.eval.queries import load_queries, slice_queries, slice_query_ids
  from ragreceipts.eval.receipts import read_run_doc, strip_for_commit
  from ragreceipts.eval.run_state import RunStore
  from ragreceipts.eval.runner import (
      AblationRunner,
      SpendCapExceeded,
      estimate_run_cost,
      new_run_id,
  )
  from ragreceipts.ingest.pipeline import run_ingest
  from ragreceipts.retrieval.core import RetrievalCore
  from ragreceipts.types import RouteMode
  from ragreceipts.vendors.cohere_client import CohereClient
  from ragreceipts.vendors.voyage_client import VoyageClient

  DEFAULT_PRESETS = "bm25-only,dense-rrf,contextual,rerank,router-on"


  def _default_data_dir() -> Path:
      """R6: RAGRECEIPTS_DATA_DIR env var, default ../data relative to api/."""
      return Path(os.environ.get("RAGRECEIPTS_DATA_DIR", "../data"))


  # --- factory seams (Plan A's two kept verbatim; Plan B adds the third) ------


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
      return QdrantClient(path=str(data_dir / "qdrant-local"))  # local file mode, no server


  def build_rerank_transport() -> CohereClient:
      api_key = os.environ.get("COHERE_API_KEY")
      if not api_key:
          raise SystemExit(
              "COHERE_API_KEY is not set — rerank cells need Cohere rerank-v4.0-pro "
              "(set it in .env)"
          )
      return CohereClient(api_key=api_key)


  def main(argv: list[str] | None = None) -> int:
      parser = argparse.ArgumentParser(prog="ragreceipts")
      sub = parser.add_subparsers(dest="command", required=True)

      # --- ingest (Plan A, preserved) ---
      ingest_p = sub.add_parser("ingest",
                                help="(re)build all index variants for a corpus")
      ingest_p.add_argument("--corpus", required=True, help="corpus id, e.g. nq-dev-300")
      ingest_p.add_argument("--data-dir", type=Path, default=_default_data_dir(),
                            help="data dir holding corpora/ (default ../data, run from api/)")
      ingest_p.add_argument("--chunk-size", type=int, default=IngestConfig().chunk_size)
      ingest_p.add_argument("--chunk-overlap", type=int,
                            default=IngestConfig().chunk_overlap)

      # --- eval (Plan B) ---
      eval_p = sub.add_parser("eval", help="run the ablation ladder on a corpus slice")
      eval_p.add_argument("--corpus", required=True,
                          help="corpus_id under {data_dir}/corpora/")
      eval_p.add_argument("--slice", choices=["smoke", "full"], default="smoke")
      eval_p.add_argument("--presets", default=DEFAULT_PRESETS,
                          help="comma-separated preset ladder subset")
      eval_p.add_argument("--spend-cap-usd", type=float, default=5.0,
                          help="hard cap; the run aborts (resumably) when reached")
      eval_p.add_argument("--run-id", default=None,
                          help="resume an aborted run by its run_id")
      eval_p.add_argument("--data-dir", type=Path, default=_default_data_dir(),
                          help="data dir holding corpora/ (default ../data, run from api/)")
      eval_p.add_argument("--ragas", action="store_true",
                          help="score RAGAS faithfulness/relevancy (extra Claude spend)")
      eval_p.add_argument("--yes", action="store_true",
                          help="skip the interactive cost confirmation gate")

      # --- receipts (Plan B) ---
      receipts_p = sub.add_parser("receipts", help="manage committed receipts")
      rsub = receipts_p.add_subparsers(dest="receipts_command", required=True)
      promote_p = rsub.add_parser(
          "promote", help="copy a local run to receipts/ stripped to IDs + metrics"
      )
      promote_p.add_argument("run_id")
      promote_p.add_argument("--data-dir", type=Path, default=_default_data_dir())
      promote_p.add_argument("--receipts-dir", type=Path, default=Path("../receipts"))

      args = parser.parse_args(argv)
      if args.command == "ingest":
          return _cmd_ingest(args)
      if args.command == "eval":
          return _cmd_eval(args)
      return _cmd_promote(args)


  def _cmd_ingest(args: argparse.Namespace) -> int:
      """Plan A's ingest behavior, moved out of main() verbatim."""
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


  def _missing_keys(preset_names: list[str]) -> list[str]:
      """Named env-var messages, never stack traces (spec §Error handling)."""
      runnable = [PRESETS[n] for n in preset_names
                  if PRESETS[n].query.route_mode is RouteMode.FORCE_S1]
      needed = {"ANTHROPIC_API_KEY": "Claude answer synthesis"}
      if any(cfg.query.dense for cfg in runnable):
          needed["VOYAGE_API_KEY"] = "voyage-context-3 query embeddings"
      if any(cfg.query.rerank for cfg in runnable):
          needed["COHERE_API_KEY"] = "Cohere rerank-v4.0-pro"
      return [f"missing env var {key} (needed for {why})"
              for key, why in needed.items() if not os.environ.get(key)]


  def _build_core_real(config: PipelineConfig, corpus_id: str,
                       data_dir: Path) -> RetrievalCore:
      """Composition root (name + signature pinned by R9).

      Assembles Plan A's real retrieval stack for one preset from the
      artifacts `ragreceipts ingest` wrote (surfaces pinned by Plan A):
        - chunks: ingest/chunk_store.read_chunks({corpus_dir}/chunks.jsonl)
        - sparse: SparseRetriever.load({corpus_dir}/sparse, chunks)
        - dense:  DenseRetriever(client, collection=corpus_id,
                  vector_name_for(config.ingest.contextual), embed)
        - rerank: RerankStage(transport)
      All vendor/Qdrant access flows through the module-level factory seams so
      tests can monkeypatch them (offline construction test in test_cli.py).
      """
      from ragreceipts.ingest.chunk_store import read_chunks
      from ragreceipts.retrieval.dense import DenseRetriever, vector_name_for
      from ragreceipts.retrieval.rerank import RerankStage
      from ragreceipts.retrieval.sparse import SparseRetriever

      corpus_dir = data_dir / "corpora" / corpus_id
      chunks = read_chunks(corpus_dir / "chunks.jsonl")
      sparse = (
          SparseRetriever.load(corpus_dir / "sparse", chunks)
          if config.query.bm25 else None
      )
      dense = (
          DenseRetriever(
              build_qdrant(data_dir),
              corpus_id,  # collection name == corpus_id (Plan A's ingest)
              vector_name_for(config.ingest.contextual),
              build_embed_transport(),
          )
          if config.query.dense else None
      )
      rerank_stage = (
          RerankStage(build_rerank_transport()) if config.query.rerank else None
      )
      return RetrievalCore(config=config, dense=dense, sparse=sparse,
                           rerank_stage=rerank_stage)


  def _make_claude():
      """The real ClaudeTransport (vendors/ seam, vendors/anthropic_client.py).

      Lazy import: offline tests never construct it, and the module follows
      Plan A's vendor naming convention (voyage_client.py / cohere_client.py).
      """
      from ragreceipts.vendors.anthropic_client import AnthropicClient

      return AnthropicClient()


  def _cmd_eval(args: argparse.Namespace) -> int:
      preset_names = [p.strip() for p in args.presets.split(",") if p.strip()]
      unknown = [p for p in preset_names if p not in PRESETS]
      if unknown:
          print(f"error: unknown presets {unknown}; valid presets: "
                f"{list(PRESETS)}", file=sys.stderr)
          return 2
      missing = _missing_keys(preset_names)
      if missing:
          for line in missing:
              print(f"error: {line}", file=sys.stderr)
          print("Set the keys in api/.env or the environment, then re-run.",
                file=sys.stderr)
          return 2

      queries = slice_queries(
          load_queries(args.data_dir, args.corpus),
          slice_query_ids(args.data_dir, args.corpus, args.slice),
      )
      estimate = estimate_run_cost(preset_names, len(queries), ragas=args.ragas)
      print(f"Run plan: corpus={args.corpus} slice={args.slice} "
            f"({len(queries)} queries) presets={preset_names}")
      print(f"Estimated cost: ${estimate:.2f}  |  hard spend cap: "
            f"${args.spend_cap_usd:.2f}")
      if args.ragas:
          print("Estimate includes a per-query RAGAS judge heuristic; actual "
                "judge spend is untracked in Plan B and NOT counted against "
                "the hard cap (disclosed per receipt).")
      if not args.yes:
          reply = input("Proceed? [y/N] ").strip().lower()
          if reply != "y":
              print("Aborted before any spend.")
              return 1

      run_id = args.run_id or new_run_id(args.corpus, args.slice)
      ragas = None
      if args.ragas:
          from ragreceipts.eval.ragas_adapter import RagasV04Judge
          from ragreceipts.vendors.ragas_clients import make_anthropic_client

          ragas = RagasV04Judge(make_anthropic_client())
      runner = AblationRunner(
          core_factory=lambda cfg: _build_core_real(cfg, args.corpus, args.data_dir),
          claude=_make_claude(),
          store=RunStore(args.data_dir / "eval-runs.db"),
          data_dir=args.data_dir,
          ragas=ragas,
      )
      try:
          doc = runner.run(run_id=run_id, corpus_id=args.corpus,
                           slice_name=args.slice, presets=preset_names,
                           spend_cap_usd=args.spend_cap_usd)
      except SpendCapExceeded as exc:
          print(f"ABORTED: {exc}", file=sys.stderr)
          return 3

      print(f"Wrote {args.data_dir / 'receipts-local' / (run_id + '.json')}")
      for skip in doc["skipped"]:
          print(f"  SKIPPED {skip['preset']}: {skip['reason']}")
      for env in doc["receipts"]:
          receipt = env["receipt"]
          m = receipt["metrics"]
          print(f"  {receipt['preset']}: recall@5={m['recall_at_5']} "
                f"mrr@3={m['mrr_at_3']} em={m['em']} f1={m['f1']} "
                f"n={receipt['n_total']} failed={receipt['n_failed']} "
                f"abstained={receipt['n_abstained']}")
          if not receipt["anchors"]:
              print("    (no anchors: ladder base, or baseline cell absent "
                    "from this run)")
      return 0


  def _cmd_promote(args: argparse.Namespace) -> int:
      src = args.data_dir / "receipts-local" / f"{args.run_id}.json"
      if not src.exists():
          print(f"error: {src} not found - run `ragreceipts eval` first "
                f"(run_id {args.run_id!r})", file=sys.stderr)
          return 2
      doc = read_run_doc(src)
      stripped = strip_for_commit(doc)
      args.receipts_dir.mkdir(parents=True, exist_ok=True)
      dst = args.receipts_dir / f"{args.run_id}.json"
      dst.write_text(json.dumps(stripped, indent=2) + "\n")
      print(f"Promoted {len(stripped['receipts'])} receipt cell(s) to {dst} "
            f"(passage text + answers stripped; IDs + metrics only). "
            f"Review, then `git add` to commit.")
      return 0


  if __name__ == "__main__":
      sys.exit(main())
  ```

- [ ] Run the tests and a smoke of the entry point (the `ragreceipts` console script was already registered in `api/pyproject.toml` by Plan A — no pyproject change in this task):

  ```bash
  uv run pytest tests/test_cli.py -q
  uv run ragreceipts ingest --help
  uv run ragreceipts eval --help
  uv run ragreceipts receipts promote --help
  ```

  Expected: all CLI tests **PASS** (Plan A's three and Plan B's appended ones); all three `--help` invocations print usage and exit 0.

- [ ] Commit:

  ```bash
  git add ragreceipts/cli.py tests/test_cli.py
  git commit -m "feat(cli): add eval + receipts promote subcommands beside ingest (R6)" \
    -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
  ```

---

### Task 10: Manual first-keyed-run runbook + final verification

**Files:**
- Create: `docs/runbooks/first-keyed-run.md` (repo root, not under `api/`)

- [ ] Create `docs/runbooks/first-keyed-run.md` with this content:

  ```markdown
  # First keyed eval run (manual - NEVER CI)

  CI runs offline with zero keys. This runbook is the one human-driven path
  that spends real money. Budget for the full flow below: under $5 of tracked
  spend, plus untracked RAGAS judge spend (see steps 4-5: the hard cap
  EXCLUDES judge spend).

  ## 0. Prerequisites
  - Spike 0's download script has materialized the raw slices:
    `data/corpora/<corpus_id>/raw/{queries.jsonl,docs.jsonl,slice-full.json,
    slice-smoke.json,download_meta.json}` (from the repo root:
    `uv run --project api python scripts/download_data.py --corpus all`).
  - Plan A ingest completed for at least one corpus (e.g. `musique-dev-300`):
    `uv run ragreceipts ingest --corpus musique-dev-300` wrote
    `data/corpora/<corpus_id>/{manifest.json,chunks.jsonl,sparse/}` plus the
    Qdrant collection.
  - `ANTHROPIC_API_KEY`, `VOYAGE_API_KEY`, `COHERE_API_KEY` exported (or in
    `api/.env`, loaded into the shell). The CLI names any missing key.
  - All commands below run from `api/`; the data dir resolves via
    RAGRECEIPTS_DATA_DIR (default `../data` - R6).

  ## 1. Verify the composition seams (no spend)
  uv run pytest tests/test_plan_a_seams.py tests/test_cli.py \
    tests/test_harness_selftest.py -q

  This covers Spike 0's alignment API, Plan A's FakeRerank scores mode, and
  the R9-pinned composition root `cli.py::_build_core_real(config, corpus_id,
  data_dir)` via its offline construction test - the upstream names are
  pinned by the seam resolutions, so there is no signature discovery step.

  ## 2. Verify rerank pricing (no spend)
  Open the Cohere billing dashboard and confirm rerank-v4.0-pro is billed at
  $0.0025 per search unit (cohere.com/pricing does not publish it; our value
  is corroborated by reseller listings - see eval/pricing.py docstring). If it
  differs: update PRICING, bump PRICING_VERSION to today's date, update the
  pricing tests, commit, and only then continue.

  ## 3. Smoke slice first (~$0.90 estimated)
  uv run ragreceipts eval --corpus musique-dev-300 --slice smoke \
    --presets bm25-only,dense-rrf,contextual,rerank --spend-cap-usd 2.50

  Review the printed estimate (15 queries x 4 presets ~= $0.90), confirm with
  `y`. Expect: 4 receipt cells; `router-on` only appears if you listed it, and
  is then SKIPPED with the "requires Plan C" reason. Inspect
  `data/receipts-local/<run_id>.json`: every envelope carries the fixed
  nondeterminism_note; receipts carry prompts_version "n/a"; anchors carry
  their caveat notes (on an nq-dev-300 run they additionally end with the
  corpus-scale caveat); n_failed / n_abstained are visible; usd_per_query is
  plausible (~$0.015).

  ## 4. RAGAS spot-check
  Re-run step 3 with `--ragas` on the smoke slice. First run downloads
  BAAI/bge-small-en-v1.5 (~130MB, local; no extra key). The printed estimate
  grows by the per-query judge heuristic (~$0.02/query x 15 queries x 4
  presets ~= $1.17 -> total ~= $2.07). IMPORTANT: the HARD SPEND CAP EXCLUDES
  judge spend - RAGAS judge calls are not token-metered in Plan B, so only
  the synthesis/embed/rerank spend counts against the cap; the omission is
  disclosed per receipt via the ragas_judge_usd_untracked flag. Confirm
  ragas_faithfulness / ragas_answer_relevancy are populated and that flag
  appears in per_query flags.

  ## 5. Headline run (full slice, ~300 queries)
  uv run ragreceipts eval --corpus musique-dev-300 --slice full \
    --presets bm25-only,dense-rrf,contextual,rerank --ragas --spend-cap-usd 25

  The printed estimate (~$41 with --ragas) includes the judge heuristic, but
  the hard cap meters only tracked spend (~$18 estimated) - budget the
  untracked judge spend separately. If the cap aborts the run mid-way, re-run
  with the printed `--run-id` and a higher cap - completed queries are never
  re-billed.

  ## 6. Promote and commit the headline receipts
  uv run ragreceipts receipts promote <run_id>
  git -C .. add receipts/<run_id>.json
  git -C .. commit -m "docs(receipts): first committed headline receipts (<run_id>)" \
    -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"

  Promotion strips passage text AND model answers - committed per-query
  records are IDs + metrics only (benchmark redistribution terms). The
  default --receipts-dir is ../receipts (R6), matching the data-dir default
  so this command works from api/.
  ```

- [ ] Run the complete Plan B verification gate:

  ```bash
  uv run ruff check ragreceipts tests
  uv run pytest -q
  ```

  Expected: ruff clean; full suite **PASS** with zero network access and zero keys set (sanity-check by running with keys explicitly unset: `env -u ANTHROPIC_API_KEY -u VOYAGE_API_KEY -u COHERE_API_KEY uv run pytest -q`).

- [ ] Commit (from repo root):

  ```bash
  git add docs/runbooks/first-keyed-run.md
  git commit -m "docs(eval): manual first-keyed-run runbook (smoke -> ragas -> full -> promote)" \
    -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
  ```

---

## Done criteria for Plan B

- `uv run pytest -q` green offline with zero keys (seam tests, metrics goldens, pricing goldens, receipt round-trip, RAGAS fake, run-state, runner, harness self-test, CLI incl. the offline `_build_core_real` composition test).
- Harness self-test proves rerank flips Recall@5 (0.75 → 1.0) and misaligned golds score 0.0 — CI fails if either detection dies.
- `uv run ragreceipts eval --corpus <id> --slice smoke|full --presets ...` reads Spike 0's `raw/queries.jsonl` + slice files directly (R2), produces `data/receipts-local/<run_id>.json` with versioned envelopes (each carrying the fixed `nondeterminism_note` and `prompts_version: "n/a"` per R11), anchors + required caveat notes (corpus-scale caveat appended on nq-dev-300), failure/abstention disclosure, the two independent router-on gates (R10: permanent multi-hop + temporary requires-Plan-C), and is resumable after a spend-cap abort.
- `uv run ragreceipts receipts promote <run_id>` writes `receipts/<run_id>.json` with IDs + metrics only; the `ingest` subcommand and Plan A's factory seams remain intact in the modified `cli.py` (R6).
- `docs/runbooks/first-keyed-run.md` documents the only keyed path (smoke → RAGAS spot-check → full → promote), including the Cohere price re-verification step and the disclosure that the hard cap excludes RAGAS judge spend.
