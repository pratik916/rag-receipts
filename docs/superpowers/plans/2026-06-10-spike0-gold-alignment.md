# Spike 0: Gold-to-Chunk Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** De-risk every future receipt by pinning exact benchmark dataset revisions, implementing the binding gold-to-chunk alignment rules with hand-computed golden tests, and hand-validating alignment on real MuSiQue and Natural Questions data before Plans A–D begin.

**Architecture:** Spike 0 bootstraps the `api/` uv project (the repo's first code) and produces two kinds of artifacts: lasting code (`ragreceipts/eval/alignment.py` hit rules, dataset normalization modules, a stub token-window chunker whose interface Plan A must keep) and throwaway-grade exploration (a download script that materializes `data/corpora/{musique-dev-300,nq-dev-300}/raw/`, and a hand-check harness that renders query/gold/chunk alignment to markdown for a mandatory human review gate). Everything testable is pure Python and runs offline.

**Tech Stack:** Python 3.12, uv, pytest, ruff, HuggingFace `datasets` (pinned `>=4.8.4,<5` — the only networked dependency, used only by the download script, never by tests).

---

## Context

### State of the repo when this plan starts

`/Users/pratiksoni/PersonalProjects/rag-receipts/` is already a git repo (branch `main`) containing **only documentation**: the design spec, the deep-research report, brainstorm mockups, and a `.gitignore` that already excludes `data/`, `.venv/`, `__pycache__/`, `.env*`. The directory `docs/superpowers/plans/` (contracts + this plan) is untracked — Task 1 commits it. There is **no code, no `api/`, no `pyproject.toml`** anywhere. Spike 0 creates the first code.

Source of truth documents (read-only for this plan):

- Spec: `docs/superpowers/specs/2026-06-10-rag-receipts-design.md`
- Contracts (binding names/paths): `docs/superpowers/plans/2026-06-10-contracts.md`

### Binding contract excerpts used by this plan

Tooling (contracts §Tooling): *"Python 3.12, managed with uv (`uv init`, `uv add`, `uv run pytest`). Package dir: `api/ragreceipts/`, tests: `api/tests/`. Lint/format: ruff (line length 100). Test: pytest."*

Core types — `api/ragreceipts/types.py` (contracts, verbatim; Task 1 creates this file):

```python
@dataclass(frozen=True)
class Chunk:
    chunk_id: str          # f"{doc_id}:{position}"
    corpus_id: str
    doc_id: str
    passage_id: str        # parent passage ID for gold alignment (== doc_id when unsegmented)
    text: str
    position: int          # chunk index within document

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

Metric hit rules (contracts §Metrics — these are exactly what `alignment.py` implements):

> A retrieved chunk is a **hit** for a gold passage iff `chunk.passage_id == gold.passage_id`; for span-format golds (NQ long answers), hit iff the chunk covers ≥50% of the gold span's tokens.

Chunking defaults (contracts `IngestConfig`): `chunk_size: int = 512` tokens, `chunk_overlap: int = 64`. The stub chunker uses the same defaults so hand-check numbers are representative of Plan A.

Corpus manifest `dataset` block (contracts §Corpus manifest) — the pins produced by this spike feed it in Plan A: `{"name": ..., "hf_id": ..., "split": ..., "revision": ...}`.

Smoke slice (spec §Eval plane): *"a first-class smoke slice (15 queries per corpus)"* — this spike defines both slices.

### What Spike 0 explicitly does NOT touch

No Anthropic/Voyage/Cohere calls (so `api/tests/fakes.py` is **not** created here — it arrives with `vendors/` in Plan A). No Qdrant, no bm25s, no LlamaIndex, no LangGraph, no FastAPI. All tests in this plan are pure-Python, offline, zero keys — the testing constraint is satisfied trivially because nothing networked is under test. The only network use is the one-shot download script run (Task 7), which is a script execution, not a test.

### New names this plan defines (binding for Plans A–D; contracts allow plans to define what contracts omit)

| Name | Where | Notes |
|---|---|---|
| `ChunkSpan` | `api/ragreceipts/ingest/chunker.py` | chunk + its whitespace-token range in the parent passage |
| `chunk_passage(*, corpus_id, doc_id, passage_id, text, chunk_size=512, chunk_overlap=64) -> list[ChunkSpan]` | same | **Plan A replaces the internals (sentence-window) but MUST keep this signature and `ChunkSpan`** |
| `GoldPassage`, `GoldSpan`, `Gold` | `api/ragreceipts/eval/alignment.py` | gold label types |
| `passage_hit`, `span_hit`, `is_hit`, `first_hit_rank` | same | the binding hit rules; Plan B builds Recall@5/MRR@3 on top of these |
| `strip_html_tokens`, `remap_span`, `select_long_answer`, `nq_doc_id` | `api/ragreceipts/ingest/nq.py` | NQ normalization, reused by Plan A loaders |
| `musique_passage_id`, `musique_records` | `api/ragreceipts/ingest/musique.py` | MuSiQue normalization, reused by Plan A loaders |
| raw slice layout | `data/corpora/{corpus_id}/raw/{queries.jsonl,docs.jsonl,slice-full.json,slice-smoke.json,download_meta.json}` | Plan A ingests from exactly these files |

`docs.jsonl` record: `{"doc_id", "passage_id", "title", "text"}` (in both corpora `doc_id == passage_id` — passages are unsegmented documents, per the `Chunk.passage_id` comment in the contracts).
`queries.jsonl` record: `{"query_id", "question", ..., "gold"}` where `gold` is either `{"type": "passage", "passage_ids": [...]}` (MuSiQue) or `{"type": "span", "doc_id": ..., "start_token": ..., "end_token": ...}` plus a `gold_text` field (NQ). Token indices are **whitespace-token indices into the doc's `text`** (i.e. indices into `text.split()`).
Slice files: JSON arrays of `query_id` strings; `slice-smoke.json` is the first 15 entries of `slice-full.json`.

### External APIs verified for this plan (verified 2026-06-10)

| What | Verified fact | Source |
|---|---|---|
| `datasets.load_dataset` | signature `load_dataset(path, name=None, ..., split=None, revision=None, streaming=False, ...)`; `revision` accepts a commit SHA; `streaming=True` returns an `IterableDataset`; JSONL and Parquet hub repos load without a loading script; Parquet streams row-group-wise | https://huggingface.co/docs/datasets/en/package_reference/loading_methods (v4.8.4 docs). PyPI latest is `datasets` 5.0.0; we pin `>=4.8.4,<5` to stay on the exact major version the docs above describe |
| MuSiQue source | `dgslibisey/MuSiQue` (most-downloaded mirror of official `musique_ans` v1.0 JSONL; files `musique_ans_v1.0_{train,dev}.jsonl`), config `default`, split `validation` = 2,417 rows; pinned revision `c8f4f8c9465fb69d31a8eae894c3fd509c4ca321` (2023-06-16) | https://huggingface.co/datasets/dgslibisey/MuSiQue + https://huggingface.co/api/datasets/dgslibisey/MuSiQue + https://datasets-server.huggingface.co/info?dataset=dgslibisey/MuSiQue |
| MuSiQue row shape | top keys `id, paragraphs, question, question_decomposition, answer, answer_aliases, answerable`; `paragraphs` = **list of dicts** `{idx:int, title:str, paragraph_text:str, is_supporting:bool}` (20 per row); `question_decomposition` = list of dicts `{id, question, answer, paragraph_support_idx}` | https://datasets-server.huggingface.co/rows?dataset=dgslibisey/MuSiQue&config=default&split=validation&offset=0&length=2 (inspected raw JSON) |
| NQ source | `google-research-datasets/natural_questions`, config **`dev`** (parquet-only, `dev/validation-0000{0..6}-of-00007.parquet`, 7,830 rows, ~3.5 GB, streamable); pinned revision `e8103d566bef4154c2c12b17c6095ec5275840cc` (2024-03-11) | https://huggingface.co/api/datasets/google-research-datasets/natural_questions + https://datasets-server.huggingface.co/info?dataset=google-research-datasets/natural_questions |
| NQ row shape | `ex["document"]["tokens"]` = **dict of lists** `{"token": [...], "is_html": [...], "start_byte": [...], "end_byte": [...]}`; `ex["annotations"]` = **dict of lists** with `"id"`: 5 strings, `"long_answer"`: list of 5 dicts `{start_token, end_token, start_byte, end_byte, candidate_index}` (null = `candidate_index == -1`), `"short_answers"`: list of 5 dicts-of-lists (key `"text"`: list[str]), `"yes_no_answer"`: list of 5 ints (-1 = none); `ex["question"]["text"]` is the question string; `ex["id"]` is a string | https://datasets-server.huggingface.co/rows?dataset=google-research-datasets/natural_questions&config=dev&split=validation&offset=0&length=1 (inspected raw JSON) |
| KILT-NQ (evaluated, **rejected** — see Task 2) | `facebook/kilt_tasks` config `nq`: validation 2,837; provenance = `wikipedia_id` + `start_paragraph_id`/`end_paragraph_id` + `bleu_score` into the KILT Wikipedia snapshot (`facebook/kilt_wikipedia`, ~37 GB) | https://datasets-server.huggingface.co/info?dataset=facebook/kilt_tasks |
| Pinned tool versions | `datasets` 5.0.0 (PyPI latest; **we pin `<5`**), `pytest` 9.0.3, `ruff` 0.15.16 | https://pypi.org/pypi/{datasets,pytest,ruff}/json |

Residual unverified detail: the exact Python-side dict shape yielded by `datasets` when *iterating* (vs. the datasets-server JSON above) is the same by construction (datasets-server uses the same library), but Task 7 still includes a `--peek` verification step that prints the first raw example's structure before any full run, so a mismatch is caught in seconds, not after a 300-query download.

Golden hash values used in tests (precomputed with Python 3 `hashlib.sha1`, reproducible):

- `sha1("T\nhello world")[:16]` = `23d5d02c9d6894dc` → `musique_passage_id("T", "hello world") == "mu-23d5d02c9d6894dc"`
- `sha1("hello world")[:16]` = `2aae6c35c94fcfb4` → `nq_doc_id("hello world") == "nq-2aae6c35c94fcfb4"`

### Command cheat sheet

All commands below are given from the repo root `/Users/pratiksoni/PersonalProjects/rag-receipts`:

```bash
cd api && uv run pytest                 # run all tests
cd api && uv run ruff check .           # lint
uv run --project api python scripts/download_data.py --corpus all   # scripts (uv resolves api/.venv)
```

---

### Task 1: Bootstrap the uv project + core types

**Files:**
- Create: `api/pyproject.toml`
- Create: `api/.python-version` (via `uv python pin`)
- Create: `api/ragreceipts/__init__.py`
- Create: `api/ragreceipts/types.py`
- Test: `api/tests/test_types.py`
- Modify: (git only) commit the untracked `docs/superpowers/plans/`

**Steps:**

- [ ] Commit the planning docs that already exist but are untracked:
  ```bash
  git add docs/superpowers/plans/
  git commit -m "docs: add shared contracts and Spike 0 implementation plan

  Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
  ```
- [ ] Create `api/pyproject.toml` with exactly this content (hatchling flat layout so the package dir is `api/ragreceipts/` per contracts, not `src/`):
  ```toml
  [project]
  name = "ragreceipts"
  version = "0.1.0"
  description = "Adaptive RAG engine with measured ablation receipts"
  requires-python = ">=3.12"
  dependencies = []

  [dependency-groups]
  dev = ["pytest>=9.0", "ruff>=0.15"]

  [build-system]
  requires = ["hatchling"]
  build-backend = "hatchling.build"

  [tool.hatch.build.targets.wheel]
  packages = ["ragreceipts"]

  [tool.ruff]
  line-length = 100

  [tool.ruff.lint]
  select = ["E", "F", "I", "UP"]

  [tool.pytest.ini_options]
  testpaths = ["tests"]
  ```
- [ ] Pin the interpreter and create the package skeleton:
  ```bash
  cd api && uv python pin 3.12
  mkdir -p ragreceipts tests
  ```
  (`uv python pin 3.12` writes `api/.python-version`; uv downloads CPython 3.12 if it is not present.)
- [ ] Create `api/ragreceipts/__init__.py`:
  ```python
  """rag-receipts backend — every RAG technique, with receipts."""
  ```
- [ ] Sync the environment: `cd api && uv sync`. Expected: uv creates `api/.venv` and `api/uv.lock`, installs `pytest` and `ruff` (dev group is installed by default), and installs `ragreceipts` itself as an editable project.
- [ ] Write the failing test `api/tests/test_types.py`:
  ```python
  import dataclasses

  import pytest

  from ragreceipts.types import Chunk, RouteMode, ScoredChunk


  def test_chunk_is_frozen_and_carries_alignment_metadata():
      c = Chunk(chunk_id="d1:0", corpus_id="musique-dev-300", doc_id="d1",
                passage_id="p1", text="hello world", position=0)
      assert c.passage_id == "p1"
      assert c.chunk_id == f"{c.doc_id}:{c.position}"
      with pytest.raises(dataclasses.FrozenInstanceError):
          c.text = "nope"


  def test_scored_chunk_and_route_mode():
      c = Chunk(chunk_id="d1:0", corpus_id="c", doc_id="d1", passage_id="d1",
                text="t", position=0)
      s = ScoredChunk(chunk=c, score=1.5, source="bm25")
      assert s.source == "bm25"
      assert RouteMode.FORCE_S1.value == "force_s1"
      assert RouteMode.AUTO.value == "auto"
  ```
- [ ] Run it and watch it fail: `cd api && uv run pytest tests/test_types.py`. Expected failure: `ModuleNotFoundError: No module named 'ragreceipts.types'`.
- [ ] Create `api/ragreceipts/types.py` — contracts-exact definitions:
  ```python
  """Core shared types (binding: docs/superpowers/plans/2026-06-10-contracts.md)."""

  from dataclasses import dataclass
  from enum import Enum


  @dataclass(frozen=True)
  class Chunk:
      chunk_id: str  # f"{doc_id}:{position}"
      corpus_id: str
      doc_id: str
      passage_id: str  # parent passage ID for gold alignment (== doc_id when unsegmented)
      text: str
      position: int  # chunk index within document


  @dataclass(frozen=True)
  class ScoredChunk:
      chunk: Chunk
      score: float
      source: str  # "bm25" | "dense" | "rrf" | "rerank"


  class RouteMode(str, Enum):
      AUTO = "auto"
      FORCE_S1 = "force_s1"
      FORCE_S2 = "force_s2"
  ```
- [ ] Run again: `cd api && uv run pytest tests/test_types.py`. Expected: `2 passed`.
- [ ] Lint: `cd api && uv run ruff check .`. Expected: `All checks passed!`.
- [ ] Commit:
  ```bash
  git add api/
  git commit -m "feat: bootstrap api uv project with core Chunk/ScoredChunk/RouteMode types

  Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
  ```

---

### Task 2: Dataset decision doc — pin MuSiQue, decide KILT-NQ vs original NQ

This task records the spike's load-bearing research decision **before** any data code is written. The research was done while authoring this plan (sources in the Context table); the engineer's job here is to re-verify the pins still resolve and commit the decision doc.

**Files:**
- Create: `docs/superpowers/specs/2026-06-10-spike0-decisions.md`

**Steps:**

- [ ] Re-verify the pinned revisions still resolve on the Hub (the API returns HTTP 200 for a tree at a commit SHA):
  ```bash
  curl -s -o /dev/null -w "%{http_code}\n" \
    "https://huggingface.co/api/datasets/dgslibisey/MuSiQue/tree/c8f4f8c9465fb69d31a8eae894c3fd509c4ca321"
  curl -s -o /dev/null -w "%{http_code}\n" \
    "https://huggingface.co/api/datasets/google-research-datasets/natural_questions/tree/e8103d566bef4154c2c12b17c6095ec5275840cc"
  ```
  Expected: `200` twice. If either returns `404` (repo history was rewritten), fetch the current sha with `curl -s https://huggingface.co/api/datasets/<id> | python3 -c "import json,sys; print(json.load(sys.stdin)['sha'])"`, use that sha everywhere this plan mentions the old one, and note the substitution in the decisions doc.
- [ ] Create `docs/superpowers/specs/2026-06-10-spike0-decisions.md` with exactly this content:
  ```markdown
  # Spike 0 Decisions — Gold-to-Chunk Alignment

  **Date:** 2026-06-10 · **Status:** draft until the human review gate passes;
  the Outcomes section is filled at the end of the spike.
  Plan: `docs/superpowers/plans/2026-06-10-spike0-gold-alignment.md`

  ## D1 — Natural Questions source: original NQ, not KILT-NQ

  Options evaluated (both verified against the HF datasets-server on 2026-06-10):

  | | KILT-NQ (`facebook/kilt_tasks`, config `nq`) | Original NQ (`google-research-datasets/natural_questions`, config `dev`) |
  |---|---|---|
  | Validation size | 2,837 | 7,830 |
  | Gold format | `provenance` = `wikipedia_id` + `start_paragraph_id`/`end_paragraph_id` + `bleu_score` into the KILT 2019/08 Wikipedia snapshot | 5 annotators' `long_answer` token spans + `short_answers`, natively indexed into the same record's `document.tokens` |
  | Corpus dependency | requires `facebook/kilt_wikipedia` (~37 GB, 5.9M pages) to materialize passage text | none — each record carries the full document |
  | Gold provenance quality | re-mapped from original NQ via BLEU matching (`bleu_score` exposes mapping confidence < 1.0) — derived, not native | native annotator labels |

  **Decision: original NQ** (`google-research-datasets/natural_questions`, config `dev`,
  split `validation`, parquet, streamed). Rationale:
  1. Self-contained: corpus and gold come from the same record; no 37 GB knowledge-source
     dependency for a laptop-scale project.
  2. Native annotator golds — no BLEU-remap noise layered under our alignment rule.
  3. It exercises the spec's span-overlap hit rule ("for span-format golds (NQ long
     answers), hit iff the chunk covers ≥50% of the gold span's tokens"). KILT's
     paragraph-id golds would leave that contract-defined rule as dead code.
  4. The `dev` config is parquet-only and streams row-group-wise, so we download only
     the prefix we read (~hundreds of MB), not 3.5 GB.

  Disclosed consequence: the corpus is the union of the selected queries' own Wikipedia
  pages (~300 pages, content-deduplicated). Retrieval difficulty is "find the right
  chunk among ~300 pages", not open-Wikipedia retrieval. Every receipt on this corpus
  must carry that scale caveat in its `published_anchor.note` (Plan B).

  ## D2 — Pinned dataset revisions

  | Corpus | HF id | Config/split | Revision (commit sha) | Verified |
  |---|---|---|---|---|
  | musique-dev-300 | `dgslibisey/MuSiQue` | `default` / `validation` (2,417 rows) | `c8f4f8c9465fb69d31a8eae894c3fd509c4ca321` | 2026-06-10 |
  | nq-dev-300 | `google-research-datasets/natural_questions` | `dev` / `validation` (7,830 rows) | `e8103d566bef4154c2c12b17c6095ec5275840cc` | 2026-06-10 |

  `dgslibisey/MuSiQue` is a community mirror of the official **musique_ans v1.0** JSONL
  (official distribution is Zenodo via `github.com/stonybrooknlp/musique`; the mirror's
  files are named `musique_ans_v1.0_{train,dev}.jsonl` and its schema matches the
  official format field-for-field). Pinning the commit sha makes the mirror tamper-evident.

  ## D3 — Gold formats and hit rules

  - **MuSiQue (passage golds):** each example carries 20 paragraphs
    `{idx, title, paragraph_text, is_supporting}`; gold = the `is_supporting` paragraphs
    (2–4 per question), cross-checked against `question_decomposition[*].paragraph_support_idx`
    (mismatch ⇒ example skipped and counted). Paragraph ids are content-addressed
    (`mu-` + sha1(title\ntext)[:16]) so identical paragraphs shared across examples
    deduplicate to one corpus passage. Hit rule: `chunk.passage_id == gold.passage_id`.
  - **NQ (span golds):** gold = the majority long answer over 5 dev annotators
    (rule: require ≥2 non-null; pick the most frequent `(start_token, end_token)` span;
    ties broken by smallest start then end). Original token indices include HTML tokens;
    they are remapped to clean-token space (D5). Hit rule: chunk covers ≥50% of the gold
    span's tokens (integer form: `2*overlap >= gold_len`), same `doc_id` required.

  ## D4 — Slice definitions (deterministic)

  - **musique-dev-300:** sort dev examples by `id`, shuffle with `random.Random(42)`,
    take the first 300 that pass the support-set cross-check. Smoke = first 15 of the
    slice order. Corpus = union of all 20 paragraphs of each selected example, deduped.
  - **nq-dev-300:** stream `dev`/`validation` in dataset order; accept examples with a
    ≥2/5-annotator long answer that remaps to a non-empty clean span of ≤1024 tokens;
    stop at 300. Smoke = first 15 accepted. Corpus = content-deduped pages of the 300.
  - Slice membership is written to `slice-full.json` / `slice-smoke.json` per corpus;
    the generating logic lives in `scripts/download_data.py` with seed 42.

  ## D5 — Normalization rules that affect alignment

  1. HTML tokens (`is_html == True`) are dropped; each surviving original token maps to
     a `(start, end)` range in clean-token space so gold spans can be remapped exactly.
  2. Tokens containing internal whitespace are split into parts so the invariant
     `" ".join(clean_tokens).split() == clean_tokens` holds — whitespace-token indices
     are therefore stable across (text ↔ token list) round trips. This invariant is what
     lets `Chunk.text` and span token indices coexist without storing offsets on `Chunk`.
  3. `MAX_GOLD_SPAN_TOKENS = 1024`: with `chunk_size=512`, a chunk covers at most 512
     tokens, so the ≥50% rule is mathematically unsatisfiable for golds longer than
     1024 tokens. Such examples (typically giant tables) are excluded at download time
     and counted in `download_meta.json` — disclosed, never silently dropped.
  4. NQ doc ids are content-addressed (`nq-` + sha1(text)[:16]) so the same Wikipedia
     page appearing under multiple queries becomes one corpus document; span golds from
     different queries against the same content share that doc and stay valid.

  ## Outcomes

  Pending — filled at the end of the spike (Task 10 of the plan) after the human
  review gate, with: actual download counts and skip statistics, gold-span length
  stats, hand-check verdicts, and surprises for Plans A/B.
  ```
- [ ] Commit:
  ```bash
  git add docs/superpowers/specs/2026-06-10-spike0-decisions.md
  git commit -m "docs: Spike 0 dataset decisions - pin MuSiQue mirror and original NQ over KILT

  Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
  ```

---

### Task 3: Stub token-window chunker (`ChunkSpan`)

A minimal whitespace-token sliding-window chunker. It exists so alignment can be tested and hand-checked now; Plan A replaces its internals with the real sentence-window chunker **but keeps `chunk_passage`'s signature and `ChunkSpan` unchanged**.

**Files:**
- Create: `api/ragreceipts/ingest/__init__.py`
- Create: `api/ragreceipts/ingest/chunker.py`
- Test: `api/tests/test_chunker.py`

**Steps:**

- [ ] Create the empty package marker `api/ragreceipts/ingest/__init__.py` (empty file).
- [ ] Write the failing test `api/tests/test_chunker.py`:
  ```python
  import pytest

  from ragreceipts.ingest.chunker import chunk_passage


  def _text(n: int) -> str:
      return " ".join(f"t{i}" for i in range(n))


  def test_single_window_when_text_fits():
      spans = chunk_passage(corpus_id="c", doc_id="d1", passage_id="p1",
                            text=_text(5), chunk_size=8, chunk_overlap=2)
      assert len(spans) == 1
      assert (spans[0].start_token, spans[0].end_token) == (0, 5)
      assert spans[0].chunk.chunk_id == "d1:0"
      assert spans[0].chunk.text == "t0 t1 t2 t3 t4"
      assert spans[0].chunk.passage_id == "p1"
      assert spans[0].chunk.corpus_id == "c"


  def test_sliding_windows_with_overlap():
      # 10 tokens, size 4, overlap 1 -> stride 3 -> windows (0,4) (3,7) (6,10)
      spans = chunk_passage(corpus_id="c", doc_id="d1", passage_id="p1",
                            text=_text(10), chunk_size=4, chunk_overlap=1)
      assert [(s.start_token, s.end_token) for s in spans] == [(0, 4), (3, 7), (6, 10)]
      assert [s.chunk.position for s in spans] == [0, 1, 2]
      assert spans[1].chunk.chunk_id == "d1:1"
      assert spans[1].chunk.text == "t3 t4 t5 t6"


  def test_empty_text_yields_no_chunks():
      assert chunk_passage(corpus_id="c", doc_id="d", passage_id="p",
                           text="   ", chunk_size=4, chunk_overlap=0) == []


  def test_invalid_params_rejected():
      with pytest.raises(ValueError):
          chunk_passage(corpus_id="c", doc_id="d", passage_id="p",
                        text="a b", chunk_size=4, chunk_overlap=4)
      with pytest.raises(ValueError):
          chunk_passage(corpus_id="c", doc_id="d", passage_id="p",
                        text="a b", chunk_size=0, chunk_overlap=0)
  ```
- [ ] Run it: `cd api && uv run pytest tests/test_chunker.py`. Expected failure: `ModuleNotFoundError: No module named 'ragreceipts.ingest.chunker'` (after creating `__init__.py`, the error names `chunker`).
- [ ] Create `api/ragreceipts/ingest/chunker.py`:
  ```python
  """Token-window chunker (Spike 0 stub).

  Whitespace-token sliding window. Plan A replaces the internals with the real
  sentence-window chunker but MUST keep `chunk_passage`'s signature and `ChunkSpan`
  unchanged - eval/alignment.py and the hand-check harness depend on them.
  Defaults match contracts IngestConfig (chunk_size=512, chunk_overlap=64).
  """

  from dataclasses import dataclass

  from ragreceipts.types import Chunk


  @dataclass(frozen=True)
  class ChunkSpan:
      """A chunk plus the whitespace-token range it covers in its parent passage text.

      start_token is inclusive, end_token exclusive; both index into passage_text.split().
      """

      chunk: Chunk
      start_token: int
      end_token: int


  def chunk_passage(
      *,
      corpus_id: str,
      doc_id: str,
      passage_id: str,
      text: str,
      chunk_size: int = 512,
      chunk_overlap: int = 64,
  ) -> list[ChunkSpan]:
      """Split `text` into overlapping windows of whitespace tokens.

      Window stride is chunk_size - chunk_overlap; the final window is truncated at the
      end of the text. Empty/whitespace-only text yields [].
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
      start = 0
      position = 0
      while True:
          end = min(start + chunk_size, len(tokens))
          chunk = Chunk(
              chunk_id=f"{doc_id}:{position}",
              corpus_id=corpus_id,
              doc_id=doc_id,
              passage_id=passage_id,
              text=" ".join(tokens[start:end]),
              position=position,
          )
          spans.append(ChunkSpan(chunk=chunk, start_token=start, end_token=end))
          if end == len(tokens):
              break
          start += stride
          position += 1
      return spans
  ```
- [ ] Run again: `cd api && uv run pytest tests/test_chunker.py`. Expected: `4 passed`.
- [ ] Lint: `cd api && uv run ruff check .` → `All checks passed!`.
- [ ] Commit:
  ```bash
  git add api/
  git commit -m "feat: token-window chunker stub with ChunkSpan token ranges

  Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
  ```

---

### Task 4: Alignment rules — the spike's lasting core (TDD, hand-computed goldens)

Implements the contracts' binding hit rules. Plan B's Recall@5/MRR@3 will be thin wrappers over `is_hit`/`first_hit_rank`.

**Files:**
- Create: `api/ragreceipts/eval/__init__.py`
- Create: `api/ragreceipts/eval/alignment.py`
- Test: `api/tests/test_alignment.py`

**Steps:**

- [ ] Create the empty package marker `api/ragreceipts/eval/__init__.py`.
- [ ] Write the failing test `api/tests/test_alignment.py`. Every expected value below is hand-computed; the 50% boundary case is tested on both sides:
  ```python
  import pytest

  from ragreceipts.eval.alignment import (
      GoldPassage,
      GoldSpan,
      first_hit_rank,
      is_hit,
      passage_hit,
      span_hit,
  )
  from ragreceipts.ingest.chunker import ChunkSpan
  from ragreceipts.types import Chunk


  def _span(doc_id: str, passage_id: str, start: int, end: int, position: int = 0) -> ChunkSpan:
      chunk = Chunk(chunk_id=f"{doc_id}:{position}", corpus_id="c", doc_id=doc_id,
                    passage_id=passage_id, text="x " * (end - start), position=position)
      return ChunkSpan(chunk=chunk, start_token=start, end_token=end)


  def test_passage_hit_is_exact_id_match():
      gold = GoldPassage(query_id="q", passage_id="p1")
      assert passage_hit(_span("d", "p1", 0, 4).chunk, gold)
      assert not passage_hit(_span("d", "p2", 0, 4).chunk, gold)


  def test_span_hit_at_exactly_50_percent_boundary():
      gold = GoldSpan(query_id="q", doc_id="d", start_token=10, end_token=20)  # 10 tokens
      assert span_hit(_span("d", "d", 0, 15), gold)      # overlap 5/10 = 50% -> hit
      assert not span_hit(_span("d", "d", 0, 14), gold)  # overlap 4/10 = 40% -> miss
      assert span_hit(_span("d", "d", 12, 30), gold)     # overlap 8/10 = 80% -> hit
      assert span_hit(_span("d", "d", 0, 100), gold)     # full cover -> hit
      assert not span_hit(_span("d", "d", 20, 30), gold)  # adjacent, overlap 0 -> miss


  def test_span_hit_requires_same_document():
      gold = GoldSpan(query_id="q", doc_id="d1", start_token=0, end_token=10)
      assert not span_hit(_span("d2", "d2", 0, 10), gold)


  def test_span_hit_rejects_empty_gold():
      gold = GoldSpan(query_id="q", doc_id="d", start_token=5, end_token=5)
      with pytest.raises(ValueError):
          span_hit(_span("d", "d", 0, 10), gold)


  def test_is_hit_dispatches_on_gold_type():
      ps = _span("d", "p1", 0, 4)
      assert is_hit(ps, GoldPassage(query_id="q", passage_id="p1"))
      assert is_hit(ps, GoldSpan(query_id="q", doc_id="d", start_token=0, end_token=4))


  def test_first_hit_rank_is_one_based_and_k_bounded():
      gold = GoldSpan(query_id="q", doc_id="d", start_token=10, end_token=20)
      ranked = [
          _span("d", "d", 30, 40, position=0),   # miss
          _span("d", "d", 8, 22, position=1),    # hit (full cover)
          _span("d", "d", 10, 20, position=2),   # hit, but rank 2 already found
      ]
      assert first_hit_rank(ranked, gold, k=3) == 2
      assert first_hit_rank(ranked, gold, k=1) is None
      assert first_hit_rank([], gold, k=5) is None


  def test_misaligned_gold_provably_never_hits():
      # Harness self-test flavor (spec section Testing): a gold pointing at a passage id
      # absent from every chunk must produce zero hits at any rank.
      gold = GoldPassage(query_id="q", passage_id="not-in-corpus")
      ranked = [_span("d", f"p{i}", 0, 4, position=i) for i in range(10)]
      assert first_hit_rank(ranked, gold, k=10) is None
      assert not any(is_hit(s, gold) for s in ranked)
  ```
- [ ] Run it: `cd api && uv run pytest tests/test_alignment.py`. Expected failure: `ModuleNotFoundError: No module named 'ragreceipts.eval.alignment'`.
- [ ] Create `api/ragreceipts/eval/alignment.py`:
  ```python
  """Gold-to-chunk alignment rules (Spike 0; lasting code).

  Binding metric definitions (contracts section Metrics):
  - passage gold: hit iff chunk.passage_id == gold.passage_id
  - span gold (NQ long answers): hit iff the chunk covers >= 50% of the gold span's
    tokens (and the chunk belongs to the gold's document)
  Token indices are whitespace-token indices into the parent document's text
  (indices into text.split()), as produced by ragreceipts.ingest.chunker.chunk_passage.
  """

  from dataclasses import dataclass

  from ragreceipts.ingest.chunker import ChunkSpan
  from ragreceipts.types import Chunk


  @dataclass(frozen=True)
  class GoldPassage:
      query_id: str
      passage_id: str


  @dataclass(frozen=True)
  class GoldSpan:
      query_id: str
      doc_id: str
      start_token: int  # inclusive
      end_token: int    # exclusive; must be > start_token


  Gold = GoldPassage | GoldSpan


  def passage_hit(chunk: Chunk, gold: GoldPassage) -> bool:
      return chunk.passage_id == gold.passage_id


  def span_hit(span: ChunkSpan, gold: GoldSpan) -> bool:
      gold_len = gold.end_token - gold.start_token
      if gold_len <= 0:
          raise ValueError(f"gold span must be non-empty: {gold}")
      if span.chunk.doc_id != gold.doc_id:
          return False
      overlap = min(span.end_token, gold.end_token) - max(span.start_token, gold.start_token)
      return 2 * overlap >= gold_len  # integer form of overlap/gold_len >= 0.5


  def is_hit(span: ChunkSpan, gold: Gold) -> bool:
      if isinstance(gold, GoldPassage):
          return passage_hit(span.chunk, gold)
      return span_hit(span, gold)


  def first_hit_rank(ranked: list[ChunkSpan], gold: Gold, k: int) -> int | None:
      """1-based rank of the first hit within the top-k of `ranked`, else None."""
      for rank, span in enumerate(ranked[:k], start=1):
          if is_hit(span, gold):
              return rank
      return None
  ```
- [ ] Run again: `cd api && uv run pytest tests/test_alignment.py`. Expected: `7 passed`.
- [ ] Run the whole suite + lint: `cd api && uv run pytest && uv run ruff check .`. Expected: `13 passed`, `All checks passed!`.
- [ ] Commit:
  ```bash
  git add api/
  git commit -m "feat: gold-to-chunk alignment rules (passage exact-match + >=50% span overlap)

  Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
  ```

---

### Task 5: MuSiQue normalization (pure functions, TDD)

**Files:**
- Create: `api/ragreceipts/ingest/musique.py`
- Test: `api/tests/test_musique.py`

**Steps:**

- [ ] Write the failing test `api/tests/test_musique.py` (the mini example mirrors the verified raw schema exactly):
  ```python
  import pytest

  from ragreceipts.ingest.musique import musique_passage_id, musique_records

  EXAMPLE = {
      "id": "2hop__460946_294723",
      "question": "Who is the spouse of the Green performer?",
      "answer": "Miquette Giraudy",
      "answer_aliases": [],
      "answerable": True,
      "paragraphs": [
          {"idx": 0, "title": "A", "paragraph_text": "Distractor paragraph.",
           "is_supporting": False},
          {"idx": 1, "title": "B", "paragraph_text": "Supporting paragraph one.",
           "is_supporting": True},
          {"idx": 2, "title": "C", "paragraph_text": "Supporting paragraph two.",
           "is_supporting": True},
      ],
      "question_decomposition": [
          {"id": 460946, "question": "sub1", "answer": "x", "paragraph_support_idx": 1},
          {"id": 294723, "question": "sub2", "answer": "y", "paragraph_support_idx": 2},
      ],
  }


  def test_passage_id_is_deterministic_content_address():
      # sha1("T\nhello world")[:16] == "23d5d02c9d6894dc", precomputed
      assert musique_passage_id("T", "hello world") == "mu-23d5d02c9d6894dc"
      assert musique_passage_id("T", "hello world!") != musique_passage_id("T", "hello world")


  def test_records_extract_golds_and_dedupable_passages():
      query, passages = musique_records(EXAMPLE)
      assert query["query_id"] == "2hop__460946_294723"
      assert query["question"] == EXAMPLE["question"]
      assert query["answer"] == "Miquette Giraudy"
      assert query["gold"]["type"] == "passage"
      expected_golds = [
          musique_passage_id("B", "Supporting paragraph one."),
          musique_passage_id("C", "Supporting paragraph two."),
      ]
      assert query["gold"]["passage_ids"] == expected_golds
      assert len(passages) == 3
      for p in passages:
          assert p["doc_id"] == p["passage_id"]  # unsegmented: passage == doc
          assert set(p) == {"doc_id", "passage_id", "title", "text"}


  def test_support_mismatch_raises():
      broken = dict(EXAMPLE)
      broken["question_decomposition"] = [
          {"id": 1, "question": "sub1", "answer": "x", "paragraph_support_idx": 0},
      ]
      with pytest.raises(ValueError):
          musique_records(broken)
  ```
- [ ] Run it: `cd api && uv run pytest tests/test_musique.py`. Expected failure: `ModuleNotFoundError: No module named 'ragreceipts.ingest.musique'`.
- [ ] Create `api/ragreceipts/ingest/musique.py`:
  ```python
  """MuSiQue (musique_ans v1.0) normalization - pure functions, no network.

  Raw example schema (verified against dgslibisey/MuSiQue revision c8f4f8c9, 2026-06-10):
    id: str, question: str, answer: str, answer_aliases: list[str], answerable: bool,
    paragraphs: list[{idx: int, title: str, paragraph_text: str, is_supporting: bool}],
    question_decomposition: list[{id, question, answer, paragraph_support_idx: int}]
  """

  import hashlib


  def musique_passage_id(title: str, text: str) -> str:
      """Deterministic content-addressed id; dedups identical paragraphs across examples."""
      digest = hashlib.sha1(f"{title}\n{text}".encode()).hexdigest()
      return f"mu-{digest[:16]}"


  def musique_records(example: dict) -> tuple[dict, list[dict]]:
      """Normalize one raw example into (query_record, passage_records).

      Raises ValueError when is_supporting disagrees with question_decomposition's
      paragraph_support_idx - the caller counts and skips such examples (degrade
      visibly, never silently).
      """
      supporting_idx = {p["idx"] for p in example["paragraphs"] if p["is_supporting"]}
      decomp_idx = {d["paragraph_support_idx"] for d in example["question_decomposition"]}
      if supporting_idx != decomp_idx:
          raise ValueError(
              f"{example['id']}: is_supporting {sorted(supporting_idx)} != "
              f"decomposition support {sorted(decomp_idx)}"
          )
      passage_records: list[dict] = []
      gold_passage_ids: list[str] = []
      for p in example["paragraphs"]:
          pid = musique_passage_id(p["title"], p["paragraph_text"])
          passage_records.append(
              {"doc_id": pid, "passage_id": pid, "title": p["title"],
               "text": p["paragraph_text"]}
          )
          if p["is_supporting"]:
              gold_passage_ids.append(pid)
      query_record = {
          "query_id": example["id"],
          "question": example["question"],
          "answer": example["answer"],
          "answer_aliases": list(example["answer_aliases"]),
          "gold": {"type": "passage", "passage_ids": gold_passage_ids},
      }
      return query_record, passage_records
  ```
- [ ] Run again: `cd api && uv run pytest tests/test_musique.py`. Expected: `3 passed`.
- [ ] Lint: `cd api && uv run ruff check .` → `All checks passed!`.
- [ ] Commit:
  ```bash
  git add api/
  git commit -m "feat: MuSiQue gold normalization with content-addressed passage ids

  Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
  ```

---

### Task 6: NQ normalization — HTML strip, span remap, majority gold (pure functions, TDD)

This is the riskiest pure logic in the project: original NQ long-answer spans index into `document.tokens` *including* HTML tokens, while our chunks live in clean whitespace-token space. These functions make that mapping exact.

**Files:**
- Create: `api/ragreceipts/ingest/nq.py`
- Test: `api/tests/test_nq.py`

**Steps:**

- [ ] Write the failing test `api/tests/test_nq.py`:
  ```python
  from ragreceipts.ingest.nq import (
      nq_doc_id,
      remap_span,
      select_long_answer,
      strip_html_tokens,
  )


  def test_strip_html_tokens_drops_html_and_maps_clean_spans():
      tokens = ["<p>", "Hello", "big world", "<table>", "x", "</p>"]
      is_html = [True, False, False, True, False, True]
      clean, spans = strip_html_tokens(tokens, is_html)
      assert clean == ["Hello", "big", "world", "x"]  # "big world" split into parts
      assert spans == [None, (0, 1), (1, 3), None, (3, 4), None]
      # round-trip invariant that makes whitespace-token indices stable:
      assert " ".join(clean).split() == clean


  def test_strip_html_tokens_drops_empty_tokens():
      clean, spans = strip_html_tokens(["a", "   ", "b"], [False, False, False])
      assert clean == ["a", "b"]
      assert spans == [(0, 1), None, (1, 2)]


  def test_remap_span_clips_html_edges():
      spans = [None, (0, 1), (1, 3), None, (3, 4), None]
      assert remap_span(spans, 0, 3) == (0, 3)   # <p> Hello big-world -> Hello big world
      assert remap_span(spans, 2, 5) == (1, 4)   # big-world <table> x -> big world x
      assert remap_span(spans, 0, 1) is None     # html-only span
      assert remap_span(spans, 3, 4) is None     # html-only span


  def _la(start: int, end: int, cand: int) -> dict:
      return {"start_token": start, "end_token": end, "start_byte": 0, "end_byte": 0,
              "candidate_index": cand}


  def test_select_long_answer_majority_of_five():
      las = [_la(-1, -1, -1), _la(10, 20, 5), _la(10, 20, 5), _la(30, 40, 9),
             _la(-1, -1, -1)]
      assert select_long_answer(las) == (10, 20)


  def test_select_long_answer_requires_two_nonnull_annotators():
      las = [_la(10, 20, 5)] + [_la(-1, -1, -1)] * 4
      assert select_long_answer(las) is None


  def test_select_long_answer_tie_breaks_deterministically():
      # 1-1 tie between (30,40) and (10,20): smallest start_token wins
      las = [_la(30, 40, 9), _la(10, 20, 5), _la(-1, -1, -1)]
      assert select_long_answer(las) == (10, 20)


  def test_nq_doc_id_is_content_addressed():
      # sha1("hello world")[:16] == "2aae6c35c94fcfb4", precomputed
      assert nq_doc_id("hello world") == "nq-2aae6c35c94fcfb4"
      assert nq_doc_id("hello world") == nq_doc_id("hello world")
      assert nq_doc_id("hello world.") != nq_doc_id("hello world")
  ```
- [ ] Run it: `cd api && uv run pytest tests/test_nq.py`. Expected failure: `ModuleNotFoundError: No module named 'ragreceipts.ingest.nq'`.
- [ ] Create `api/ragreceipts/ingest/nq.py`:
  ```python
  """Natural Questions (original NQ) normalization - pure functions, no network.

  Verified row shape (datasets-server, google-research-datasets/natural_questions,
  config "dev", revision e8103d56, 2026-06-10):
    ex["document"]["tokens"] = {"token": list[str], "is_html": list[bool],
                                "start_byte": list[int], "end_byte": list[int]}
    ex["annotations"] = {"id": list[str] (5 annotators on dev),
                         "long_answer": list[{start_token, end_token, start_byte,
                                              end_byte, candidate_index}],
                         "short_answers": list[{... lists, incl. "text": list[str]}],
                         "yes_no_answer": list[int]}  # null = candidate_index == -1
  Long-answer token indices include HTML tokens; these helpers map them into
  clean whitespace-token space (the space chunk_passage operates in).
  """

  import hashlib
  from collections import Counter

  TokenSpan = tuple[int, int]


  def strip_html_tokens(
      tokens: list[str], is_html: list[bool]
  ) -> tuple[list[str], list[TokenSpan | None]]:
      """Drop HTML tokens; split tokens containing internal whitespace into parts.

      Returns (clean_tokens, token_spans) where token_spans[i] is the (start, end)
      range the i-th ORIGINAL token occupies in clean_tokens, or None if dropped
      (html or whitespace-only). Guarantees " ".join(clean_tokens).split() == clean_tokens.
      """
      clean: list[str] = []
      spans: list[TokenSpan | None] = []
      for token, html in zip(tokens, is_html, strict=True):
          if html:
              spans.append(None)
              continue
          parts = token.split()
          if not parts:
              spans.append(None)
              continue
          start = len(clean)
          clean.extend(parts)
          spans.append((start, len(clean)))
      return clean, spans


  def remap_span(
      token_spans: list[TokenSpan | None], start_token: int, end_token: int
  ) -> TokenSpan | None:
      """Map an original-token [start_token, end_token) range to clean-token space.

      Returns None when the range contains no visible (non-html) tokens.
      """
      visible = [s for s in token_spans[start_token:end_token] if s is not None]
      if not visible:
          return None
      return visible[0][0], visible[-1][1]


  def select_long_answer(long_answers: list[dict]) -> TokenSpan | None:
      """Majority gold over the 5 dev annotators (rule recorded in the decisions doc).

      Require >= 2 annotators with a non-null long answer (candidate_index != -1);
      gold = the most frequent (start_token, end_token) span; ties broken by smallest
      start_token then end_token. Returns None otherwise.
      """
      non_null = [
          (la["start_token"], la["end_token"])
          for la in long_answers
          if la["candidate_index"] != -1
      ]
      if len(non_null) < 2:
          return None
      counts = Counter(non_null)
      return min(counts, key=lambda span: (-counts[span], span[0], span[1]))


  def nq_doc_id(text: str) -> str:
      """Content-addressed doc id; dedups the same Wikipedia page across queries."""
      return f"nq-{hashlib.sha1(text.encode()).hexdigest()[:16]}"
  ```
- [ ] Run again: `cd api && uv run pytest tests/test_nq.py`. Expected: `7 passed`.
- [ ] Full suite + lint: `cd api && uv run pytest && uv run ruff check .`. Expected: `23 passed`, `All checks passed!`.
- [ ] Commit:
  ```bash
  git add api/
  git commit -m "feat: NQ long-answer normalization (html strip, span remap, majority gold)

  Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
  ```

---

### Task 7: Download script + live data pull (network step)

The script is thin glue around the tested pure functions. It pins revisions, applies the deterministic selection rules from the decisions doc, and writes the raw slice layout. The `--peek` mode verifies the live row structure before any full run — this is the designed verification step for the one thing docs alone could not prove (the exact dict shape `datasets` yields when iterating).

**Files:**
- Create: `scripts/download_data.py`
- Modify: `api/pyproject.toml` (+ `api/uv.lock`) via `uv add`
- Test: none new (logic was tested in Tasks 5–6; this task validates outputs by inspection commands)

**Steps:**

- [ ] Add the pinned dependency: `cd api && uv add "datasets>=4.8.4,<5"`. Expected: resolves to datasets 4.8.x; lockfile updated. (We pin `<5` because the verified docs are v4.8.4; PyPI's latest is 5.0.0 — do not upgrade inside the spike.)
- [ ] Create `scripts/download_data.py`:
  ```python
  #!/usr/bin/env python
  """Download + normalize the Spike 0 benchmark slices (network required, no API keys).

  Produces (gitignored):
    data/corpora/musique-dev-300/raw/{queries.jsonl,docs.jsonl,slice-full.json,
                                      slice-smoke.json,download_meta.json}
    data/corpora/nq-dev-300/raw/{...same files...}

  Run from repo root:
    uv run --project api python scripts/download_data.py --peek          # schema check
    uv run --project api python scripts/download_data.py --corpus all    # full pull
  Selection rules and pins are documented in
  docs/superpowers/specs/2026-06-10-spike0-decisions.md (D2, D4, D5).
  """

  import argparse
  import json
  import random
  import sys
  from datetime import UTC, datetime
  from pathlib import Path

  import datasets as hf_datasets
  # load_dataset(path, name=None, split=..., revision=..., streaming=...) verified against
  # https://huggingface.co/docs/datasets/en/package_reference/loading_methods (v4.8.4)
  from datasets import load_dataset

  from ragreceipts.ingest.musique import musique_records
  from ragreceipts.ingest.nq import (
      nq_doc_id,
      remap_span,
      select_long_answer,
      strip_html_tokens,
  )

  REPO_ROOT = Path(__file__).resolve().parents[1]

  MUSIQUE_HF_ID = "dgslibisey/MuSiQue"  # mirror of official musique_ans v1.0 jsonl
  MUSIQUE_REVISION = "c8f4f8c9465fb69d31a8eae894c3fd509c4ca321"  # 2023-06-16
  NQ_HF_ID = "google-research-datasets/natural_questions"
  NQ_CONFIG = "dev"  # parquet-only config: validation split, 7,830 rows
  NQ_REVISION = "e8103d566bef4154c2c12b17c6095ec5275840cc"  # 2024-03-11
  N_QUERIES = 300
  N_SMOKE = 15
  SELECTION_SEED = 42
  # With chunk_size=512 a chunk covers at most 512 tokens, so the >=50% rule is
  # unsatisfiable for golds longer than 2*512 tokens. Exclude and count them.
  MAX_GOLD_SPAN_TOKENS = 1024


  def _write_outputs(out_dir: Path, queries: list[dict], docs: list[dict],
                     meta: dict) -> None:
      out_dir.mkdir(parents=True, exist_ok=True)
      with (out_dir / "queries.jsonl").open("w") as f:
          for q in queries:
              f.write(json.dumps(q) + "\n")
      with (out_dir / "docs.jsonl").open("w") as f:
          for d in docs:
              f.write(json.dumps(d) + "\n")
      slice_full = [q["query_id"] for q in queries]
      (out_dir / "slice-full.json").write_text(json.dumps(slice_full, indent=2))
      (out_dir / "slice-smoke.json").write_text(json.dumps(slice_full[:N_SMOKE], indent=2))
      (out_dir / "download_meta.json").write_text(json.dumps(meta, indent=2))
      print(f"{meta['corpus_id']}: {len(queries)} queries, {len(docs)} docs -> {out_dir}")


  def _base_meta(corpus_id: str, hf_id: str, config: str, revision: str,
                 selection_rule: str, extra: dict) -> dict:
      return {
          "corpus_id": corpus_id,
          "dataset": {"hf_id": hf_id, "config": config, "split": "validation",
                      "revision": revision},
          "selection_rule": selection_rule,
          "seed": SELECTION_SEED,
          "n_queries": N_QUERIES,
          "n_smoke": N_SMOKE,
          "datasets_lib_version": hf_datasets.__version__,
          "created_at": datetime.now(UTC).isoformat(),
          **extra,
      }


  def download_musique(peek: bool = False) -> None:
      ds = load_dataset(MUSIQUE_HF_ID, split="validation", revision=MUSIQUE_REVISION)
      if peek:
          ex = ds[0]
          print("musique keys:", sorted(ex.keys()))
          print("paragraph keys:", sorted(ex["paragraphs"][0].keys()))
          print("decomposition keys:", sorted(ex["question_decomposition"][0].keys()))
          return
      examples = sorted(ds, key=lambda ex: ex["id"])
      random.Random(SELECTION_SEED).shuffle(examples)
      queries: list[dict] = []
      docs_by_id: dict[str, dict] = {}
      n_skipped = 0
      for ex in examples:
          if len(queries) == N_QUERIES:
              break
          try:
              query_record, passage_records = musique_records(ex)
          except ValueError as err:
              print(f"skip {ex['id']}: {err}", file=sys.stderr)
              n_skipped += 1
              continue
          queries.append(query_record)
          for p in passage_records:
              docs_by_id.setdefault(p["doc_id"], p)
      if len(queries) < N_QUERIES:
          raise SystemExit(f"musique: only {len(queries)} qualifying queries, need {N_QUERIES}")
      meta = _base_meta(
          "musique-dev-300", MUSIQUE_HF_ID, "default", MUSIQUE_REVISION,
          "sort dev by id, shuffle with seed, take first 300 whose is_supporting set "
          "matches question_decomposition support idx; corpus = union of all 20 "
          "paragraphs per selected example, deduped by content-addressed passage_id",
          {"n_docs": len(docs_by_id), "n_skipped_support_mismatch": n_skipped},
      )
      _write_outputs(REPO_ROOT / "data/corpora/musique-dev-300/raw",
                     queries, list(docs_by_id.values()), meta)


  def download_nq(peek: bool = False) -> None:
      ds = load_dataset(NQ_HF_ID, NQ_CONFIG, split="validation", streaming=True,
                        revision=NQ_REVISION)
      it = iter(ds)
      if peek:
          ex = next(it)
          print("nq keys:", sorted(ex.keys()))
          print("annotations keys:", sorted(ex["annotations"].keys()))
          print("first long_answer:", ex["annotations"]["long_answer"][0])
          print("document.tokens keys:", sorted(ex["document"]["tokens"].keys()))
          print("question keys:", sorted(ex["question"].keys()))
          return
      queries: list[dict] = []
      docs_by_id: dict[str, dict] = {}
      skip = {"no_majority_long_answer": 0, "empty_doc": 0, "unmappable_span": 0,
              "gold_too_long": 0}
      n_seen = 0
      for ex in it:
          if len(queries) == N_QUERIES:
              break
          n_seen += 1
          gold = select_long_answer(ex["annotations"]["long_answer"])
          if gold is None:
              skip["no_majority_long_answer"] += 1
              continue
          toks = ex["document"]["tokens"]
          clean, token_spans = strip_html_tokens(toks["token"], toks["is_html"])
          if not clean:
              skip["empty_doc"] += 1
              continue
          mapped = remap_span(token_spans, gold[0], gold[1])
          if mapped is None:
              skip["unmappable_span"] += 1
              continue
          start, end = mapped
          if end - start > MAX_GOLD_SPAN_TOKENS:
              skip["gold_too_long"] += 1
              continue
          text = " ".join(clean)
          doc_id = nq_doc_id(text)  # content-addressed: dedups repeated pages
          docs_by_id.setdefault(doc_id, {"doc_id": doc_id, "passage_id": doc_id,
                                         "title": ex["document"]["title"], "text": text})
          short_answers = sorted(
              {t for sa in ex["annotations"]["short_answers"] for t in sa["text"]}
          )
          queries.append({
              "query_id": f"nqq-{ex['id']}",
              "question": ex["question"]["text"],
              "answer_texts": short_answers,
              "gold": {"type": "span", "doc_id": doc_id,
                       "start_token": start, "end_token": end},
              "gold_text": " ".join(clean[start:end]),
          })
      if len(queries) < N_QUERIES:
          raise SystemExit(f"nq: only {len(queries)} qualifying queries, need {N_QUERIES}")
      meta = _base_meta(
          "nq-dev-300", NQ_HF_ID, NQ_CONFIG, NQ_REVISION,
          "stream dev/validation in dataset order; accept examples with a >=2/5-annotator "
          "long answer that remaps to a non-empty clean-token span of <= "
          f"{MAX_GOLD_SPAN_TOKENS} tokens; stop at 300",
          {"n_docs": len(docs_by_id), "n_seen": n_seen, "skip_counts": skip},
      )
      _write_outputs(REPO_ROOT / "data/corpora/nq-dev-300/raw",
                     queries, list(docs_by_id.values()), meta)


  def main() -> None:
      parser = argparse.ArgumentParser(description=__doc__)
      parser.add_argument("--corpus", choices=["musique", "nq", "all"], default="all")
      parser.add_argument("--peek", action="store_true",
                          help="print the first raw example's structure and exit")
      args = parser.parse_args()
      if args.corpus in ("musique", "all"):
          download_musique(peek=args.peek)
      if args.corpus in ("nq", "all"):
          download_nq(peek=args.peek)


  if __name__ == "__main__":
      main()
  ```
- [ ] **Schema verification step** — run the peek before any full pull (network on; no token needed, both datasets are public):
  ```bash
  uv run --project api python scripts/download_data.py --peek
  ```
  Expected output (order of keys may differ only by sorting):
  ```
  musique keys: ['answer', 'answer_aliases', 'answerable', 'id', 'paragraphs', 'question', 'question_decomposition']
  paragraph keys: ['idx', 'is_supporting', 'paragraph_text', 'title']
  decomposition keys: ['answer', 'id', 'paragraph_support_idx', 'question']
  nq keys: ['annotations', 'document', 'id', 'long_answer_candidates', 'question']
  annotations keys: ['id', 'long_answer', 'short_answers', 'yes_no_answer']
  first long_answer: {'start_token': ..., 'end_token': ..., 'start_byte': ..., 'end_byte': ..., 'candidate_index': ...}
  document.tokens keys: ['end_byte', 'is_html', 'start_byte', 'token']
  question keys: ['text', 'tokens']
  ```
  If any key set differs, STOP: the accessors in `download_nq`/`download_musique` are the only place to adjust; fix them to the printed shape, re-run `--peek`, and record the discrepancy in the decisions doc Outcomes section later. Do not change the tested pure functions' contracts.
- [ ] Pull MuSiQue (fast — the dev JSONL is small): `uv run --project api python scripts/download_data.py --corpus musique`. Expected final line: `musique-dev-300: 300 queries, <N> docs -> .../data/corpora/musique-dev-300/raw` with N roughly 5,000–6,000 (20 paragraphs × 300 queries, minus content dedup).
- [ ] Pull NQ (streams the parquet prefix; expect 5–20 minutes depending on connection — roughly 600–900 raw examples are read to find 300 qualifying): `uv run --project api python scripts/download_data.py --corpus nq`. Expected final line: `nq-dev-300: 300 queries, <N> docs -> ...` with N ≤ 300 (deduped pages).
- [ ] Validate outputs:
  ```bash
  wc -l data/corpora/musique-dev-300/raw/queries.jsonl data/corpora/nq-dev-300/raw/queries.jsonl
  # expected: 300 each
  python3 - <<'EOF'
  import json
  from pathlib import Path
  for name in ("musique-dev-300", "nq-dev-300"):
      raw = Path(f"data/corpora/{name}/raw")
      full = json.loads((raw / "slice-full.json").read_text())
      smoke = json.loads((raw / "slice-smoke.json").read_text())
      assert len(full) == 300 and len(smoke) == 15 and smoke == full[:15], name
      docs = {json.loads(line)["doc_id"] for line in (raw / "docs.jsonl").open()}
      for line in (raw / "queries.jsonl").open():
          q = json.loads(line)
          g = q["gold"]
          if g["type"] == "passage":
              assert set(g["passage_ids"]) <= docs, q["query_id"]
          else:
              assert g["doc_id"] in docs and g["end_token"] > g["start_token"], q["query_id"]
      print(name, "OK:", len(full), "queries,", len(docs), "docs")
  EOF
  ```
  Expected: `musique-dev-300 OK: 300 queries, <N> docs` and `nq-dev-300 OK: 300 queries, <N> docs` with no assertion errors (this proves every gold references a doc that exists in its corpus — the precondition for alignment).
- [ ] Confirm `data/` stayed untracked: `git status --short` shows only `scripts/` and `api/` changes.
- [ ] Commit (script + lockfile only; data is gitignored by design and benchmark redistribution terms):
  ```bash
  git add scripts/download_data.py api/pyproject.toml api/uv.lock
  git commit -m "feat: pinned benchmark slice download script (musique-dev-300, nq-dev-300)

  Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
  ```

---

### Task 8: Hand-check harness — render alignment for human eyes

Samples 20 queries per corpus, chunks the gold documents with the production default chunking (512/64), applies the real `is_hit` rules, and renders everything a human needs to judge whether the alignment is trustworthy. For MuSiQue it also chunks 2 distractor passages per query so the reviewer can see the rule *discriminating*, not just matching.

**Files:**
- Create: `scripts/handcheck_alignment.py`
- Output (gitignored): `data/handcheck/musique-dev-300.md`, `data/handcheck/nq-dev-300.md`

**Steps:**

- [ ] Create `scripts/handcheck_alignment.py`:
  ```python
  #!/usr/bin/env python
  """Render gold-to-chunk alignment samples for human review (Spike 0 acceptance gate).

  Offline: reads only data/corpora/ produced by scripts/download_data.py.
  Run from repo root:
    uv run --project api python scripts/handcheck_alignment.py
  Writes data/handcheck/{corpus}.md and prints a one-line JSON summary per corpus.
  """

  import argparse
  import json
  import random
  import statistics
  from pathlib import Path

  from ragreceipts.eval.alignment import Gold, GoldPassage, GoldSpan, is_hit
  from ragreceipts.ingest.chunker import ChunkSpan, chunk_passage

  REPO_ROOT = Path(__file__).resolve().parents[1]


  def _load_jsonl(path: Path) -> list[dict]:
      with path.open() as f:
          return [json.loads(line) for line in f if line.strip()]


  def _golds_for(q: dict) -> list[Gold]:
      g = q["gold"]
      if g["type"] == "passage":
          return [GoldPassage(query_id=q["query_id"], passage_id=pid)
                  for pid in g["passage_ids"]]
      return [GoldSpan(query_id=q["query_id"], doc_id=g["doc_id"],
                       start_token=g["start_token"], end_token=g["end_token"])]


  def render_corpus(corpus_dir: Path, out_path: Path, *, n_sample: int, seed: int,
                    chunk_size: int, chunk_overlap: int) -> dict:
      raw = corpus_dir / "raw"
      queries = _load_jsonl(raw / "queries.jsonl")
      docs = {d["doc_id"]: d for d in _load_jsonl(raw / "docs.jsonl")}
      by_id = {q["query_id"]: q for q in queries}
      slice_full: list[str] = json.loads((raw / "slice-full.json").read_text())

      # corpus-wide gold-span stats (span golds only) - feeds the decisions doc
      span_lens = [q["gold"]["end_token"] - q["gold"]["start_token"]
                   for q in queries if q["gold"]["type"] == "span"]
      rng = random.Random(seed)
      sample_ids = rng.sample(slice_full, n_sample)

      lines = [f"# Hand-check: {corpus_dir.name}",
               f"chunk_size={chunk_size} chunk_overlap={chunk_overlap} "
               f"seed={seed} n_sample={n_sample}", ""]
      if span_lens:
          lines.append(f"Gold span tokens over all {len(span_lens)} queries: "
                       f"min={min(span_lens)} median={int(statistics.median(span_lens))} "
                       f"max={max(span_lens)}")
      n_golds = n_golds_hit = n_queries_ok = 0
      for query_id in sample_ids:
          q = by_id[query_id]
          golds = _golds_for(q)
          answer = q.get("answer") or ", ".join(q.get("answer_texts", [])) or "(none)"
          lines += ["", f"## {query_id}", f"**Q:** {q['question']}",
                    f"**Gold answer:** {answer}"]
          if q["gold"]["type"] == "passage":
              doc_ids = list(q["gold"]["passage_ids"])
              non_gold = sorted(set(docs) - set(doc_ids))
              doc_ids += rng.sample(non_gold, min(2, len(non_gold)))  # distractors
          else:
              g = q["gold"]
              lines += [f"**Gold span:** doc={g['doc_id']} tokens "
                        f"[{g['start_token']}, {g['end_token']}) "
                        f"len={g['end_token'] - g['start_token']}",
                        f"**Gold text:** {q['gold_text'][:600]}"]
              doc_ids = [g["doc_id"]]
          spans: list[ChunkSpan] = []
          for doc_id in doc_ids:
              d = docs[doc_id]
              doc_spans = chunk_passage(corpus_id=corpus_dir.name, doc_id=d["doc_id"],
                                        passage_id=d["passage_id"], text=d["text"],
                                        chunk_size=chunk_size, chunk_overlap=chunk_overlap)
              spans.extend(doc_spans)
              lines.append(f"\n**Doc {doc_id}** ({d['title']!r}, {len(doc_spans)} chunks):")
              for s in doc_spans:
                  mark = "HIT " if any(is_hit(s, g) for g in golds) else "miss"
                  lines.append(f"- [{mark}] {s.chunk.chunk_id} tokens "
                               f"[{s.start_token},{s.end_token}) - {s.chunk.text[:240]}")
          hit_flags = [any(is_hit(s, g) for s in spans) for g in golds]
          n_golds += len(golds)
          n_golds_hit += sum(hit_flags)
          if all(hit_flags):
              n_queries_ok += 1
              lines.append("\nALIGNMENT OK - every gold has at least one hitting chunk.")
          else:
              lines.append("\n**NO HIT for some gold - INVESTIGATE BEFORE SIGN-OFF.**")
      summary = {"corpus": corpus_dir.name, "queries_sampled": n_sample,
                 "golds": n_golds, "golds_hit": n_golds_hit,
                 "queries_all_golds_hit": n_queries_ok,
                 "span_len_max": max(span_lens) if span_lens else None}
      lines += ["", "---", f"Summary: {json.dumps(summary)}"]
      out_path.parent.mkdir(parents=True, exist_ok=True)
      out_path.write_text("\n".join(lines) + "\n")
      return summary


  def main() -> None:
      parser = argparse.ArgumentParser(description=__doc__)
      parser.add_argument("--n", type=int, default=20)
      parser.add_argument("--seed", type=int, default=7)
      parser.add_argument("--chunk-size", type=int, default=512)
      parser.add_argument("--chunk-overlap", type=int, default=64)
      args = parser.parse_args()
      for name in ("musique-dev-300", "nq-dev-300"):
          corpus_dir = REPO_ROOT / "data" / "corpora" / name
          out_path = REPO_ROOT / "data" / "handcheck" / f"{name}.md"
          summary = render_corpus(corpus_dir, out_path, n_sample=args.n, seed=args.seed,
                                  chunk_size=args.chunk_size,
                                  chunk_overlap=args.chunk_overlap)
          print(json.dumps(summary))
          print(f"wrote {out_path}")


  if __name__ == "__main__":
      main()
  ```
  (No new unit tests: every decision the harness makes flows through `chunk_passage` and `is_hit`, which carry golden tests from Tasks 3–4; the harness itself is presentation glue whose output is judged by a human in Task 9.)
- [ ] Run it: `uv run --project api python scripts/handcheck_alignment.py`. Expected: two JSON summary lines and two `wrote data/handcheck/....md` lines. Healthy values: `golds_hit == golds` and `queries_all_golds_hit == 20` for **musique** (passage golds hit by construction — anything less means an id-plumbing bug); for **nq** expect `golds_hit` ≥ 19/20 (a miss is possible if a gold span straddles chunk windows worse than 50% — each miss must be inspected in the markdown, understood, and written up in the Outcomes section; it is a finding about the rule, not necessarily a bug).
- [ ] Open both files and skim 3 queries each to confirm the rendering itself is legible (gold text visible, HIT/miss markers present, distractor docs show all-miss for MuSiQue):
  ```bash
  head -n 60 data/handcheck/musique-dev-300.md
  head -n 60 data/handcheck/nq-dev-300.md
  ```
- [ ] Commit the harness (outputs stay gitignored):
  ```bash
  git add scripts/handcheck_alignment.py
  git commit -m "feat: hand-check harness rendering gold-to-chunk alignment for review

  Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
  ```

---

### Task 9: STOP — mandatory human review gate

**Files:** none (review of `data/handcheck/*.md`)

**Steps:**

- [ ] **STOP. Do not proceed to Task 10 and do not declare Spike 0 done until a human has reviewed the hand-check files and explicitly approved.** Present to the human reviewer:
  - `data/handcheck/musique-dev-300.md` and `data/handcheck/nq-dev-300.md`
  - the two JSON summary lines from Task 8
  - this checklist for the reviewer to answer:
    1. For ~5 MuSiQue queries: do the HIT chunks actually contain the supporting facts for the question, and do the 2 distractor docs show all-miss?
    2. For ~5 NQ queries: does the gold text shown match the content of the HIT chunk(s)? Is the gold text a sensible long answer to the question (not navigation junk or a mangled table)?
    3. For every `NO HIT ... INVESTIGATE` marker (if any): is the explanation in front of you acceptable (e.g. span straddling at <50%), or does it indicate an off-by-one / remap bug?
    4. Are token spans plausible (no negative lengths, no spans past document end)?
  - The reviewer must reply with explicit approval (e.g. "alignment approved") or concrete objections.
- [ ] If the reviewer raises objections: fix the implicated code (most likely `nq.py` remap or the chunker), re-run Tasks 7–8 outputs as needed, and repeat this gate. Each fix follows TDD: encode the broken case as a failing golden test first.
- [ ] Record the reviewer's verdict verbatim (name/handle, date, notes) — it goes into the decisions doc in Task 10.

---

### Task 10: Record outcomes, final sweep, close the spike

**Files:**
- Modify: `docs/superpowers/specs/2026-06-10-spike0-decisions.md` (replace the `## Outcomes` section)

**Steps:**

- [ ] Replace the `## Outcomes` section of the decisions doc with the filled version of this template — every blank is mechanically sourced from a named artifact (no estimates):
  ```markdown
  ## Outcomes (recorded <today's date>, after the human review gate)

  ### Download results (source: data/corpora/*/raw/download_meta.json)
  - musique-dev-300: n_queries=300, n_docs=<meta n_docs>,
    n_skipped_support_mismatch=<meta value>, datasets_lib_version=<meta value>
  - nq-dev-300: n_queries=300, n_docs=<meta n_docs>, n_seen=<meta value>,
    skip_counts: no_majority_long_answer=<v>, empty_doc=<v>, unmappable_span=<v>,
    gold_too_long=<v>

  ### Gold span stats (source: header of data/handcheck/nq-dev-300.md)
  - NQ gold span tokens over all 300 queries: min=<v>, median=<v>, max=<v>
    (cap MAX_GOLD_SPAN_TOKENS=1024 enforced at download; <gold_too_long count>
    examples excluded by it)

  ### Hand-check verdicts (source: Task 8 JSON summaries + Task 9 review)
  - musique: queries_sampled=20, golds=<v>, golds_hit=<v>, queries_all_golds_hit=<v>
  - nq: queries_sampled=20, golds=<v>, golds_hit=<v>, queries_all_golds_hit=<v>
  - Per-miss explanations (one line each, or "none"): <...>
  - Reviewer: <name/handle>, <date> - verdict: <verbatim approval/notes>

  ### Surprises & follow-ups for Plans A/B
  <List every observation that future plans must know: tokenization quirks, table-heavy
  long answers, how many duplicate pages were deduped, paragraphs shared across MuSiQue
  examples, peek-step schema discrepancies (if any), NO-HIT cases and their causes.
  If genuinely nothing surprised you, instead describe the two ugliest examples you
  reviewed and why they still pass - an empty surprises section is not credible.>
  ```
- [ ] Final verification sweep:
  ```bash
  cd api && uv run pytest && uv run ruff check .
  ```
  Expected: `23 passed`, `All checks passed!`.
- [ ] Confirm the lasting artifacts are all in place for Plan A: `api/ragreceipts/{types.py,ingest/chunker.py,ingest/musique.py,ingest/nq.py,eval/alignment.py}` committed; `data/corpora/{musique-dev-300,nq-dev-300}/raw/` populated locally (gitignored — Plan A's ingest reads them in place); decisions doc complete.
- [ ] Commit:
  ```bash
  git add docs/superpowers/specs/2026-06-10-spike0-decisions.md
  git commit -m "docs: record Spike 0 outcomes - alignment validated on real slices

  Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
  ```
- [ ] Spike 0 is done. Plans A–D may begin.
