# LangGraph Router + System-1/System-2 + Trace Store Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the adaptive query plane — a Claude-routed LangGraph state machine that sends each query down a System-1 fast path or a hard-bounded System-2 agentic loop, with every node emitting TraceEvents to a SQLite WAL store, and wire it into the Plan B eval runner so the `router-on` preset produces receipts with route-distribution stats.

**Architecture:** `agents/` is pure orchestration: LangGraph nodes (`route`, `s1_retrieve`, `s1_answer`, `decompose`, `retrieve_hop`, `grade`, `refine`, `synthesize`) call retrieval only through the injected `RetrievalCore` and call Claude only through the `ClaudeTransport` protocol (real impl: `vendors/anthropic_client.py`). Budgets (3 hops, 50K tokens) live in graph state and are enforced by conditional edges fed by node bookkeeping; a per-query `TraceRecorder` stamps `trace_id`/`seq` onto every event and appends to `TraceStore` (SQLite, WAL). The eval runner's generation seam is replaced by `agents/service.run_query`, enabling the `router-on` preset.

**Tech Stack:** Python 3.12 + uv, LangGraph 1.2.x (`StateGraph`), Pydantic v2 structured outputs via the contracts-pinned `messages.parse()` pattern, `anthropic` SDK, sqlite3 (stdlib, WAL mode), pytest with offline fakes (zero API keys in CI).

---

## Context

### Where this plan starts (per plan ordering)

Plan A (ingestion + retrieval core + PipelineConfig + golden tests) and Plan B (eval CLI producing `receipts.json` + first committed receipts) are complete. The repo is a git repo rooted at `rag-receipts/` with a uv project at `api/` (`uv run pytest` green, `ruff` clean, line length 100). All contracts below are quoted from `docs/superpowers/plans/2026-06-10-contracts.md` (binding — never rename or redefine).

Already existing and used (not created) by this plan:

- `api/ragreceipts/constants.py`:

  ```python
  ROUTER_MODEL = "claude-haiku-4-5-20251001"   # routing + CRAG grading, temperature=0
  SYNTH_MODEL = "claude-sonnet-4-6"            # answer synthesis
  ROUTE_CONFIDENCE_THRESHOLD = 0.7             # below this, escalate to System-2
  S2_MAX_HOPS = 3
  S2_TOKEN_CEILING = 50_000                    # input+output summed across all Claude calls per query
  ```

- `api/ragreceipts/types.py`: frozen dataclasses `Chunk` (`chunk_id, corpus_id, doc_id, passage_id, text, position, start_token, end_token` — the two token fields are R3's whitespace-token offsets within the parent passage, persisted to `chunks.jsonl` and the Qdrant payload), `ScoredChunk` (`chunk, score, source`), and `RouteMode` enum (`AUTO="auto"`, `FORCE_S1="force_s1"`, `FORCE_S2="force_s2"`).

- `api/ragreceipts/config.py`: frozen `IngestConfig`, `QueryConfig` (`bm25, dense, rerank, route_mode, top_k_fuse, top_k_final`), `PipelineConfig(name, ingest, query)`, and `PRESETS` with keys `"bm25-only"`, `"dense-rrf"`, `"contextual"`, `"rerank"`, `"router-on"` (router-on: all flags on, `route_mode=RouteMode.AUTO`).

- `api/ragreceipts/retrieval/core.py` (Plan A; trace wiring pinned by R9):

  ```python
  class RetrievalCore:
      def __init__(self, config: PipelineConfig, dense: Retriever | None,
                   sparse: Retriever | None, rerank_stage: "RerankStage | None",
                   on_trace: "TraceCallback | None" = None): ...
      def retrieve(self, query: str) -> list[ScoredChunk]: ...
      # honors config.query flags; returns top_k_final chunks; emits TraceEvents
      # through on_trace (TraceCallback = Callable[[TraceEvent], None])
  ```

  Trace wiring is the constructor kwarg `on_trace` (R9): a caller that wants the core's intra-retrieval events on a query's trace constructs `RetrievalCore(..., on_trace=recorder)` — never a private-attribute assignment. `TraceRecorder.__call__` (Task 3) accepts the emitted `TraceEvent` and re-stamps `trace_id`/`seq` onto the per-query trace.

- `api/ragreceipts/vendors/base.py`:

  ```python
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

- `api/tests/fakes.py`: `FakeEmbed`, `FakeRerank`, and `FakeClaude(script: list)`. Per R5, Plan A authors `FakeClaude` with the ordered script from the start — one list consumed across both `complete()` and `parse()`: a `str` item → `ClaudeResult`, a Pydantic instance → `ParsedResult`, optional `(item, input_tokens, output_tokens)` tuples for token accounting. There is NO constructor migration, and Plan B never constructs `FakeClaude` (its runner tests use their own local `StubClaude` and synthesize via `parse()`). Task 7 adds `FakeCore` + `make_chunk` and extends `FakeClaude` only if a capability is missing.

- `api/ragreceipts/eval/`: Plan B's runner (CLI producing `receipts.json`), metrics (`recall_at_5`, `mrr_at_3`, EM/F1, RAGAS adapter), `receipts.py` (`Receipt` frozen dataclass — note `metrics: dict` and `per_query: list[dict]` are open dicts; this plan adds keys, never fields; per R11 `Receipt` already carries `prompts_version: str`, set to `"n/a"` by Plan B), `pricing.py`. The runner is `eval/runner.py::AblationRunner` with `_run_preset` and `estimate_run_cost` (names pinned by R9). It currently generates S1 answers via its `synthesize()` helper — a `ClaudeTransport.parse(..., output_format=S1Answer)` call, NOT `complete()` — and keeps TWO independent gates on `router-on` (R10): the temporary "requires Plan C" skip and the permanent `MULTI_HOP_DATASETS` gate. Task 12 replaces the synthesize seam with `agents/service.run_query` and deletes only the temporary skip.

### Contracts this plan implements

- `api/ragreceipts/traces/models.py`:

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

- LangGraph contract: nodes named exactly `route`, `s1_retrieve`, `s1_answer`, `decompose`, `retrieve_hop`, `grade`, `refine`, `synthesize`. Budget enforced in graph state (`hops_used`, `tokens_used`); bounded loops via conditional edges + `recursion_limit`. Abstention surfaced as structured `abstained: bool` on the answer object, never prose-only.

- Anthropic SDK usage (binding, verified against the claude-api skill 2026-06-10 in the contracts — do not substitute other patterns): `anthropic.Anthropic()` reads `ANTHROPIC_API_KEY`; structured outputs via `client.messages.parse(model=..., max_tokens=..., messages=[...], output_format=PydanticModel)` → `response.parsed_output`; `temperature=0` is supported on Sonnet 4.6 / Haiku 4.5; typed exceptions `anthropic.RateLimitError` / `anthropic.APIStatusError`; the SDK auto-retries 429/5xx with backoff (`max_retries` configurable) honoring `retry-after`; `max_tokens` 1024 for routing/grading, 4096 for synthesis; no assistant prefills; `anthropic` is imported only inside `vendors/`.

- `router-on` cell metrics (contract): primary = answer-level EM/F1 + RAGAS; retrieval recall over the **union of per-hop top-5** is a secondary diagnostic flagged `union_of_hops: true`. Abstentions excluded from RAGAS, reported as `n_abstained`.

### Spec behaviors implemented here (spec §Query plane)

- `route` (Haiku, temperature 0) → `simple|complex` + confidence; confidence `< 0.7` escalates to System-2; `route_mode ∈ {auto, force_s1, force_s2}`.
- System-2 hard bounds: 3 hops max + 50K-token ceiling summed input+output across all Claude calls. Exhaustion → caveated synthesis with `unresolved_subqueries`, never papered over.
- CRAG-style grading: `sufficient` → proceed; `insufficient` → refine + re-retrieve while budget remains; `contradictory` → **one** re-retrieve attempt, then synthesis citing both sources with an explicit contradiction flag in answer **and** trace.
- Every node emits TraceEvents (inputs, outputs, scores, model, tokens, ms) → SQLite (WAL).

### Verified external APIs (verified 2026-06-10)

| Library | Verified at | What was verified |
|---|---|---|
| langgraph 1.2.4 | https://docs.langchain.com/oss/python/langgraph/graph-api (redirect target of langchain-ai.github.io/langgraph); https://pypi.org/pypi/langgraph/json | `from langgraph.graph import StateGraph, START, END`; state as `TypedDict`; `builder.add_node("name", fn)`; `add_edge(START, "a")`; `add_conditional_edges("node", routing_fn, path_map_dict)` (routing fn returns a key of the path map); nodes return **partial state updates** (dict merged into state); `builder.compile()`; `graph.invoke(inputs, config={"recursion_limit": N})` — `recursion_limit` is a standalone top-level `config` key, not under `configurable`. |
| anthropic 0.109.0 | https://pypi.org/pypi/anthropic/json | Latest version for pinning only. Usage patterns are NOT taken from the web — they are binding from the contracts file. Task 1 asserts `messages.parse` exists at the pinned version; if that assert fails, STOP and escalate (contract mismatch) rather than improvising. |
| pydantic 2.13.4 | https://pypi.org/pypi/pydantic/json | Latest v2 for pinning (`Literal` fields and `Field(ge=, le=)` are stable v2 features). |

**Qdrant note:** Plan C's tests never touch Qdrant. Graph tests inject `FakeCore` (duck-types `RetrievalCore.retrieve`); the eval-integration test reuses Plan B's existing offline harness (which already fakes vendors per Plan A/B). The `QdrantClient(":memory:")` named-vector question is Plan A/B's concern and is not re-litigated here.

### Conventions for every task

- Shell commands run from `/Users/pratiksoni/PersonalProjects/rag-receipts/api` unless a step says otherwise; git commands run from `/Users/pratiksoni/PersonalProjects/rag-receipts`.
- `api/tests/` is a package (R8): `api/tests/__init__.py` exists and `[tool.pytest.ini_options] pythonpath = ["."]` is set in `api/pyproject.toml`. Every test file imports fakes as `from tests.fakes import ...` — no other form, no fallback shims.
- Offline always: no test may require a network or an API key.

---

### Task 1: Dependencies + API availability checks

**Files:**
- Modify: `api/pyproject.toml` (via `uv add`)

- [ ] Add the pinned dependencies (versions verified above):

  ```bash
  uv add "langgraph>=1.2,<2" "anthropic>=0.109" "pydantic>=2.7"
  ```

- [ ] Verify the LangGraph imports this plan uses exist:

  ```bash
  uv run python -c "from langgraph.graph import StateGraph, START, END; print('langgraph ok')"
  ```

  Expected output: `langgraph ok`.

- [ ] Verify the contracts-pinned structured-output entry point exists on the installed SDK (constructing the client makes no network call; the dummy key only satisfies client construction):

  ```bash
  uv run python -c "
  import anthropic
  c = anthropic.Anthropic(api_key='offline-check')
  assert hasattr(c.messages, 'parse'), anthropic.__version__
  assert hasattr(anthropic, 'RateLimitError') and hasattr(anthropic, 'APIStatusError')
  print('anthropic ok', anthropic.__version__)
  "
  ```

  Expected output: `anthropic ok 0.109.0` (or newer). **If the `messages.parse` assert fails, STOP — the binding contract pattern and the installed SDK disagree; escalate to the contracts owner instead of substituting a different API.**

- [ ] Confirm the existing suite still passes: `uv run pytest -q` → all Plan A/B tests PASS.
- [ ] Commit:

  ```bash
  git add api/pyproject.toml api/uv.lock
  git commit -m "chore(deps): add langgraph, anthropic, pydantic for the agent layer

  Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
  ```

---

### Task 2: TraceEvent model + SQLite WAL TraceStore

**Files:**
- Create: `api/ragreceipts/traces/__init__.py`, `api/ragreceipts/traces/models.py`, `api/ragreceipts/traces/store.py`
- Test: `api/tests/test_trace_store.py`

- [ ] Check whether Plan A already created `api/ragreceipts/traces/models.py` (its `RetrievalCore` "emits TraceEvents via callback", so it may have): `ls api/ragreceipts/traces/ 2>/dev/null` (run from repo root) . If `models.py` exists, diff it against the contract dataclass below — it must be field-for-field identical (it is contract-pinned); if identical, skip creating it. If the directory is missing, create `api/ragreceipts/traces/__init__.py` (empty file).
- [ ] Write the failing test:

  ```python
  # api/tests/test_trace_store.py
  import threading

  from ragreceipts.traces.models import TraceEvent
  from ragreceipts.traces.store import TraceStore


  def make_event(seq: int, trace_id: str = "t-1", node: str = "route") -> TraceEvent:
      return TraceEvent(
          trace_id=trace_id, seq=seq, node=node,
          payload={"query": "q", "n": seq}, model="claude-haiku-4-5-20251001",
          input_tokens=10, output_tokens=5, duration_ms=12.5,
      )


  def test_append_get_roundtrip(tmp_path):
      store = TraceStore(tmp_path / "traces.sqlite3")
      store.append(make_event(0))
      store.append(make_event(1))
      events = store.get("t-1")
      assert [e.seq for e in events] == [0, 1]
      assert events[0] == make_event(0)          # frozen dataclass equality incl. payload
      assert events[1].payload == {"query": "q", "n": 1}


  def test_get_orders_by_seq_and_isolates_traces(tmp_path):
      store = TraceStore(tmp_path / "traces.sqlite3")
      store.append(make_event(2))
      store.append(make_event(0))
      store.append(make_event(1, trace_id="t-2"))
      assert [e.seq for e in store.get("t-1")] == [0, 2]
      assert [e.trace_id for e in store.get("t-2")] == ["t-2"]
      assert store.get("missing") == []


  def test_wal_mode_enabled(tmp_path):
      store = TraceStore(tmp_path / "traces.sqlite3")
      mode = store._conn.execute("PRAGMA journal_mode").fetchone()[0]
      assert mode == "wal"


  def test_concurrent_appends(tmp_path):
      # The server (Plan D) runs jobs in a worker thread next to request handlers.
      store = TraceStore(tmp_path / "traces.sqlite3")

      def worker(offset: int) -> None:
          for i in range(20):
              store.append(make_event(offset + i))

      threads = [threading.Thread(target=worker, args=(o,)) for o in (0, 100)]
      for t in threads:
          t.start()
      for t in threads:
          t.join()
      assert len(store.get("t-1")) == 40
  ```

- [ ] Run it: `uv run pytest tests/test_trace_store.py -q` → expected failure: `ModuleNotFoundError: No module named 'ragreceipts.traces.store'` (or `.models` if Plan A didn't create it).
- [ ] Create `api/ragreceipts/traces/models.py` (contract-exact, only if missing per step 1):

  ```python
  """TraceEvent — contract-pinned in docs/superpowers/plans/2026-06-10-contracts.md."""
  from __future__ import annotations

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
  ```

- [ ] Create `api/ragreceipts/traces/store.py`:

  ```python
  """SQLite trace store. WAL mode per spec (single-worker uvicorn + one job thread)."""
  from __future__ import annotations

  import json
  import sqlite3
  import threading
  from pathlib import Path

  from ragreceipts.traces.models import TraceEvent

  _SCHEMA = """
  CREATE TABLE IF NOT EXISTS trace_events (
      trace_id TEXT NOT NULL,
      seq INTEGER NOT NULL,
      node TEXT NOT NULL,
      payload TEXT NOT NULL,
      model TEXT,
      input_tokens INTEGER NOT NULL,
      output_tokens INTEGER NOT NULL,
      duration_ms REAL NOT NULL,
      PRIMARY KEY (trace_id, seq)
  )
  """


  class TraceStore:
      def __init__(self, db_path: str | Path):
          Path(db_path).parent.mkdir(parents=True, exist_ok=True)
          # One shared connection + lock: WAL allows concurrent readers, and our
          # writers are few (request thread + one worker thread). check_same_thread
          # is safe because every access is serialized by the lock.
          self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
          self._lock = threading.Lock()
          with self._lock:
              self._conn.execute("PRAGMA journal_mode=WAL")
              self._conn.execute(_SCHEMA)
              self._conn.commit()

      def append(self, event: TraceEvent) -> None:
          with self._lock:
              self._conn.execute(
                  "INSERT INTO trace_events VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                  (event.trace_id, event.seq, event.node, json.dumps(event.payload),
                   event.model, event.input_tokens, event.output_tokens,
                   event.duration_ms),
              )
              self._conn.commit()

      def get(self, trace_id: str) -> list[TraceEvent]:
          with self._lock:
              rows = self._conn.execute(
                  "SELECT trace_id, seq, node, payload, model, input_tokens,"
                  " output_tokens, duration_ms"
                  " FROM trace_events WHERE trace_id = ? ORDER BY seq",
                  (trace_id,),
              ).fetchall()
          return [
              TraceEvent(trace_id=r[0], seq=r[1], node=r[2], payload=json.loads(r[3]),
                         model=r[4], input_tokens=r[5], output_tokens=r[6],
                         duration_ms=r[7])
              for r in rows
          ]

      def close(self) -> None:
          with self._lock:
              self._conn.close()
  ```

- [ ] Run again: `uv run pytest tests/test_trace_store.py -q` → expected: **4 passed**.
- [ ] Lint: `uv run ruff check ragreceipts/traces tests/test_trace_store.py` → clean.
- [ ] Commit:

  ```bash
  git add api/ragreceipts/traces api/tests/test_trace_store.py
  git commit -m "feat(traces): TraceEvent model and SQLite WAL TraceStore

  Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
  ```

---

### Task 3: Per-query TraceRecorder

**Files:**
- Create: `api/ragreceipts/traces/recorder.py`
- Test: `api/tests/test_trace_recorder.py`

- [ ] Write the failing test:

  ```python
  # api/tests/test_trace_recorder.py
  from ragreceipts.traces.models import TraceEvent
  from ragreceipts.traces.recorder import TraceRecorder
  from ragreceipts.traces.store import TraceStore


  def test_emit_stamps_trace_id_and_increments_seq(tmp_path):
      store = TraceStore(tmp_path / "t.sqlite3")
      rec = TraceRecorder(store, "trace-9")
      rec.emit("route", {"a": 1}, model="m", input_tokens=3, output_tokens=4,
               duration_ms=1.0)
      rec.emit("s1_retrieve", {"b": 2})
      events = store.get("trace-9")
      assert [(e.seq, e.node) for e in events] == [(0, "route"), (1, "s1_retrieve")]
      assert events[0].model == "m" and events[0].input_tokens == 3
      assert events[1].model is None and events[1].input_tokens == 0


  def test_call_accepts_trace_event_and_restamps(tmp_path):
      # Plan A's RetrievalCore on_trace callback delivers a ready TraceEvent (R9).
      store = TraceStore(tmp_path / "t.sqlite3")
      rec = TraceRecorder(store, "trace-9")
      foreign = TraceEvent(trace_id="other", seq=99, node="s1_retrieve",
                           payload={"k": 1}, model=None, input_tokens=0,
                           output_tokens=0, duration_ms=2.0)
      rec(foreign)
      events = store.get("trace-9")
      assert events[0].trace_id == "trace-9" and events[0].seq == 0
      assert events[0].node == "s1_retrieve" and events[0].payload == {"k": 1}


  def test_call_accepts_kwargs_dict(tmp_path):
      # The recorder also accepts a kwargs dict for emit() — used by test doubles.
      store = TraceStore(tmp_path / "t.sqlite3")
      rec = TraceRecorder(store, "trace-9")
      rec({"node": "retrieve_hop", "payload": {"hop": 0}, "duration_ms": 3.0})
      events = store.get("trace-9")
      assert events[0].node == "retrieve_hop" and events[0].duration_ms == 3.0
  ```

- [ ] Run it: `uv run pytest tests/test_trace_recorder.py -q` → expected failure: `ModuleNotFoundError: No module named 'ragreceipts.traces.recorder'`.
- [ ] Create `api/ragreceipts/traces/recorder.py`:

  ```python
  """Per-query trace recorder: stamps trace_id + a monotonically increasing seq.

  Also serves as the RetrievalCore trace callback (Plan A): __call__ accepts either
  a ready TraceEvent (re-stamped onto this trace) or a kwargs dict for emit().
  """
  from __future__ import annotations

  import dataclasses
  import itertools

  from ragreceipts.traces.models import TraceEvent
  from ragreceipts.traces.store import TraceStore


  class TraceRecorder:
      def __init__(self, store: TraceStore, trace_id: str):
          self.store = store
          self.trace_id = trace_id
          self._seq = itertools.count()

      def emit(self, node: str, payload: dict, *, model: str | None = None,
               input_tokens: int = 0, output_tokens: int = 0,
               duration_ms: float = 0.0) -> None:
          self.store.append(TraceEvent(
              trace_id=self.trace_id, seq=next(self._seq), node=node, payload=payload,
              model=model, input_tokens=input_tokens, output_tokens=output_tokens,
              duration_ms=duration_ms,
          ))

      def __call__(self, event: TraceEvent | dict) -> None:
          if isinstance(event, TraceEvent):
              self.store.append(dataclasses.replace(
                  event, trace_id=self.trace_id, seq=next(self._seq)))
          else:
              self.emit(**event)
  ```

- [ ] Run again: `uv run pytest tests/test_trace_recorder.py -q` → expected: **3 passed**.
- [ ] Commit:

  ```bash
  git add api/ragreceipts/traces/recorder.py api/tests/test_trace_recorder.py
  git commit -m "feat(traces): per-query TraceRecorder with seq stamping and callback adapter

  Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
  ```

---

### Task 4: AnthropicClient (real ClaudeTransport)

**Files:**
- Create: `api/ragreceipts/vendors/anthropic_client.py`
- Test: `api/tests/test_anthropic_client.py`

The SDK patterns here are **binding from the contracts file** (already verified against the claude-api skill on 2026-06-10). Do not consult other Anthropic documentation; do not invent other call shapes. Tests are offline: a stub object is injected in place of the real SDK client, so the mapping logic (content-block joining, usage extraction, kwargs passthrough) is tested without any network or key.

- [ ] Write the failing test:

  ```python
  # api/tests/test_anthropic_client.py
  from types import SimpleNamespace

  from pydantic import BaseModel

  from ragreceipts.vendors.anthropic_client import AnthropicClient


  class _Shape(BaseModel):
      answer: str


  class _StubMessages:
      def __init__(self):
          self.create_kwargs = None
          self.parse_kwargs = None

      def create(self, **kwargs):
          self.create_kwargs = kwargs
          return SimpleNamespace(
              content=[SimpleNamespace(type="text", text="hello "),
                       SimpleNamespace(type="thinking", thinking="x"),
                       SimpleNamespace(type="text", text="world")],
              usage=SimpleNamespace(input_tokens=12, output_tokens=7),
          )

      def parse(self, **kwargs):
          self.parse_kwargs = kwargs
          return SimpleNamespace(
              parsed_output=kwargs["output_format"](answer="42"),
              usage=SimpleNamespace(input_tokens=20, output_tokens=9),
          )


  def make_client():
      stub = SimpleNamespace(messages=_StubMessages())
      return AnthropicClient(client=stub), stub


  def test_complete_maps_text_and_usage():
      client, stub = make_client()
      res = client.complete(model="claude-sonnet-4-6", system="sys", user="hi",
                            max_tokens=4096)
      assert res.text == "hello world"            # text blocks joined, others skipped
      assert (res.input_tokens, res.output_tokens) == (12, 7)
      kw = stub.messages.create_kwargs
      assert kw["model"] == "claude-sonnet-4-6"
      assert kw["system"] == "sys"
      assert kw["messages"] == [{"role": "user", "content": "hi"}]
      assert kw["max_tokens"] == 4096
      assert kw["temperature"] == 0.0             # default per ClaudeTransport contract


  def test_parse_returns_parsed_output_and_usage():
      client, stub = make_client()
      res = client.parse(model="claude-haiku-4-5-20251001", system="sys", user="hi",
                         max_tokens=1024, output_format=_Shape)
      assert res.parsed == _Shape(answer="42")
      assert (res.input_tokens, res.output_tokens) == (20, 9)
      assert stub.messages.parse_kwargs["output_format"] is _Shape
      assert stub.messages.parse_kwargs["messages"] == [{"role": "user", "content": "hi"}]
  ```

- [ ] Run it: `uv run pytest tests/test_anthropic_client.py -q` → expected failure: `ModuleNotFoundError: No module named 'ragreceipts.vendors.anthropic_client'`.
- [ ] Create `api/ragreceipts/vendors/anthropic_client.py`:

  ```python
  """ClaudeTransport over the official `anthropic` SDK.

  Binding usage per docs/superpowers/plans/2026-06-10-contracts.md (verified against
  the claude-api skill 2026-06-10):
  - complete() -> client.messages.create(...)
  - parse()    -> client.messages.parse(..., output_format=Model) -> resp.parsed_output
  - The SDK auto-retries 429/5xx with exponential backoff honoring retry-after;
    `max_retries` is configurable on the client constructor.
  - Typed exceptions (anthropic.RateLimitError, anthropic.APIStatusError) propagate
    after retries are exhausted — spec: Claude failure is surfaced, never fabricated.
  - Constructing without a key fails fast with the SDK's message naming
    ANTHROPIC_API_KEY (spec: named env-var message, not a stack trace mystery).
  - `anthropic` is imported ONLY here (vendors/ boundary rule).
  """
  from __future__ import annotations

  import anthropic

  from ragreceipts.vendors.base import ClaudeResult, ParsedResult


  class AnthropicClient:
      def __init__(self, api_key: str | None = None, max_retries: int = 4,
                   client: "anthropic.Anthropic | None" = None):
          # `client` injection exists for offline tests only.
          self._client = client or anthropic.Anthropic(api_key=api_key,
                                                       max_retries=max_retries)

      def complete(self, *, model: str, system: str, user: str, max_tokens: int,
                   temperature: float = 0.0) -> ClaudeResult:
          resp = self._client.messages.create(
              model=model, system=system, max_tokens=max_tokens,
              temperature=temperature,
              messages=[{"role": "user", "content": user}],
          )
          text = "".join(block.text for block in resp.content if block.type == "text")
          return ClaudeResult(text=text, input_tokens=resp.usage.input_tokens,
                              output_tokens=resp.usage.output_tokens)

      def parse(self, *, model: str, system: str, user: str, max_tokens: int,
                output_format: type, temperature: float = 0.0) -> ParsedResult:
          resp = self._client.messages.parse(
              model=model, system=system, max_tokens=max_tokens,
              temperature=temperature,
              messages=[{"role": "user", "content": user}],
              output_format=output_format,
          )
          return ParsedResult(parsed=resp.parsed_output,
                              input_tokens=resp.usage.input_tokens,
                              output_tokens=resp.usage.output_tokens)
  ```

- [ ] Run again: `uv run pytest tests/test_anthropic_client.py -q` → expected: **2 passed**.
- [ ] Commit:

  ```bash
  git add api/ragreceipts/vendors/anthropic_client.py api/tests/test_anthropic_client.py
  git commit -m "feat(vendors): AnthropicClient implementing ClaudeTransport via messages.create/parse

  Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
  ```

---

### Task 5: Structured-output schemas (agents/schemas.py)

**Files:**
- Create: `api/ragreceipts/agents/__init__.py`, `api/ragreceipts/agents/schemas.py`
- Test: `api/tests/test_agent_schemas.py`

- [ ] Create `api/ragreceipts/agents/__init__.py` (empty file) if `agents/` does not exist yet.
- [ ] Write the failing test:

  ```python
  # api/tests/test_agent_schemas.py
  import pytest
  from pydantic import ValidationError

  from ragreceipts.agents.schemas import (
      FinalAnswer,
      GradeResult,
      RouteDecision,
      SubQueries,
  )


  def test_route_decision_literal_and_bounds():
      d = RouteDecision(route="simple", confidence=0.9)
      assert d.route == "simple" and d.confidence == 0.9
      with pytest.raises(ValidationError):
          RouteDecision(route="medium", confidence=0.5)
      with pytest.raises(ValidationError):
          RouteDecision(route="simple", confidence=1.5)


  def test_grade_result_verdicts():
      for v in ("sufficient", "insufficient", "contradictory"):
          assert GradeResult(verdict=v).verdict == v
      with pytest.raises(ValidationError):
          GradeResult(verdict="maybe")


  def test_subqueries():
      assert SubQueries(items=["a", "b"]).items == ["a", "b"]


  def test_final_answer_defaults_and_fields():
      a = FinalAnswer(text="Paris [1]", citations=[1])
      assert a.abstained is False
      assert a.unresolved_subqueries == []
      assert a.contradiction_flag is False
      b = FinalAnswer(text="cannot answer", abstained=True,
                      unresolved_subqueries=["who founded X"], contradiction_flag=True)
      assert b.abstained and b.contradiction_flag
  ```

- [ ] Run it: `uv run pytest tests/test_agent_schemas.py -q` → expected failure: `ModuleNotFoundError: No module named 'ragreceipts.agents.schemas'`.
- [ ] Create `api/ragreceipts/agents/schemas.py`:

  ```python
  """Pydantic response models for Claude structured outputs (messages.parse).

  Field names are part of the prompt contract in agents/prompts.py — the prompts
  reference `abstained`, `citations`, `unresolved_subqueries`, `contradiction_flag`
  by name. Change them in lockstep or not at all.
  """
  from __future__ import annotations

  from typing import Literal

  from pydantic import BaseModel, Field


  class RouteDecision(BaseModel):
      route: Literal["simple", "complex"]
      confidence: float = Field(ge=0.0, le=1.0)


  class SubQueries(BaseModel):
      items: list[str]


  class GradeResult(BaseModel):
      verdict: Literal["sufficient", "insufficient", "contradictory"]


  class FinalAnswer(BaseModel):
      text: str
      citations: list[int] = Field(default_factory=list)
      abstained: bool = False
      unresolved_subqueries: list[str] = Field(default_factory=list)
      contradiction_flag: bool = False
  ```

- [ ] Run again: `uv run pytest tests/test_agent_schemas.py -q` → expected: **4 passed**.
- [ ] Commit:

  ```bash
  git add api/ragreceipts/agents api/tests/test_agent_schemas.py
  git commit -m "feat(agents): structured-output schemas for router, grader, and synthesis

  Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
  ```

---

### Task 6: Prompts (agents/prompts.py)

**Files:**
- Create: `api/ragreceipts/agents/prompts.py`
- Test: `api/tests/test_prompts.py`

Full prompt texts live in this task — they are versioned artifacts (receipts record prompt versions, spec §Error handling). The `[n]` citation format and the structured-abstention instruction are load-bearing.

- [ ] Write the failing test:

  ```python
  # api/tests/test_prompts.py
  from ragreceipts.agents import prompts
  from ragreceipts.types import Chunk, ScoredChunk


  def sc(i: int, text: str) -> ScoredChunk:
      return ScoredChunk(
          chunk=Chunk(chunk_id=f"d:{i}", corpus_id="c", doc_id="d", passage_id="d",
                      text=text, position=i, start_token=0,
                      end_token=len(text.split())),
          score=1.0, source="rrf",
      )


  def test_format_numbered_context():
      out = prompts.format_numbered_context([sc(0, "alpha"), sc(1, "beta")])
      assert out == "[1] alpha\n\n[2] beta"
      assert prompts.format_numbered_context([]) == ""


  def test_format_hop_context_global_numbering_and_dedupe():
      h1 = {"subquery": "q1", "chunks": [sc(0, "alpha"), sc(1, "beta")]}
      h2 = {"subquery": "q2", "chunks": [sc(1, "beta"), sc(2, "gamma")]}
      text, ordered = prompts.format_hop_context([h1, h2])
      assert [s.chunk.chunk_id for s in ordered] == ["d:0", "d:1", "d:2"]
      assert "[3]" in text and "[4]" not in text   # dedupe: beta numbered once
      assert '(hop: "q2") gamma' in text


  def test_prompts_carry_required_instructions():
      # Citation format [n] and structured abstention are load-bearing (spec).
      assert "[1]" in prompts.S1_ANSWER_SYSTEM
      assert "abstained" in prompts.S1_ANSWER_SYSTEM
      assert "abstained" in prompts.SYNTHESIZE_SYSTEM
      assert "contradiction_flag" in prompts.SYNTHESIZE_SYSTEM
      assert "contradictory" in prompts.GRADE_SYSTEM
      assert "{query}" in prompts.ROUTE_USER
      assert "{max_hops}" in prompts.DECOMPOSE_USER
      for name in ("ROUTE_SYSTEM", "DECOMPOSE_SYSTEM", "GRADE_SYSTEM",
                   "REFINE_SYSTEM", "SYNTHESIZE_SYSTEM", "S1_ANSWER_SYSTEM"):
          assert len(getattr(prompts, name)) > 100, name
  ```

- [ ] Run it: `uv run pytest tests/test_prompts.py -q` → expected failure: `ModuleNotFoundError: No module named 'ragreceipts.agents.prompts'`.
- [ ] Create `api/ragreceipts/agents/prompts.py` with the complete prompt set:

  ```python
  """Prompt set for the agent graph. PROMPTS_VERSION is recorded in receipts.

  All structured-output prompts rely on messages.parse() enforcing the schema, so
  prompts describe SEMANTICS (when to abstain, how to cite) rather than JSON shape.
  """
  from __future__ import annotations

  from typing import Sequence

  from ragreceipts.types import ScoredChunk

  PROMPTS_VERSION = "2026-06-10.c1"

  # ---------------------------------------------------------------- route
  ROUTE_SYSTEM = """\
  You are a query-complexity router for a retrieval-augmented QA system.

  Classify the user's question:
  - "simple": answerable from a single passage of text — one fact, one entity, \
  one lookup.
  - "complex": requires combining evidence from multiple passages — multi-hop \
  reasoning, comparisons between entities, or chains like "the director of the \
  film that won X".

  Also report your confidence in this classification as a number between 0.0 and
  1.0. Be honest about uncertainty: if the question is ambiguous or could go either
  way, report low confidence. The system escalates low-confidence questions to a
  slower, more careful pipeline, so an honest low score is useful and a falsely
  high score is harmful."""

  ROUTE_USER = "Question: {query}"

  # ---------------------------------------------------------------- System-1 answer
  S1_ANSWER_SYSTEM = """\
  You answer questions using ONLY the numbered context passages provided.

  Rules:
  - Cite evidence inline with bracketed passage numbers, e.g. "Paris [1]" or
    "in 1969 [2][3]". Set `citations` to the list of passage numbers you actually
    used.
  - Keep answers short and factual: benchmark answers are typically a few words.
  - ABSTENTION: if the context does not contain the information needed to answer,
    set `abstained` to true, set `text` to one sentence explaining what is missing,
    and leave `citations` empty. Never guess and never use outside knowledge.
  - This is the single-hop path: leave `unresolved_subqueries` empty and
    `contradiction_flag` false."""

  S1_ANSWER_USER = """\
  Question: {query}

  Context passages:
  {context}"""

  # ---------------------------------------------------------------- decompose
  DECOMPOSE_SYSTEM = """\
  You decompose a multi-hop question into ordered sub-queries for a retrieval
  system. Each sub-query will be sent to a search engine ON ITS OWN, with no memory
  of the other sub-queries or their answers.

  Rules:
  - Order sub-queries so earlier ones establish the entities later ones need.
  - Phrase each as a standalone factual search query: name entities explicitly;
    no pronouns; never write "the answer from step 1".
  - Use the smallest number of sub-queries that covers the question; a two-hop
    question needs two, not four."""

  DECOMPOSE_USER = """\
  Question: {query}

  Produce at most {max_hops} ordered sub-queries."""

  # ---------------------------------------------------------------- grade (CRAG-style)
  GRADE_SYSTEM = """\
  You grade whether retrieved passages are adequate to answer a search query.
  Return exactly one verdict:
  - "sufficient": the passages contain the information needed to answer the query.
  - "insufficient": the passages are off-topic, or on-topic but missing the needed
    fact.
  - "contradictory": two or more passages make incompatible claims about the queried
    fact (e.g. different dates, different people for the same role).

  Judge only adequacy for THIS query. Ignore style, length, and whether the passages
  cover other topics."""

  GRADE_USER = """\
  Search query: {subquery}

  Retrieved passages:
  {context}"""

  # ---------------------------------------------------------------- refine
  REFINE_SYSTEM = """\
  You rewrite a search query whose retrieval results were inadequate. Produce ONE
  improved query: add disambiguating entities, synonyms, or more specific phrasing
  likely to match the missing evidence. Output ONLY the rewritten query text — no
  quotes, no explanation, one line."""

  REFINE_USER = """\
  Original query: {subquery}

  Inadequate passages retrieved for it:
  {context}"""

  # ---------------------------------------------------------------- synthesize
  SYNTHESIZE_SYSTEM = """\
  You write the final answer to a multi-hop question from evidence gathered over
  several retrieval hops. The evidence passages are numbered globally, e.g. "[1]".

  Rules:
  - Use ONLY the numbered evidence. Cite inline with bracketed passage numbers,
    e.g. "Bong Joon-ho [3]". Set `citations` to the passage numbers you actually
    used.
  - Keep the answer short and factual.
  - If sub-queries are listed as UNRESOLVED, do not invent their answers: state the
    limitation in `text` (one clause is enough) and copy them into
    `unresolved_subqueries`.
  - If the grader flagged the evidence as CONTRADICTORY, present both claims with
    their citations and set `contradiction_flag` to true.
  - ABSTENTION: if the evidence cannot support any answer at all, set `abstained`
    to true and explain what is missing in `text`. Never use outside knowledge."""

  SYNTHESIZE_USER = """\
  Question: {query}

  Evidence passages (numbered globally across hops):
  {context}

  Unresolved sub-queries: {unresolved}
  Contradiction detected by the grader: {contradiction}"""


  # ---------------------------------------------------------------- formatting helpers
  def format_numbered_context(chunks: Sequence[ScoredChunk]) -> str:
      """'[1] text' blocks, 1-based to match the [n] citation format."""
      return "\n\n".join(f"[{i}] {sc.chunk.text}" for i, sc in enumerate(chunks, 1))


  def format_hop_context(
      hop_records: Sequence[dict],
  ) -> tuple[str, list[ScoredChunk]]:
      """Globally numbered context across hops; dedupes by chunk_id (first wins).

      Returns (formatted context, ordered chunks) so that citation [n] maps to
      ordered[n-1] — the eval/UI layers rely on this mapping.
      """
      seen: set[str] = set()
      ordered: list[ScoredChunk] = []
      blocks: list[str] = []
      for rec in hop_records:
          for sc in rec["chunks"]:
              if sc.chunk.chunk_id in seen:
                  continue
              seen.add(sc.chunk.chunk_id)
              ordered.append(sc)
              blocks.append(f'[{len(ordered)}] (hop: "{rec["subquery"]}") {sc.chunk.text}')
      return "\n\n".join(blocks), ordered
  ```

- [ ] Run again: `uv run pytest tests/test_prompts.py -q` → expected: **3 passed**.
- [ ] Commit:

  ```bash
  git add api/ragreceipts/agents/prompts.py api/tests/test_prompts.py
  git commit -m "feat(agents): full prompt set for route/decompose/grade/refine/synthesize/s1

  Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
  ```

---

### Task 7: FakeCore + make_chunk test doubles (FakeClaude is already scripted per R5)

**Files:**
- Modify: `api/tests/fakes.py`
- Test: `api/tests/test_fakes_claude.py`

Per R5, Plan A authored `FakeClaude(script: list)` from the start: one ordered script consumed across both `complete()` and `parse()` — a `str` item returns a `ClaudeResult`, a Pydantic instance returns a `ParsedResult`, and `(item, input_tokens, output_tokens)` tuples control token accounting; under- or mis-scripted tests fail loudly with `AssertionError`. **There is no constructor migration and there are no Plan B call sites to update** — Plan B never constructs `FakeClaude` (its runner tests use their own local `StubClaude` and synthesize via `parse()`). This task adds the two doubles the graph tests still need — `make_chunk` and `FakeCore` — and extends `FakeClaude` ONLY if a capability the graph tests rely on is missing (the unified `.calls` recording).

- [ ] Write the failing test:

  ```python
  # api/tests/test_fakes_claude.py
  import pytest

  from ragreceipts.agents.schemas import RouteDecision
  from tests.fakes import FakeClaude, FakeCore, make_chunk


  def test_script_pops_in_order_across_parse_and_complete():
      fc = FakeClaude(script=[RouteDecision(route="simple", confidence=0.9),
                              "refined query"])
      r1 = fc.parse(model="m", system="s", user="u", max_tokens=10,
                    output_format=RouteDecision)
      assert r1.parsed.route == "simple"
      r2 = fc.complete(model="m", system="s", user="u", max_tokens=10)
      assert r2.text == "refined query"
      assert [c["method"] for c in fc.calls] == ["parse", "complete"]
      assert fc.calls[0]["output_format"] == "RouteDecision"


  def test_token_tuples_and_exhaustion():
      fc = FakeClaude(script=[("answer", 1000, 2000)])
      r = fc.complete(model="m", system="s", user="u", max_tokens=10)
      assert (r.input_tokens, r.output_tokens) == (1000, 2000)
      with pytest.raises(AssertionError):        # script ran dry -> loud failure
          fc.complete(model="m", system="s", user="u", max_tokens=10)


  def test_type_mismatch_fails_loud():
      fc = FakeClaude(script=["not a model"])
      with pytest.raises(AssertionError):
          fc.parse(model="m", system="s", user="u", max_tokens=10,
                   output_format=RouteDecision)


  def test_fake_core_scripts_and_records_queries():
      hit = [make_chunk(0)]
      core = FakeCore(by_query={"q1": hit})
      assert core.retrieve("q1") == hit
      assert len(core.retrieve("unknown")) == 2   # default corpus
      assert core.queries == ["q1", "unknown"]
  ```

- [ ] Run it: `uv run pytest tests/test_fakes_claude.py -q` → expected failure: `ImportError: cannot import name 'FakeCore' from 'tests.fakes'`. The three `FakeClaude` tests are executable verification of the R5 contract — once the import error is fixed they must pass against Plan A's fake as authored (only the `.calls` capability may be missing; see below).
- [ ] Append `make_chunk` + `FakeCore` to `api/tests/fakes.py` (keep `FakeEmbed`/`FakeRerank`/`FakeClaude` untouched):

  ```python
  # --- appended to api/tests/fakes.py ---
  from ragreceipts.types import Chunk, ScoredChunk


  def make_chunk(i: int, *, doc: str = "d1", text: str | None = None) -> ScoredChunk:
      """Tiny ScoredChunk fixture; chunk_id f'{doc}:{i}', passage_id == doc.

      start_token/end_token (R3) are consecutive positional ranges so span math
      stays valid: chunk i covers tokens [i*n, (i+1)*n) of its parent passage.
      """
      body = text or f"passage text {i}"
      n = len(body.split())
      return ScoredChunk(
          chunk=Chunk(chunk_id=f"{doc}:{i}", corpus_id="test", doc_id=doc,
                      passage_id=doc, text=body, position=i,
                      start_token=i * n, end_token=(i + 1) * n),
          score=1.0 / (i + 1), source="rrf",
      )


  class FakeCore:
      """Duck-types RetrievalCore.retrieve for agent-graph tests (no Qdrant, no keys).

      by_query maps exact query text -> scripted results; anything else returns the
      two-chunk default corpus. Records every query for transition assertions.
      """

      def __init__(self, by_query: dict[str, list[ScoredChunk]] | None = None,
                   default: list[ScoredChunk] | None = None):
          self.by_query = by_query or {}
          self.default = default if default is not None else [make_chunk(0), make_chunk(1)]
          self.queries: list[str] = []

      def retrieve(self, query: str) -> list[ScoredChunk]:
          self.queries.append(query)
          return self.by_query.get(query, list(self.default))
  ```

- [ ] Re-run: `uv run pytest tests/test_fakes_claude.py -q`. ONLY IF the two `FakeClaude` call-recording assertions fail because the unified `.calls` list is missing (Plan A's fake may record per-method `complete_calls`/`parse_calls` only — a missing capability, not a constructor change), extend `FakeClaude` additively. Do not touch the constructor, the script semantics, or any existing recording attribute:

  ```python
  # inside FakeClaude.__init__, add:
          self.calls: list[dict] = []
  # as the first statement of FakeClaude.complete(), add:
          self.calls.append({"method": "complete", "model": model,
                             "system": system, "user": user})
  # as the first statement of FakeClaude.parse(), add:
          self.calls.append({"method": "parse", "model": model,
                             "output_format": output_format.__name__,
                             "system": system, "user": user})
  ```

- [ ] Run the new test: `uv run pytest tests/test_fakes_claude.py -q` → expected: **4 passed**.
- [ ] Run the FULL suite: `uv run pytest -q` → green with zero changes outside `api/tests/fakes.py` (R5: no migration exists; Plan B has no `FakeClaude` call sites).
- [ ] Commit:

  ```bash
  git add api/tests/fakes.py api/tests/test_fakes_claude.py
  git commit -m "test(fakes): FakeCore and make_chunk for offline agent-graph tests

  Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
  ```

---

### Task 8: LangGraph state machine (full graph) + System-1 path tests

**Files:**
- Create: `api/ragreceipts/agents/graph.py`
- Test: `api/tests/test_graph_s1.py`

The state machine is one coherent unit, so `graph.py` lands complete in this task (a half-registered graph fails LangGraph compile validation). TDD discipline holds at the task level: the S1 tests are written first and fail on import; Tasks 9–10 add the System-2 behavioral coverage as test-only slices over this same implementation.

**Budget semantics (decided here, consistent across tasks):** every execution of `retrieve_hop` consumes one hop — including refine re-retrievals and the contradiction re-retrieve. `tokens_used` sums input+output of every Claude call in the trace (including `route`). Budgets are checked at decision points (`decompose`/`grade` bookkeeping), so the call that crosses the ceiling completes and the overshoot is at most one call — exhaustion then forces caveated synthesis with the not-yet-resolved sub-queries disclosed. Conditional-edge functions never mutate state (LangGraph routing functions are read-only); all bookkeeping happens in nodes, which set `next_action` for the edges to read.

- [ ] Write the failing test (the `run` helper here is reused by Tasks 9–10):

  ```python
  # api/tests/test_graph_s1.py
  from ragreceipts.agents.graph import build_graph, initial_state
  from ragreceipts.agents.schemas import FinalAnswer, RouteDecision
  from ragreceipts.traces.recorder import TraceRecorder
  from ragreceipts.traces.store import TraceStore
  from ragreceipts.types import RouteMode
  from tests.fakes import FakeClaude, FakeCore


  def run(tmp_path, script, route_mode, query="what is the capital of France?",
          core=None, **graph_kwargs):
      """Build graph with fakes, invoke once, return (final_state, store, core, claude)."""
      store = TraceStore(tmp_path / "t.sqlite3")
      recorder = TraceRecorder(store, "t-1")
      core = core or FakeCore()
      claude = FakeClaude(script=script)
      graph = build_graph(core=core, claude=claude, recorder=recorder,
                          route_mode=route_mode, **graph_kwargs)
      out = graph.invoke(initial_state(query), config={"recursion_limit": 50})
      return out, store, core, claude


  def test_simple_routes_to_s1(tmp_path):
      out, store, core, claude = run(
          tmp_path,
          [RouteDecision(route="simple", confidence=0.95),
           FinalAnswer(text="Paris [1]", citations=[1])],
          RouteMode.AUTO,
      )
      assert out["chosen_system"] == "s1"
      assert out["final"].text == "Paris [1]"
      assert core.queries == ["what is the capital of France?"]
      assert [e.node for e in store.get("t-1")] == ["route", "s1_retrieve", "s1_answer"]
      # route used Haiku, answer used Sonnet (contract model split)
      assert claude.calls[0]["model"] == "claude-haiku-4-5-20251001"
      assert claude.calls[1]["model"] == "claude-sonnet-4-6"


  def test_force_s1_skips_route_node(tmp_path):
      out, store, _, claude = run(
          tmp_path, [FinalAnswer(text="Paris [1]", citations=[1])], RouteMode.FORCE_S1)
      assert [e.node for e in store.get("t-1")] == ["s1_retrieve", "s1_answer"]
      assert claude.calls[0]["output_format"] == "FinalAnswer"   # no route call happened


  def test_s1_abstention_is_structured(tmp_path):
      out, store, _, _ = run(
          tmp_path,
          [RouteDecision(route="simple", confidence=0.9),
           FinalAnswer(text="The context does not mention this.", abstained=True)],
          RouteMode.AUTO,
      )
      assert out["final"].abstained is True
      assert out["final"].citations == []
      assert store.get("t-1")[-1].payload["abstained"] is True   # surfaced in trace too


  def test_tokens_accumulate_across_calls(tmp_path):
      out, _, _, _ = run(
          tmp_path,
          [(RouteDecision(route="simple", confidence=0.9), 100, 20),
           (FinalAnswer(text="Paris [1]", citations=[1]), 300, 50)],
          RouteMode.AUTO,
      )
      assert out["tokens_used"] == 470


  def test_trace_events_carry_chunk_scores(tmp_path):
      _, store, _, _ = run(
          tmp_path, [FinalAnswer(text="Paris [1]", citations=[1])], RouteMode.FORCE_S1)
      retrieve_event = store.get("t-1")[0]
      assert retrieve_event.node == "s1_retrieve"
      assert retrieve_event.payload["chunks"][0]["chunk_id"] == "d1:0"
      assert "score" in retrieve_event.payload["chunks"][0]
  ```

- [ ] Run it: `uv run pytest tests/test_graph_s1.py -q` → expected failure: `ModuleNotFoundError: No module named 'ragreceipts.agents.graph'`.
- [ ] Create `api/ragreceipts/agents/graph.py` — the complete state machine:

  ```python
  """LangGraph state machine: route -> System-1 fast path | System-2 agentic loop.

  Pure orchestration (spec boundary rule): retrieval happens only through the
  injected core's .retrieve(); every Claude call goes through ClaudeTransport.

  LangGraph API verified 2026-06-10 against
  https://docs.langchain.com/oss/python/langgraph/graph-api (langgraph 1.2.x):
  nodes return partial state updates; add_conditional_edges(source, fn, path_map);
  recursion_limit is a standalone top-level config key on invoke().
  """
  from __future__ import annotations

  import time
  from typing import Protocol, TypedDict

  from langgraph.graph import END, START, StateGraph

  from ragreceipts.agents import prompts
  from ragreceipts.agents.schemas import (
      FinalAnswer,
      GradeResult,
      RouteDecision,
      SubQueries,
  )
  from ragreceipts.constants import (
      ROUTE_CONFIDENCE_THRESHOLD,
      ROUTER_MODEL,
      S2_MAX_HOPS,
      S2_TOKEN_CEILING,
      SYNTH_MODEL,
  )
  from ragreceipts.traces.recorder import TraceRecorder
  from ragreceipts.types import RouteMode, ScoredChunk
  from ragreceipts.vendors.base import ClaudeTransport


  class SupportsRetrieve(Protocol):
      """Structural match for retrieval.core.RetrievalCore — tests inject fakes."""

      def retrieve(self, query: str) -> list[ScoredChunk]: ...


  class GraphState(TypedDict, total=False):
      query: str
      route: str                   # "simple" | "complex" (set by route node)
      confidence: float
      chosen_system: str           # "s1" | "s2"
      retrieved: list              # list[ScoredChunk] — S1 top-k
      subqueries: list[str]
      hop_index: int               # index of the sub-query currently being retrieved
      hop_records: list[dict]      # {"subquery","original","chunks","verdict"}
      hops_used: int               # every retrieve_hop execution counts (incl. retries)
      tokens_used: int             # input+output summed over every Claude call
      refined_query: str | None    # set by refine, consumed by next retrieve_hop
      contradiction_retried: bool  # the one re-retrieve attempt was used
      contradiction_flag: bool
      unresolved: list[str]
      budget_exhausted: bool
      next_action: str             # set by decompose/grade, read by conditional edges
      final: FinalAnswer | None


  def initial_state(query: str) -> GraphState:
      return GraphState(
          query=query, retrieved=[], subqueries=[], hop_index=0, hop_records=[],
          hops_used=0, tokens_used=0, refined_query=None,
          contradiction_retried=False, contradiction_flag=False, unresolved=[],
          budget_exhausted=False, final=None,
      )


  def _chunk_payload(chunks: list[ScoredChunk]) -> list[dict]:
      return [{"chunk_id": c.chunk.chunk_id, "score": c.score, "source": c.source}
              for c in chunks]


  def build_graph(
      *,
      core: SupportsRetrieve,
      claude: ClaudeTransport,
      recorder: TraceRecorder,
      route_mode: RouteMode = RouteMode.AUTO,
      confidence_threshold: float = ROUTE_CONFIDENCE_THRESHOLD,
      max_hops: int = S2_MAX_HOPS,
      token_ceiling: int = S2_TOKEN_CEILING,
  ):
      """Compile the query graph. Dependencies are closed over; state holds data only."""

      # ---------------------------------------------------------------- route
      def route_node(state: GraphState) -> dict:
          t0 = time.perf_counter()
          res = claude.parse(
              model=ROUTER_MODEL, system=prompts.ROUTE_SYSTEM,
              user=prompts.ROUTE_USER.format(query=state["query"]),
              max_tokens=1024, output_format=RouteDecision, temperature=0.0,
          )
          decision: RouteDecision = res.parsed
          recorder.emit(
              "route",
              {"query": state["query"], "route": decision.route,
               "confidence": decision.confidence},
              model=ROUTER_MODEL, input_tokens=res.input_tokens,
              output_tokens=res.output_tokens,
              duration_ms=(time.perf_counter() - t0) * 1000,
          )
          return {
              "route": decision.route, "confidence": decision.confidence,
              "tokens_used": state["tokens_used"] + res.input_tokens + res.output_tokens,
          }

      def after_route(state: GraphState) -> str:
          # Confidence is consumed, not decorative (spec): low confidence escalates.
          if state["route"] == "complex" or state["confidence"] < confidence_threshold:
              return "decompose"
          return "s1_retrieve"

      # ---------------------------------------------------------------- System-1
      def s1_retrieve_node(state: GraphState) -> dict:
          t0 = time.perf_counter()
          chunks = core.retrieve(state["query"])
          recorder.emit(
              "s1_retrieve",
              {"query": state["query"], "chunks": _chunk_payload(chunks)},
              duration_ms=(time.perf_counter() - t0) * 1000,
          )
          return {"retrieved": chunks, "chosen_system": "s1"}

      def s1_answer_node(state: GraphState) -> dict:
          t0 = time.perf_counter()
          context = prompts.format_numbered_context(state["retrieved"])
          res = claude.parse(
              model=SYNTH_MODEL, system=prompts.S1_ANSWER_SYSTEM,
              user=prompts.S1_ANSWER_USER.format(query=state["query"], context=context),
              max_tokens=4096, output_format=FinalAnswer, temperature=0.0,
          )
          final: FinalAnswer = res.parsed
          recorder.emit(
              "s1_answer",
              {"text": final.text, "citations": final.citations,
               "abstained": final.abstained},
              model=SYNTH_MODEL, input_tokens=res.input_tokens,
              output_tokens=res.output_tokens,
              duration_ms=(time.perf_counter() - t0) * 1000,
          )
          return {"final": final,
                  "tokens_used": state["tokens_used"] + res.input_tokens + res.output_tokens}

      # ---------------------------------------------------------------- System-2
      def decompose_node(state: GraphState) -> dict:
          t0 = time.perf_counter()
          res = claude.parse(
              model=ROUTER_MODEL, system=prompts.DECOMPOSE_SYSTEM,
              user=prompts.DECOMPOSE_USER.format(query=state["query"], max_hops=max_hops),
              max_tokens=1024, output_format=SubQueries, temperature=0.0,
          )
          items = list(res.parsed.items)[:max_hops]   # hard cap (spec S2 bound)
          tokens = state["tokens_used"] + res.input_tokens + res.output_tokens
          recorder.emit(
              "decompose",
              {"query": state["query"], "subqueries": items,
               "truncated": len(res.parsed.items) > max_hops},
              model=ROUTER_MODEL, input_tokens=res.input_tokens,
              output_tokens=res.output_tokens,
              duration_ms=(time.perf_counter() - t0) * 1000,
          )
          update: dict = {"subqueries": items, "hop_index": 0, "chosen_system": "s2",
                          "tokens_used": tokens, "next_action": "retrieve_hop"}
          if not items:
              # Degenerate decomposition: nothing to retrieve; synthesize will abstain.
              update["next_action"] = "synthesize"
          elif tokens >= token_ceiling:
              # Ceiling crossed before any retrieval: caveated synthesis, all unresolved.
              update.update(next_action="synthesize", budget_exhausted=True,
                            unresolved=items)
          return update

      def after_decompose(state: GraphState) -> str:
          return state["next_action"]

      def retrieve_hop_node(state: GraphState) -> dict:
          t0 = time.perf_counter()
          original = state["subqueries"][state["hop_index"]]
          sub = state["refined_query"] or original
          chunks = core.retrieve(sub)
          record = {"subquery": sub, "original": original, "chunks": chunks,
                    "verdict": None}
          recorder.emit(
              "retrieve_hop",
              {"hop_index": state["hop_index"], "subquery": sub,
               "chunks": _chunk_payload(chunks)},
              duration_ms=(time.perf_counter() - t0) * 1000,
          )
          return {"hop_records": state["hop_records"] + [record],
                  "hops_used": state["hops_used"] + 1,
                  "refined_query": None}

      def grade_node(state: GraphState) -> dict:
          t0 = time.perf_counter()
          record = state["hop_records"][-1]
          res = claude.parse(
              model=ROUTER_MODEL, system=prompts.GRADE_SYSTEM,
              user=prompts.GRADE_USER.format(
                  subquery=record["subquery"],
                  context=prompts.format_numbered_context(record["chunks"]),
              ),
              max_tokens=1024, output_format=GradeResult, temperature=0.0,
          )
          verdict = res.parsed.verdict
          tokens = state["tokens_used"] + res.input_tokens + res.output_tokens
          budget_ok = state["hops_used"] < max_hops and tokens < token_ceiling
          remaining = state["subqueries"][state["hop_index"] + 1:]
          update: dict = {
              "hop_records": state["hop_records"][:-1] + [dict(record, verdict=verdict)],
              "tokens_used": tokens,
          }

          def advance() -> None:
              if remaining and budget_ok:
                  update.update(next_action="retrieve_hop",
                                hop_index=state["hop_index"] + 1,
                                contradiction_retried=False)
              elif remaining:   # budget exhausted mid-plan: disclose the rest
                  update.update(next_action="synthesize", budget_exhausted=True,
                                unresolved=state["unresolved"] + remaining)
              else:
                  update["next_action"] = "synthesize"

          if verdict == "sufficient":
              advance()
          elif verdict == "insufficient":
              if budget_ok:
                  update["next_action"] = "refine"
              else:
                  update.update(
                      next_action="synthesize", budget_exhausted=True,
                      unresolved=state["unresolved"] + [record["original"]] + remaining)
          else:  # contradictory — one re-retrieve attempt, then flagged synthesis
              if not state["contradiction_retried"] and budget_ok:
                  update.update(next_action="retrieve_hop", contradiction_retried=True)
              else:
                  update["contradiction_flag"] = True
                  advance()

          recorder.emit(
              "grade",
              {"subquery": record["subquery"], "verdict": verdict,
               "next_action": update["next_action"], "budget_ok": budget_ok},
              model=ROUTER_MODEL, input_tokens=res.input_tokens,
              output_tokens=res.output_tokens,
              duration_ms=(time.perf_counter() - t0) * 1000,
          )
          return update

      def after_grade(state: GraphState) -> str:
          return state["next_action"]

      def refine_node(state: GraphState) -> dict:
          t0 = time.perf_counter()
          record = state["hop_records"][-1]
          res = claude.complete(
              model=ROUTER_MODEL, system=prompts.REFINE_SYSTEM,
              user=prompts.REFINE_USER.format(
                  subquery=record["subquery"],
                  context=prompts.format_numbered_context(record["chunks"]),
              ),
              max_tokens=1024, temperature=0.0,
          )
          refined = res.text.strip()
          recorder.emit(
              "refine", {"original": record["subquery"], "refined": refined},
              model=ROUTER_MODEL, input_tokens=res.input_tokens,
              output_tokens=res.output_tokens,
              duration_ms=(time.perf_counter() - t0) * 1000,
          )
          return {"refined_query": refined,
                  "tokens_used": state["tokens_used"] + res.input_tokens + res.output_tokens}

      def synthesize_node(state: GraphState) -> dict:
          t0 = time.perf_counter()
          context, _ordered = prompts.format_hop_context(state["hop_records"])
          unresolved = list(state["unresolved"])
          res = claude.parse(
              model=SYNTH_MODEL, system=prompts.SYNTHESIZE_SYSTEM,
              user=prompts.SYNTHESIZE_USER.format(
                  query=state["query"],
                  context=context or "(no evidence retrieved)",
                  unresolved=", ".join(unresolved) or "(none)",
                  contradiction="yes" if state["contradiction_flag"] else "no",
              ),
              max_tokens=4096, output_format=FinalAnswer, temperature=0.0,
          )
          final: FinalAnswer = res.parsed
          # State-enforced disclosure: never trust the model alone for budget or
          # contradiction flags (spec: flagged, never papered over).
          final = final.model_copy(update={
              "contradiction_flag": final.contradiction_flag or state["contradiction_flag"],
              "unresolved_subqueries": sorted(
                  set(final.unresolved_subqueries) | set(unresolved)),
          })
          recorder.emit(
              "synthesize",
              {"text": final.text, "citations": final.citations,
               "abstained": final.abstained,
               "contradiction_flag": final.contradiction_flag,
               "unresolved_subqueries": final.unresolved_subqueries,
               "budget_exhausted": state["budget_exhausted"]},
              model=SYNTH_MODEL, input_tokens=res.input_tokens,
              output_tokens=res.output_tokens,
              duration_ms=(time.perf_counter() - t0) * 1000,
          )
          return {"final": final,
                  "tokens_used": state["tokens_used"] + res.input_tokens + res.output_tokens}

      # ---------------------------------------------------------------- wiring
      def select_entry(state: GraphState) -> str:
          if route_mode is RouteMode.FORCE_S1:
              return "s1_retrieve"
          if route_mode is RouteMode.FORCE_S2:
              return "decompose"
          return "route"

      builder = StateGraph(GraphState)
      builder.add_node("route", route_node)
      builder.add_node("s1_retrieve", s1_retrieve_node)
      builder.add_node("s1_answer", s1_answer_node)
      builder.add_node("decompose", decompose_node)
      builder.add_node("retrieve_hop", retrieve_hop_node)
      builder.add_node("grade", grade_node)
      builder.add_node("refine", refine_node)
      builder.add_node("synthesize", synthesize_node)

      # Conditional entry keeps all nodes statically reachable in every mode.
      builder.add_conditional_edges(START, select_entry, {
          "route": "route", "s1_retrieve": "s1_retrieve", "decompose": "decompose"})
      builder.add_conditional_edges("route", after_route, {
          "s1_retrieve": "s1_retrieve", "decompose": "decompose"})
      builder.add_edge("s1_retrieve", "s1_answer")
      builder.add_edge("s1_answer", END)
      builder.add_conditional_edges("decompose", after_decompose, {
          "retrieve_hop": "retrieve_hop", "synthesize": "synthesize"})
      builder.add_edge("retrieve_hop", "grade")
      builder.add_conditional_edges("grade", after_grade, {
          "retrieve_hop": "retrieve_hop", "refine": "refine",
          "synthesize": "synthesize"})
      builder.add_edge("refine", "retrieve_hop")
      builder.add_edge("synthesize", END)
      return builder.compile()
  ```

- [ ] Run again: `uv run pytest tests/test_graph_s1.py -q` → expected: **5 passed**.
- [ ] Lint: `uv run ruff check ragreceipts/agents` → clean.
- [ ] Commit:

  ```bash
  git add api/ragreceipts/agents/graph.py api/tests/test_graph_s1.py
  git commit -m "feat(agents): LangGraph state machine with System-1 fast path and bounded System-2 loop

  Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
  ```

---

### Task 9: System-2 happy-path state-transition tests

**Files:**
- Test: `api/tests/test_graph_s2.py`

Test-only slice over Task 8's implementation. Expected to pass on first run; any failure means `decompose`/`grade` bookkeeping deviates from the scenario table in Task 8 — fix `graph.py` (most common defects: `budget_ok` off-by-one, `remaining` computed from the wrong `hop_index`) before proceeding.

- [ ] Write the tests:

  ```python
  # api/tests/test_graph_s2.py
  from ragreceipts.agents.schemas import FinalAnswer, GradeResult, RouteDecision, SubQueries
  from ragreceipts.types import RouteMode
  from tests.test_graph_s1 import run


  def test_complex_routes_to_s2_two_hops(tmp_path):
      script = [
          RouteDecision(route="complex", confidence=0.9),
          SubQueries(items=["who directed Film X", "what else did that director direct"]),
          GradeResult(verdict="sufficient"),
          GradeResult(verdict="sufficient"),
          FinalAnswer(text="Director Y [1]; also Film Z [3]", citations=[1, 3]),
      ]
      out, store, core, _ = run(tmp_path, script, RouteMode.AUTO, query="multi-hop?")
      assert out["chosen_system"] == "s2"
      assert out["hops_used"] == 2
      assert core.queries == ["who directed Film X",
                              "what else did that director direct"]
      assert [e.node for e in store.get("t-1")] == [
          "route", "decompose", "retrieve_hop", "grade",
          "retrieve_hop", "grade", "synthesize"]
      assert out["final"].unresolved_subqueries == []
      assert out["budget_exhausted"] is False


  def test_low_confidence_escalates_to_s2(tmp_path):
      script = [
          RouteDecision(route="simple", confidence=0.4),   # below 0.7 threshold
          SubQueries(items=["sq1"]),
          GradeResult(verdict="sufficient"),
          FinalAnswer(text="answer [1]", citations=[1]),
      ]
      out, store, _, _ = run(tmp_path, script, RouteMode.AUTO)
      assert out["chosen_system"] == "s2"
      assert store.get("t-1")[1].node == "decompose"       # route -> decompose


  def test_force_s2_skips_route(tmp_path):
      script = [SubQueries(items=["sq1"]), GradeResult(verdict="sufficient"),
                FinalAnswer(text="a [1]", citations=[1])]
      out, store, _, claude = run(tmp_path, script, RouteMode.FORCE_S2)
      assert store.get("t-1")[0].node == "decompose"
      assert claude.calls[0]["output_format"] == "SubQueries"


  def test_decompose_truncated_to_max_hops(tmp_path):
      script = [SubQueries(items=["a", "b", "c", "d", "e"]),
                GradeResult(verdict="sufficient"), GradeResult(verdict="sufficient"),
                GradeResult(verdict="sufficient"),
                FinalAnswer(text="x [1]", citations=[1])]
      out, store, _, _ = run(tmp_path, script, RouteMode.FORCE_S2)
      assert out["subqueries"] == ["a", "b", "c"]          # S2_MAX_HOPS = 3
      assert out["hops_used"] == 3
      assert store.get("t-1")[0].payload["truncated"] is True


  def test_empty_decomposition_goes_straight_to_synthesize(tmp_path):
      script = [SubQueries(items=[]),
                FinalAnswer(text="No retrievable sub-questions.", abstained=True)]
      out, store, core, _ = run(tmp_path, script, RouteMode.FORCE_S2)
      assert core.queries == []                            # nothing retrieved
      assert out["final"].abstained is True
      assert [e.node for e in store.get("t-1")] == ["decompose", "synthesize"]
  ```

- [ ] Run: `uv run pytest tests/test_graph_s2.py -q` → expected: **5 passed** (verifying Task 8's implementation; on failure, fix `graph.py`, not the test, unless the test contradicts the Task 8 scenario table).
- [ ] Commit:

  ```bash
  git add api/tests/test_graph_s2.py
  git commit -m "test(agents): System-2 happy-path state transitions incl. escalation and hop cap

  Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
  ```

---

### Task 10: System-2 edge cases — refine loop, contradiction, budget exhaustion

**Files:**
- Test: `api/tests/test_graph_s2_edges.py`

Covers the four contract-mandated edge transitions: insufficient→refine loop, contradictory→re-retrieve→flagged synthesis, hop-budget exhaustion→caveated synthesis, token-ceiling exhaustion→caveated synthesis. All offline via scripted `FakeClaude`.

- [ ] Write the tests:

  ```python
  # api/tests/test_graph_s2_edges.py
  from ragreceipts.agents.schemas import FinalAnswer, GradeResult, SubQueries
  from ragreceipts.types import RouteMode
  from tests.test_graph_s1 import run


  def test_insufficient_triggers_refine_loop(tmp_path):
      script = [
          SubQueries(items=["sq1"]),
          GradeResult(verdict="insufficient"),
          "sq1 rewritten with entities",            # refine goes through complete()
          GradeResult(verdict="sufficient"),
          FinalAnswer(text="a [1]", citations=[1]),
      ]
      out, store, core, _ = run(tmp_path, script, RouteMode.FORCE_S2)
      assert core.queries == ["sq1", "sq1 rewritten with entities"]
      assert [e.node for e in store.get("t-1")] == [
          "decompose", "retrieve_hop", "grade", "refine",
          "retrieve_hop", "grade", "synthesize"]
      assert out["final"].unresolved_subqueries == []
      refine_event = store.get("t-1")[3]
      assert refine_event.payload == {"original": "sq1",
                                      "refined": "sq1 rewritten with entities"}


  def test_contradictory_re_retrieves_once_then_flags(tmp_path):
      script = [
          SubQueries(items=["sq1"]),
          GradeResult(verdict="contradictory"),
          GradeResult(verdict="contradictory"),
          # Model "forgets" to set the flag — the graph must enforce it from state.
          FinalAnswer(text="Source A says 1990 [1]; source B says 1992 [2]",
                      citations=[1, 2]),
      ]
      out, store, core, _ = run(tmp_path, script, RouteMode.FORCE_S2)
      assert core.queries == ["sq1", "sq1"]               # exactly ONE re-retrieve
      assert out["final"].contradiction_flag is True       # state-enforced
      assert [e.node for e in store.get("t-1")] == [
          "decompose", "retrieve_hop", "grade",
          "retrieve_hop", "grade", "synthesize"]
      synth = store.get("t-1")[-1]
      assert synth.payload["contradiction_flag"] is True   # flagged in trace too


  def test_hop_budget_exhaustion_yields_caveated_synthesis(tmp_path):
      script = [
          SubQueries(items=["sq1", "sq2", "sq3"]),
          GradeResult(verdict="sufficient"),       # hop 1 (hops_used=1) ok
          GradeResult(verdict="insufficient"),     # hop 2 (hops_used=2) weak -> refine
          "sq2 rewritten",
          GradeResult(verdict="insufficient"),     # hops_used=3 == max -> stop
          FinalAnswer(text="partial answer [1]", citations=[1]),
      ]
      out, store, _, _ = run(tmp_path, script, RouteMode.FORCE_S2)
      assert out["hops_used"] == 3
      assert out["budget_exhausted"] is True
      # sq2 (original phrasing, not the refined one) and never-reached sq3 disclosed.
      assert out["final"].unresolved_subqueries == ["sq2", "sq3"]
      assert store.get("t-1")[-1].payload["budget_exhausted"] is True


  def test_token_ceiling_exhaustion(tmp_path):
      script = [
          (SubQueries(items=["sq1", "sq2"]), 50, 10),   # tokens 60 < 100 -> proceed
          (GradeResult(verdict="sufficient"), 80, 20),  # tokens 160 >= 100 -> stop
          (FinalAnswer(text="partial [1]", citations=[1]), 10, 10),
      ]
      out, _, core, _ = run(tmp_path, script, RouteMode.FORCE_S2, token_ceiling=100)
      assert core.queries == ["sq1"]                    # sq2 never retrieved
      assert out["budget_exhausted"] is True
      assert out["final"].unresolved_subqueries == ["sq2"]


  def test_token_ceiling_before_first_hop(tmp_path):
      script = [
          (SubQueries(items=["sq1", "sq2"]), 90, 20),   # tokens 110 >= 100 at decompose
          (FinalAnswer(text="cannot pursue sub-queries", abstained=True), 5, 5),
      ]
      out, store, core, _ = run(tmp_path, script, RouteMode.FORCE_S2,
                                token_ceiling=100)
      assert core.queries == []
      assert out["budget_exhausted"] is True
      assert out["final"].unresolved_subqueries == ["sq1", "sq2"]
      assert [e.node for e in store.get("t-1")] == ["decompose", "synthesize"]
  ```

- [ ] Run: `uv run pytest tests/test_graph_s2_edges.py -q` → expected: **5 passed** (on failure, debug `grade_node`'s `update` logic in `graph.py` against the Task 8 budget-semantics block).
- [ ] Run the whole agent suite together: `uv run pytest tests/test_graph_s1.py tests/test_graph_s2.py tests/test_graph_s2_edges.py -q` → all pass.
- [ ] Commit:

  ```bash
  git add api/tests/test_graph_s2_edges.py
  git commit -m "test(agents): refine loop, contradiction re-retrieve+flag, hop and token budget exhaustion

  Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
  ```

---

### Task 11: run_query service — GraphResult, union-of-hops, trace wiring

**Files:**
- Create: `api/ragreceipts/agents/service.py`
- Test: `api/tests/test_run_query.py`

The single entry point Plan D's `POST /query` and Task 12's eval runner will call. Trace wiring follows R9: `RetrievalCore` takes `on_trace` as a **constructor kwarg** (no private-attribute assignment, no discovery step), so `run_query` accepts either a ready core or a per-query factory — `run_query` invokes the factory with this query's `TraceRecorder`, and the factory constructs `RetrievalCore(..., on_trace=recorder)` itself. That lands Plan A's intra-retrieval events (per-retriever timings, degraded flags) on the same trace as the node-emitted events; the node-emitted `s1_retrieve`/`retrieve_hop` events (Task 8) already satisfy "every node emits TraceEvents" on their own, so callers passing a ready core (the eval runner) lose nothing required.

- [ ] Write the failing test:

  ```python
  # api/tests/test_run_query.py
  import dataclasses

  from ragreceipts.agents.schemas import (
      FinalAnswer,
      GradeResult,
      RouteDecision,
      SubQueries,
  )
  from ragreceipts.agents.service import GraphResult, route_counts, run_query
  from ragreceipts.config import PRESETS
  from ragreceipts.traces.store import TraceStore
  from ragreceipts.types import RouteMode
  from tests.fakes import FakeClaude, FakeCore, make_chunk


  def force_s2(config):
      return dataclasses.replace(
          config, query=dataclasses.replace(config.query,
                                            route_mode=RouteMode.FORCE_S2))


  def test_run_query_s1_result_and_trace(tmp_path):
      store = TraceStore(tmp_path / "t.sqlite3")
      core = FakeCore()
      claude = FakeClaude(script=[RouteDecision(route="simple", confidence=0.9),
                                  FinalAnswer(text="Paris [1]", citations=[1])])
      result = run_query(query="capital of France?", core=core, claude=claude,
                         store=store, config=PRESETS["router-on"])
      assert result.system == "s1"
      assert result.final.text == "Paris [1]"
      assert [s.chunk.chunk_id for s in result.retrieved] == ["d1:0", "d1:1"]
      events = store.get(result.trace_id)
      assert [e.node for e in events] == ["route", "s1_retrieve", "s1_answer"]
      assert [e.seq for e in events] == [0, 1, 2]
      assert result.tokens_used == 40              # 2 calls x default 10/10 tokens


  def test_run_query_s2_union_of_hops_dedupes(tmp_path):
      store = TraceStore(tmp_path / "t.sqlite3")
      a, b, c = make_chunk(0), make_chunk(1), make_chunk(2)
      core = FakeCore(by_query={"sq1": [a, b], "sq2": [b, c]})
      claude = FakeClaude(script=[SubQueries(items=["sq1", "sq2"]),
                                  GradeResult(verdict="sufficient"),
                                  GradeResult(verdict="sufficient"),
                                  FinalAnswer(text="x [1][3]", citations=[1, 3])])
      result = run_query(query="multi?", core=core, claude=claude, store=store,
                         config=force_s2(PRESETS["router-on"]))
      assert result.system == "s2"
      # union of per-hop top-k, deduped (b appears once), first-seen order
      assert [s.chunk.chunk_id for s in result.retrieved] == ["d1:0", "d1:1", "d1:2"]
      assert result.hops_used == 2


  def test_trace_ids_are_distinct_per_query(tmp_path):
      store = TraceStore(tmp_path / "t.sqlite3")
      script = [RouteDecision(route="simple", confidence=0.9),
                FinalAnswer(text="a [1]", citations=[1]),
                RouteDecision(route="simple", confidence=0.9),
                FinalAnswer(text="b [1]", citations=[1])]
      claude = FakeClaude(script=script)
      r1 = run_query(query="q1", core=FakeCore(), claude=claude, store=store,
                     config=PRESETS["router-on"])
      r2 = run_query(query="q2", core=FakeCore(), claude=claude, store=store,
                     config=PRESETS["router-on"])
      assert r1.trace_id != r2.trace_id
      assert len(store.get(r1.trace_id)) == 3 and len(store.get(r2.trace_id)) == 3


  def test_core_factory_receives_per_query_recorder(tmp_path):
      # R9: on_trace is wired at RetrievalCore CONSTRUCTION. run_query accepts a
      # per-query factory, calls it with this query's TraceRecorder, and the core
      # built with on_trace=recorder lands intra-retrieval events on the same trace.
      store = TraceStore(tmp_path / "t.sqlite3")
      captured = []

      class TracingCore(FakeCore):
          """Stands in for RetrievalCore(config, dense, sparse, stage, on_trace=...)."""

          def __init__(self, *, on_trace):
              super().__init__()
              self._on_trace = on_trace

          def retrieve(self, query):
              self._on_trace({"node": "s1_retrieve",
                              "payload": {"stage": "inner", "query": query}})
              return super().retrieve(query)

      def per_query_factory(recorder):
          captured.append(recorder)
          return TracingCore(on_trace=recorder)

      claude = FakeClaude(script=[FinalAnswer(text="a [1]", citations=[1])])
      result = run_query(query="q", core=per_query_factory, claude=claude,
                         store=store, config=PRESETS["rerank"])   # force_s1 preset
      assert captured[0].trace_id == result.trace_id
      events = store.get(result.trace_id)
      assert [(e.seq, e.node) for e in events] == [
          (0, "s1_retrieve"),   # the core's inner event, stamped by the recorder
          (1, "s1_retrieve"),   # the graph node's own event
          (2, "s1_answer"),
      ]
      assert events[0].payload == {"stage": "inner", "query": "q"}


  def test_route_counts():
      def fake_result(system: str) -> GraphResult:
          return GraphResult(final=FinalAnswer(text="x"), system=system,
                             trace_id="t", tokens_used=0, hops_used=0, retrieved=[])
      assert route_counts([fake_result("s1"), fake_result("s2"),
                           fake_result("s2")]) == {"n_s1": 1, "n_s2": 2}
      assert route_counts([]) == {"n_s1": 0, "n_s2": 0}
  ```

- [ ] Run it: `uv run pytest tests/test_run_query.py -q` → expected failure: `ModuleNotFoundError: No module named 'ragreceipts.agents.service'`.
- [ ] Create `api/ragreceipts/agents/service.py`:

  ```python
  """run_query: the one entry point that drives the graph for a single query.

  Used by the eval runner (Plan C Task 12) and the FastAPI POST /query (Plan D),
  so both execute the identical retrieval+agent code path (spec invariant).
  """
  from __future__ import annotations

  import uuid
  from dataclasses import dataclass
  from typing import Callable, Iterable

  from ragreceipts.agents.graph import SupportsRetrieve, build_graph, initial_state
  from ragreceipts.agents.schemas import FinalAnswer
  from ragreceipts.config import PipelineConfig
  from ragreceipts.traces.recorder import TraceRecorder
  from ragreceipts.traces.store import TraceStore
  from ragreceipts.types import ScoredChunk
  from ragreceipts.vendors.base import ClaudeTransport

  # R9: RetrievalCore's trace wiring is the constructor kwarg `on_trace` — never a
  # private-attribute assignment. A caller that wants the core's intra-retrieval
  # events (per-retriever timings, degraded flags) on a query's trace passes a
  # per-query factory: run_query invokes it with this query's TraceRecorder and
  # the factory constructs RetrievalCore(..., on_trace=recorder) itself.
  type CoreOrFactory = SupportsRetrieve | Callable[[TraceRecorder], SupportsRetrieve]


  @dataclass(frozen=True)
  class GraphResult:
      final: FinalAnswer
      system: str                    # "s1" | "s2"
      trace_id: str
      tokens_used: int
      hops_used: int                 # 0 on the S1 path
      retrieved: list[ScoredChunk]   # S1 top-k, or S2 union-of-hops (deduped,
                                     # first-seen order) for the eval diagnostic


  def union_of_hops(hop_records: list[dict]) -> list[ScoredChunk]:
      seen: set[str] = set()
      out: list[ScoredChunk] = []
      for rec in hop_records:
          for sc in rec["chunks"]:
              if sc.chunk.chunk_id not in seen:
                  seen.add(sc.chunk.chunk_id)
                  out.append(sc)
      return out


  def run_query(*, query: str, core: CoreOrFactory, claude: ClaudeTransport,
                store: TraceStore, config: PipelineConfig,
                trace_id: str | None = None) -> GraphResult:
      trace_id = trace_id or uuid.uuid4().hex
      recorder = TraceRecorder(store, trace_id)
      if not hasattr(core, "retrieve"):
          core = core(recorder)      # per-query factory (see CoreOrFactory above)
      graph = build_graph(core=core, claude=claude, recorder=recorder,
                          route_mode=config.query.route_mode)
      out = graph.invoke(initial_state(query), config={"recursion_limit": 50})
      system = out.get("chosen_system", "s1")
      retrieved = (out["retrieved"] if system == "s1"
                   else union_of_hops(out["hop_records"]))
      return GraphResult(final=out["final"], system=system, trace_id=trace_id,
                         tokens_used=out["tokens_used"], hops_used=out["hops_used"],
                         retrieved=retrieved)


  def route_counts(results: Iterable[GraphResult]) -> dict[str, int]:
      """Receipt route-distribution stats: {"n_s1": ..., "n_s2": ...}."""
      rs = list(results)
      n_s2 = sum(1 for r in rs if r.system == "s2")
      return {"n_s1": len(rs) - n_s2, "n_s2": n_s2}
  ```

- [ ] Run again: `uv run pytest tests/test_run_query.py -q` → expected: **5 passed**.
- [ ] Full suite: `uv run pytest -q` → green.
- [ ] Commit:

  ```bash
  git add api/ragreceipts/agents/service.py api/tests/test_run_query.py
  git commit -m "feat(agents): run_query service with per-query traces and union-of-hops result

  Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
  ```

---

### Task 12: Eval integration — runner drives the graph, router-on preset live

**Files:**
- Modify: `api/ragreceipts/eval/runner.py` (R9-pinned names: `AblationRunner`, `_run_preset`, `estimate_run_cost`)
- Modify: `api/ragreceipts/eval/run_state.py` (persist per-query `route`)
- Modify: `api/tests/test_runner.py`, `api/tests/test_harness_selftest.py` (graph-driven stubs)
- Test: `api/tests/test_eval_router_on.py`

**Outcome (binding):** the runner's per-query answer generation goes through `run_query` for every preset (S1-only presets run `force_s1` through the graph — identical retrieval+synthesis path, satisfying the shared-core invariant). Of the runner's two `router-on` gates (R10), ONLY the temporary "requires Plan C" skip is deleted; the permanent `MULTI_HOP_DATASETS` gate stays and is re-tested. Receipts gain `metrics["n_s1"]` / `metrics["n_s2"]` and per-query `"route"`; for runs containing any S2 query, retrieval recall is reported as the secondary union-of-hops diagnostic flagged `union_of_hops: true`, with `recall_at_5`/`mrr_at_3` nulled (ill-defined across decomposed hops, per contract). `prompts_version` flips from Plan B's `"n/a"` to `agents.prompts.PROMPTS_VERSION` (R11). `estimate_run_cost` gains a System-2 estimate for AUTO presets, and actual per-query usd is computed from TraceStore events' `(model, input_tokens, output_tokens)` (R10). No grep discovery anywhere in this task: every seam name used below is pinned by R9.

- [ ] Write the failing integration test — create `api/tests/test_eval_router_on.py` (complete file; the corpus fixture follows Spike 0's R1 raw layout: `raw/queries.jsonl` with typed golds plus the slice-id files, never "first N lines"):

  ```python
  # api/tests/test_eval_router_on.py
  """Router-on eval integration: the graph-driven runner produces route stats.

  Offline: FakeClaude scripts every Claude call; FakeCore replaces retrieval;
  the corpus fixture is written in Spike 0's raw/ layout (R1/R2).
  """
  import json
  from pathlib import Path

  import pytest

  from ragreceipts.agents.prompts import PROMPTS_VERSION
  from ragreceipts.agents.schemas import (
      FinalAnswer,
      GradeResult,
      RouteDecision,
      SubQueries,
  )
  from ragreceipts.eval.run_state import RunStore
  from ragreceipts.eval.runner import AblationRunner
  from ragreceipts.traces.store import TraceStore
  from tests.fakes import FakeClaude, FakeCore, make_chunk


  def write_corpus(tmp_path: Path, dataset_name: str = "musique") -> Path:
      """R1 raw layout: raw/{queries.jsonl, slice-*.json} + manifest.json."""
      raw = tmp_path / "corpora" / "c1" / "raw"
      raw.mkdir(parents=True)
      records = [
          {"query_id": "q0", "question": "question 0?", "answer": "answer zero",
           "answer_aliases": [],
           "gold": {"type": "passage", "passage_ids": ["p0"]}},
          {"query_id": "q1", "question": "question 1?", "answer": "answer two",
           "answer_aliases": [],
           "gold": {"type": "passage", "passage_ids": ["p1"]}},
      ]
      (raw / "queries.jsonl").write_text(
          "\n".join(json.dumps(r) for r in records) + "\n")
      (raw / "slice-full.json").write_text(json.dumps(["q0", "q1"]))
      (raw / "slice-smoke.json").write_text(json.dumps(["q0", "q1"]))
      (tmp_path / "corpora" / "c1" / "manifest.json").write_text(json.dumps({
          "corpus_id": "c1",
          "dataset": {"name": dataset_name, "hf_id": "x", "split": "dev",
                      "revision": "r"},
          "index_hashes": {"dense_contextual": "sha256:c",
                           "dense_isolated": "sha256:i", "sparse": "sha256:s"},
          "n_queries": 2,
      }))
      return tmp_path


  def make_runner(tmp_path: Path, claude: FakeClaude,
                  dataset_name: str = "musique") -> AblationRunner:
      data_dir = write_corpus(tmp_path, dataset_name)
      core = FakeCore(by_query={
          "question 0?": [make_chunk(0, doc="p0")],          # S1: gold p0 at rank 1
          "hop one": [make_chunk(0, doc="p1"), make_chunk(1, doc="f1")],
          "hop two": [make_chunk(0, doc="f2")],              # union still holds gold p1
      })
      return AblationRunner(
          core_factory=lambda cfg: core,
          claude=claude,
          store=RunStore(tmp_path / "runs.db"),
          data_dir=data_dir,
          trace_store=TraceStore(tmp_path / "traces.sqlite3"),
      )


  def router_on_script() -> list:
      return [
          # q0 -> S1
          RouteDecision(route="simple", confidence=0.95),
          FinalAnswer(text="answer one [1]", citations=[1]),
          # q1 -> S2, two hops, both sufficient
          RouteDecision(route="complex", confidence=0.9),
          SubQueries(items=["hop one", "hop two"]),
          GradeResult(verdict="sufficient"),
          GradeResult(verdict="sufficient"),
          FinalAnswer(text="answer two [1][3]", citations=[1, 3]),
      ]


  def test_router_on_preset_produces_route_stats(tmp_path: Path) -> None:
      runner = make_runner(tmp_path, FakeClaude(script=router_on_script()))
      doc = runner.run(run_id="r1", corpus_id="c1", slice_name="smoke",
                       presets=["router-on"], spend_cap_usd=5.0)
      assert doc["skipped"] == []                  # the temporary skip is gone
      receipt = doc["receipts"][0]["receipt"]
      assert receipt["preset"] == "router-on"
      m = receipt["metrics"]
      assert m["n_s1"] + m["n_s2"] == receipt["n_total"]
      assert m["n_s2"] >= 1
      assert m.get("union_of_hops") is True
      assert m["recall_at_5"] is None              # ill-defined across hops
      assert m["mrr_at_3"] is None
      assert m["recall_union_of_hops"] == pytest.approx(1.0)
      assert m["usd_per_query"] > 0                # R10: priced from TraceEvents
      assert all("route" in pq for pq in receipt["per_query"])
      assert {pq["route"] for pq in receipt["per_query"]} == {"s1", "s2"}
      assert receipt["prompts_version"] == PROMPTS_VERSION   # R11


  def test_router_on_still_skips_on_single_hop_corpus(tmp_path: Path) -> None:
      # R10: the MULTI_HOP_DATASETS gate is permanent — nq corpora never run AUTO.
      runner = make_runner(tmp_path, FakeClaude(script=[]), dataset_name="nq")
      doc = runner.run(run_id="r1", corpus_id="c1", slice_name="smoke",
                       presets=["router-on"], spend_cap_usd=5.0)
      assert doc["receipts"] == []
      assert doc["skipped"][0]["preset"] == "router-on"
      assert "multi-hop" in doc["skipped"][0]["reason"]
  ```

- [ ] Run it: `uv run pytest tests/test_eval_router_on.py -q` → expected failure: `TypeError: AblationRunner.__init__() got an unexpected keyword argument 'trace_store'`.
- [ ] Give the runner its `TraceStore` (`api/ragreceipts/eval/runner.py`). New imports at the top:

  ```python
  from ragreceipts.agents.prompts import PROMPTS_VERSION
  from ragreceipts.agents.service import run_query
  from ragreceipts.traces.store import TraceStore
  ```

  and add `S2_MAX_HOPS` to the existing `from ragreceipts.constants import (...)` block. Then extend `AblationRunner.__init__` — the only signature change is the trailing kwarg (`run(...)` keeps its signature; the store rides on the instance):

  ```python
      def __init__(
          self,
          *,
          core_factory: Callable[[PipelineConfig], RetrievalCore],
          claude: ClaudeTransport,
          store: RunStore,
          data_dir: Path,
          ragas: RagasJudge | None = None,
          clock: Callable[[], float] = time.perf_counter,
          trace_store: TraceStore | None = None,
      ) -> None:
          self._core_factory = core_factory
          self._claude = claude
          self._store = store
          self._data_dir = data_dir
          self._ragas = ragas
          self._clock = clock
          # Default resolves next to the corpora dirs (the same data_dir the
          # runner already uses); tests pass a tmp_path-backed store.
          self._trace_store = trace_store or TraceStore(data_dir / "traces-eval.sqlite3")
  ```

- [ ] In `AblationRunner.run`, delete ONLY the temporary "requires Plan C" skip arm; the `MULTI_HOP_DATASETS` gate is permanent (R10). The preset loop becomes:

  ```python
          for name in presets:
              cfg = PRESETS[name]
              if cfg.query.route_mode is not RouteMode.FORCE_S1:
                  # R10: permanent gate — AUTO presets run on multi-hop corpora only.
                  # (The temporary "requires Plan C" skip that used to follow is gone.)
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
              self._run_preset(
                  run_id=run_id, cfg=cfg, queries=queries, spend_cap_usd=spend_cap_usd
              )
              receipt = self._build_receipt(
                  run_id=run_id, corpus_id=corpus_id, cfg=cfg, manifest=manifest,
                  queries=queries, results_by_preset=results_by_preset,
              )
              results_by_preset[name] = receipt.metrics
              receipts.append(receipt)
  ```

- [ ] Replace `_run_preset` wholesale — the retrieval + `synthesize()` block becomes one `run_query` call, and actual usd comes from this query's TraceEvents (R10). The complete replacement method:

  ```python
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
                  result = run_query(query=q.question, core=core, claude=self._claude,
                                     store=self._trace_store, config=cfg)
                  latency_ms = (self._clock() - t0) * 1000.0
                  events = self._trace_store.get(result.trace_id)
                  tin = sum(e.input_tokens for e in events)
                  tout = sum(e.output_tokens for e in events)
                  # R10: actual per-query usd from traced (model, in, out) tokens.
                  usd = sum(
                      usd_for_tokens(e.model, e.input_tokens, e.output_tokens)
                      for e in events
                      if e.model is not None
                  )
                  n_retrievals = result.hops_used if result.system == "s2" else 1
                  if cfg.query.rerank:
                      usd += usd_for_rerank(n_retrievals)
                  if cfg.query.dense:
                      usd += usd_for_tokens(
                          EMBED_MODEL, n_retrievals * EST_QUERY_EMBED_TOKENS, 0)
                  self._store.record_result(
                      run_id=run_id, preset=cfg.name, query_id=q.query_id,
                      status="abstained" if result.final.abstained else "ok",
                      retrieved=[
                          {
                              "chunk_id": sc.chunk.chunk_id,
                              "passage_id": sc.chunk.passage_id,
                              "start_token": sc.chunk.start_token,   # R3: span-gold
                              "end_token": sc.chunk.end_token,       # hits stay computable
                              "text": sc.chunk.text,
                          }
                          for sc in result.retrieved
                      ],
                      answer=result.final.text, latency_ms=latency_ms, usd=usd,
                      input_tokens=tin, output_tokens=tout, error=None,
                      route=result.system,
                  )
              except Exception as exc:  # disclosed, never batch-fatal
                  latency_ms = (self._clock() - t0) * 1000.0
                  self._store.record_result(
                      run_id=run_id, preset=cfg.name, query_id=q.query_id,
                      status="failed", retrieved=[], answer=None,
                      latency_ms=latency_ms, usd=0.0, input_tokens=0,
                      output_tokens=0, error=repr(exc), route=None,
                  )
  ```

  Then delete the now-dead `synthesize()` helper, the `S1Answer` model, and the `S1_SYSTEM` prompt from `runner.py`, plus the now-unused `from pydantic import BaseModel` and `ScoredChunk` imports (the graph's `agents/prompts.py::S1_ANSWER_SYSTEM` from Task 6 carries the same citation + structured-abstention rules, so no behavior is lost).

- [ ] Persist the per-query route in `api/ragreceipts/eval/run_state.py` so receipts stay correct across resumed runs. Three edits — the `eval_results` schema gains a trailing `route` column:

  ```sql
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
    route TEXT,                        -- 's1' | 's2' | NULL (failed before routing)
    PRIMARY KEY (run_id, preset, query_id)
  );
  ```

  `record_result` gains the kwarg and the twelfth placeholder:

  ```python
      def record_result(self, *, run_id: str, preset: str, query_id: str, status: str,
                        retrieved: list[dict], answer: str | None, latency_ms: float,
                        usd: float, input_tokens: int, output_tokens: int,
                        error: str | None, route: str | None = None) -> None:
          self._conn.execute(
              "INSERT OR REPLACE INTO eval_results "
              "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
              (run_id, preset, query_id, status, json.dumps(retrieved), answer,
               latency_ms, usd, input_tokens, output_tokens, error, route),
          )
          self._conn.commit()
  ```

  and `results_for` selects + returns it:

  ```python
      def results_for(self, run_id: str, preset: str) -> list[dict]:
          rows = self._conn.execute(
              "SELECT query_id, status, retrieved, answer, latency_ms, usd, "
              "input_tokens, output_tokens, error, route "
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
                  "route": r[9],
              }
              for r in rows
          ]
  ```

- [ ] Extend `estimate_run_cost` with the System-2 estimate (R10). New constants next to the existing `EST_*` block, then the complete replacement function:

  ```python
  # System-2 estimate inputs (R10): route + decompose + one grade per hop, all on
  # Haiku; synthesis is the Sonnet base already counted below.
  EST_S2_HAIKU_CALLS = 2 + S2_MAX_HOPS    # route + decompose + S2_MAX_HOPS grades
  EST_S2_HAIKU_INPUT_TOKENS = 1_200       # ~5 chunks x 200 tokens + prompt
  EST_S2_HAIKU_OUTPUT_TOKENS = 100
  ```

  ```python
  def estimate_run_cost(
      preset_names: list[str], n_queries: int, *, ragas: bool = False
  ) -> float:
      """Pre-run cost estimate (spec: estimate + confirmation gate + hard cap).

      AUTO presets are priced as a System-2 upper bound (R10): every query
      escalates and spends all S2_MAX_HOPS hops. ragas=True keeps Plan B's
      per-ok-query judge heuristic (conservative; the HARD CAP still meters
      only tracked spend — RAGAS judge usage stays untracked and disclosed).
      """
      total = 0.0
      for name in preset_names:
          cfg = PRESETS[name]
          per_q = usd_for_tokens(
              SYNTH_MODEL, EST_SYNTH_INPUT_TOKENS, EST_SYNTH_OUTPUT_TOKENS
          )
          if cfg.query.dense:
              per_q += usd_for_tokens(EMBED_MODEL, EST_QUERY_EMBED_TOKENS, 0)
          if cfg.query.rerank:
              per_q += usd_for_rerank(1)
          if cfg.query.route_mode is not RouteMode.FORCE_S1:
              per_q += EST_S2_HAIKU_CALLS * usd_for_tokens(
                  ROUTER_MODEL, EST_S2_HAIKU_INPUT_TOKENS, EST_S2_HAIKU_OUTPUT_TOKENS
              )
              extra_hops = S2_MAX_HOPS - 1   # the first hop's retrieval is the base
              if cfg.query.dense:
                  per_q += extra_hops * usd_for_tokens(
                      EMBED_MODEL, EST_QUERY_EMBED_TOKENS, 0
                  )
              if cfg.query.rerank:
                  per_q += extra_hops * usd_for_rerank(1)
          if ragas:
              per_q += usd_for_tokens(
                  JUDGE_MODEL, EST_RAGAS_INPUT_TOKENS, EST_RAGAS_OUTPUT_TOKENS
              )
          total += per_q * n_queries
      return total
  ```

  The signature keeps Plan B's `*, ragas: bool = False` keyword and judge-heuristic
  block verbatim (Plan B's CLI calls `estimate_run_cost(..., ragas=args.ragas)`; R10
  says the function *gains* the System-2 estimate, it does not lose the ragas one).
  Only the `route_mode` branch is new — it replaces Plan B's temporary
  `continue`-skip for AUTO presets.

  ```python
  # imports already present in runner.py from Plan B; JUDGE_MODEL and the
  # EST_RAGAS_* constants are Plan B's — do not redefine them.
  ```

- [ ] In `_build_receipt`, three edits. First, route stats + the union-of-hops disclosure immediately after the `metrics = {...}` dict is built (Receipt's `metrics`/`per_query` are open dicts — add keys, never dataclass fields):

  ```python
          n_s1 = sum(1 for r in rows if r.get("route") == "s1")
          n_s2 = sum(1 for r in rows if r.get("route") == "s2")
          metrics["n_s1"] = n_s1
          metrics["n_s2"] = n_s2
          if n_s2 > 0:
              # Contract: router-on retrieval recall is a secondary diagnostic over
              # the union of per-hop top-5, flagged union_of_hops; primary metrics
              # are EM/F1 + RAGAS.
              metrics["union_of_hops"] = True
              metrics["recall_union_of_hops"] = metrics["recall_at_5"]
              metrics["recall_at_5"] = None
              metrics["mrr_at_3"] = None
  ```

  Second, add `"route"` to each per-query record in the existing `per_query` loop:

  ```python
              per_query.append({
                  "query_id": r["query_id"],
                  "retrieved_chunk_ids": [d["chunk_id"] for d in r["retrieved"]],
                  "answer": r["answer"],
                  "latency_ms": r["latency_ms"],
                  "usd": r["usd"],
                  "route": r.get("route"),
                  "flags": flags,
              })
  ```

  Third, the one-line R11 change in the `Receipt(...)` constructor:

  ```python
              prompts_version=PROMPTS_VERSION,   # was: prompts_version="n/a"
  ```

  The existing recall/MRR computation needs no change: it already scores whatever `retrieved` list each query stored (the union for S2 queries), and the relabeling above discloses the semantics.

- [ ] Run the integration test: `uv run pytest tests/test_eval_router_on.py -q` → expected: **2 passed**.
- [ ] Update `api/tests/test_runner.py` for the graph-driven path. Plan B already synthesized via `parse()`; what changes is the response model (`S1Answer` → `FinalAnswer`) and the user-prompt layout (the question now sits on the FIRST line of `prompts.S1_ANSWER_USER`). Four mechanical edits:

  1. DELETE `test_router_on_skipped_requires_plan_c` — its skip no longer exists. KEEP `test_router_on_skipped_on_simple_corpus` unchanged: it pins the permanent R10 gate.
  2. Replace `StubClaude` and the `answers` values:

     ```python
     class StubClaude:
         """ClaudeTransport stub for the graph's S1 path; answers keyed by question.

         force_s1 presets enter the graph at s1_retrieve, so the only Claude call
         per query is s1_answer: parse(output_format=FinalAnswer) with the
         S1_ANSWER_USER prompt ('Question: {query}\\n\\nContext passages:\\n{context}').
         """

         def __init__(self, answers: dict[str, FinalAnswer]) -> None:
             self._answers = answers
             self.parse_calls = 0

         def complete(self, *, model, system, user, max_tokens, temperature=0.0):
             raise AssertionError("the S1 graph path uses parse(), not complete()")

         def parse(self, *, model, system, user, max_tokens, output_format,
                   temperature=0.0):
             self.parse_calls += 1
             question = user.split("Question: ", 1)[1].split("\n", 1)[0]
             return ParsedResult(
                 parsed=self._answers[question], input_tokens=1000, output_tokens=100
             )
     ```

     ```python
     answers = {
         "question 0?": FinalAnswer(text="Answer 0 [1]", citations=[1]),
         "question 1?": FinalAnswer(
             text="The passages do not contain this.", abstained=abstain_q1
         ),
     }
     ```

     Import changes: drop `S1Answer` from the `ragreceipts.eval.runner` import and add `from ragreceipts.agents.schemas import FinalAnswer`. Collapse any `try/except ImportError` fake-import shim to the plain `from tests.fakes import ...` form (R8). The usd/token assertions survive untouched: the graph's `s1_answer` is still exactly one Sonnet `parse()` per query at 1000/100 tokens.
  3. Replace the `router-on` line of `test_estimate_run_cost_hand_computed`:

     ```python
     # router-on/query (R10 S2 upper bound): rerank base 0.0169072
     #   + 5 haiku calls x (1200 x $1/M + 100 x $5/M = 0.0017) = 0.0085
     #   + 2 extra hops x (embed 0.0000072 + rerank 0.0025)    = 0.0050144
     #   = 0.0304216
     assert estimate_run_cost(["router-on"], 100) == pytest.approx(3.04216)
     ```

     And replace the AUTO-preset line of
     `test_estimate_includes_ragas_judge_heuristic_when_enabled` (its
     `== 0.0` assertion encoded the deleted temporary skip):

     ```python
     # router-on with ragas: S2 upper bound 0.0304216 + judge 0.0195
     #   = 0.0499216/query -> x100 = 4.99216
     assert estimate_run_cost(["router-on"], 100, ragas=True) == pytest.approx(4.99216)
     ```

     The `bm25-only` ragas assertion in that test survives unchanged
     (0.339 — the S1 path and judge heuristic are untouched).

  4. Any Plan B assertion that a runner-produced receipt has `prompts_version == "n/a"` becomes `== PROMPTS_VERSION` (import `from ragreceipts.agents.prompts import PROMPTS_VERSION`); Receipts constructed directly in `test_receipts.py` fixtures keep whatever literal they pass.

- [ ] Update `api/tests/test_harness_selftest.py` the same way. `EchoClaude` becomes route-aware (the harness fixture's dataset is `"musique"`, so the permanent gate now ADMITS `router-on` — the ladder's disclosed-skip assertion is replaced by a live router-on cell where every query routes `"simple"`):

  ```python
  class EchoClaude:
      """ClaudeTransport stub: routes everything 'simple', then answers each
      fixture question with its gold answer (graph S1 path)."""

      def complete(self, *, model, system, user, max_tokens, temperature=0.0):
          raise AssertionError("self-test synthesis uses parse(), not complete()")

      def parse(self, *, model, system, user, max_tokens, output_format,
                temperature=0.0):
          if output_format is RouteDecision:
              return ParsedResult(
                  parsed=RouteDecision(route="simple", confidence=0.95),
                  input_tokens=50, output_tokens=10,
              )
          question = user.split("Question: ", 1)[1].split("\n", 1)[0]
          i = question.split()[-1].rstrip("?")
          return ParsedResult(
              parsed=FinalAnswer(text=f"gold answer {i}", citations=[1]),
              input_tokens=500, output_tokens=50,
          )
  ```

  ```python
  def test_full_ladder_runs_offline_all_presets(tmp_path: Path) -> None:
      runner = make_runner(tmp_path)
      doc = runner.run(
          run_id="ladder", corpus_id="harness", slice_name="smoke",
          presets=["bm25-only", "dense-rrf", "contextual", "rerank", "router-on"],
          spend_cap_usd=5.0,
      )
      assert [e["receipt"]["preset"] for e in doc["receipts"]] == [
          "bm25-only", "dense-rrf", "contextual", "rerank", "router-on",
      ]
      assert doc["skipped"] == []          # musique fixture passes the R10 gate
      m = metrics_for(doc, "router-on")
      assert (m["n_s1"], m["n_s2"]) == (4, 0)
      # no S2 query -> primary retrieval metrics keep their normal semantics
      assert m["recall_at_5"] == pytest.approx(1.0)
      assert "union_of_hops" not in m
      assert (tmp_path / "receipts-local" / "ladder.json").exists()
  ```

  (renames `test_full_ladder_runs_offline_with_disclosed_skip`; import changes mirror step 2: drop `S1Answer`, add `from ragreceipts.agents.schemas import FinalAnswer, RouteDecision`, collapse the import shim per R8.)

- [ ] Run everything: `uv run pytest -q` → green. Plan B's force_s1 preset tests now go through the graph (`s1_retrieve` → `s1_answer`); answers are unchanged in shape (`text`, `abstained`), so metric assertions hold as updated above.
- [ ] Commit:

  ```bash
  git add api/ragreceipts/eval api/tests
  git commit -m "feat(eval): drive generation through the agent graph; enable router-on with route stats and union-of-hops disclosure

  Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
  ```

---

### Task 13: Live System-2 smoke script (manual, keyed, never CI)

**Files:**
- Create: `api/scripts/smoke_s2.py`

One real multi-hop query through the full S2 path with live Anthropic/Voyage/Cohere calls. Manual/nightly use only — it is a script, not a test; nothing under `api/tests/` references it, so CI stays offline and keyless.

- [ ] The core comes from Plan B's composition root — `cli.py::_build_core_real(config, corpus_id, data_dir)` (name and signature pinned by R9). It is the same factory the eval CLI uses, so the smoke exercises the production construction path; there is nothing to discover. Data-dir resolution follows R6: `RAGRECEIPTS_DATA_DIR` env var, default `../data` relative to `api/`.
- [ ] Create `api/scripts/smoke_s2.py`:

  ```python
  """Live System-2 smoke — MANUAL ONLY, never CI.

  Sends ONE real multi-hop query through the LangGraph System-2 path against an
  ingested corpus (real Anthropic/Voyage/Cohere calls), then prints the answer and
  the full trace with per-node timings and token counts.

  Prerequisites:
    - an ingested corpus; Qdrant running (docker compose up qdrant) with
      QDRANT_URL set, or QDRANT_URL unset to use the CLI's local-file fallback
      at {data_dir}/qdrant-local (R7)
    - ANTHROPIC_API_KEY, VOYAGE_API_KEY, COHERE_API_KEY set (one .env, spec)

  Usage (from rag-receipts/api/):
    uv run python scripts/smoke_s2.py --corpus musique-dev-300 \\
        --query "Who is the spouse of the director of the film Parasite?"
  """
  from __future__ import annotations

  import argparse
  import dataclasses
  import os
  import uuid
  from pathlib import Path

  from ragreceipts.agents.service import run_query
  from ragreceipts.cli import _build_core_real  # Plan B's composition root (R9)
  from ragreceipts.config import PRESETS
  from ragreceipts.traces.store import TraceStore
  from ragreceipts.types import RouteMode
  from ragreceipts.vendors.anthropic_client import AnthropicClient

  # R6 data-dir resolution: RAGRECEIPTS_DATA_DIR env var, default ../data from api/.
  DATA_DIR = Path(os.environ.get("RAGRECEIPTS_DATA_DIR")
                  or Path(__file__).resolve().parents[2] / "data")


  def main() -> None:
      parser = argparse.ArgumentParser(description="Live System-2 smoke (manual only)")
      parser.add_argument("--corpus", required=True, help="ingested corpus_id")
      parser.add_argument("--query", required=True, help="a multi-hop question")
      args = parser.parse_args()

      preset = PRESETS["router-on"]
      config = dataclasses.replace(
          preset,
          query=dataclasses.replace(preset.query, route_mode=RouteMode.FORCE_S2))
      core = _build_core_real(config, args.corpus, DATA_DIR)
      claude = AnthropicClient()                      # reads ANTHROPIC_API_KEY
      store = TraceStore(DATA_DIR / "traces-smoke.sqlite3")
      trace_id = f"smoke-{uuid.uuid4().hex[:8]}"

      result = run_query(query=args.query, core=core, claude=claude, store=store,
                         config=config, trace_id=trace_id)

      print(f"\nsystem={result.system}  hops={result.hops_used}  "
            f"tokens={result.tokens_used}  trace={trace_id}")
      print(f"abstained={result.final.abstained}  "
            f"contradiction={result.final.contradiction_flag}")
      print(f"unresolved={result.final.unresolved_subqueries}")
      print(f"citations={result.final.citations}")
      print(f"\nANSWER:\n{result.final.text}\n\nTRACE:")
      for e in store.get(trace_id):
          print(f"  [{e.seq:02d}] {e.node:<12} {e.duration_ms:7.1f}ms "
                f"in={e.input_tokens:<6} out={e.output_tokens:<5} "
                f"model={e.model or '-'}")
          if e.node == "grade":
              print(f"        verdict={e.payload['verdict']} "
                    f"-> {e.payload['next_action']}")


  if __name__ == "__main__":
      main()
  ```

- [ ] Static checks only (no live run in this plan's execution): `uv run ruff check scripts/smoke_s2.py` and `uv run python -c "import ast; ast.parse(open('scripts/smoke_s2.py').read()); print('parses ok')"` → clean / `parses ok`. The import line is exercised by `uv run python -c "import scripts.smoke_s2"` only if `scripts/` has no `__init__.py` ambiguity — simpler equivalent: `uv run python scripts/smoke_s2.py --help` → prints usage and exits 0 **without** touching any vendor (argparse runs before client construction; `AnthropicClient()` is only built inside `main()` after parse — and `--help` exits first).
- [ ] Optional manual verification (requires keys + ingested corpus; never CI): run the usage command from the docstring against the MuSiQue corpus; eyeball that the trace shows `decompose → retrieve_hop → grade → ... → synthesize`, hops ≤ 3, tokens ≤ 50_000, and the answer carries `[n]` citations.
- [ ] Final full check: `uv run pytest -q && uv run ruff check .` → green and clean, zero keys required.
- [ ] Commit:

  ```bash
  git add api/scripts/smoke_s2.py
  git commit -m "feat(scripts): manual live System-2 smoke script with trace printout

  Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
  ```

---

## Done criteria (whole plan)

- `uv run pytest -q` green offline with zero API keys; `uv run ruff check .` clean.
- All eight contract node names exist in `agents/graph.py`; every node emits a TraceEvent; `TraceStore.get(trace_id)` replays a query's full node sequence in `seq` order from SQLite (WAL).
- State-transition coverage: simple→S1, force_s1, complex→S2, low-confidence→S2, insufficient→refine loop, contradictory→one re-retrieve→flagged synthesis, hop exhaustion→caveated synthesis, token-ceiling exhaustion→caveated synthesis, abstention as structured field.
- `router-on` preset produces a Receipt with `n_s1`/`n_s2`, per-query `route`, nulled `recall_at_5`/`mrr_at_3`, `recall_union_of_hops` flagged `union_of_hops: true`, `prompts_version` populated from `agents.prompts.PROMPTS_VERSION` (R11), and per-query `usd` computed from TraceStore events' `(model, input_tokens, output_tokens)` (R10).
- The permanent `MULTI_HOP_DATASETS` gate still skips `router-on` on single-hop corpora (R10, tested), and `estimate_run_cost` prices AUTO presets with the System-2 upper bound.
- `api/scripts/smoke_s2.py --help` works keyless; the live path is documented for manual use only.






