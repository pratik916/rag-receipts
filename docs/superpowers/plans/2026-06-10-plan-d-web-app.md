# Web App, Docker Compose, E2E & BYO Ingest (Plan D) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the user-facing layer of rag-receipts — a FastAPI server exposing the query/trace/eval/receipts API, a Next.js web app (Playground, Ablation Lab, Corpora), docker compose deployment, offline Playwright e2e, and bring-your-own-documents ingestion — on top of the engine built in Plans A–C.

**Architecture:** A single-worker FastAPI app (`api/ragreceipts/server/`) fronts the Plan A–C engine through three narrow seams (`QueryRunner`, `EvalRunner`, `IngestSink`) and runs ingest/eval work on a SQLite-backed worker thread. The Next.js app (`web/`) computes nothing: it consumes a typed client generated from FastAPI's OpenAPI 3.1 schema via openapi-typescript v7. docker compose wires api + web + qdrant with healthcheck-gated startup; a `TESTING=1` mode swaps all vendor transports for the contracts' fakes so Playwright e2e runs offline with zero keys.

**Tech Stack:** Python 3.12 + uv, FastAPI + Pydantic + uvicorn (single worker), SQLite (WAL), LlamaIndex file readers (BYO), Next.js App Router + TypeScript + pnpm, openapi-typescript v7 + openapi-fetch, recharts, Playwright, docker compose.

---

## Context

### Where this plan starts

Plans A–C are complete. The repo is a git repo rooted at `rag-receipts/` with:

- `api/` — uv-managed Python 3.12 package `ragreceipts` with `ingest/`, `retrieval/`, `agents/`, `eval/`, `traces/`, `vendors/`, plus `api/tests/` (pytest, all offline). `ruff` line length 100.
- `receipts/` — committed headline receipts from Plan B (versioned envelope `{"schema_version": 1, "receipt": {...}}`).
- `data/` — gitignored; corpora manifests at `data/corpora/{corpus_id}/manifest.json`, local receipts at `data/receipts-local/`.
- No `web/`, no `docker-compose.yml`, no `README.md` at root, no `server/` package yet — this plan builds all of those.

### Binding contracts this plan consumes (quoted from `docs/superpowers/plans/2026-06-10-contracts.md`)

Core types (`api/ragreceipts/types.py`):

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

Config (`api/ragreceipts/config.py`): `IngestConfig`, `QueryConfig`, `PipelineConfig`, and
`PRESETS: dict[str, PipelineConfig]` with keys, in ladder order:
`"bm25-only"`, `"dense-rrf"`, `"contextual"`, `"rerank"`, `"router-on"`.

Constants (`api/ragreceipts/constants.py`): `ROUTER_MODEL = "claude-haiku-4-5-20251001"`,
`SYNTH_MODEL = "claude-sonnet-4-6"`, `JUDGE_MODEL = "claude-sonnet-4-6"`,
`EMBED_MODEL = "voyage-context-3"`, `RERANK_MODEL = "rerank-v4.0-pro"`,
`RAGAS_EMBED_MODEL = "BAAI/bge-small-en-v1.5"`, `ROUTE_CONFIDENCE_THRESHOLD = 0.7`,
`S2_MAX_HOPS = 3`, `S2_TOKEN_CEILING = 50_000`.

Vendor transports (`api/ragreceipts/vendors/base.py`):

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
    parsed: object
    input_tokens: int
    output_tokens: int
```

(`EmbedTransport`, `RerankTransport` exist in the same module; real impls `VoyageClient`,
`CohereClient`, `AnthropicClient`; fakes `FakeEmbed`, `FakeRerank`, `FakeClaude` in
`api/tests/fakes.py`. Application code never imports `anthropic` outside `vendors/`; the
Anthropic SDK is used only via `client.messages.parse(...)` / typed exceptions as pinned in
the contracts — this plan adds **no** new Anthropic API usage.)

Traces (`api/ragreceipts/traces/models.py` + `traces/store.py`):

```python
@dataclass(frozen=True)
class TraceEvent:
    trace_id: str
    seq: int
    node: str              # "route"|"s1_retrieve"|"s1_answer"|"decompose"|"retrieve_hop"|"grade"|"refine"|"synthesize"
    payload: dict
    model: str | None
    input_tokens: int
    output_tokens: int
    duration_ms: float
```

`TraceStore.append(event)`, `TraceStore.get(trace_id) -> list[TraceEvent]` (SQLite WAL).

Receipts (`api/ragreceipts/eval/receipts.py`): `Receipt` and `PublishedAnchor` dataclasses
(fields per contracts §receipts.json schema; `PublishedAnchor.note` is REQUIRED), serialized
as `{"schema_version": 1, "receipt": {...}}`. Pricing (`api/ragreceipts/eval/pricing.py`):
`PRICING: dict[str, dict]` with `usd_per_mtok_input` / `usd_per_mtok_output` keys for Claude
models, `PRICING_VERSION = "2026-06-10"`.

Server endpoints (binding paths, contracts §Server): `GET /health`, `POST /query`
(`{"query": str, "corpus_id": str, "preset": str}` → answer + trace_id + degraded flags),
`GET /traces/{trace_id}`, `GET /corpora`, `POST /corpora/ingest` (BYO, job),
`GET /jobs/{job_id}`, `POST /eval/runs` (cost estimate + confirmation), `GET /eval/runs`,
`GET /receipts` (committed + local). Frontend pages: `/` Playground, `/ablation` Ablation
Lab, `/corpora` Corpora. The api runs **single-worker uvicorn**; jobs run in a dedicated
worker thread keyed by SQLite job rows (`server/jobs.py`).

### The seam principle (read this before Task 1)

The contracts pin the *engine's* types, the *server's* endpoint paths, **and** — per
contracts §Seam Resolutions **R9** — the exact engine entry points the server wires to:

- `agents/service.py::run_query(query=, core=, claude=, store=, config=) -> GraphResult`
  (Plan C; `GraphResult` carries `final` (a `FinalAnswer` with `text`/`citations`/`abstained`),
  `system` (`"s1" | "s2"`), `trace_id`, `tokens_used`, `hops_used`, `retrieved: list[ScoredChunk]`)
- `eval/runner.py::AblationRunner` (constructor kwargs `core_factory=, claude=, store=,
  data_dir=, ragas=None`; `run(run_id=, corpus_id=, slice_name=, presets=, spend_cap_usd=)`)
  and `eval/runner.py::estimate_run_cost(preset_names: list[str], n_queries: int) -> float`
- `ingest/pipeline.py::run_ingest(corpus_id=, data_dir=, ingest_config=, embed=, qdrant=) -> dict`
  (Plan A; returns the corpus manifest)
- Plan B's composition root `cli.py::_build_core_real(config, corpus_id, data_dir)`

This plan still defines three narrow Protocols owned by the server — `QueryRunner` (Task 5),
`EvalRunner` (Task 7), `IngestSink` (Task 13) — and all endpoint code and all endpoint tests
target those Protocols with fully-specified fakes. The production adapters behind them
(`build_real_query_runner`, `RealEvalRunner`, `build_real_ingest_sink`) are written as
**complete code** in their tasks against the R9-pinned names — no grep-discovery steps. Each
adapter task additionally keeps (a) a short **signature-drift verification step** (an
`inspect.signature` one-liner with the R9-expected output; if anything drifted, reconcile
ONLY the adapter — the Protocols and the endpoints never change) and (b) **offline
construction/marshalling tests with fakes**, so the wiring is proven without keys or network.

### External APIs verified for this plan (2026-06-10)

| Library | What was verified | Source |
|---|---|---|
| llama-index-readers-file | `from llama_index.readers.file import PDFReader, MarkdownReader, HTMLTagReader, FlatReader`; `PDFReader(return_full_document=True).load_data(file: Path)`; `MarkdownReader().load_data(file: str)`; `HTMLTagReader` is tag-configurable (default `<section>`), needs `beautifulsoup4`; `PDFReader` needs `pypdf`; all return `Document` objects with `.text` | https://developers.llamaindex.ai/python/framework-api-reference/readers/file/ |
| openapi-typescript v7 | CLI: `npx openapi-typescript schema.json -o schema.d.ts`; single-schema CLI form (globbing removed in v7) | https://openapi-ts.dev/cli |
| openapi-fetch (0.17.x) | `createClient<paths>({ baseUrl })`; `client.GET("/path/{id}", { params: { path: { id } } })`; `client.POST("/path", { body })`; returns `{ data, error, response }` | https://openapi-ts.dev/openapi-fetch/ |
| qdrant-client local mode | `QdrantClient(":memory:")` **does support named vectors**: `local/local_collection.py` handles `vectors: dict[str, models.VectorParams]` and `using: str | None` in search | https://github.com/qdrant/qdrant-client (qdrant_client/local/local_collection.py) |
| FastAPI | Generates **OpenAPI 3.1.0** by default; schema at `/openapi.json` and programmatically via `app.openapi()` | https://fastapi.tiangolo.com/how-to/extending-openapi/ |
| FastAPI uploads | `UploadFile` + `File()` + `Form()` multipart requires `python-multipart` | https://fastapi.tiangolo.com/tutorial/request-files/ |
| Playwright | `webServer` accepts an **array** of servers (each with `command`, `cwd`, `env`, `url`, `reuseExistingServer`); `use.baseURL` must be set explicitly when array form is used | https://playwright.dev/docs/test-webserver |
| create-next-app | `--typescript --eslint --app --src-dir --no-tailwind --import-alias "@/*" --use-pnpm --yes` non-interactive flags | https://nextjs.org/docs/app/api-reference/cli/create-next-app |
| recharts | `<ResponsiveContainer><BarChart data={...}><XAxis dataKey/><Bar dataKey/></BarChart></ResponsiveContainer>`; multiple `<Bar>` elements without `stackId` render grouped bars | https://recharts.github.io/en-US/api/BarChart/ |
| qdrant docker image | Image ships **no curl/wget by design** (healthcheck tooling rejected as security risk), so compose healthcheck uses bash's `/dev/tcp` TCP-connect probe; the tag is pinned by contracts **R7** to `qdrant/qdrant:v1.18.0` — the minor matching the binding qdrant-client pin `>=1.18,<2` | https://github.com/qdrant/qdrant/issues/4250 , contracts §Seam Resolutions R7 |
| uv docker image | `ghcr.io/astral-sh/uv:python3.12-bookworm-slim` base + `uv sync --frozen` layer pattern | https://docs.astral.sh/uv/guides/integration/docker/ |

**Previously discovery-bound items, now pinned by contracts §Seam Resolutions:** the qdrant
server tag (**R7**: `qdrant/qdrant:v1.18.0`, matching the binding client pin `>=1.18,<2`);
the Plan A/B/C entry points wired in Tasks 5, 7, and 13 (**R9** pins
`agents/service.py::run_query`, `eval/runner.py::AblationRunner` + `estimate_run_cost`,
`ingest/pipeline.py::run_ingest`, and `cli.py::_build_core_real` — each task writes the
adapter as complete code and keeps only a short signature-drift verification step); and the
cost estimate (the server delegates to Plan B's `estimate_run_cost` directly, so no
voyage/cohere pricing-key discovery remains — `estimate_run_cost` already prices them).

### Conventions used by every task below

- Python commands run from `api/` (`cd api` first). Per contracts §Seam Resolutions **R8**,
  `api/tests/` IS a package: `api/tests/__init__.py` already exists (Spike 0/Plan A) and
  `api/pyproject.toml` sets `[tool.pytest.ini_options] pythonpath = ["."]`, so every test
  file imports fakes with `from tests.fakes import ...` — the repo-wide convention this plan
  follows (Task 1 has an idempotent guard; it never re-creates the marker). TESTING mode
  launches uvicorn with `python -m uvicorn` from `api/` (`python -m` puts the cwd on
  `sys.path`, making the `tests` package importable in the *server* process too — guarded
  by a named RuntimeError in `build_deps`, per R8).
- Web commands run from `web/`.
- Git commits run from the repo root `rag-receipts/`.
- **The single-worker run command (load-bearing, contracts §Server):**

  ```bash
  cd api && uv run python -m uvicorn ragreceipts.server.app:app --host 0.0.0.0 --port 8000 --workers 1
  ```

  One uvicorn worker is mandatory: the `JobRunner` worker thread and its dispatch queue are
  in-process state. With >1 workers, jobs inserted by one process would never be executed by
  the process that owns the queue, and TESTING mode's in-memory trace store would shard.

- **`TESTING=1` mode (precise definition):** when the environment variable `TESTING` equals
  the string `"1"`, `ragreceipts.server.deps.build_deps()` does not construct any real vendor
  client or touch the network. It imports `tests.e2e_fixture.build_testing_deps()` which
  wires: `ScriptedTransport` (a `ClaudeTransport` fake added to `api/tests/fakes.py` in Task
  3) for all Claude-shaped calls, a deterministic lexical fixture retriever over
  `api/tests/fixtures/e2e_corpus.json`, `QdrantClient(":memory:")` (so `/health` reports
  qdrant truthfully), an in-memory trace store, a real `JobRunner` on a temp SQLite file, and
  fixture eval/ingest runners that write real files under a temp `data/` dir. Vendor-shaped
  behavior flows only through the contracts' transport protocols, satisfying the offline/no-keys
  testing constraint. `TESTING=1` exists for Playwright e2e and screenshots; it is never the
  default and `/health` discloses it via `testing_mode: true`.

---

### Task 1: Server dependencies + SQLite-backed job runner

**Files:**
- Create: `api/ragreceipts/server/__init__.py`
- Create: `api/ragreceipts/server/jobs.py`
- Test: `api/tests/test_server_jobs.py`

**Step 1 — install server dependencies (pinned to the majors verified above):**

- [ ] Run:

  ```bash
  cd api
  uv add "fastapi>=0.115" "uvicorn>=0.30" "python-multipart>=0.0.9"
  uv add --dev "httpx>=0.27"   # required by fastapi.testclient.TestClient
  ```

- [ ] Create the empty package marker `api/ragreceipts/server/__init__.py` (empty file).

- [ ] R8 guard (idempotent): `api/tests/__init__.py` already exists — Spike 0/Plan A created
  it and `pythonpath = ["."]` is already in `api/pyproject.toml`. Create the marker ONLY if
  it is somehow missing; never overwrite or re-create it:

  ```bash
  cd api && test -f tests/__init__.py || touch tests/__init__.py
  # EXPECTED: no output; `ls tests/__init__.py` shows the file (pre-existing or just created)
  ```

**Step 2 — write the failing test (COMPLETE code):**

- [ ] Create `api/tests/test_server_jobs.py`:

  ```python
  """JobRunner: SQLite-keyed jobs on a single worker thread (contracts §Server)."""
  import sqlite3
  import time

  import pytest

  from ragreceipts.server.jobs import JobRunner, JobStatus


  def wait_for(predicate, timeout: float = 10.0) -> None:
      deadline = time.time() + timeout
      while time.time() < deadline:
          if predicate():
              return
          time.sleep(0.05)
      raise AssertionError("condition not met within timeout")


  def test_submit_runs_handler_and_records_events(tmp_path):
      runner = JobRunner(tmp_path / "jobs.sqlite")
      seen = {}

      def handler(ctx):
          seen["params"] = ctx.params
          ctx.emit("halfway", 0.5)
          ctx.emit("done", 1.0)

      runner.register("demo", handler)
      runner.start()
      try:
          job_id = runner.submit("demo", {"corpus_id": "c1"})
          wait_for(lambda: runner.get(job_id).status == JobStatus.SUCCEEDED)
      finally:
          runner.stop()
      assert seen["params"] == {"corpus_id": "c1"}
      events = runner.events(job_id)
      assert [e.message for e in events] == ["halfway", "done"]
      assert events[-1].progress == 1.0
      assert events[0].seq == 1 and events[1].seq == 2


  def test_failed_job_records_error(tmp_path):
      runner = JobRunner(tmp_path / "jobs.sqlite")

      def handler(ctx):
          raise RuntimeError("boom")

      runner.register("demo", handler)
      runner.start()
      try:
          job_id = runner.submit("demo", {})
          wait_for(lambda: runner.get(job_id).status == JobStatus.FAILED)
      finally:
          runner.stop()
      assert "boom" in runner.get(job_id).error


  def test_submit_unknown_kind_raises(tmp_path):
      runner = JobRunner(tmp_path / "jobs.sqlite")
      with pytest.raises(ValueError, match="no handler registered"):
          runner.submit("nope", {})


  def test_crash_recovery_marks_interrupted_and_resume_requeues(tmp_path):
      db = tmp_path / "jobs.sqlite"
      first = JobRunner(db)
      first.register("demo", lambda ctx: None)
      job_id = first.submit("demo", {"n": 1})  # never started -> stays QUEUED
      # Simulate a crash mid-run: force the row to RUNNING, then "restart" the process
      # by constructing a fresh JobRunner over the same DB.
      with sqlite3.connect(db) as conn:
          conn.execute("UPDATE jobs SET status = 'running' WHERE job_id = ?", (job_id,))
      second = JobRunner(db)
      assert second.get(job_id).status == JobStatus.INTERRUPTED

      ran = []
      second.register("demo", lambda ctx: ran.append(ctx.params["n"]))
      second.start()
      try:
          second.resume(job_id)
          wait_for(lambda: second.get(job_id).status == JobStatus.SUCCEEDED)
      finally:
          second.stop()
      assert ran == [1]


  def test_resume_rejects_jobs_that_are_not_resumable(tmp_path):
      runner = JobRunner(tmp_path / "jobs.sqlite")
      runner.register("demo", lambda ctx: None)
      job_id = runner.submit("demo", {})  # QUEUED, not started
      with pytest.raises(ValueError, match="not resumable"):
          runner.resume(job_id)


  def test_list_filters_by_kind_and_orders_newest_first(tmp_path):
      runner = JobRunner(tmp_path / "jobs.sqlite")
      runner.register("a", lambda ctx: None)
      runner.register("b", lambda ctx: None)
      ja = runner.submit("a", {})
      time.sleep(0.01)
      jb = runner.submit("b", {})
      assert [r.job_id for r in runner.list()] == [jb, ja]
      assert [r.job_id for r in runner.list(kind="a")] == [ja]
  ```

- [ ] Run it and confirm the expected failure:

  ```bash
  cd api && uv run pytest tests/test_server_jobs.py -q
  # EXPECTED: ModuleNotFoundError: No module named 'ragreceipts.server.jobs'
  ```

**Step 3 — implement `server/jobs.py` (COMPLETE code):**

- [ ] Create `api/ragreceipts/server/jobs.py`:

  ```python
  """Single-worker background jobs keyed by SQLite rows.

  The api runs single-worker uvicorn (contracts §Server): exactly one process owns this
  JobRunner. SQLite rows are the durable source of truth; the in-process queue is only the
  dispatch mechanism. On construction, rows left RUNNING by a crashed/stopped process are
  marked INTERRUPTED; resume() re-enqueues them. Handlers must therefore be idempotent —
  that is what makes "resumable from job state" real (ingest rebuilds from saved uploads,
  eval resumes from Plan B's per-query SQLite state).
  """
  from __future__ import annotations

  import json
  import queue
  import sqlite3
  import threading
  import time
  import traceback
  import uuid
  from dataclasses import dataclass
  from enum import Enum
  from pathlib import Path
  from typing import Callable


  class JobStatus(str, Enum):
      QUEUED = "queued"
      RUNNING = "running"
      SUCCEEDED = "succeeded"
      FAILED = "failed"
      INTERRUPTED = "interrupted"


  @dataclass(frozen=True)
  class JobRow:
      job_id: str
      kind: str
      status: JobStatus
      params: dict
      error: str | None
      created_at: float
      updated_at: float


  @dataclass(frozen=True)
  class JobEvent:
      seq: int
      ts: float
      message: str
      progress: float  # 0.0 .. 1.0


  class JobContext:
      """Handed to job handlers; the only API a handler needs."""

      def __init__(self, runner: "JobRunner", job_id: str, params: dict) -> None:
          self.job_id = job_id
          self.params = params
          self._runner = runner

      def emit(self, message: str, progress: float) -> None:
          self._runner.append_event(self.job_id, message, progress)


  _SCHEMA = """
  CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    status TEXT NOT NULL,
    params_json TEXT NOT NULL,
    error TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
  );
  CREATE TABLE IF NOT EXISTS job_events (
    job_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    ts REAL NOT NULL,
    message TEXT NOT NULL,
    progress REAL NOT NULL,
    PRIMARY KEY (job_id, seq)
  );
  """


  class JobRunner:
      def __init__(self, db_path: Path) -> None:
          db_path.parent.mkdir(parents=True, exist_ok=True)
          self._db_path = db_path
          self._handlers: dict[str, Callable[[JobContext], None]] = {}
          self._queue: "queue.Queue[str]" = queue.Queue()
          self._stop = threading.Event()
          self._thread: threading.Thread | None = None
          with self._conn() as conn:
              conn.executescript(_SCHEMA)
              conn.execute(  # crash recovery: a RUNNING row from a dead process never finished
                  "UPDATE jobs SET status = ?, updated_at = ? WHERE status = ?",
                  (JobStatus.INTERRUPTED.value, time.time(), JobStatus.RUNNING.value),
              )

      def _conn(self) -> sqlite3.Connection:
          conn = sqlite3.connect(self._db_path)
          conn.execute("PRAGMA journal_mode=WAL")
          return conn

      # -- registration / lifecycle -------------------------------------------------

      def register(self, kind: str, handler: Callable[[JobContext], None]) -> None:
          self._handlers[kind] = handler

      def start(self) -> None:
          if self._thread is not None:
              return
          self._thread = threading.Thread(target=self._run_loop, name="job-worker", daemon=True)
          self._thread.start()

      def stop(self) -> None:
          self._stop.set()
          if self._thread is not None:
              self._thread.join(timeout=5)
              self._thread = None
          self._stop.clear()

      # -- job API -------------------------------------------------------------------

      def submit(self, kind: str, params: dict) -> str:
          if kind not in self._handlers:
              raise ValueError(f"no handler registered for job kind {kind!r}")
          job_id = uuid.uuid4().hex
          now = time.time()
          with self._conn() as conn:
              conn.execute(
                  "INSERT INTO jobs VALUES (?, ?, ?, ?, NULL, ?, ?)",
                  (job_id, kind, JobStatus.QUEUED.value, json.dumps(params), now, now),
              )
          self._queue.put(job_id)
          return job_id

      def resume(self, job_id: str) -> None:
          row = self.get(job_id)
          if row is None:
              raise KeyError(job_id)
          if row.status not in (JobStatus.INTERRUPTED, JobStatus.FAILED):
              raise ValueError(f"job {job_id} is {row.status.value}, not resumable")
          self._set_status(job_id, JobStatus.QUEUED, error=None)
          self._queue.put(job_id)

      def get(self, job_id: str) -> JobRow | None:
          with self._conn() as conn:
              cur = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
              row = cur.fetchone()
          return self._to_row(row) if row else None

      def list(self, kind: str | None = None) -> list[JobRow]:
          q = "SELECT * FROM jobs"
          args: tuple = ()
          if kind is not None:
              q += " WHERE kind = ?"
              args = (kind,)
          q += " ORDER BY created_at DESC"
          with self._conn() as conn:
              rows = conn.execute(q, args).fetchall()
          return [self._to_row(r) for r in rows]

      def events(self, job_id: str) -> list[JobEvent]:
          with self._conn() as conn:
              rows = conn.execute(
                  "SELECT seq, ts, message, progress FROM job_events"
                  " WHERE job_id = ? ORDER BY seq",
                  (job_id,),
              ).fetchall()
          return [JobEvent(seq=r[0], ts=r[1], message=r[2], progress=r[3]) for r in rows]

      def append_event(self, job_id: str, message: str, progress: float) -> None:
          with self._conn() as conn:
              conn.execute(
                  "INSERT INTO job_events (job_id, seq, ts, message, progress)"
                  " SELECT ?, COALESCE(MAX(seq), 0) + 1, ?, ?, ?"
                  " FROM job_events WHERE job_id = ?",
                  (job_id, time.time(), message, progress, job_id),
              )

      # -- internals -------------------------------------------------------------------

      @staticmethod
      def _to_row(row: tuple) -> JobRow:
          return JobRow(
              job_id=row[0],
              kind=row[1],
              status=JobStatus(row[2]),
              params=json.loads(row[3]),
              error=row[4],
              created_at=row[5],
              updated_at=row[6],
          )

      def _set_status(self, job_id: str, status: JobStatus, error: str | None = None) -> None:
          with self._conn() as conn:
              conn.execute(
                  "UPDATE jobs SET status = ?, error = ?, updated_at = ? WHERE job_id = ?",
                  (status.value, error, time.time(), job_id),
              )

      def _run_loop(self) -> None:
          while not self._stop.is_set():
              try:
                  job_id = self._queue.get(timeout=0.2)
              except queue.Empty:
                  continue
              row = self.get(job_id)
              if row is None or row.status != JobStatus.QUEUED:
                  continue
              self._set_status(job_id, JobStatus.RUNNING)
              try:
                  self._handlers[row.kind](JobContext(self, job_id, row.params))
              except Exception:
                  self._set_status(job_id, JobStatus.FAILED, error=traceback.format_exc(limit=5))
              else:
                  self._set_status(job_id, JobStatus.SUCCEEDED)
  ```

- [ ] Run the tests again — all pass:

  ```bash
  cd api && uv run pytest tests/test_server_jobs.py -q
  # EXPECTED: 6 passed
  ```

- [ ] Lint: `cd api && uv run ruff check ragreceipts/server tests/test_server_jobs.py`

- [ ] Commit:

  ```bash
  git add api/pyproject.toml api/uv.lock api/ragreceipts/server api/tests/test_server_jobs.py
  git commit -m "feat(server): SQLite-keyed single-worker job runner with events and resume" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
  ```

---

### Task 2: Pydantic API models (fully typed OpenAPI 3.1)

**Files:**
- Create: `api/ragreceipts/server/models.py`
- Test: `api/tests/test_server_models.py`

Every request/response below is a Pydantic model so FastAPI's OpenAPI 3.1 output is fully
typed end-to-end — that is what makes the Task 8 codegen produce a real typed client
("`web/` computes nothing").

**Step 1 — failing test:**

- [ ] Create `api/tests/test_server_models.py`:

  ```python
  """API models: validation rules the endpoints rely on."""
  import pytest
  from pydantic import ValidationError

  from ragreceipts.server import models as m


  def test_query_request_rejects_unknown_preset():
      with pytest.raises(ValidationError, match="unknown preset"):
          m.QueryRequest(query="q", corpus_id="c1", preset="not-a-preset")


  def test_query_request_accepts_every_contract_preset():
      for preset in ["bm25-only", "dense-rrf", "contextual", "rerank", "router-on"]:
          req = m.QueryRequest(query="q", corpus_id="c1", preset=preset)
          assert req.preset == preset


  def test_corpus_id_must_be_a_safe_slug():
      with pytest.raises(ValidationError, match="corpus_id"):
          m.QueryRequest(query="q", corpus_id="../etc", preset="rerank")


  def test_eval_run_request_defaults():
      req = m.EvalRunRequest(corpus_id="c1", preset="rerank")
      assert req.slice == "smoke"
      assert req.confirm is False
      assert req.spend_cap_usd == 5.0


  def test_eval_run_request_rejects_unknown_slice():
      with pytest.raises(ValidationError):
          m.EvalRunRequest(corpus_id="c1", preset="rerank", slice="huge")
  ```

- [ ] Run: `cd api && uv run pytest tests/test_server_models.py -q`
  — EXPECTED: `ModuleNotFoundError: No module named 'ragreceipts.server.models'`.

**Step 2 — implement `server/models.py` (COMPLETE code):**

- [ ] Create `api/ragreceipts/server/models.py`:

  ```python
  """Pydantic request/response models for every endpoint.

  FastAPI derives the OpenAPI 3.1 schema from these models; web/ generates its typed
  client from that schema (Task 8). Keep every field typed — no bare dict responses
  except where the payload is by-design open (trace payloads, manifests, receipts).
  """
  from __future__ import annotations

  import re
  from typing import Literal

  from pydantic import BaseModel, Field, field_validator

  from ragreceipts.config import PRESETS

  _SLUG = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


  def _validate_corpus_id(v: str) -> str:
      if not _SLUG.match(v):
          raise ValueError("corpus_id must match ^[a-z0-9][a-z0-9_-]{0,63}$")
      return v


  def _validate_preset(v: str) -> str:
      if v not in PRESETS:
          raise ValueError(f"unknown preset {v!r}; valid: {sorted(PRESETS)}")
      return v


  # -- health ------------------------------------------------------------------------


  class VendorStatusModel(BaseModel):
      name: str
      configured: bool
      env_var: str


  class HealthResponse(BaseModel):
      status: Literal["ok", "degraded"]
      vendors: list[VendorStatusModel]
      qdrant_ok: bool
      missing_env_vars: list[str]
      testing_mode: bool


  # -- query / traces -----------------------------------------------------------------


  class QueryRequest(BaseModel):
      query: str = Field(min_length=1, max_length=4000)
      corpus_id: str
      preset: str

      _corpus = field_validator("corpus_id")(_validate_corpus_id)
      _preset = field_validator("preset")(_validate_preset)


  class CitationModel(BaseModel):
      n: int
      chunk_id: str
      passage_id: str
      text: str
      score: float


  class QueryResponse(BaseModel):
      answer: str
      abstained: bool
      route: Literal["s1", "s2"]
      degraded: list[str]  # e.g. ["rerank-skipped"] — degrade visibly, never silently
      citations: list[CitationModel]
      trace_id: str


  class TraceEventModel(BaseModel):
      trace_id: str
      seq: int
      node: str
      payload: dict
      model: str | None
      input_tokens: int
      output_tokens: int
      duration_ms: float


  class TraceResponse(BaseModel):
      trace_id: str
      events: list[TraceEventModel]


  # -- corpora --------------------------------------------------------------------------


  class CorpusModel(BaseModel):
      corpus_id: str
      manifest: dict  # contracts §Corpus manifest (open by design)


  class CorporaResponse(BaseModel):
      corpora: list[CorpusModel]


  class IngestResponse(BaseModel):
      job_id: str
      corpus_id: str


  # -- jobs -------------------------------------------------------------------------------


  class JobEventModel(BaseModel):
      seq: int
      ts: float
      message: str
      progress: float


  class JobResponse(BaseModel):
      job_id: str
      kind: str
      status: Literal["queued", "running", "succeeded", "failed", "interrupted"]
      params: dict
      error: str | None
      events: list[JobEventModel]


  # -- eval runs ---------------------------------------------------------------------------


  class CostEstimateModel(BaseModel):
      n_queries: int
      est_tokens: int
      est_usd: float
      pricing_table_version: str


  class EvalRunRequest(BaseModel):
      corpus_id: str
      preset: str
      slice: Literal["smoke", "full"] = "smoke"
      confirm: bool = False  # confirmation gate: first call returns the estimate only
      spend_cap_usd: float = Field(default=5.0, gt=0)

      _corpus = field_validator("corpus_id")(_validate_corpus_id)
      _preset = field_validator("preset")(_validate_preset)


  class EvalRunResponse(BaseModel):
      status: Literal["needs_confirmation", "started"]
      estimate: CostEstimateModel
      job_id: str | None = None


  class EvalRunListItem(BaseModel):
      job_id: str
      status: str
      corpus_id: str
      preset: str
      slice: str
      created_at: float


  class EvalRunsResponse(BaseModel):
      runs: list[EvalRunListItem]


  # -- receipts -------------------------------------------------------------------------------


  class ReceiptEntryModel(BaseModel):
      source: Literal["committed", "local"]
      path: str
      schema_version: int
      receipt: dict  # contracts §receipts.json schema (open by design)


  class ReceiptsResponse(BaseModel):
      receipts: list[ReceiptEntryModel]
      errors: list[str]  # unparseable receipt files, disclosed — never silently dropped
  ```

- [ ] Run: `cd api && uv run pytest tests/test_server_models.py -q` — EXPECTED: `5 passed`.
- [ ] Lint: `cd api && uv run ruff check ragreceipts/server/models.py`
- [ ] Commit:

  ```bash
  git add api/ragreceipts/server/models.py api/tests/test_server_models.py
  git commit -m "feat(server): typed Pydantic models for all endpoints (OpenAPI 3.1 source of truth)" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
  ```

---

### Task 3: Dependency container + TESTING=1 seam

**Files:**
- Create: `api/ragreceipts/server/deps.py`
- Modify: `api/tests/fakes.py` (ADD two classes; rename nothing)
- Create: `api/tests/e2e_fixture.py`
- Create: `api/tests/helpers_server.py`
- Test: `api/tests/test_server_deps.py`

**Step 1 — failing test:**

- [ ] Create `api/tests/test_server_deps.py`:

  ```python
  """build_deps(): env-driven vendor capability + the TESTING=1 seam."""
  from ragreceipts.server.deps import VENDOR_ENV_VARS, AppPaths, build_deps


  def test_vendor_env_var_names_are_the_official_sdk_names():
      assert VENDOR_ENV_VARS == {
          "voyage": "VOYAGE_API_KEY",
          "cohere": "COHERE_API_KEY",
          "anthropic": "ANTHROPIC_API_KEY",
      }


  def test_app_paths_layout(tmp_path):
      paths = AppPaths(data_dir=tmp_path / "data", receipts_committed_dir=tmp_path / "receipts")
      paths.ensure()
      assert paths.corpora_dir == tmp_path / "data" / "corpora"
      assert paths.receipts_local_dir == tmp_path / "data" / "receipts-local"
      assert paths.uploads_dir == tmp_path / "data" / "uploads"
      assert paths.jobs_db == tmp_path / "data" / "server-jobs.sqlite"
      assert paths.corpora_dir.is_dir() and paths.uploads_dir.is_dir()


  def test_build_deps_reports_missing_keys_without_touching_network(tmp_path, monkeypatch):
      for env in VENDOR_ENV_VARS.values():
          monkeypatch.delenv(env, raising=False)
      monkeypatch.delenv("TESTING", raising=False)
      monkeypatch.delenv("QDRANT_URL", raising=False)
      monkeypatch.setenv("RAGRECEIPTS_DATA_DIR", str(tmp_path / "data"))
      monkeypatch.setenv("RAGRECEIPTS_RECEIPTS_DIR", str(tmp_path / "receipts"))
      deps = build_deps()
      assert deps.testing_mode is False
      assert [v.configured for v in deps.vendors] == [False, False, False]
      assert deps.qdrant is None  # R7: QDRANT_URL unset -> NO silent localhost default
      assert deps.query_runner is None  # no keys -> endpoint will 503 with named env vars
      deps.job_runner.stop()


  def test_testing_env_wires_fixture_deps(tmp_path, monkeypatch):
      monkeypatch.setenv("TESTING", "1")
      monkeypatch.setenv("RAGRECEIPTS_DATA_DIR", str(tmp_path / "data"))
      monkeypatch.setenv("RAGRECEIPTS_RECEIPTS_DIR", str(tmp_path / "receipts"))
      deps = build_deps()
      assert deps.testing_mode is True
      assert all(v.configured for v in deps.vendors)  # fakes count as configured
      assert deps.qdrant is not None  # QdrantClient(":memory:") — named vectors verified
      manifest = deps.paths.corpora_dir / "fixture-corpus" / "manifest.json"
      assert manifest.exists()
      deps.job_runner.stop()
  ```

- [ ] Run: `cd api && uv run pytest tests/test_server_deps.py -q`
  — EXPECTED: `ModuleNotFoundError: No module named 'ragreceipts.server.deps'`.

**Step 2 — add fakes (ADDITIVE edit to `api/tests/fakes.py`):**

- [ ] Append to `api/tests/fakes.py` (keep all Plan A fakes untouched; extend imports to
  cover `ClaudeResult`, `ParsedResult`, `TraceEvent` if not already imported):

  ```python
  class InMemoryTraceStore:
      """Duck-typed TraceStore (append/get per contracts §Traces) for server tests and
      TESTING mode. Single-process only — which is exactly the single-worker constraint."""

      def __init__(self) -> None:
          self._events: dict[str, list[TraceEvent]] = {}

      def append(self, event: TraceEvent) -> None:
          self._events.setdefault(event.trace_id, []).append(event)

      def get(self, trace_id: str) -> list[TraceEvent]:
          return sorted(self._events.get(trace_id, []), key=lambda e: e.seq)


  class ScriptedTransport:
      """ClaudeTransport fake with cycling scripts (never exhausts across e2e runs).

      parse() validates the scripted payload into the *requested* output_format, so it
      stays correct even though Plan C owns the route/grade Pydantic models.
      """

      def __init__(self, completions: list[str], parse_payloads: list[dict]) -> None:
          self._completions = completions
          self._parse_payloads = parse_payloads
          self._c = 0
          self._p = 0

      def complete(self, *, model: str, system: str, user: str, max_tokens: int,
                   temperature: float = 0.0) -> ClaudeResult:
          text = self._completions[self._c % len(self._completions)]
          self._c += 1
          return ClaudeResult(text=text, input_tokens=120, output_tokens=40)

      def parse(self, *, model: str, system: str, user: str, max_tokens: int,
                output_format: type, temperature: float = 0.0) -> ParsedResult:
          payload = self._parse_payloads[self._p % len(self._parse_payloads)]
          self._p += 1
          return ParsedResult(parsed=output_format.model_validate(payload),
                              input_tokens=80, output_tokens=20)
  ```

**Step 3 — implement `server/deps.py` (COMPLETE code):**

- [ ] Create `api/ragreceipts/server/deps.py`:

  ```python
  """Application dependency container.

  Everything the endpoints need arrives through AppDeps — never module globals — so unit
  tests construct it directly with fakes, and TESTING=1 swaps the whole container.
  """
  from __future__ import annotations

  import os
  from dataclasses import dataclass
  from pathlib import Path
  from typing import Protocol

  from ragreceipts.server.jobs import JobRunner
  from ragreceipts.traces.models import TraceEvent

  VENDOR_ENV_VARS = {
      "voyage": "VOYAGE_API_KEY",
      "cohere": "COHERE_API_KEY",
      "anthropic": "ANTHROPIC_API_KEY",
  }


  class TraceReadWrite(Protocol):
      """Structural type of the contracts' TraceStore (append/get)."""

      def append(self, event: TraceEvent) -> None: ...
      def get(self, trace_id: str) -> list[TraceEvent]: ...


  @dataclass(frozen=True)
  class VendorCapability:
      name: str
      configured: bool
      env_var: str


  @dataclass(frozen=True)
  class AppPaths:
      data_dir: Path
      receipts_committed_dir: Path

      @property
      def corpora_dir(self) -> Path:
          return self.data_dir / "corpora"

      @property
      def receipts_local_dir(self) -> Path:
          return self.data_dir / "receipts-local"

      @property
      def uploads_dir(self) -> Path:
          return self.data_dir / "uploads"

      @property
      def jobs_db(self) -> Path:
          return self.data_dir / "server-jobs.sqlite"

      @property
      def traces_db(self) -> Path:
          return self.data_dir / "traces.sqlite"

      def ensure(self) -> None:
          for d in (self.data_dir, self.corpora_dir, self.receipts_local_dir, self.uploads_dir):
              d.mkdir(parents=True, exist_ok=True)

      @classmethod
      def from_env(cls) -> "AppPaths":
          return cls(
              data_dir=Path(os.environ.get("RAGRECEIPTS_DATA_DIR", "../data")).resolve(),
              receipts_committed_dir=Path(
                  os.environ.get("RAGRECEIPTS_RECEIPTS_DIR", "../receipts")
              ).resolve(),
          )


  @dataclass
  class AppDeps:
      paths: AppPaths
      vendors: list[VendorCapability]
      qdrant: object | None          # qdrant_client.QdrantClient when wired
      trace_store: TraceReadWrite
      job_runner: JobRunner
      query_runner: object | None    # server.pipeline.QueryRunner (Task 5)
      eval_runner: object | None     # server.evalruns.EvalRunner (Task 7)
      ingest_sink: object | None     # server.ingest_byo.IngestSink (Task 13, built last)
      testing_mode: bool


  def build_deps() -> AppDeps:
      """Build production deps from env — or fixture deps when TESTING=1 (see plan §Context)."""
      if os.environ.get("TESTING") == "1":
          try:
              from tests.e2e_fixture import build_testing_deps
          except ImportError as exc:  # degrade with a named cause, not a stack trace
              raise RuntimeError(
                  "TESTING=1 requires launching from api/ via 'uv run python -m uvicorn ...'"
                  " so the tests package is importable"
              ) from exc
          return build_testing_deps()

      paths = AppPaths.from_env()
      paths.ensure()
      vendors = [
          VendorCapability(name, bool(os.environ.get(env)), env)
          for name, env in VENDOR_ENV_VARS.items()
      ]
      # R7: the server REQUIRES QDRANT_URL — never a silent http://localhost:6333 default.
      # When unset, qdrant stays None and the healthcheck (/health) plus every gated
      # endpoint discloses the missing variable BY NAME (compose sets it; see Task 12).
      qdrant = None
      qdrant_url = os.environ.get("QDRANT_URL")
      if qdrant_url:
          from qdrant_client import QdrantClient  # Plan A dependency; constructor does not connect

          qdrant = QdrantClient(url=qdrant_url)
      from ragreceipts.traces.store import TraceStore

      trace_store = TraceStore(paths.traces_db)
      job_runner = JobRunner(paths.jobs_db)
      return AppDeps(
          paths=paths,
          vendors=vendors,
          qdrant=qdrant,
          trace_store=trace_store,
          job_runner=job_runner,
          query_runner=None,   # wired in Task 5 when all keys + QDRANT_URL are present
          eval_runner=None,    # wired in Task 7 when all keys + QDRANT_URL are present
          ingest_sink=None,    # wired in Task 13 when all keys + QDRANT_URL are present
          testing_mode=False,
      )
  ```

- [ ] **DISCOVERY (constructor check, 2 min):** open `api/ragreceipts/traces/store.py` (Plan
  C) and confirm `TraceStore`'s constructor takes the SQLite path. The contracts pin only
  `append`/`get`; if Plan C's constructor differs (e.g. takes a connection), adapt the single
  construction call in `build_deps` — nothing else in this plan touches it.

**Step 4 — implement the TESTING fixture module and the test helper:**

- [ ] Create `api/tests/e2e_fixture.py`:

  ```python
  """TESTING=1 wiring: the whole AppDeps container backed by offline fakes.

  Used by Playwright e2e (via `TESTING=1 uv run python -m uvicorn ...`) and by
  tests/test_testing_mode.py. No network, no keys. Extended by later plan tasks:
  Task 5 adds FixtureQueryRunner, Task 7 adds FixtureEvalRunner, Task 10 seeds a
  local receipt, Task 14 adds TestingIngestSink.
  """
  from __future__ import annotations

  import json
  import os
  import tempfile
  from datetime import datetime, timezone
  from pathlib import Path

  from qdrant_client import QdrantClient

  from ragreceipts.constants import EMBED_MODEL
  from ragreceipts.server.deps import VENDOR_ENV_VARS, AppDeps, AppPaths, VendorCapability
  from ragreceipts.server.jobs import JobRunner
  from tests.fakes import InMemoryTraceStore, ScriptedTransport

  FIXTURE_CORPUS_ID = "fixture-corpus"
  ANSWER_TEXT = "Paris is the capital of France [1][2]. It lies on the Seine [3]."
  ROUTE_PAYLOAD = {"complexity": "simple", "confidence": 0.95}


  def _write_fixture_manifest(paths: AppPaths) -> None:
      d = paths.corpora_dir / FIXTURE_CORPUS_ID
      d.mkdir(parents=True, exist_ok=True)
      manifest = {
          "corpus_id": FIXTURE_CORPUS_ID,
          "dataset": {"name": "e2e-fixture", "hf_id": None, "split": None, "revision": None},
          "chunking": {"chunk_size": 512, "chunk_overlap": 64},
          "embed_model": EMBED_MODEL,
          "index_hashes": {
              "dense_contextual": "sha256:fixture",
              "dense_isolated": "sha256:fixture",
              "sparse": "sha256:fixture",
          },
          "tokenizer_artifact": "fixture",
          "n_docs": 6, "n_chunks": 7, "n_queries": 0,
          "created_at": datetime.now(timezone.utc).isoformat(),
      }
      (d / "manifest.json").write_text(json.dumps(manifest, indent=2))


  def build_testing_deps() -> AppDeps:
      data_dir = Path(
          os.environ.get("RAGRECEIPTS_DATA_DIR", tempfile.mkdtemp(prefix="ragreceipts-testing-"))
      )
      receipts_dir = Path(os.environ.get("RAGRECEIPTS_RECEIPTS_DIR", "../receipts")).resolve()
      paths = AppPaths(data_dir=data_dir, receipts_committed_dir=receipts_dir)
      paths.ensure()
      _write_fixture_manifest(paths)
      return AppDeps(
          paths=paths,
          vendors=[VendorCapability(name, True, env) for name, env in VENDOR_ENV_VARS.items()],
          qdrant=QdrantClient(":memory:"),  # local mode supports named vectors (verified)
          trace_store=InMemoryTraceStore(),
          job_runner=JobRunner(paths.jobs_db),
          query_runner=None,   # FixtureQueryRunner wired in Task 5
          eval_runner=None,    # FixtureEvalRunner wired in Task 7
          ingest_sink=None,    # TestingIngestSink wired in Task 14
          testing_mode=True,
      )
  ```

- [ ] Create `api/tests/helpers_server.py`:

  ```python
  """Construct AppDeps wired entirely with offline fakes for endpoint unit tests."""
  from __future__ import annotations

  from pathlib import Path

  from ragreceipts.server.deps import AppDeps, AppPaths, VendorCapability
  from ragreceipts.server.jobs import JobRunner
  from tests.fakes import InMemoryTraceStore


  def make_test_deps(tmp_path: Path, *, configured: bool = False) -> AppDeps:
      paths = AppPaths(data_dir=tmp_path / "data", receipts_committed_dir=tmp_path / "receipts")
      paths.ensure()
      return AppDeps(
          paths=paths,
          vendors=[
              VendorCapability("voyage", configured, "VOYAGE_API_KEY"),
              VendorCapability("cohere", configured, "COHERE_API_KEY"),
              VendorCapability("anthropic", configured, "ANTHROPIC_API_KEY"),
          ],
          qdrant=None,
          trace_store=InMemoryTraceStore(),
          job_runner=JobRunner(paths.jobs_db),
          query_runner=None,
          eval_runner=None,
          ingest_sink=None,
          testing_mode=False,
      )
  ```

- [ ] Run: `cd api && uv run pytest tests/test_server_deps.py tests/test_server_jobs.py -q`
  — EXPECTED: all pass (4 + 6).
- [ ] Lint: `cd api && uv run ruff check ragreceipts/server tests`
- [ ] Commit:

  ```bash
  git add api/ragreceipts/server/deps.py api/tests/fakes.py api/tests/e2e_fixture.py api/tests/helpers_server.py api/tests/test_server_deps.py
  git commit -m "feat(server): AppDeps container, vendor capability detection, TESTING=1 fixture seam" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
  ```

---

### Task 4: FastAPI app, `GET /health`, OpenAPI export

**Files:**
- Create: `api/ragreceipts/server/app.py`
- Create: `api/ragreceipts/server/export_openapi.py`
- Test: `api/tests/test_server_health.py`

**Step 1 — failing test:**

- [ ] Create `api/tests/test_server_health.py`:

  ```python
  """/health: per-vendor capability with NAMED env vars (spec §Error handling)."""
  from fastapi.testclient import TestClient
  from qdrant_client import QdrantClient

  from ragreceipts.server.app import create_app
  from tests.helpers_server import make_test_deps


  def test_health_names_missing_env_vars(tmp_path):
      app = create_app(deps_factory=lambda: make_test_deps(tmp_path, configured=False))
      with TestClient(app) as client:  # context manager runs lifespan
          r = client.get("/health")
      assert r.status_code == 200
      body = r.json()
      assert body["status"] == "degraded"
      # R7: deps.qdrant is None (no QDRANT_URL) -> the healthcheck fails NAMING the env var
      assert body["missing_env_vars"] == [
          "VOYAGE_API_KEY", "COHERE_API_KEY", "ANTHROPIC_API_KEY", "QDRANT_URL",
      ]
      assert {v["name"]: v["configured"] for v in body["vendors"]} == {
          "voyage": False, "cohere": False, "anthropic": False,
      }
      assert body["qdrant_ok"] is False
      assert body["testing_mode"] is False


  def test_health_ok_when_configured_and_qdrant_reachable(tmp_path):
      deps = make_test_deps(tmp_path, configured=True)
      deps.qdrant = QdrantClient(":memory:")  # in-process local mode (named vectors verified)
      app = create_app(deps_factory=lambda: deps)
      with TestClient(app) as client:
          body = client.get("/health").json()
      assert body["status"] == "ok"
      assert body["missing_env_vars"] == []
      assert body["qdrant_ok"] is True


  def test_openapi_is_31_and_lists_health():
      app = create_app(deps_factory=lambda: None)  # schema generation never builds deps
      schema = app.openapi()
      assert schema["openapi"].startswith("3.1")  # FastAPI default (verified, see plan table)
      assert "/health" in schema["paths"]
  ```

- [ ] Run: `cd api && uv run pytest tests/test_server_health.py -q`
  — EXPECTED: `ModuleNotFoundError: No module named 'ragreceipts.server.app'`.

**Step 2 — implement `server/app.py` (COMPLETE code):**

- [ ] Create `api/ragreceipts/server/app.py`:

  ```python
  """FastAPI app (contracts §Server). Run SINGLE-WORKER only:

      cd api && uv run python -m uvicorn ragreceipts.server.app:app \
          --host 0.0.0.0 --port 8000 --workers 1

  Single worker is load-bearing: the JobRunner worker thread and its dispatch queue are
  in-process state; with more workers, jobs visible in SQLite would belong to a process
  that will never execute them. `python -m uvicorn` (not bare `uvicorn`) puts api/ on
  sys.path, which the TESTING=1 seam needs to import the tests package.
  """
  from __future__ import annotations

  import os
  from contextlib import asynccontextmanager
  from typing import Callable

  from fastapi import APIRouter, Depends, FastAPI, Request
  from fastapi.middleware.cors import CORSMiddleware

  from ragreceipts.server import models as m
  from ragreceipts.server.deps import AppDeps, build_deps

  router = APIRouter()


  def get_deps(request: Request) -> AppDeps:
      return request.app.state.deps


  def _missing_env_vars(deps: AppDeps) -> list[str]:
      """Vendor keys plus QDRANT_URL (R7: the server REQUIRES it; when unset, the
      healthcheck and every gated endpoint disclose it BY NAME — no silent default)."""
      missing = [v.env_var for v in deps.vendors if not v.configured]
      if deps.qdrant is None:
          missing.append("QDRANT_URL")
      return missing


  @router.get("/health", response_model=m.HealthResponse)
  def health(deps: AppDeps = Depends(get_deps)) -> m.HealthResponse:
      qdrant_ok = False
      if deps.qdrant is not None:
          try:
              deps.qdrant.get_collections()
              qdrant_ok = True
          except Exception:
              qdrant_ok = False  # report capability, never raise from a healthcheck
      missing = _missing_env_vars(deps)
      return m.HealthResponse(
          status="ok" if qdrant_ok and not missing else "degraded",
          vendors=[
              m.VendorStatusModel(name=v.name, configured=v.configured, env_var=v.env_var)
              for v in deps.vendors
          ],
          qdrant_ok=qdrant_ok,
          missing_env_vars=missing,
          testing_mode=deps.testing_mode,
      )


  def create_app(deps_factory: Callable[[], AppDeps] = build_deps) -> FastAPI:
      @asynccontextmanager
      async def lifespan(app: FastAPI):
          deps = deps_factory()
          app.state.deps = deps
          deps.job_runner.start()
          try:
              yield
          finally:
              deps.job_runner.stop()

      app = FastAPI(title="rag-receipts", version="0.1.0", lifespan=lifespan)
      app.add_middleware(
          CORSMiddleware,
          allow_origins=os.environ.get(
              "RAGRECEIPTS_CORS_ORIGINS", "http://localhost:3000"
          ).split(","),
          allow_methods=["*"],
          allow_headers=["*"],
      )
      app.include_router(router)
      return app


  app = create_app()
  ```

- [ ] Create `api/ragreceipts/server/export_openapi.py`:

  ```python
  """Print the OpenAPI 3.1 schema without starting the server or building deps.

  FastAPI emits OpenAPI 3.1.0 by default and serves it at /openapi.json; app.openapi()
  returns the same document programmatically (verified:
  https://fastapi.tiangolo.com/how-to/extending-openapi/). Used by web/ codegen:

      cd api && uv run python -m ragreceipts.server.export_openapi > ../web/openapi.json
  """
  import json

  from ragreceipts.server.app import create_app


  def main() -> None:
      print(json.dumps(create_app().openapi(), indent=2))


  if __name__ == "__main__":
      main()
  ```

- [ ] Run: `cd api && uv run pytest tests/test_server_health.py -q` — EXPECTED: `3 passed`.
- [ ] Sanity-run the export: `cd api && uv run python -m ragreceipts.server.export_openapi | head -5`
  — EXPECTED: JSON starting with `{"openapi": "3.1...`.
- [ ] Lint: `cd api && uv run ruff check ragreceipts/server`
- [ ] Commit:

  ```bash
  git add api/ragreceipts/server/app.py api/ragreceipts/server/export_openapi.py api/tests/test_server_health.py
  git commit -m "feat(server): FastAPI app with lifespan-managed deps, /health with named env vars, OpenAPI export" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
  ```

---

### Task 5: `POST /query` + `GET /traces/{trace_id}` + QueryRunner seam

**Files:**
- Create: `api/ragreceipts/server/pipeline.py`
- Create: `api/tests/fixtures/e2e_corpus.json`
- Modify: `api/ragreceipts/server/app.py` (add routes)
- Modify: `api/ragreceipts/server/deps.py` (wire real runner behind key check)
- Modify: `api/tests/e2e_fixture.py` (add FixtureQueryRunner)
- Test: `api/tests/test_server_query.py`, `api/tests/test_testing_mode.py`,
  `api/tests/test_real_query_runner.py` (Step 5 — offline adapter tests)

**Step 1 — failing endpoint tests:**

- [ ] Create `api/tests/test_server_query.py`:

  ```python
  """/query and /traces/{trace_id} against a stub QueryRunner (offline)."""
  import json

  from fastapi.testclient import TestClient

  from ragreceipts.constants import ROUTER_MODEL
  from ragreceipts.server.app import create_app
  from ragreceipts.server.pipeline import Citation, QueryResult
  from ragreceipts.traces.models import TraceEvent
  from tests.helpers_server import make_test_deps


  class StubQueryRunner:
      def __init__(self, trace_store) -> None:
          self._ts = trace_store

      def run(self, *, query: str, corpus_id: str, preset: str) -> QueryResult:
          trace_id = "t-123"
          self._ts.append(TraceEvent(
              trace_id=trace_id, seq=0, node="route", payload={"route": "s1"},
              model=ROUTER_MODEL, input_tokens=10, output_tokens=2, duration_ms=5.0,
          ))
          return QueryResult(
              answer="Paris [1].", abstained=False, route="s1",
              degraded=["rerank-skipped"],
              citations=[Citation(n=1, chunk_id="geo-001:0", passage_id="geo-001",
                                  text="Paris is the capital of France.", score=0.91)],
              trace_id=trace_id,
          )


  def make_app(tmp_path, *, with_runner: bool = True):
      deps = make_test_deps(tmp_path, configured=with_runner)
      if with_runner:
          deps.query_runner = StubQueryRunner(deps.trace_store)
      corpus = deps.paths.corpora_dir / "fixture-corpus"
      corpus.mkdir(parents=True, exist_ok=True)
      (corpus / "manifest.json").write_text(json.dumps({"corpus_id": "fixture-corpus"}))
      return create_app(deps_factory=lambda: deps)


  def test_query_round_trip_returns_answer_trace_and_degraded_flags(tmp_path):
      with TestClient(make_app(tmp_path)) as client:
          r = client.post("/query", json={
              "query": "capital of France?", "corpus_id": "fixture-corpus", "preset": "rerank",
          })
          assert r.status_code == 200, r.text
          body = r.json()
          assert body["answer"] == "Paris [1]."
          assert body["route"] == "s1"
          assert body["degraded"] == ["rerank-skipped"]
          assert body["citations"][0]["chunk_id"] == "geo-001:0"
          t = client.get(f"/traces/{body['trace_id']}")
          assert t.status_code == 200
          assert t.json()["events"][0]["node"] == "route"


  def test_unknown_corpus_404(tmp_path):
      with TestClient(make_app(tmp_path)) as client:
          r = client.post("/query", json={
              "query": "q", "corpus_id": "nope", "preset": "rerank",
          })
      assert r.status_code == 404
      assert "unknown corpus" in r.json()["detail"]


  def test_unknown_preset_422(tmp_path):
      with TestClient(make_app(tmp_path)) as client:
          r = client.post("/query", json={
              "query": "q", "corpus_id": "fixture-corpus", "preset": "bad",
          })
      assert r.status_code == 422


  def test_query_unavailable_names_missing_env_vars(tmp_path):
      with TestClient(make_app(tmp_path, with_runner=False)) as client:
          r = client.post("/query", json={
              "query": "q", "corpus_id": "fixture-corpus", "preset": "rerank",
          })
      assert r.status_code == 503
      assert "VOYAGE_API_KEY" in r.json()["detail"]


  def test_unknown_trace_404(tmp_path):
      with TestClient(make_app(tmp_path)) as client:
          assert client.get("/traces/missing").status_code == 404
  ```

- [ ] Run: `cd api && uv run pytest tests/test_server_query.py -q`
  — EXPECTED: `ModuleNotFoundError: No module named 'ragreceipts.server.pipeline'`.

**Step 2 — implement the seam (`server/pipeline.py`, COMPLETE code):**

- [ ] Create `api/ragreceipts/server/pipeline.py`:

  ```python
  """Query execution seam between HTTP and the Plan C agent graph."""
  from __future__ import annotations

  from dataclasses import dataclass
  from typing import Protocol


  @dataclass(frozen=True)
  class Citation:
      n: int               # matches the [n] markers in the answer text
      chunk_id: str
      passage_id: str
      text: str
      score: float


  @dataclass(frozen=True)
  class QueryResult:
      answer: str
      abstained: bool      # structured field per spec — never prose-only
      route: str           # "s1" | "s2"
      degraded: list[str]  # e.g. ["rerank-skipped"] (contracts §Vendor protocols)
      citations: list[Citation]
      trace_id: str


  class QueryRunner(Protocol):
      def run(self, *, query: str, corpus_id: str, preset: str) -> QueryResult: ...
  ```

**Step 3 — add the routes (Modify `api/ragreceipts/server/app.py`):**

- [ ] Add imports near the top of `app.py` (after the existing imports):

  ```python
  from dataclasses import asdict

  from fastapi import HTTPException
  ```

  (merge `HTTPException` into the existing `from fastapi import ...` line) and add the two
  routes after the `health` function:

  ```python
  @router.post("/query", response_model=m.QueryResponse)
  def post_query(req: m.QueryRequest, deps: AppDeps = Depends(get_deps)) -> m.QueryResponse:
      if deps.query_runner is None:
          missing = ", ".join(_missing_env_vars(deps))
          raise HTTPException(503, detail=f"query unavailable; missing env vars: {missing}")
      if not (deps.paths.corpora_dir / req.corpus_id / "manifest.json").exists():
          raise HTTPException(404, detail=f"unknown corpus: {req.corpus_id}")
      result = deps.query_runner.run(query=req.query, corpus_id=req.corpus_id, preset=req.preset)
      return m.QueryResponse(
          answer=result.answer,
          abstained=result.abstained,
          route=result.route,
          degraded=result.degraded,
          citations=[m.CitationModel(**asdict(c)) for c in result.citations],
          trace_id=result.trace_id,
      )


  @router.get("/traces/{trace_id}", response_model=m.TraceResponse)
  def get_trace(trace_id: str, deps: AppDeps = Depends(get_deps)) -> m.TraceResponse:
      events = deps.trace_store.get(trace_id)
      if not events:
          raise HTTPException(404, detail=f"unknown trace: {trace_id}")
      return m.TraceResponse(
          trace_id=trace_id,
          events=[m.TraceEventModel(**asdict(e)) for e in events],
      )
  ```

- [ ] Run: `cd api && uv run pytest tests/test_server_query.py -q` — EXPECTED: `5 passed`.

**Step 4 — fixture corpus + FixtureQueryRunner for TESTING mode:**

- [ ] Create `api/tests/fixtures/e2e_corpus.json`:

  ```json
  {
    "corpus_id": "fixture-corpus",
    "chunks": [
      {"doc_id": "geo-001", "passage_id": "geo-001", "position": 0,
       "text": "Paris is the capital and most populous city of France, on the river Seine."},
      {"doc_id": "geo-001", "passage_id": "geo-001", "position": 1,
       "text": "France is a country in Western Europe; its capital city is Paris."},
      {"doc_id": "geo-002", "passage_id": "geo-002", "position": 0,
       "text": "Berlin is the capital of Germany and its largest city."},
      {"doc_id": "geo-003", "passage_id": "geo-003", "position": 0,
       "text": "The Seine is a river in northern France flowing through Paris."},
      {"doc_id": "geo-004", "passage_id": "geo-004", "position": 0,
       "text": "Madrid is the capital of Spain, located on the Manzanares river."},
      {"doc_id": "geo-005", "passage_id": "geo-005", "position": 0,
       "text": "Mont Blanc is the highest mountain in the Alps, on the French-Italian border."},
      {"doc_id": "geo-006", "passage_id": "geo-006", "position": 0,
       "text": "Rome is the capital of Italy, on the river Tiber."}
    ]
  }
  ```

- [ ] Append to `api/tests/e2e_fixture.py` (and extend its imports with
  `import time`, `import uuid`, `from pydantic import BaseModel`,
  `from ragreceipts.constants import RERANK_MODEL, ROUTER_MODEL, SYNTH_MODEL`,
  `from ragreceipts.server.pipeline import Citation, QueryResult`,
  `from ragreceipts.traces.models import TraceEvent`,
  `from ragreceipts.types import Chunk`):

  ```python
  class RouteDecision(BaseModel):
      """Local stand-in for parse() output in TESTING mode; field names live in
      ROUTE_PAYLOAD so this never has to match Plan C's internal route model."""
      complexity: str
      confidence: float


  def load_fixture_chunks() -> list[Chunk]:
      raw = json.loads((Path(__file__).parent / "fixtures" / "e2e_corpus.json").read_text())
      # R3: Chunk carries whitespace-token offsets within its parent passage. Fixture
      # chunks are laid out back-to-back per doc, so the offsets are the running sum.
      offsets: dict[str, int] = {}
      chunks: list[Chunk] = []
      for c in raw["chunks"]:
          start = offsets.get(c["doc_id"], 0)
          end = start + len(c["text"].split())
          offsets[c["doc_id"]] = end
          chunks.append(Chunk(
              chunk_id=f"{c['doc_id']}:{c['position']}",
              corpus_id=raw["corpus_id"],
              doc_id=c["doc_id"],
              passage_id=c["passage_id"],
              text=c["text"],
              position=c["position"],
              start_token=start,
              end_token=end,
          ))
      return chunks


  class FixtureQueryRunner:
      """QueryRunner over the fixture corpus, FakeClaude-backed.

      Routing and the answer text come from a ClaudeTransport fake (ScriptedTransport);
      retrieval is deterministic lexical word-overlap so trace scores are stable. A query
      containing the word "degrade" triggers the rerank-skipped degraded path so e2e can
      assert the badge. This exercises the HTTP+UI contract; the real agent graph has its
      own offline tests in Plan C.
      """

      def __init__(self, transport: ScriptedTransport, chunks: list[Chunk],
                   trace_store) -> None:
          self._transport = transport
          self._chunks = chunks
          self._traces = trace_store

      def run(self, *, query: str, corpus_id: str, preset: str) -> QueryResult:
          trace_id = uuid.uuid4().hex
          t0 = time.perf_counter()
          parsed = self._transport.parse(
              model=ROUTER_MODEL, system="route", user=query,
              max_tokens=1024, output_format=RouteDecision,
          )
          decision = parsed.parsed
          self._traces.append(TraceEvent(
              trace_id=trace_id, seq=0, node="route",
              payload={"complexity": decision.complexity,
                       "confidence": decision.confidence, "route": "s1"},
              model=ROUTER_MODEL, input_tokens=parsed.input_tokens,
              output_tokens=parsed.output_tokens,
              duration_ms=(time.perf_counter() - t0) * 1000,
          ))

          degraded = ["rerank-skipped"] if "degrade" in query.lower() else []
          words = set(query.lower().split())
          scored = sorted(
              ((len(words & set(c.text.lower().split())), c) for c in self._chunks),
              key=lambda pair: (-pair[0], pair[1].chunk_id),
          )[:5]
          top = [(score / max(len(words), 1), c) for score, c in scored]
          self._traces.append(TraceEvent(
              trace_id=trace_id, seq=1, node="s1_retrieve",
              payload={
                  "k": 5,
                  "degraded": degraded,
                  "rerank_model": None if degraded else RERANK_MODEL,
                  "chunks": [
                      {"chunk_id": c.chunk_id, "passage_id": c.passage_id,
                       "score": round(s, 4), "text": c.text}
                      for s, c in top
                  ],
              },
              model=None, input_tokens=0, output_tokens=0, duration_ms=2.0,
          ))

          completion = self._transport.complete(
              model=SYNTH_MODEL, system="answer", user=query, max_tokens=4096,
          )
          self._traces.append(TraceEvent(
              trace_id=trace_id, seq=2, node="s1_answer",
              payload={"answer": completion.text, "abstained": False, "degraded": degraded},
              model=SYNTH_MODEL, input_tokens=completion.input_tokens,
              output_tokens=completion.output_tokens, duration_ms=8.0,
          ))
          citations = [
              Citation(n=i + 1, chunk_id=c.chunk_id, passage_id=c.passage_id,
                       text=c.text, score=round(s, 4))
              for i, (s, c) in enumerate(top)
          ]
          return QueryResult(
              answer=completion.text, abstained=False, route="s1",
              degraded=degraded, citations=citations, trace_id=trace_id,
          )
  ```

- [ ] In `build_testing_deps()`, replace `query_runner=None,   # FixtureQueryRunner wired in Task 5`
  with:

  ```python
          query_runner=FixtureQueryRunner(
              ScriptedTransport(completions=[ANSWER_TEXT], parse_payloads=[ROUTE_PAYLOAD]),
              load_fixture_chunks(),
              trace_store,
          ),
  ```

  hoisting `trace_store = InMemoryTraceStore()` to a local variable above the `return`
  (and passing the same object to `trace_store=trace_store`).

- [ ] Create `api/tests/test_testing_mode.py` — the offline integration test that protects
  Playwright from wiring drift:

  ```python
  """TESTING=1 end-to-end (in-process): the exact stack Playwright runs against."""
  from fastapi.testclient import TestClient

  from ragreceipts.server.app import create_app
  from ragreceipts.server.deps import build_deps


  def make_client(tmp_path, monkeypatch) -> TestClient:
      monkeypatch.setenv("TESTING", "1")
      monkeypatch.setenv("RAGRECEIPTS_DATA_DIR", str(tmp_path / "data"))
      monkeypatch.setenv("RAGRECEIPTS_RECEIPTS_DIR", str(tmp_path / "receipts"))
      return TestClient(create_app(deps_factory=build_deps))


  def test_testing_query_round_trip(tmp_path, monkeypatch):
      with make_client(tmp_path, monkeypatch) as client:
          assert client.get("/health").json()["testing_mode"] is True
          r = client.post("/query", json={
              "query": "What is the capital of France?",
              "corpus_id": "fixture-corpus", "preset": "rerank",
          })
          assert r.status_code == 200, r.text
          body = r.json()
          assert "Paris" in body["answer"]
          assert body["route"] == "s1"
          assert body["citations"][0]["passage_id"] == "geo-001"
          nodes = [e["node"] for e in client.get(f"/traces/{body['trace_id']}").json()["events"]]
          assert nodes == ["route", "s1_retrieve", "s1_answer"]


  def test_testing_degraded_path(tmp_path, monkeypatch):
      with make_client(tmp_path, monkeypatch) as client:
          body = client.post("/query", json={
              "query": "degrade: capital of France?",
              "corpus_id": "fixture-corpus", "preset": "rerank",
          }).json()
          assert body["degraded"] == ["rerank-skipped"]
  ```

- [ ] Run: `cd api && uv run pytest tests/test_server_query.py tests/test_testing_mode.py -q`
  — EXPECTED: `7 passed`.

**Step 5 — wire the real Plan C graph (R9-pinned entry points; production path only):**

- [ ] Verify the pinned signatures before wiring (drift guard — if anything differs from
  R9, reconcile ONLY the adapter below; the `QueryRunner` seam and the `/query` endpoint
  MUST NOT change):

  ```bash
  cd api && uv run python -c "
  import dataclasses, inspect
  from ragreceipts.agents.service import GraphResult, run_query
  from ragreceipts.cli import _build_core_real, _make_claude
  print(inspect.signature(run_query))
  print([f.name for f in dataclasses.fields(GraphResult)])
  print(inspect.signature(_build_core_real))"
  # EXPECTED (R9):
  #   (*, query: str, core: ..., claude: ..., store: ..., config: ..., trace_id: str | None = None) -> GraphResult
  #   ['final', 'system', 'trace_id', 'tokens_used', 'hops_used', 'retrieved']
  #   (cfg: PipelineConfig, corpus_id: str, data_dir: Path) -> RetrievalCore
  ```

- [ ] Write the failing offline adapter test `api/tests/test_real_query_runner.py`
  (COMPLETE code — construction + marshalling against fakes, zero keys, zero network):

  ```python
  """RealQueryRunner: GraphResult -> QueryResult marshalling against the R9 pins."""
  from ragreceipts.agents.schemas import FinalAnswer
  from ragreceipts.agents.service import GraphResult
  from ragreceipts.server.pipeline import RealQueryRunner
  from ragreceipts.traces.models import TraceEvent
  from ragreceipts.types import Chunk, ScoredChunk
  from tests.fakes import InMemoryTraceStore


  def _scored(doc_id: str, position: int, text: str, score: float) -> ScoredChunk:
      n_tokens = len(text.split())
      chunk = Chunk(chunk_id=f"{doc_id}:{position}", corpus_id="c1", doc_id=doc_id,
                    passage_id=doc_id, text=text, position=position,
                    start_token=0, end_token=n_tokens)
      return ScoredChunk(chunk=chunk, score=score, source="rerank")


  def test_run_marshals_graph_result_and_trace_degraded_flags(tmp_path):
      store = InMemoryTraceStore()
      store.append(TraceEvent(trace_id="t-1", seq=0, node="s1_retrieve",
                              payload={"degraded": ["rerank-skipped"]}, model=None,
                              input_tokens=0, output_tokens=0, duration_ms=1.0))
      retrieved = [_scored("geo-001", 0, "Paris is the capital of France.", 0.91)]
      graph_result = GraphResult(
          final=FinalAnswer(text="Paris [1].", citations=[1], abstained=False),
          system="s1", trace_id="t-1", tokens_used=160, hops_used=0, retrieved=retrieved,
      )
      seen: dict = {}

      def fake_run_query(*, query, core, claude, store, config):
          seen.update(query=query, core=core, preset=config.name)
          return graph_result

      runner = RealQueryRunner(
          data_dir=tmp_path, trace_store=store, claude="claude-transport",
          core_factory=lambda config, corpus_id, data_dir: f"core:{corpus_id}:{config.name}",
          run_query_fn=fake_run_query,
      )
      result = runner.run(query="capital of France?", corpus_id="c1", preset="rerank")
      assert seen == {"query": "capital of France?", "core": "core:c1:rerank",
                      "preset": "rerank"}
      assert result.answer == "Paris [1]." and result.route == "s1"
      assert result.abstained is False
      assert result.degraded == ["rerank-skipped"]  # collected from the trace, not invented
      assert result.citations[0].n == 1
      assert result.citations[0].chunk_id == "geo-001:0"
      assert result.trace_id == "t-1"


  def test_out_of_range_citation_indices_are_dropped(tmp_path):
      store = InMemoryTraceStore()
      graph_result = GraphResult(
          final=FinalAnswer(text="x [9].", citations=[9], abstained=True),
          system="s2", trace_id="t-2", tokens_used=10, hops_used=2, retrieved=[],
      )
      runner = RealQueryRunner(
          data_dir=tmp_path, trace_store=store, claude=None,
          core_factory=lambda config, corpus_id, data_dir: None,
          run_query_fn=lambda **kwargs: graph_result,
      )
      result = runner.run(query="q", corpus_id="c1", preset="router-on")
      assert result.citations == [] and result.route == "s2"
      assert result.abstained is True and result.degraded == []


  def test_construction_resolves_pinned_entry_points(tmp_path):
      from ragreceipts.agents.service import run_query  # noqa: F401  (R9 drift guard)
      from ragreceipts.cli import _build_core_real  # noqa: F401

      runner = RealQueryRunner(data_dir=tmp_path, trace_store=InMemoryTraceStore(),
                               claude=None)
      assert runner._run_query is run_query
      assert runner._core_factory is _build_core_real
  ```

  Run: `cd api && uv run pytest tests/test_real_query_runner.py -q`
  — EXPECTED: `ImportError: cannot import name 'RealQueryRunner'`.

- [ ] Append to `api/ragreceipts/server/pipeline.py` (extend the module imports at the top
  with `from pathlib import Path`, `from typing import Callable` — merged into the existing
  `typing` import — and `from ragreceipts.config import PRESETS`):

  ```python
  def _collect_degraded(events) -> list[str]:
      """Union of degraded flags recorded in the query's TraceEvents, first-seen order
      (degrade visibly, never silently — the flags live in the trace, not invented here)."""
      out: list[str] = []
      for ev in events:
          for flag in ev.payload.get("degraded") or []:
              if flag not in out:
                  out.append(flag)
      return out


  class RealQueryRunner:
      """QueryRunner over the R9-pinned production entry points.

      Pins (contracts §Seam Resolutions R9):
        - agents/service.py::run_query(query=, core=, claude=, store=, config=) -> GraphResult
        - cli.py::_build_core_real(config, corpus_id, data_dir) -> RetrievalCore
      GraphResult: final (FinalAnswer: text/citations/abstained), system ("s1"|"s2"),
      trace_id, tokens_used, hops_used, retrieved (list[ScoredChunk]).
      Constructor seams default to the real entry points; tests inject fakes.
      """

      def __init__(self, *, data_dir: Path, trace_store, claude,
                   core_factory: Callable | None = None,
                   run_query_fn: Callable | None = None) -> None:
          if core_factory is None:
              from ragreceipts.cli import _build_core_real  # R9 composition root

              core_factory = _build_core_real
          if run_query_fn is None:
              from ragreceipts.agents.service import run_query  # R9 graph entry point

              run_query_fn = run_query
          self._data_dir = data_dir
          self._trace_store = trace_store
          self._claude = claude
          self._core_factory = core_factory
          self._run_query = run_query_fn

      def run(self, *, query: str, corpus_id: str, preset: str) -> QueryResult:
          config = PRESETS[preset]
          core = self._core_factory(config, corpus_id, self._data_dir)
          result = self._run_query(query=query, core=core, claude=self._claude,
                                   store=self._trace_store, config=config)
          citations: list[Citation] = []
          for n in result.final.citations:
              if 1 <= n <= len(result.retrieved):  # out-of-range [n] markers are dropped
                  sc = result.retrieved[n - 1]
                  citations.append(Citation(n=n, chunk_id=sc.chunk.chunk_id,
                                            passage_id=sc.chunk.passage_id,
                                            text=sc.chunk.text, score=sc.score))
          return QueryResult(
              answer=result.final.text,
              abstained=result.final.abstained,
              route=result.system,
              degraded=_collect_degraded(self._trace_store.get(result.trace_id)),
              citations=citations,
              trace_id=result.trace_id,
          )


  def build_real_query_runner(*, paths, qdrant, trace_store) -> RealQueryRunner:
      """Production constructor — wired by deps.build_deps when all three vendor keys AND
      QDRANT_URL are present (R7). `qdrant` is accepted for parity with the deps container
      (it backs /health); core construction goes through Plan B's composition root
      `_build_core_real`, which reads QDRANT_URL itself — guaranteed set on this path.
      """
      from ragreceipts.cli import _make_claude  # Plan B's real ClaudeTransport factory

      return RealQueryRunner(data_dir=paths.data_dir, trace_store=trace_store,
                             claude=_make_claude())
  ```

- [ ] Run: `cd api && uv run pytest tests/test_real_query_runner.py -q`
  — EXPECTED: `3 passed`.

- [ ] In `deps.py` `build_deps()`, replace `query_runner=None,   # wired in Task 5 ...`:
  hoist a plain block above the `return AppDeps(`:

  ```python
      query_runner = None
      if qdrant is not None and all(v.configured for v in vendors):
          from ragreceipts.server.pipeline import build_real_query_runner

          query_runner = build_real_query_runner(
              paths=paths, qdrant=qdrant, trace_store=trace_store
          )
  ```

  and pass `query_runner=query_runner,` in the return — wired only when all three keys
  AND `QDRANT_URL` (R7) are present, else `None` → 503 with named vars.
- [ ] Verify construction offline (no network happens at build time):
  `cd api && VOYAGE_API_KEY=x COHERE_API_KEY=x ANTHROPIC_API_KEY=x QDRANT_URL=http://localhost:6333 uv run python -c "from ragreceipts.server.deps import build_deps; d = build_deps(); print(type(d.query_runner).__name__); d.job_runner.stop()"`
  — EXPECTED: `RealQueryRunner`, no exception.
- [ ] Re-run the full suite: `cd api && uv run pytest -q` — EXPECTED: all pass.
- [ ] Commit:

  ```bash
  git add api/ragreceipts/server api/tests
  git commit -m "feat(server): /query and /traces endpoints, QueryRunner seam, TESTING fixture runner, real graph wiring" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
  ```

---

### Task 6: `GET /corpora`, `GET /receipts` (committed + local merge), `GET /eval/runs`

**Files:**
- Modify: `api/ragreceipts/server/app.py`
- Test: `api/tests/test_server_catalog.py`

**Step 1 — failing test:**

- [ ] Create `api/tests/test_server_catalog.py`:

  ```python
  """Read-only catalog endpoints: corpora, receipts (committed+local), eval runs."""
  import json

  from fastapi.testclient import TestClient

  from ragreceipts.server.app import create_app
  from tests.helpers_server import make_test_deps


  def seed(deps):
      c = deps.paths.corpora_dir / "musique-dev-300"
      c.mkdir(parents=True)
      (c / "manifest.json").write_text(json.dumps({
          "corpus_id": "musique-dev-300", "n_docs": 10, "n_chunks": 42,
      }))
      deps.paths.receipts_committed_dir.mkdir(parents=True, exist_ok=True)
      (deps.paths.receipts_committed_dir / "headline.json").write_text(json.dumps({
          "schema_version": 1,
          "receipt": {"run_id": "r1", "preset": "rerank", "metrics": {"recall_at_5": 0.78}},
      }))
      (deps.paths.receipts_committed_dir / "corrupt.json").write_text("{not json")
      (deps.paths.receipts_local_dir / "local.json").write_text(json.dumps({
          "schema_version": 1,
          "receipt": {"run_id": "r2", "preset": "bm25-only", "metrics": {"recall_at_5": 0.55}},
      }))


  def test_corpora_lists_manifests(tmp_path):
      deps = make_test_deps(tmp_path)
      seed(deps)
      with TestClient(create_app(deps_factory=lambda: deps)) as client:
          body = client.get("/corpora").json()
      assert [c["corpus_id"] for c in body["corpora"]] == ["musique-dev-300"]
      assert body["corpora"][0]["manifest"]["n_chunks"] == 42


  def test_receipts_merges_committed_and_local_and_discloses_corrupt_files(tmp_path):
      deps = make_test_deps(tmp_path)
      seed(deps)
      with TestClient(create_app(deps_factory=lambda: deps)) as client:
          body = client.get("/receipts").json()
      by_source = {(r["source"], r["receipt"]["run_id"]) for r in body["receipts"]}
      assert by_source == {("committed", "r1"), ("local", "r2")}
      assert len(body["errors"]) == 1 and "corrupt.json" in body["errors"][0]


  def test_eval_runs_empty_initially(tmp_path):
      deps = make_test_deps(tmp_path)
      with TestClient(create_app(deps_factory=lambda: deps)) as client:
          assert client.get("/eval/runs").json() == {"runs": []}
  ```

- [ ] Run: `cd api && uv run pytest tests/test_server_catalog.py -q`
  — EXPECTED: 3 failures with `404 Not Found` (routes missing).

**Step 2 — implement (Modify `api/ragreceipts/server/app.py`):**

- [ ] Add `import json` and `from pathlib import Path` to the imports, then add after
  `get_trace`:

  ```python
  @router.get("/corpora", response_model=m.CorporaResponse)
  def list_corpora(deps: AppDeps = Depends(get_deps)) -> m.CorporaResponse:
      out: list[m.CorpusModel] = []
      if deps.paths.corpora_dir.exists():
          for manifest_path in sorted(deps.paths.corpora_dir.glob("*/manifest.json")):
              try:
                  manifest = json.loads(manifest_path.read_text())
              except json.JSONDecodeError:
                  manifest = {"error": "unreadable manifest"}  # disclose, don't hide
              out.append(m.CorpusModel(corpus_id=manifest_path.parent.name, manifest=manifest))
      return m.CorporaResponse(corpora=out)


  def _load_receipts(directory: Path, source: str, errors: list[str]) -> list[m.ReceiptEntryModel]:
      entries: list[m.ReceiptEntryModel] = []
      if not directory.exists():
          return entries
      for path in sorted(directory.glob("*.json")):
          try:
              data = json.loads(path.read_text())
              entries.append(m.ReceiptEntryModel(
                  source=source, path=str(path),
                  schema_version=int(data["schema_version"]), receipt=data["receipt"],
              ))
          except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
              errors.append(f"{path}: {exc.__class__.__name__}: {exc}")
      return entries


  @router.get("/receipts", response_model=m.ReceiptsResponse)
  def list_receipts(deps: AppDeps = Depends(get_deps)) -> m.ReceiptsResponse:
      errors: list[str] = []
      receipts = _load_receipts(deps.paths.receipts_committed_dir, "committed", errors)
      receipts += _load_receipts(deps.paths.receipts_local_dir, "local", errors)
      return m.ReceiptsResponse(receipts=receipts, errors=errors)


  @router.get("/eval/runs", response_model=m.EvalRunsResponse)
  def list_eval_runs(deps: AppDeps = Depends(get_deps)) -> m.EvalRunsResponse:
      return m.EvalRunsResponse(runs=[
          m.EvalRunListItem(
              job_id=r.job_id, status=r.status.value,
              corpus_id=r.params.get("corpus_id", "?"),
              preset=r.params.get("preset", "?"),
              slice=r.params.get("slice", "?"),
              created_at=r.created_at,
          )
          for r in deps.job_runner.list(kind="eval")
      ])
  ```

- [ ] Run: `cd api && uv run pytest tests/test_server_catalog.py -q` — EXPECTED: `3 passed`.
- [ ] Lint + full suite: `cd api && uv run ruff check ragreceipts/server && uv run pytest -q`
- [ ] Commit:

  ```bash
  git add api/ragreceipts/server/app.py api/tests/test_server_catalog.py
  git commit -m "feat(server): corpora, receipts (committed+local merge with disclosed errors), eval-runs listing" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
  ```

---

### Task 7: `POST /eval/runs` confirm gate + jobs endpoints + EvalRunner seam

**Files:**
- Create: `api/ragreceipts/server/evalruns.py`
- Modify: `api/ragreceipts/server/app.py` (eval + jobs routes, handler registration)
- Modify: `api/tests/e2e_fixture.py` (FixtureEvalRunner)
- Modify: `api/ragreceipts/server/deps.py` (wire real eval runner behind key check)
- Test: `api/tests/test_server_eval.py`,
  `api/tests/test_real_eval_runner.py` (Step 5 — offline adapter tests)

**Step 1 — failing test:**

- [ ] Create `api/tests/test_server_eval.py`:

  ```python
  """Eval runs: pre-run cost estimate, confirmation gate, job execution, resume API."""
  import time

  from fastapi.testclient import TestClient

  from ragreceipts.server.app import create_app
  from ragreceipts.server.evalruns import CostEstimate
  from tests.helpers_server import make_test_deps


  class StubEvalRunner:
      def __init__(self) -> None:
          self.ran: list[dict] = []

      def estimate(self, *, corpus_id: str, preset: str, slice_name: str) -> CostEstimate:
          n = 15 if slice_name == "smoke" else 300
          return CostEstimate(n_queries=n, est_tokens=n * 4700,
                              est_usd=round(n * 0.02, 2), pricing_table_version="2026-06-10")

      def run(self, *, corpus_id, preset, slice_name, spend_cap_usd, emit) -> str:
          emit("scoring", 0.5)
          self.ran.append({"corpus_id": corpus_id, "preset": preset, "slice": slice_name})
          return "run-001"


  def make_client(tmp_path):
      deps = make_test_deps(tmp_path, configured=True)
      deps.eval_runner = StubEvalRunner()
      return TestClient(create_app(deps_factory=lambda: deps)), deps


  def wait_status(client, job_id, want, timeout=10.0):
      deadline = time.time() + timeout
      while time.time() < deadline:
          body = client.get(f"/jobs/{job_id}").json()
          if body["status"] == want:
              return body
          time.sleep(0.05)
      raise AssertionError(f"job never reached {want}")


  def test_estimate_without_confirm_creates_no_job(tmp_path):
      client, deps = make_client(tmp_path)
      with client:
          r = client.post("/eval/runs", json={"corpus_id": "c1", "preset": "rerank"})
          assert r.status_code == 200
          body = r.json()
          assert body["status"] == "needs_confirmation"
          assert body["job_id"] is None
          assert body["estimate"]["n_queries"] == 15
          assert client.get("/eval/runs").json()["runs"] == []
      assert deps.eval_runner.ran == []


  def test_confirm_starts_job_that_runs_and_lists(tmp_path):
      client, deps = make_client(tmp_path)
      with client:
          r = client.post("/eval/runs", json={
              "corpus_id": "c1", "preset": "rerank", "confirm": True,
          })
          body = r.json()
          assert body["status"] == "started" and body["job_id"]
          done = wait_status(client, body["job_id"], "succeeded")
          assert any("run-001" in e["message"] for e in done["events"])
          runs = client.get("/eval/runs").json()["runs"]
          assert runs[0]["preset"] == "rerank" and runs[0]["status"] == "succeeded"
      assert deps.eval_runner.ran[0]["slice"] == "smoke"


  def test_estimate_over_spend_cap_refused(tmp_path):
      client, _ = make_client(tmp_path)
      with client:
          r = client.post("/eval/runs", json={
              "corpus_id": "c1", "preset": "rerank", "slice": "full",
              "confirm": True, "spend_cap_usd": 1.0,
          })
      assert r.status_code == 400
      assert "exceeds spend cap" in r.json()["detail"]


  def test_eval_unavailable_names_missing_env_vars(tmp_path):
      deps = make_test_deps(tmp_path, configured=False)
      with TestClient(create_app(deps_factory=lambda: deps)) as client:
          r = client.post("/eval/runs", json={"corpus_id": "c1", "preset": "rerank"})
      assert r.status_code == 503
      assert "ANTHROPIC_API_KEY" in r.json()["detail"]


  def test_jobs_404_and_resume_409(tmp_path):
      client, _ = make_client(tmp_path)
      with client:
          assert client.get("/jobs/none").status_code == 404
          r = client.post("/eval/runs", json={
              "corpus_id": "c1", "preset": "rerank", "confirm": True,
          })
          job_id = r.json()["job_id"]
          wait_status(client, job_id, "succeeded")
          assert client.post(f"/jobs/{job_id}/resume").status_code == 409
  ```

- [ ] Run: `cd api && uv run pytest tests/test_server_eval.py -q`
  — EXPECTED: `ModuleNotFoundError: No module named 'ragreceipts.server.evalruns'`.

**Step 2 — implement `server/evalruns.py` (COMPLETE code):**

- [ ] Create `api/ragreceipts/server/evalruns.py`:

  ```python
  """Eval-run seam (spec: estimate -> confirm -> hard cap).

  Plan B owns BOTH the authoritative runner and the authoritative cost estimator:
  `eval/runner.py::estimate_run_cost(preset_names, n_queries) -> float` (R9-pinned).
  The server never re-implements a pricing formula — `RealEvalRunner.estimate` (Step 5)
  delegates to `estimate_run_cost`, which already prices Claude synthesis + voyage query
  embeddings + cohere rerank per query (and, after Plan C, the System-2 estimate for AUTO
  presets — R10). The mid-run hard spend cap also lives in Plan B's runner
  (`SpendCapExceeded`); a raise fails the job with the named error.
  """
  from __future__ import annotations

  from dataclasses import dataclass
  from typing import Callable, Protocol


  @dataclass(frozen=True)
  class CostEstimate:
      n_queries: int
      est_tokens: int
      est_usd: float
      pricing_table_version: str


  class EvalRunner(Protocol):
      def estimate(self, *, corpus_id: str, preset: str, slice_name: str) -> CostEstimate: ...

      def run(self, *, corpus_id: str, preset: str, slice_name: str,
              spend_cap_usd: float, emit: Callable[[str, float], None]) -> str:
          """Execute the run; emit(message, progress) streams job events; returns run_id."""
          ...
  ```

**Step 3 — eval + jobs routes and handler registration (Modify `app.py`):**

- [ ] Add the routes after `list_eval_runs`:

  ```python
  @router.post("/eval/runs", response_model=m.EvalRunResponse)
  def create_eval_run(req: m.EvalRunRequest, deps: AppDeps = Depends(get_deps)) -> m.EvalRunResponse:
      if deps.eval_runner is None:
          missing = ", ".join(_missing_env_vars(deps))
          raise HTTPException(503, detail=f"eval unavailable; missing env vars: {missing}")
      est = deps.eval_runner.estimate(
          corpus_id=req.corpus_id, preset=req.preset, slice_name=req.slice,
      )
      estimate = m.CostEstimateModel(**asdict(est))
      if not req.confirm:  # confirmation gate: nothing runs until confirm=true
          return m.EvalRunResponse(status="needs_confirmation", estimate=estimate, job_id=None)
      if est.est_usd > req.spend_cap_usd:
          raise HTTPException(
              400,
              detail=f"estimated cost ${est.est_usd:.2f} exceeds spend cap"
                     f" ${req.spend_cap_usd:.2f}; raise spend_cap_usd to proceed",
          )
      job_id = deps.job_runner.submit("eval", req.model_dump())
      return m.EvalRunResponse(status="started", estimate=estimate, job_id=job_id)


  def _job_response(deps: AppDeps, job_id: str) -> m.JobResponse:
      row = deps.job_runner.get(job_id)
      if row is None:
          raise HTTPException(404, detail=f"unknown job: {job_id}")
      return m.JobResponse(
          job_id=row.job_id, kind=row.kind, status=row.status.value,
          params=row.params, error=row.error,
          events=[m.JobEventModel(**asdict(e)) for e in deps.job_runner.events(job_id)],
      )


  @router.get("/jobs/{job_id}", response_model=m.JobResponse)
  def get_job(job_id: str, deps: AppDeps = Depends(get_deps)) -> m.JobResponse:
      return _job_response(deps, job_id)


  @router.post("/jobs/{job_id}/resume", response_model=m.JobResponse)
  def resume_job(job_id: str, deps: AppDeps = Depends(get_deps)) -> m.JobResponse:
      try:
          deps.job_runner.resume(job_id)
      except KeyError:
          raise HTTPException(404, detail=f"unknown job: {job_id}")
      except ValueError as exc:
          raise HTTPException(409, detail=str(exc))
      return _job_response(deps, job_id)
  ```

- [ ] Register the eval handler in `create_app`'s lifespan — replace
  `deps.job_runner.start()` with:

  ```python
          if deps.eval_runner is not None:
              from ragreceipts.server.evalruns import EvalRunner  # noqa: F401  (protocol doc)

              eval_runner = deps.eval_runner

              def eval_handler(ctx):
                  p = ctx.params
                  ctx.emit(f"eval start preset={p['preset']} slice={p['slice']}", 0.0)
                  run_id = eval_runner.run(
                      corpus_id=p["corpus_id"], preset=p["preset"], slice_name=p["slice"],
                      spend_cap_usd=p["spend_cap_usd"], emit=ctx.emit,
                  )
                  ctx.emit(f"eval complete run_id={run_id}", 1.0)

              deps.job_runner.register("eval", eval_handler)
          deps.job_runner.start()
  ```

- [ ] Run: `cd api && uv run pytest tests/test_server_eval.py -q` — EXPECTED: `5 passed`.

**Step 4 — FixtureEvalRunner for TESTING mode (Modify `api/tests/e2e_fixture.py`):**

- [ ] Append (extend imports with `from ragreceipts.eval.pricing import PRICING_VERSION`
  and `from ragreceipts.server.evalruns import CostEstimate`):

  ```python
  _PRESET_RECALL = {"bm25-only": 0.55, "dense-rrf": 0.63, "contextual": 0.66,
                    "rerank": 0.78, "router-on": 0.80}
  # Per-preset index hashes mirror the contracts: IngestConfig.contextual selects the
  # named vector AND the matching manifest hash. dense-rrf queries dense_isolated;
  # contextual/rerank/router-on query dense_contextual — the differing dense hash is
  # what drives the Ablation Lab's cell-level cross-index marker (R11).
  _PRESET_INDEX_HASHES = {
      "bm25-only": {"sparse": "sha256:fixture-sparse"},
      "dense-rrf": {"dense_isolated": "sha256:fixture-iso", "sparse": "sha256:fixture-sparse"},
      "contextual": {"dense_contextual": "sha256:fixture-ctx", "sparse": "sha256:fixture-sparse"},
      "rerank": {"dense_contextual": "sha256:fixture-ctx", "sparse": "sha256:fixture-sparse"},
      "router-on": {"dense_contextual": "sha256:fixture-ctx", "sparse": "sha256:fixture-sparse"},
  }


  def _fixture_receipt(preset: str, run_id: str) -> dict:
      r5 = _PRESET_RECALL.get(preset, 0.6)
      return {
          "run_id": run_id, "corpus_id": FIXTURE_CORPUS_ID, "preset": preset,
          "config": {"name": preset},
          "index_hashes": _PRESET_INDEX_HASHES.get(preset, {"sparse": "sha256:fixture-sparse"}),
          "models": {"router": "claude-haiku-4-5-20251001", "synth": "claude-sonnet-4-6",
                     "judge": "claude-sonnet-4-6", "rerank": "rerank-v4.0-pro",
                     "embed": "voyage-context-3"},
          "pricing_table_version": PRICING_VERSION,
          "prompts_version": "n/a",
          "n_total": 15, "n_failed": 0, "n_abstained": 1,
          "metrics": {"recall_at_5": r5, "mrr_at_3": round(r5 - 0.14, 2), "em": 0.33,
                      "f1": 0.46, "ragas_faithfulness": 0.79, "ragas_answer_relevancy": 0.74,
                      "latency_p50_ms": 820, "latency_p95_ms": 1900, "usd_per_query": 0.011},
          # R11: per_query rows use the committed schema exactly —
          # {query_id, retrieved_chunk_ids, latency_ms, usd, flags: {...}}
          "per_query": [{"query_id": "q-001", "retrieved_chunk_ids": ["geo-001:0"],
                         "latency_ms": 800, "usd": 0.01, "flags": {}}],
          "anchors": [],
      }


  class FixtureEvalRunner:
      """Deterministic EvalRunner for TESTING mode: fixed estimate; writes a complete
      schema_version-1 receipt to data/receipts-local/ so the Ablation Lab local toggle
      has real files to render."""

      def __init__(self, paths: AppPaths) -> None:
          self._paths = paths

      def estimate(self, *, corpus_id: str, preset: str, slice_name: str) -> CostEstimate:
          n = 15 if slice_name == "smoke" else 300
          return CostEstimate(n_queries=n, est_tokens=n * 4700,
                              est_usd=round(n * 0.018, 2),
                              pricing_table_version=PRICING_VERSION)

      def run(self, *, corpus_id, preset, slice_name, spend_cap_usd, emit) -> str:
          import uuid as _uuid

          run_id = f"local-{preset}-{_uuid.uuid4().hex[:8]}"
          emit("scoring fixture queries", 0.5)
          path = self._paths.receipts_local_dir / f"{run_id}.json"
          # R11 committed-envelope schema: nondeterminism_note is Plan B's fixed
          # constant — import it rather than duplicating the literal.
          from ragreceipts.eval.receipts import NONDETERMINISM_NOTE

          path.write_text(json.dumps(
              {"schema_version": 1, "nondeterminism_note": NONDETERMINISM_NOTE,
               "receipt": _fixture_receipt(preset, run_id)}, indent=2,
          ))
          emit("receipt written", 1.0)
          return run_id
  ```

- [ ] In `build_testing_deps()`, replace `eval_runner=None,    # FixtureEvalRunner wired in Task 7`
  with `eval_runner=FixtureEvalRunner(paths),`.

**Step 5 — wire Plan B's real runner + estimator (R9-pinned; production path only):**

- [ ] Verify the pinned signatures (drift guard — if anything differs from R9, reconcile
  ONLY the adapter below; the `EvalRunner` protocol and the routes MUST NOT change):

  ```bash
  cd api && uv run python -c "
  import inspect
  from ragreceipts.eval.run_state import RunStore
  from ragreceipts.eval.runner import AblationRunner, estimate_run_cost, new_run_id
  print(inspect.signature(estimate_run_cost))
  print(inspect.signature(AblationRunner.__init__))
  print(inspect.signature(AblationRunner.run))
  print(inspect.signature(new_run_id))"
  # EXPECTED (R9):
  #   (preset_names: list[str], n_queries: int) -> float
  #   (self, *, core_factory: ..., claude: ..., store: ..., data_dir: ..., ragas: ... = None, clock: ... = ...)
  #   (self, *, run_id: str, corpus_id: str, slice_name: str, presets: list[str], spend_cap_usd: float) -> dict
  #   (corpus_id: str, slice_name: str) -> str
  ```

- [ ] Write the failing offline adapter test `api/tests/test_real_eval_runner.py`
  (COMPLETE code — construction + marshalling against fakes, zero keys, zero network):

  ```python
  """RealEvalRunner: delegates estimate/run to Plan B's R9-pinned entry points."""
  from ragreceipts.eval.pricing import PRICING_VERSION
  from ragreceipts.eval.runner import (
      EST_QUERY_EMBED_TOKENS,
      EST_SYNTH_INPUT_TOKENS,
      EST_SYNTH_OUTPUT_TOKENS,
  )
  from ragreceipts.server.evalruns import RealEvalRunner


  class RecordingRunner:
      def __init__(self) -> None:
          self.calls: list[dict] = []

      def run(self, **kwargs):
          self.calls.append(kwargs)
          return {"receipts": [], "skipped": []}


  def test_estimate_delegates_to_plan_b_estimate_run_cost(tmp_path):
      estimate_calls: list[tuple] = []

      def fake_estimate(preset_names, n_queries):
          estimate_calls.append((preset_names, n_queries))
          return 0.2536

      runner = RealEvalRunner(
          data_dir=tmp_path,
          n_queries_fn=lambda corpus_id, slice_name: 15,
          estimate_fn=fake_estimate,
      )
      est = runner.estimate(corpus_id="c1", preset="rerank", slice_name="smoke")
      assert estimate_calls == [(["rerank"], 15)]  # NOT a re-implemented formula
      assert est.n_queries == 15
      assert est.est_usd == 0.2536
      # rerank preset queries dense, so the token estimate includes the query embed
      assert est.est_tokens == 15 * (
          EST_SYNTH_INPUT_TOKENS + EST_SYNTH_OUTPUT_TOKENS + EST_QUERY_EMBED_TOKENS
      )
      assert est.pricing_table_version == PRICING_VERSION


  def test_run_invokes_ablation_runner_with_single_preset(tmp_path):
      rec = RecordingRunner()
      messages: list[str] = []
      runner = RealEvalRunner(
          data_dir=tmp_path,
          runner_factory=lambda corpus_id: rec,
          run_id_fn=lambda corpus_id, slice_name: "run-xyz",
      )
      run_id = runner.run(corpus_id="c1", preset="rerank", slice_name="smoke",
                          spend_cap_usd=5.0, emit=lambda msg, p: messages.append(msg))
      assert run_id == "run-xyz"
      assert rec.calls == [{
          "run_id": "run-xyz", "corpus_id": "c1", "slice_name": "smoke",
          "presets": ["rerank"], "spend_cap_usd": 5.0,
      }]
      assert any("run-xyz" in msg for msg in messages)


  def test_construction_resolves_pinned_entry_points(tmp_path):
      # Drift guard in test form: the R9 names must import; signature drift reconciles
      # ONLY the adapter, never the EvalRunner protocol or the routes.
      from ragreceipts.cli import _build_core_real, _make_claude  # noqa: F401
      from ragreceipts.eval.run_state import RunStore  # noqa: F401
      from ragreceipts.eval.runner import (  # noqa: F401
          AblationRunner,
          estimate_run_cost,
          new_run_id,
      )

      RealEvalRunner(data_dir=tmp_path)  # constructs without touching Plan B or network
  ```

  Run: `cd api && uv run pytest tests/test_real_eval_runner.py -q`
  — EXPECTED: `ImportError: cannot import name 'RealEvalRunner'`.

- [ ] Append `RealEvalRunner` to `api/ragreceipts/server/evalruns.py` (extend the module
  imports with `from pathlib import Path`, `from ragreceipts.config import PRESETS`, and
  `from ragreceipts.eval.pricing import PRICING_VERSION`):

  ```python
  class RealEvalRunner:
      """EvalRunner over Plan B's R9-pinned entry points.

      Pins (contracts §Seam Resolutions R9/R10):
        - eval/runner.py::estimate_run_cost(preset_names, n_queries) -> float (USD) —
          the ONLY estimator; it prices Claude + voyage + cohere and (post-Plan C) the
          System-2 estimate for AUTO presets. No server-side pricing formula exists.
        - eval/runner.py::AblationRunner(core_factory=, claude=, store=, data_dir=,
          ragas=None) with .run(run_id=, corpus_id=, slice_name=, presets=, spend_cap_usd=)
        - cli.py::_build_core_real(config, corpus_id, data_dir) (composition root)
      Plan B's hard mid-run spend cap (SpendCapExceeded) propagates out of run() and
      fails the job with the named error. Constructor seams default to the real entry
      points; tests inject fakes.
      """

      def __init__(self, *, data_dir: Path,
                   n_queries_fn: Callable[[str, str], int] | None = None,
                   estimate_fn: Callable[[list[str], int], float] | None = None,
                   runner_factory: Callable[[str], object] | None = None,
                   run_id_fn: Callable[[str, str], str] | None = None) -> None:
          self._data_dir = data_dir
          self._n_queries_fn = n_queries_fn
          self._estimate_fn = estimate_fn
          self._runner_factory = runner_factory
          self._run_id_fn = run_id_fn

      def estimate(self, *, corpus_id: str, preset: str, slice_name: str) -> CostEstimate:
          from ragreceipts.eval.runner import (
              EST_QUERY_EMBED_TOKENS,
              EST_SYNTH_INPUT_TOKENS,
              EST_SYNTH_OUTPUT_TOKENS,
              estimate_run_cost,
          )

          n_queries_fn = self._n_queries_fn or self._count_slice_queries
          estimate_fn = self._estimate_fn or estimate_run_cost
          n = n_queries_fn(corpus_id, slice_name)
          per_q_tokens = EST_SYNTH_INPUT_TOKENS + EST_SYNTH_OUTPUT_TOKENS
          if PRESETS[preset].query.dense:
              per_q_tokens += EST_QUERY_EMBED_TOKENS
          return CostEstimate(
              n_queries=n,
              est_tokens=n * per_q_tokens,
              est_usd=round(estimate_fn([preset], n), 4),
              pricing_table_version=PRICING_VERSION,
          )

      def run(self, *, corpus_id: str, preset: str, slice_name: str,
              spend_cap_usd: float, emit: Callable[[str, float], None]) -> str:
          if self._run_id_fn is None:
              from ragreceipts.eval.runner import new_run_id as run_id_fn
          else:
              run_id_fn = self._run_id_fn
          runner_factory = self._runner_factory or self._build_runner
          run_id = run_id_fn(corpus_id, slice_name)
          emit(f"eval run {run_id}: preset={preset} slice={slice_name}", 0.05)
          runner_factory(corpus_id).run(
              run_id=run_id, corpus_id=corpus_id, slice_name=slice_name,
              presets=[preset], spend_cap_usd=spend_cap_usd,
          )
          emit(f"receipt written: receipts-local/{run_id}.json", 0.95)
          return run_id

      # -- real-entry-point defaults (overridden by fakes in tests) --------------------

      def _count_slice_queries(self, corpus_id: str, slice_name: str) -> int:
          from ragreceipts.eval.queries import load_queries, slice_queries, slice_query_ids

          # slice_queries takes query IDs, not a slice NAME — resolve the name first
          # via the slice files (Plan B's slice_query_ids).
          return len(slice_queries(
              load_queries(self._data_dir, corpus_id),
              slice_query_ids(self._data_dir, corpus_id, slice_name),
          ))

      def _build_runner(self, corpus_id: str):
          from ragreceipts.cli import _build_core_real, _make_claude
          from ragreceipts.eval.run_state import RunStore
          from ragreceipts.eval.runner import AblationRunner

          return AblationRunner(
              core_factory=lambda cfg: _build_core_real(cfg, corpus_id, self._data_dir),
              claude=_make_claude(),
              store=RunStore(self._data_dir / "eval-runs.db"),
              data_dir=self._data_dir,
          )
  ```

- [ ] Run: `cd api && uv run pytest tests/test_real_eval_runner.py -q`
  — EXPECTED: `3 passed`.

- [ ] In `deps.py` `build_deps()`, wire it inside the same key+`QDRANT_URL` check added
  for `query_runner` in Task 5:

  ```python
      eval_runner = None
      if qdrant is not None and all(v.configured for v in vendors):
          from ragreceipts.server.evalruns import RealEvalRunner

          eval_runner = RealEvalRunner(data_dir=paths.data_dir)
  ```

  (fold into the same `if` block as `query_runner`; pass `eval_runner=eval_runner,`).
- [ ] Note (no code): the server runs Plan B's runner with `ragas=None` — RAGAS judging
  stays a CLI concern (`--ragas`), and Plan B nulls those receipt metrics when absent. A
  `SpendCapExceeded` raise propagates out of the job handler and fails the job with the
  named mid-run cap error — the disclosed behavior the spec requires.
- [ ] Run the full suite: `cd api && uv run pytest -q` — EXPECTED: all pass.
- [ ] Commit:

  ```bash
  git add api/ragreceipts/server api/tests
  git commit -m "feat(server): eval runs with cost estimate + confirmation gate, jobs API with resume, EvalRunner seam" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
  ```

---

### Task 8: Next.js scaffold, typed OpenAPI client, layout, Playwright harness

**Files:**
- Create: `web/` (scaffolded), `web/src/lib/api/client.ts`, `web/src/lib/api/schema.d.ts`
  (generated), `web/openapi.json` (generated), `web/src/app/layout.tsx`,
  `web/src/app/globals.css`, `web/playwright.config.ts`
- Test: `web/e2e/nav.spec.ts`

**Step 1 — scaffold (exact commands; flags verified against nextjs.org):**

- [ ] From the repo root:

  ```bash
  pnpm create next-app@15 web --typescript --eslint --app --src-dir --no-tailwind --import-alias "@/*" --use-pnpm --yes
  cd web
  pnpm add openapi-fetch@^0.17 recharts@^3
  pnpm add -D openapi-typescript@^7 @playwright/test@^1.50
  pnpm exec playwright install chromium
  ```

- [ ] Delete the scaffold's demo content: `web/src/app/page.tsx` body (replaced in Task 9)
  and any `page.module.css` / hero assets it created. Keep `globals.css` (replaced below).

**Step 2 — generate the typed client (exact codegen commands):**

- [ ] Export the schema and generate types (re-run these two commands whenever an endpoint
  changes; both outputs are committed so `docker build` never needs a running api):

  ```bash
  cd api && uv run python -m ragreceipts.server.export_openapi > ../web/openapi.json
  cd ../web && pnpm exec openapi-typescript openapi.json -o src/lib/api/schema.d.ts
  ```

- [ ] Add the codegen script to `web/package.json` `"scripts"`:

  ```json
  "gen:api": "openapi-typescript openapi.json -o src/lib/api/schema.d.ts",
  "e2e": "playwright test"
  ```

- [ ] Create `web/src/lib/api/client.ts`:

  ```ts
  import createClient from "openapi-fetch";
  import type { paths } from "./schema";

  export const API_BASE =
    process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

  // Typed client generated from FastAPI's OpenAPI 3.1 schema.
  // Usage verified: https://openapi-ts.dev/openapi-fetch/
  export const api = createClient<paths>({ baseUrl: API_BASE });
  ```

**Step 3 — layout and global styles (COMPLETE code):**

- [ ] Replace `web/src/app/layout.tsx`:

  ```tsx
  import type { Metadata } from "next";
  import Link from "next/link";
  import "./globals.css";

  export const metadata: Metadata = {
    title: "rag-receipts",
    description: "Every RAG technique, with receipts.",
  };

  export default function RootLayout({ children }: { children: React.ReactNode }) {
    return (
      <html lang="en">
        <body>
          <header className="nav">
            <span className="brand">
              rag-receipts <span className="tagline">every technique, with receipts</span>
            </span>
            <nav>
              <Link href="/">Playground</Link>
              <Link href="/ablation">Ablation Lab</Link>
              <Link href="/corpora">Corpora</Link>
            </nav>
          </header>
          <main className="main">{children}</main>
        </body>
      </html>
    );
  }
  ```

- [ ] Replace `web/src/app/globals.css`:

  ```css
  :root {
    --ink: #18181b;
    --muted: #71717a;
    --line: #e4e4e7;
    --bg: #fafafa;
    --card: #ffffff;
    --accent: #2563eb;
    --ok: #16a34a;
    --warn: #b45309;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    color: var(--ink);
    background: var(--bg);
    font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  }
  code, .mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.86em; }
  .nav {
    display: flex; justify-content: space-between; align-items: baseline;
    padding: 14px 28px; border-bottom: 1px solid var(--line); background: var(--card);
  }
  .brand { font-weight: 700; }
  .tagline { font-weight: 400; color: var(--muted); font-size: 13px; margin-left: 8px; }
  .nav nav a { margin-left: 20px; color: var(--ink); text-decoration: none; }
  .nav nav a:hover { color: var(--accent); }
  .main { max-width: 980px; margin: 28px auto; padding: 0 20px; }
  .card {
    background: var(--card); border: 1px solid var(--line); border-radius: 10px;
    padding: 18px 20px; margin-bottom: 18px;
  }
  .muted { color: var(--muted); }
  .row { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
  .badge {
    display: inline-block; padding: 1px 9px; border-radius: 999px; font-size: 12px;
    border: 1px solid var(--line); background: var(--bg);
  }
  .badge-s1 { border-color: var(--ok); color: var(--ok); }
  .badge-s2 { border-color: var(--accent); color: var(--accent); }
  .badge-degraded { border-color: var(--warn); color: var(--warn); }
  .badge-ok { border-color: var(--ok); color: var(--ok); }
  button, select, input[type="text"] {
    font: inherit; padding: 7px 12px; border: 1px solid var(--line); border-radius: 8px;
    background: var(--card);
  }
  button.primary { background: var(--ink); color: #fff; border-color: var(--ink); cursor: pointer; }
  button.primary:disabled { opacity: 0.5; cursor: default; }
  textarea {
    font: inherit; width: 100%; padding: 10px 12px; border: 1px solid var(--line);
    border-radius: 8px; resize: vertical;
  }
  .cite {
    border: none; background: none; color: var(--accent); cursor: pointer; padding: 0 1px;
    font: inherit;
  }
  .popover { border: 1px solid var(--line); border-radius: 8px; padding: 12px; background: var(--bg); margin-top: 10px; }
  .popover-head { display: flex; justify-content: space-between; color: var(--muted); margin-bottom: 6px; }
  .trace { list-style: none; padding: 0; margin: 0; }
  .trace-event { border-top: 1px solid var(--line); padding: 10px 2px; }
  .trace-head { display: flex; gap: 12px; align-items: baseline; flex-wrap: wrap; }
  .trace-head .node { font-weight: 600; }
  .trace-head .ms, .trace-head .tokens { color: var(--muted); font-size: 13px; }
  table.chunks { width: 100%; border-collapse: collapse; margin-top: 8px; font-size: 13px; }
  table.chunks td { padding: 3px 8px 3px 0; border-top: 1px dashed var(--line); }
  .error { color: #b91c1c; }
  .anchor { border-top: 1px solid var(--line); padding: 10px 0; }
  .anchor-head { display: flex; gap: 14px; flex-wrap: wrap; align-items: baseline; }
  blockquote { margin: 8px 0 0; padding: 8px 12px; border-left: 3px solid var(--line); color: var(--muted); }
  .progress { height: 8px; background: var(--line); border-radius: 999px; overflow: hidden; }
  .progress > div { height: 100%; background: var(--accent); transition: width 0.3s; }
  ```

**Step 4 — Playwright harness (array webServer, verified):**

- [ ] Create `web/playwright.config.ts`:

  ```ts
  import { defineConfig } from "@playwright/test";

  // Two-server harness per https://playwright.dev/docs/test-webserver (array form
  // requires explicit baseURL). The api runs in TESTING=1 mode: vendor transports are
  // the contracts' fakes, zero keys, fully offline. RAGRECEIPTS_RECEIPTS_DIR points at
  // the committed-format fixture receipts so assertions are hermetic and deterministic.
  export default defineConfig({
    testDir: "./e2e",
    timeout: 60_000,
    use: { baseURL: "http://localhost:3000" },
    webServer: [
      {
        command:
          "uv run python -m uvicorn ragreceipts.server.app:app --port 8000 --workers 1",
        cwd: "../api",
        url: "http://localhost:8000/health",
        reuseExistingServer: !process.env.CI,
        timeout: 60_000,
        env: {
          TESTING: "1",
          RAGRECEIPTS_RECEIPTS_DIR: "tests/fixtures/receipts",
        },
      },
      {
        command: "pnpm dev",
        url: "http://localhost:3000",
        reuseExistingServer: !process.env.CI,
        timeout: 120_000,
        env: { NEXT_PUBLIC_API_BASE_URL: "http://localhost:8000" },
      },
    ],
  });
  ```

- [ ] Create `web/e2e/nav.spec.ts`:

  ```ts
  import { expect, test } from "@playwright/test";

  test("layout renders brand and all three page links", async ({ page }) => {
    await page.goto("/");
    await expect(page.locator(".brand")).toContainText("rag-receipts");
    await expect(page.getByRole("link", { name: "Playground" })).toBeVisible();
    await expect(page.getByRole("link", { name: "Ablation Lab" })).toBeVisible();
    await expect(page.getByRole("link", { name: "Corpora" })).toBeVisible();
  });
  ```

  Note: `tests/fixtures/receipts/` does not exist until Task 10 — create it now as an empty
  placeholder dir with `mkdir -p api/tests/fixtures/receipts && touch api/tests/fixtures/receipts/.gitkeep`
  so the TESTING server boots.

- [ ] Verify: `cd web && pnpm build` — EXPECTED: compiles with no type errors. Then
  `cd web && pnpm e2e` — EXPECTED: `1 passed` (boots both servers, offline).
- [ ] Commit:

  ```bash
  git add web api/tests/fixtures/receipts
  git commit -m "feat(web): Next.js app-router scaffold, typed OpenAPI client, layout, offline Playwright harness" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
  ```

---

### Task 9: Playground page (query, route badge, cited answer, trace viewer)

**Files:**
- Create: `web/src/components/AnswerView.tsx`, `web/src/components/TraceViewer.tsx`
- Modify: `web/src/app/page.tsx`
- Test: `web/e2e/playground.spec.ts`

**Step 1 — failing e2e (write first):**

- [ ] Create `web/e2e/playground.spec.ts`:

  ```ts
  import { expect, test } from "@playwright/test";

  test("query renders route badge, cited answer with popover, and trace", async ({ page }) => {
    await page.goto("/");
    await page.getByTestId("query-input").fill("What is the capital of France?");
    await page.getByTestId("preset-select").selectOption("rerank");
    await page.getByTestId("run-query").click();

    await expect(page.getByTestId("route-badge")).toHaveText("System-1");
    await expect(page.getByTestId("answer")).toContainText("Paris");

    await page.getByTestId("cite-1").click();
    await expect(page.getByTestId("citation-popover")).toBeVisible();
    await expect(page.getByTestId("citation-popover")).toContainText("geo-001");

    const nodes = page.getByTestId("trace-event");
    await expect(nodes).toHaveCount(3);
    await expect(nodes.first()).toContainText("route");
    await expect(nodes.nth(1)).toContainText("s1_retrieve");
  });

  test("degraded retrieval shows a visible badge, never silent", async ({ page }) => {
    await page.goto("/");
    await page.getByTestId("query-input").fill("degrade: capital of France?");
    await page.getByTestId("run-query").click();
    await expect(page.getByTestId("degraded-flag")).toHaveText("rerank-skipped");
    await expect(page.getByTestId("degraded-badge").first()).toBeVisible();
  });
  ```

- [ ] Run: `cd web && pnpm e2e e2e/playground.spec.ts` — EXPECTED: both tests FAIL
  (page still shows scaffold placeholder).

**Step 2 — components (COMPLETE code):**

- [ ] Create `web/src/components/AnswerView.tsx`:

  ```tsx
  "use client";
  import { useState } from "react";
  import type { components } from "@/lib/api/schema";

  type Citation = components["schemas"]["CitationModel"];

  export default function AnswerView({
    answer,
    citations,
    abstained,
  }: {
    answer: string;
    citations: Citation[];
    abstained: boolean;
  }) {
    const [open, setOpen] = useState<number | null>(null);
    const byN = new Map(citations.map((c) => [c.n, c]));
    const parts = answer.split(/(\[\d+\])/g);
    const current = open !== null ? byN.get(open) : undefined;
    return (
      <div className="card" data-testid="answer">
        {abstained && (
          <span className="badge badge-degraded" data-testid="abstained-badge">
            abstained
          </span>
        )}
        <p>
          {parts.map((part, i) => {
            const match = part.match(/^\[(\d+)\]$/);
            if (!match) return <span key={i}>{part}</span>;
            const n = Number(match[1]);
            return (
              <button
                key={i}
                className="cite"
                data-testid={`cite-${n}`}
                onClick={() => setOpen(open === n ? null : n)}
              >
                [{n}]
              </button>
            );
          })}
        </p>
        {current && (
          <div className="popover" data-testid="citation-popover">
            <div className="popover-head">
              <code>{current.chunk_id}</code>
              <span>score {current.score.toFixed(3)}</span>
            </div>
            <p>{current.text}</p>
          </div>
        )}
      </div>
    );
  }
  ```

- [ ] Create `web/src/components/TraceViewer.tsx`:

  ```tsx
  import type { components } from "@/lib/api/schema";

  type TraceEvent = components["schemas"]["TraceEventModel"];
  type ChunkRow = { chunk_id: string; score: number };

  const HOP_NODES = new Set(["retrieve_hop", "grade", "refine"]);

  export default function TraceViewer({ events }: { events: TraceEvent[] }) {
    return (
      <ol className="trace">
        {events.map((ev) => {
          const payload = ev.payload as Record<string, unknown>;
          const degraded = (payload.degraded as string[] | undefined) ?? [];
          const chunks = (payload.chunks as ChunkRow[] | undefined) ?? [];
          const hop = payload.hop as number | undefined;
          return (
            <li key={ev.seq} className="trace-event" data-testid="trace-event">
              <div className="trace-head">
                <span className="node">{ev.node}</span>
                {HOP_NODES.has(ev.node) && hop !== undefined && (
                  <span className="badge">hop {hop}</span>
                )}
                {ev.model && <code className="model">{ev.model}</code>}
                <span className="ms">{ev.duration_ms.toFixed(0)} ms</span>
                {(ev.input_tokens > 0 || ev.output_tokens > 0) && (
                  <span className="tokens">
                    {ev.input_tokens}→{ev.output_tokens} tok
                  </span>
                )}
                {degraded.map((d) => (
                  <span key={d} className="badge badge-degraded" data-testid="degraded-badge">
                    {d}
                  </span>
                ))}
              </div>
              {chunks.length > 0 && (
                <table className="chunks">
                  <tbody>
                    {chunks.map((c) => (
                      <tr key={c.chunk_id}>
                        <td>
                          <code>{c.chunk_id}</code>
                        </td>
                        <td>{c.score.toFixed(4)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </li>
          );
        })}
      </ol>
    );
  }
  ```

**Step 3 — the page (Modify `web/src/app/page.tsx`, COMPLETE code):**

- [ ] Replace `web/src/app/page.tsx`:

  ```tsx
  "use client";
  import { useEffect, useState } from "react";
  import AnswerView from "@/components/AnswerView";
  import TraceViewer from "@/components/TraceViewer";
  import { api } from "@/lib/api/client";
  import type { components } from "@/lib/api/schema";

  type QueryResponse = components["schemas"]["QueryResponse"];
  type TraceEvent = components["schemas"]["TraceEventModel"];

  // Preset ladder is fixed by contract (api/ragreceipts/config.py PRESETS).
  const PRESETS = ["bm25-only", "dense-rrf", "contextual", "rerank", "router-on"];

  export default function Playground() {
    const [corpora, setCorpora] = useState<string[]>([]);
    const [corpusId, setCorpusId] = useState("");
    const [preset, setPreset] = useState("rerank");
    const [query, setQuery] = useState("");
    const [result, setResult] = useState<QueryResponse | null>(null);
    const [events, setEvents] = useState<TraceEvent[]>([]);
    const [error, setError] = useState<string | null>(null);
    const [busy, setBusy] = useState(false);

    useEffect(() => {
      api.GET("/corpora").then(({ data }) => {
        const ids = data?.corpora.map((c) => c.corpus_id) ?? [];
        setCorpora(ids);
        if (ids.length > 0) setCorpusId((cur) => cur || ids[0]);
      });
    }, []);

    async function run() {
      setBusy(true);
      setError(null);
      setResult(null);
      setEvents([]);
      const { data, error: err } = await api.POST("/query", {
        body: { query, corpus_id: corpusId, preset },
      });
      if (err || !data) {
        const detail =
          err && typeof err === "object" && "detail" in err
            ? JSON.stringify((err as { detail: unknown }).detail)
            : "request failed";
        setError(detail);
        setBusy(false);
        return;
      }
      setResult(data);
      const trace = await api.GET("/traces/{trace_id}", {
        params: { path: { trace_id: data.trace_id } },
      });
      setEvents(trace.data?.events ?? []);
      setBusy(false);
    }

    return (
      <>
        <section className="card">
          <textarea
            data-testid="query-input"
            rows={2}
            placeholder="Ask the corpus anything…"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          <div className="row" style={{ marginTop: 10 }}>
            <select
              data-testid="corpus-select"
              value={corpusId}
              onChange={(e) => setCorpusId(e.target.value)}
            >
              {corpora.map((c) => (
                <option key={c}>{c}</option>
              ))}
            </select>
            <select
              data-testid="preset-select"
              value={preset}
              onChange={(e) => setPreset(e.target.value)}
            >
              {PRESETS.map((p) => (
                <option key={p}>{p}</option>
              ))}
            </select>
            <button
              className="primary"
              data-testid="run-query"
              disabled={busy || !query || !corpusId}
              onClick={run}
            >
              {busy ? "Running…" : "Run"}
            </button>
          </div>
        </section>

        {error && <p className="error">{error}</p>}

        {result && (
          <>
            <div className="row">
              <span
                className={result.route === "s1" ? "badge badge-s1" : "badge badge-s2"}
                data-testid="route-badge"
              >
                {result.route === "s1" ? "System-1" : "System-2"}
              </span>
              {result.degraded.map((d) => (
                <span key={d} className="badge badge-degraded" data-testid="degraded-flag">
                  {d}
                </span>
              ))}
            </div>
            <AnswerView
              answer={result.answer}
              citations={result.citations}
              abstained={result.abstained}
            />
            <section className="card">
              <h2>Trace</h2>
              <TraceViewer events={events} />
            </section>
          </>
        )}
      </>
    );
  }
  ```

- [ ] Run: `cd web && pnpm e2e e2e/playground.spec.ts` — EXPECTED: `2 passed`.
- [ ] Typecheck/build: `cd web && pnpm build` — EXPECTED: success.
- [ ] Commit:

  ```bash
  git add web/src web/e2e/playground.spec.ts
  git commit -m "feat(web): Playground with preset picker, route badge, cited answer popovers, trace viewer" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
  ```

---

### Task 10: Ablation Lab (recharts, anchors panel, committed/local toggle)

**Files:**
- Create: `api/tests/fixtures/receipts/receipt-bm25-only.json`,
  `api/tests/fixtures/receipts/receipt-contextual.json`,
  `api/tests/fixtures/receipts/receipt-rerank.json`
- Create: `web/src/app/ablation/page.tsx`
- Modify: `api/tests/e2e_fixture.py` (seed one local receipt at startup)
- Test: `web/e2e/ablation.spec.ts`

**Step 1 — committed-format fixture receipts (hermetic e2e data):**

These follow the contracts' receipt schema exactly (IDs + metrics in `per_query`, never
passage text) so the e2e exercises the same loader that renders the real `receipts/` dir;
the Playwright config already points `RAGRECEIPTS_RECEIPTS_DIR` here. Per **R11**: every
`per_query` row uses the committed shape exactly —
`{"query_id", "retrieved_chunk_ids", "latency_ms", "usd", "flags": {...}}` (flags is a
dict, never a list); receipts carry `prompts_version` ("n/a" for fixtures); and a third
committed fixture covers the `contextual` preset so the Ablation Lab's cell-level
**cross-index** disclosure (driven by differing `index_hashes`) has real data to render.

- [ ] Create `api/tests/fixtures/receipts/receipt-bm25-only.json`:

  ```json
  {
    "schema_version": 1,
    "nondeterminism_note": "LLM calls are nondeterministic even at temperature=0: answer-dependent metrics (em, f1, ragas_*) can shift slightly between identical runs. Retrieval metrics (recall_at_5, mrr_at_3) are deterministic for a fixed index. Treat small answer-metric deltas as noise, not findings.",
    "receipt": {
      "run_id": "fixture-committed-bm25-only",
      "corpus_id": "fixture-corpus",
      "preset": "bm25-only",
      "config": {"name": "bm25-only", "ingest": {"contextual": false, "chunk_size": 512, "chunk_overlap": 64}, "query": {"bm25": true, "dense": false, "rerank": false, "route_mode": "force_s1", "top_k_fuse": 50, "top_k_final": 5}},
      "index_hashes": {"sparse": "sha256:fixture-sparse"},
      "models": {"router": "claude-haiku-4-5-20251001", "synth": "claude-sonnet-4-6", "judge": "claude-sonnet-4-6", "rerank": "rerank-v4.0-pro", "embed": "voyage-context-3"},
      "pricing_table_version": "2026-06-10",
      "prompts_version": "n/a",
      "n_total": 15,
      "n_failed": 0,
      "n_abstained": 1,
      "metrics": {"recall_at_5": 0.55, "mrr_at_3": 0.41, "em": 0.27, "f1": 0.39, "ragas_faithfulness": 0.71, "ragas_answer_relevancy": 0.68, "latency_p50_ms": 740, "latency_p95_ms": 1620, "usd_per_query": 0.006},
      "per_query": [{"query_id": "q-001", "retrieved_chunk_ids": ["geo-001:0", "geo-003:0"], "latency_ms": 712, "usd": 0.005, "flags": {}}],
      "anchors": [{"source": "arXiv 2604.01733 Table I (BM25 vs dense, financial domain)", "published_value": 0.644, "measured_value": 0.55, "direction_match": true, "note": "T2-RAGBench is financial-domain; this corpus is not. Cross-domain anchors claim direction-match only, never magnitude reproduction. Source is a single non-peer-reviewed Apr 2026 preprint."}]
    }
  }
  ```

- [ ] Create `api/tests/fixtures/receipts/receipt-contextual.json` (R11: the `contextual`
  cell queries the `dense_contextual` named vector while `dense-rrf` queries
  `dense_isolated` — its differing dense index hash is what drives the cell-level
  cross-index marker):

  ```json
  {
    "schema_version": 1,
    "nondeterminism_note": "LLM calls are nondeterministic even at temperature=0: answer-dependent metrics (em, f1, ragas_*) can shift slightly between identical runs. Retrieval metrics (recall_at_5, mrr_at_3) are deterministic for a fixed index. Treat small answer-metric deltas as noise, not findings.",
    "receipt": {
      "run_id": "fixture-committed-contextual",
      "corpus_id": "fixture-corpus",
      "preset": "contextual",
      "config": {"name": "contextual", "ingest": {"contextual": true, "chunk_size": 512, "chunk_overlap": 64}, "query": {"bm25": true, "dense": true, "rerank": false, "route_mode": "force_s1", "top_k_fuse": 50, "top_k_final": 5}},
      "index_hashes": {"dense_contextual": "sha256:fixture-ctx", "sparse": "sha256:fixture-sparse"},
      "models": {"router": "claude-haiku-4-5-20251001", "synth": "claude-sonnet-4-6", "judge": "claude-sonnet-4-6", "rerank": "rerank-v4.0-pro", "embed": "voyage-context-3"},
      "pricing_table_version": "2026-06-10",
      "prompts_version": "n/a",
      "n_total": 15,
      "n_failed": 0,
      "n_abstained": 1,
      "metrics": {"recall_at_5": 0.66, "mrr_at_3": 0.52, "em": 0.33, "f1": 0.47, "ragas_faithfulness": 0.76, "ragas_answer_relevancy": 0.72, "latency_p50_ms": 860, "latency_p95_ms": 1880, "usd_per_query": 0.008},
      "per_query": [{"query_id": "q-001", "retrieved_chunk_ids": ["geo-001:0", "geo-001:1"], "latency_ms": 845, "usd": 0.008, "flags": {}}],
      "anchors": [{"source": "Anthropic contextual-retrieval post (Sep 2024): +2-3pp recall from contextualization", "published_value": 0.025, "measured_value": 0.03, "direction_match": true, "note": "The independent +2-3pp figure is for LLM-prefix contextualization; this cell measures voyage-context-3 document-context embeddings against the SAME model embedding isolated chunks - a different technique, direction-match only. Cross-index cell: contextual queries the dense_contextual named vector while dense-rrf queries dense_isolated, so the adjacent ladder cells were measured against different index hashes."}]
    }
  }
  ```

- [ ] Create `api/tests/fixtures/receipts/receipt-rerank.json`:

  ```json
  {
    "schema_version": 1,
    "nondeterminism_note": "LLM calls are nondeterministic even at temperature=0: answer-dependent metrics (em, f1, ragas_*) can shift slightly between identical runs. Retrieval metrics (recall_at_5, mrr_at_3) are deterministic for a fixed index. Treat small answer-metric deltas as noise, not findings.",
    "receipt": {
      "run_id": "fixture-committed-rerank",
      "corpus_id": "fixture-corpus",
      "preset": "rerank",
      "config": {"name": "rerank", "ingest": {"contextual": true, "chunk_size": 512, "chunk_overlap": 64}, "query": {"bm25": true, "dense": true, "rerank": true, "route_mode": "force_s1", "top_k_fuse": 50, "top_k_final": 5}},
      "index_hashes": {"dense_contextual": "sha256:fixture-ctx", "sparse": "sha256:fixture-sparse"},
      "models": {"router": "claude-haiku-4-5-20251001", "synth": "claude-sonnet-4-6", "judge": "claude-sonnet-4-6", "rerank": "rerank-v4.0-pro", "embed": "voyage-context-3"},
      "pricing_table_version": "2026-06-10",
      "prompts_version": "n/a",
      "n_total": 15,
      "n_failed": 0,
      "n_abstained": 0,
      "metrics": {"recall_at_5": 0.78, "mrr_at_3": 0.62, "em": 0.4, "f1": 0.55, "ragas_faithfulness": 0.81, "ragas_answer_relevancy": 0.77, "latency_p50_ms": 980, "latency_p95_ms": 2140, "usd_per_query": 0.012},
      "per_query": [{"query_id": "q-001", "retrieved_chunk_ids": ["geo-001:0", "geo-001:1"], "latency_ms": 990, "usd": 0.012, "flags": {}}],
      "anchors": [{"source": "arXiv 2604.01733 Table I (rerank vs hybrid RRF: +12.1pp Recall@5)", "published_value": 0.121, "measured_value": 0.23, "direction_match": true, "note": "Published delta measured with Cohere Rerank v4.0 Pro on T2-RAGBench (financial). Our corpus differs in domain; direction-match only. Reranking being the largest single gain is the claim under test, not the magnitude."}]
    }
  }
  ```

- [ ] Remove the `.gitkeep` from Task 8: `git rm api/tests/fixtures/receipts/.gitkeep`

- [ ] Seed one **local** receipt at TESTING startup so the committed/local toggle always has
  both sources. In `api/tests/e2e_fixture.py` `build_testing_deps()`, immediately before the
  `return AppDeps(` line, add:

  ```python
      # Seed a local run so the Ablation Lab committed/local toggle has both sources.
      FixtureEvalRunner(paths).run(
          corpus_id=FIXTURE_CORPUS_ID, preset="dense-rrf", slice_name="smoke",
          spend_cap_usd=1.0, emit=lambda message, progress: None,
      )
  ```

**Step 2 — failing e2e:**

- [ ] Create `web/e2e/ablation.spec.ts`:

  ```ts
  import { expect, test } from "@playwright/test";

  test("ablation lab renders committed receipts, charts, and verbatim anchor notes", async ({ page }) => {
    await page.goto("/ablation");
    await expect(page.getByTestId("metric-chart-recall_at_5")).toBeVisible();
    // Committed fixture receipts (served from RAGRECEIPTS_RECEIPTS_DIR) render
    await expect(page.getByTestId("receipt-row").filter({ hasText: "bm25-only" })).toBeVisible();
    await expect(page.getByTestId("receipt-row").filter({ hasText: "committed" }).first()).toBeVisible();
    // PublishedAnchor.note rendered VERBATIM
    await expect(page.getByTestId("anchor-note").first()).toContainText(
      "direction-match only, never magnitude reproduction"
    );
    // R11: the contextual cell carries the CELL-level cross-index marker (its dense
    // index hash differs from the preceding dense-bearing ladder cell, dense-rrf),
    // and with the fixture data it is the ONLY flagged cell.
    const contextualRow = page.getByTestId("receipt-row").filter({ hasText: "contextual" });
    await expect(contextualRow).toBeVisible();
    await expect(contextualRow.getByTestId("cross-index-badge")).toHaveText("cross-index");
    await expect(page.getByTestId("cross-index-badge")).toHaveCount(1);
    await expect(page.getByTestId("cross-index-note").first()).toContainText("contextual");
  });

  test("committed/local toggle filters sources", async ({ page }) => {
    await page.goto("/ablation");
    await expect(page.getByTestId("receipt-row").filter({ hasText: "local" }).first()).toBeVisible();
    await page.getByTestId("toggle-local").uncheck();
    await expect(page.getByTestId("receipt-row").filter({ hasText: "local" })).toHaveCount(0);
    await page.getByTestId("toggle-committed").uncheck();
    await expect(page.getByTestId("receipt-row")).toHaveCount(0);
  });
  ```

- [ ] Run: `cd web && pnpm e2e e2e/ablation.spec.ts` — EXPECTED: FAIL (404 page).

**Step 3 — implement the page (COMPLETE code):**

- [ ] Create `web/src/app/ablation/page.tsx`:

  ```tsx
  "use client";
  import { useEffect, useState } from "react";
  import {
    Bar,
    BarChart,
    CartesianGrid,
    Legend,
    ResponsiveContainer,
    Tooltip,
    XAxis,
    YAxis,
  } from "recharts";
  import { api } from "@/lib/api/client";
  import type { components } from "@/lib/api/schema";

  type ReceiptEntry = components["schemas"]["ReceiptEntryModel"];
  // Matches the contracts' PublishedAnchor dataclass (receipt payload is open by design).
  type Anchor = {
    source: string;
    published_value: number;
    measured_value: number;
    direction_match: boolean;
    note: string;
  };
  type ReceiptBody = {
    run_id?: string;
    preset?: string;
    n_total?: number;
    n_failed?: number;
    n_abstained?: number;
    index_hashes?: Record<string, string>;
    metrics?: Record<string, number>;
    anchors?: Anchor[];
  };

  const METRICS = [
    "recall_at_5",
    "mrr_at_3",
    "em",
    "f1",
    "ragas_faithfulness",
    "ragas_answer_relevancy",
    "usd_per_query",
  ];
  // Ladder order is fixed by contract (api/ragreceipts/config.py PRESETS).
  const PRESET_ORDER = ["bm25-only", "dense-rrf", "contextual", "rerank", "router-on"];

  export default function AblationLab() {
    const [receipts, setReceipts] = useState<ReceiptEntry[]>([]);
    const [errors, setErrors] = useState<string[]>([]);
    const [showCommitted, setShowCommitted] = useState(true);
    const [showLocal, setShowLocal] = useState(true);

    useEffect(() => {
      api.GET("/receipts").then(({ data }) => {
        setReceipts(data?.receipts ?? []);
        setErrors(data?.errors ?? []);
      });
    }, []);

    const visible = receipts.filter((r) =>
      r.source === "committed" ? showCommitted : showLocal
    );

    // Cell-level "cross-index" disclosure (contracts R11): a cell is flagged when its
    // dense index hash (index_hashes.dense_contextual ?? index_hashes.dense_isolated)
    // differs from the previous dense-bearing preset in ladder order — i.e. the ladder
    // step it is read against was measured on a DIFFERENT index. With the fixtures,
    // exactly the contextual cell is flagged (dense-rrf:iso -> contextual:ctx).
    const crossIndex = (() => {
      const flagged = new Set<string>();
      let prev: string | null = null;
      for (const preset of PRESET_ORDER) {
        let presetDense: string | null = null;
        for (const r of visible) {
          const body = r.receipt as ReceiptBody;
          if (body.preset !== preset) continue;
          const hashes = body.index_hashes ?? {};
          const dense = hashes["dense_contextual"] ?? hashes["dense_isolated"];
          if (!dense) continue;
          presetDense = dense;
          if (prev !== null && dense !== prev) flagged.add(r.path);
        }
        if (presetDense !== null) prev = presetDense;
      }
      return flagged;
    })();
    const crossIndexPresets = Array.from(
      new Set(
        visible
          .filter((r) => crossIndex.has(r.path))
          .map((r) => (r.receipt as ReceiptBody).preset ?? "?")
      )
    );

    // Grouped bars: one group per preset, one bar per source (recharts: multiple <Bar>
    // elements without stackId render side by side — verified, recharts BarChart API).
    function chartData(metric: string) {
      return PRESET_ORDER.map((preset) => {
        const row: Record<string, string | number> = { preset };
        for (const r of visible) {
          const body = r.receipt as ReceiptBody;
          const value = body.preset === preset ? body.metrics?.[metric] : undefined;
          if (value !== undefined) row[r.source] = value;
        }
        return row;
      }).filter((row) => "committed" in row || "local" in row);
    }

    const anchorRows = visible.flatMap((e) => {
      const body = e.receipt as ReceiptBody;
      return (body.anchors ?? []).map((anchor) => ({
        preset: body.preset ?? "?",
        source: e.source,
        anchor,
      }));
    });

    return (
      <>
        <section className="card">
          <div className="row">
            <h1 style={{ margin: 0, fontSize: 20 }}>Ablation Lab</h1>
            <label>
              <input
                type="checkbox"
                data-testid="toggle-committed"
                checked={showCommitted}
                onChange={(e) => setShowCommitted(e.target.checked)}
              />{" "}
              committed
            </label>
            <label>
              <input
                type="checkbox"
                data-testid="toggle-local"
                checked={showLocal}
                onChange={(e) => setShowLocal(e.target.checked)}
              />{" "}
              local
            </label>
          </div>
          <p className="muted">
            Each preset&apos;s receipt: measured contribution on labeled data. Committed =
            headline runs from <code>receipts/</code>; local = your runs.
          </p>
          {errors.map((err) => (
            <p key={err} className="error">
              unreadable receipt: {err}
            </p>
          ))}
          <table className="chunks">
            <tbody>
              {visible.map((r) => {
                const body = r.receipt as ReceiptBody;
                return (
                  <tr key={r.path} data-testid="receipt-row">
                    <td>{body.preset}</td>
                    <td>{r.source}</td>
                    <td>
                      <code>{body.run_id}</code>
                    </td>
                    <td className="muted">
                      n={body.n_total} failed={body.n_failed} abstained={body.n_abstained}
                    </td>
                    <td>
                      {crossIndex.has(r.path) && (
                        <span
                          className="badge badge-degraded"
                          data-testid="cross-index-badge"
                          title="Measured against a different dense index than the preceding ladder cell (see index_hashes)"
                        >
                          cross-index
                        </span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </section>

        {METRICS.map((metric) => {
          const data = chartData(metric);
          if (data.length === 0) return null;
          return (
            <section className="card" key={metric} data-testid={`metric-chart-${metric}`}>
              <h2 style={{ marginTop: 0 }}>{metric}</h2>
              <ResponsiveContainer width="100%" height={240}>
                <BarChart data={data}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="preset" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="committed" fill="#2563eb" />
                  <Bar dataKey="local" fill="#9ca3af" />
                </BarChart>
              </ResponsiveContainer>
              {crossIndexPresets.length > 0 && (
                <p className="muted" data-testid="cross-index-note">
                  cross-index: {crossIndexPresets.join(", ")} — measured against a
                  different dense index than the preceding ladder cell (the receipt&apos;s
                  index_hashes differ), so read that step as a cross-index comparison.
                </p>
              )}
            </section>
          );
        })}

        {anchorRows.length > 0 && (
          <section className="card">
            <h2 style={{ marginTop: 0 }}>Ours vs published anchors</h2>
            <p className="muted">
              Published numbers are anchors, not targets — the note explains why magnitudes
              are not directly comparable.
            </p>
            {anchorRows.map((row, i) => (
              <div key={i} className="anchor" data-testid="anchor-row">
                <div className="anchor-head">
                  <strong>{row.preset}</strong>
                  <span className="muted">{row.anchor.source}</span>
                  <span>
                    published {row.anchor.published_value} · measured{" "}
                    {row.anchor.measured_value}
                  </span>
                  <span
                    className={
                      row.anchor.direction_match ? "badge badge-ok" : "badge badge-degraded"
                    }
                  >
                    {row.anchor.direction_match ? "direction match" : "direction mismatch"}
                  </span>
                </div>
                <blockquote data-testid="anchor-note">{row.anchor.note}</blockquote>
              </div>
            ))}
          </section>
        )}
      </>
    );
  }
  ```

- [ ] Run: `cd api && uv run pytest tests/test_testing_mode.py -q` — EXPECTED: still passes
  (fixture seeding added). Then `cd web && pnpm e2e e2e/ablation.spec.ts` — EXPECTED:
  `2 passed`.
- [ ] Commit:

  ```bash
  git add web/src/app/ablation web/e2e/ablation.spec.ts api/tests/fixtures/receipts api/tests/e2e_fixture.py
  git commit -m "feat(web): Ablation Lab with grouped metric charts, verbatim anchor notes, committed/local toggle" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
  ```

---

### Task 11: Corpora page (read-only: list + manifest disclosure)

**Files:**
- Create: `web/src/app/corpora/page.tsx`
- Test: `web/e2e/corpora.spec.ts`

**Step 1 — failing e2e:**

- [ ] Create `web/e2e/corpora.spec.ts`:

  ```ts
  import { expect, test } from "@playwright/test";

  test("corpora page lists manifests with chunking and hash disclosure", async ({ page }) => {
    await page.goto("/corpora");
    const card = page.getByTestId("corpus-card").filter({ hasText: "fixture-corpus" });
    await expect(card).toBeVisible();
    await expect(card).toContainText("chunk_size 512");
    await expect(card).toContainText("voyage-context-3");
    await expect(card).toContainText("sha256:fixture");
  });
  ```

- [ ] Run: `cd web && pnpm e2e e2e/corpora.spec.ts` — EXPECTED: FAIL (404 page).

**Step 2 — implement (COMPLETE code):**

- [ ] Create `web/src/app/corpora/page.tsx`:

  ```tsx
  "use client";
  import { useCallback, useEffect, useState } from "react";
  import { api } from "@/lib/api/client";
  import type { components } from "@/lib/api/schema";

  type Corpus = components["schemas"]["CorpusModel"];
  // Matches the contracts' corpus manifest (open payload by design).
  type Manifest = {
    dataset?: { name?: string };
    chunking?: { chunk_size?: number; chunk_overlap?: number };
    embed_model?: string;
    index_hashes?: Record<string, string>;
    n_docs?: number;
    n_chunks?: number;
    n_queries?: number;
    created_at?: string;
    byo?: {
      source_files?: string[];
      split_documents?: { doc_id: string; source_file: string; n_splits: number }[];
      failures?: { file: string; error: string }[];
    };
  };

  export default function Corpora() {
    const [corpora, setCorpora] = useState<Corpus[]>([]);

    const reload = useCallback(() => {
      api.GET("/corpora").then(({ data }) => setCorpora(data?.corpora ?? []));
    }, []);

    useEffect(() => {
      reload();
    }, [reload]);

    return (
      <>
        <h1 style={{ fontSize: 20 }}>Corpora</h1>
        {corpora.length === 0 && (
          <p className="muted">No corpora ingested yet.</p>
        )}
        {corpora.map((c) => {
          const man = c.manifest as Manifest;
          return (
            <section className="card" key={c.corpus_id} data-testid="corpus-card">
              <div className="row">
                <strong>{c.corpus_id}</strong>
                <span className="badge">{man.dataset?.name ?? "unknown dataset"}</span>
                <span className="muted">
                  {man.n_docs ?? "?"} docs · {man.n_chunks ?? "?"} chunks ·{" "}
                  {man.n_queries ?? "?"} queries
                </span>
              </div>
              <p className="muted">
                chunk_size {man.chunking?.chunk_size ?? "?"} · overlap{" "}
                {man.chunking?.chunk_overlap ?? "?"} · {man.embed_model ?? "?"} · created{" "}
                {man.created_at ?? "?"}
              </p>
              <table className="chunks">
                <tbody>
                  {Object.entries(man.index_hashes ?? {}).map(([name, hash]) => (
                    <tr key={name}>
                      <td>{name}</td>
                      <td>
                        <code>{hash}</code>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {man.byo?.split_documents && man.byo.split_documents.length > 0 && (
                <p className="muted" data-testid="split-disclosure">
                  Oversized documents split at ingest (disclosed per manifest):{" "}
                  {man.byo.split_documents
                    .map((s) => `${s.source_file} → ${s.n_splits} parts`)
                    .join(", ")}
                </p>
              )}
              {man.byo?.failures && man.byo.failures.length > 0 && (
                <div data-testid="ingest-failures">
                  {man.byo.failures.map((f) => (
                    <p key={f.file} className="error">
                      failed: {f.file} — {f.error}
                    </p>
                  ))}
                </div>
              )}
            </section>
          );
        })}
      </>
    );
  }
  ```

- [ ] Run: `cd web && pnpm e2e e2e/corpora.spec.ts` — EXPECTED: `1 passed`.
- [ ] Full web check: `cd web && pnpm build && pnpm e2e` — EXPECTED: all specs pass.
- [ ] Commit:

  ```bash
  git add web/src/app/corpora web/e2e/corpora.spec.ts
  git commit -m "feat(web): Corpora page with manifest, hash, split and failure disclosure" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
  ```

---

### Task 12: docker compose (api + web + qdrant), Dockerfiles, `.env.example`

**Files:**
- Create: `docker-compose.yml`, `.env.example`, `api/Dockerfile`, `api/.dockerignore`,
  `web/Dockerfile`, `web/.dockerignore`

**Step 1 — env template:**

- [ ] Create `.env.example` (the three keys, api-only per spec):

  ```bash
  # Copy to .env and fill in. These are passed to the api container ONLY —
  # the web container never sees vendor keys.
  ANTHROPIC_API_KEY=
  VOYAGE_API_KEY=
  COHERE_API_KEY=
  ```

- [ ] Confirm `.gitignore` covers `.env` (add the line if Plan A didn't).

**Step 2 — api image (uv pattern per docs.astral.sh/uv/guides/integration/docker/):**

- [ ] Create `api/Dockerfile`:

  ```dockerfile
  FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

  WORKDIR /app
  ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

  # Dependency layer first for caching
  COPY pyproject.toml uv.lock ./
  RUN uv sync --frozen --no-dev --no-install-project

  COPY . .
  RUN uv sync --frozen --no-dev

  EXPOSE 8000
  # SINGLE worker (load-bearing): JobRunner thread + queue are in-process state.
  CMD ["uv", "run", "--no-dev", "python", "-m", "uvicorn", "ragreceipts.server.app:app", \
       "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
  ```

- [ ] Create `api/.dockerignore`:

  ```
  .venv
  __pycache__
  .pytest_cache
  .ruff_cache
  ```

**Step 3 — web image:**

- [ ] Create `web/Dockerfile` (NEXT_PUBLIC_* vars are inlined at **build** time, hence the
  build arg):

  ```dockerfile
  FROM node:22-alpine AS build
  RUN corepack enable
  WORKDIR /app
  COPY package.json pnpm-lock.yaml ./
  RUN pnpm install --frozen-lockfile
  COPY . .
  ARG NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
  ENV NEXT_PUBLIC_API_BASE_URL=$NEXT_PUBLIC_API_BASE_URL
  RUN pnpm build

  FROM node:22-alpine
  RUN corepack enable
  WORKDIR /app
  COPY --from=build /app ./
  EXPOSE 3000
  CMD ["pnpm", "start"]
  ```

- [ ] Create `web/.dockerignore`:

  ```
  node_modules
  .next
  e2e
  test-results
  playwright-report
  ```

**Step 4 — compose file:**

- [ ] The qdrant server tag is pinned by contracts **R7**: `qdrant/qdrant:v1.18.0`, the
  minor matching the binding qdrant-client pin `>=1.18,<2`. No discovery step — use it
  verbatim below. (Also per R7: the api service MUST receive `QDRANT_URL` — compose sets
  it; without it the server's healthcheck reports `QDRANT_URL` by name, never a silent
  localhost default.)
- [ ] Create `docker-compose.yml`:

  ```yaml
  services:
    qdrant:
      image: qdrant/qdrant:v1.18.0   # R7: matches qdrant-client >=1.18,<2
      volumes:
        - qdrant-storage:/qdrant/storage
      # The qdrant image ships no curl/wget by design (security stance,
      # github.com/qdrant/qdrant/issues/4250). Bash /dev/tcp TCP-connect probe instead.
      healthcheck:
        test: ["CMD-SHELL", "bash -c ':> /dev/tcp/127.0.0.1/6333' || exit 1"]
        interval: 5s
        timeout: 3s
        retries: 24

    api:
      build: ./api
      env_file: .env            # vendor keys reach the api ONLY
      environment:
        QDRANT_URL: http://qdrant:6333   # REQUIRED by the server (R7) — no silent default
        RAGRECEIPTS_DATA_DIR: /data
        RAGRECEIPTS_RECEIPTS_DIR: /receipts
        RAGRECEIPTS_CORS_ORIGINS: http://localhost:3000
      volumes:
        - app-data:/data                  # SQLite (traces/jobs), bm25s indexes, corpora, local receipts
        - ./receipts:/receipts:ro         # committed receipts, read-only at runtime (spec)
      ports:
        - "8000:8000"
      depends_on:
        qdrant:
          condition: service_healthy
      healthcheck:
        test: ["CMD", "python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"]
        interval: 10s
        timeout: 5s
        retries: 12
        start_period: 20s

    web:
      build:
        context: ./web
        args:
          NEXT_PUBLIC_API_BASE_URL: http://localhost:8000   # browser-facing URL
      ports:
        - "3000:3000"
      depends_on:
        api:
          condition: service_healthy

  volumes:
    qdrant-storage:
    app-data:
  ```

**Step 5 — verify:**

- [ ] `docker compose config -q` — EXPECTED: exit 0 (valid file).
- [ ] `cp .env.example .env && docker compose up --build -d` (keys may be empty — the stack
  must still boot; /health reports the missing names).
- [ ] `docker compose ps` — EXPECTED: qdrant **healthy**, api **healthy**, web **running**.
  If the qdrant healthcheck shows unhealthy (image variant without bash), remove the
  healthcheck block, change the api `depends_on` to plain list form, and note the change in
  the compose file comment — the api's own qdrant ping in `/health` still reports capability.
- [ ] `curl -s localhost:8000/health | python3 -m json.tool` — EXPECTED: JSON with
  `missing_env_vars` naming any unfilled keys; `"qdrant_ok": true`.
- [ ] Open `http://localhost:3000` — EXPECTED: layout renders; Playground shows "No corpora"
  state cleanly.
- [ ] `docker compose down`
- [ ] Commit:

  ```bash
  git add docker-compose.yml .env.example api/Dockerfile api/.dockerignore web/Dockerfile web/.dockerignore .gitignore
  git commit -m "feat(deploy): docker compose api+web+qdrant with healthcheck-gated startup and api-only keys" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
  ```

---

### Task 13: BYO ingest — readers, 120K-token split, failure collection, endpoint (built last per spec)

**Files:**
- Create: `api/ragreceipts/server/ingest_byo.py`
- Modify: `api/ragreceipts/server/app.py` (endpoint + handler registration)
- Modify: `api/ragreceipts/server/deps.py` (wire real sink behind key check)
- Test: `api/tests/test_ingest_byo.py`,
  `api/tests/test_real_ingest_sink.py` (Step 5 — offline adapter tests)

This is deliberately the LAST feature (spec decision #1: BYO is "built last; designated
first cut if v1 drags"). Everything before this task ships without it.

**Step 1 — dependencies (verified imports, pinned):**

- [ ] ```bash
  cd api
  uv add "llama-index-core>=0.12" "llama-index-readers-file>=0.4" "pypdf>=5" "beautifulsoup4>=4.12"
  ```

**Step 2 — failing test:**

- [ ] Create `api/tests/test_ingest_byo.py`:

  ```python
  """BYO ingest: reader dispatch, oversized-doc split, per-file failure collection,
  multipart endpoint + job."""
  import json
  import time

  from fastapi.testclient import TestClient

  from ragreceipts.server.app import create_app
  from ragreceipts.server.ingest_byo import (
      DOC_TOKEN_LIMIT,
      approx_token_count,
      load_documents,
      split_oversized,
  )
  from tests.helpers_server import make_test_deps


  def test_approx_token_count_is_conservative():
      assert approx_token_count("abcd" * 100) == 100  # 4 chars/token heuristic


  def test_split_oversized_discloses_parts():
      text = ("para. " * 200 + "\n\n") * 500  # well above DOC_TOKEN_LIMIT
      docs = split_oversized("big-doc", text, source_file="big.txt")
      assert len(docs) > 1
      assert all(d.n_splits == len(docs) for d in docs)
      assert [d.split_index for d in docs] == list(range(len(docs)))
      assert all(approx_token_count(d.text) <= DOC_TOKEN_LIMIT for d in docs)
      assert "".join(d.text for d in docs).replace("\n\n", "") == text.replace("\n\n", "")


  def test_small_doc_is_not_split():
      docs = split_oversized("small", "hello world", source_file="s.txt")
      assert len(docs) == 1 and docs[0].n_splits == 1


  def test_load_documents_collects_failures_never_batch_fatal(tmp_path):
      good = tmp_path / "good.txt"
      good.write_text("plain text content")
      md = tmp_path / "notes.md"
      md.write_text("# Title\n\nbody text")
      missing = tmp_path / "missing.pdf"  # never created -> reader raises
      docs, failures = load_documents([good, md, missing])
      assert {d.source_file for d in docs} == {"good.txt", "notes.md"}
      assert len(failures) == 1 and failures[0].file == "missing.pdf"


  def test_unsupported_extension_is_a_failure_not_a_crash(tmp_path):
      weird = tmp_path / "data.xyz"
      weird.write_text("???")
      docs, failures = load_documents([weird])
      assert docs == [] and "unsupported" in failures[0].error


  class RecordingSink:
      def __init__(self) -> None:
          self.calls: list[dict] = []

      def write_corpus(self, *, corpus_id, docs, emit):
          self.calls.append({"corpus_id": corpus_id, "n_docs": len(docs)})
          emit("indexed", 0.9)
          return {
              "corpus_id": corpus_id,
              "dataset": {"name": "byo"},
              "chunking": {"chunk_size": 512, "chunk_overlap": 64},
              "embed_model": "voyage-context-3",
              "index_hashes": {"sparse": "sha256:test"},
              "tokenizer_artifact": "test",
              "n_docs": len(docs), "n_chunks": len(docs), "n_queries": 0,
              "created_at": "2026-06-10T00:00:00+00:00",
          }


  def wait_succeeded(client, job_id, timeout=15.0):
      deadline = time.time() + timeout
      while time.time() < deadline:
          body = client.get(f"/jobs/{job_id}").json()
          if body["status"] in ("succeeded", "failed"):
              return body
          time.sleep(0.05)
      raise AssertionError("job did not finish")


  def test_ingest_endpoint_runs_job_and_writes_manifest_with_disclosures(tmp_path):
      deps = make_test_deps(tmp_path, configured=True)
      sink = RecordingSink()
      deps.ingest_sink = sink
      app = create_app(deps_factory=lambda: deps)
      with TestClient(app) as client:
          r = client.post(
              "/corpora/ingest",
              data={"corpus_id": "my-docs"},
              files=[
                  ("files", ("a.txt", b"alpha document text", "text/plain")),
                  ("files", ("b.md", b"# beta\n\nbody", "text/markdown")),
              ],
          )
          assert r.status_code == 200, r.text
          job = wait_succeeded(client, r.json()["job_id"])
          assert job["status"] == "succeeded"
          listed = client.get("/corpora").json()["corpora"]
          assert any(c["corpus_id"] == "my-docs" for c in listed)
      manifest = json.loads(
          (deps.paths.corpora_dir / "my-docs" / "manifest.json").read_text()
      )
      assert manifest["byo"]["source_files"] == ["a.txt", "b.md"]
      assert manifest["byo"]["failures"] == []
      assert sink.calls[0]["n_docs"] == 2


  def test_ingest_endpoint_rejects_when_sink_unavailable(tmp_path):
      deps = make_test_deps(tmp_path, configured=False)
      with TestClient(create_app(deps_factory=lambda: deps)) as client:
          r = client.post(
              "/corpora/ingest",
              data={"corpus_id": "my-docs"},
              files=[("files", ("a.txt", b"x", "text/plain"))],
          )
      assert r.status_code == 503
      assert "VOYAGE_API_KEY" in r.json()["detail"]
  ```

- [ ] Run: `cd api && uv run pytest tests/test_ingest_byo.py -q`
  — EXPECTED: `ModuleNotFoundError: No module named 'ragreceipts.server.ingest_byo'`.

**Step 3 — implement `server/ingest_byo.py` (COMPLETE code):**

- [ ] Create `api/ragreceipts/server/ingest_byo.py`:

  ```python
  """BYO document ingestion: PDF/MD/HTML/TXT via LlamaIndex readers (verified imports:
  https://developers.llamaindex.ai/python/framework-api-reference/readers/file/).

  Documents above the voyage-context-3 contextualization window are split into multiple
  logical documents at ingest and DISCLOSED in the manifest (spec §Ingestion plane). Token
  counts use a conservative 4-chars-per-token heuristic with a 100K limit, keeping real
  token counts safely under the 120K window without a vendor tokenizer dependency.
  Per-document failures are collected, never batch-fatal.
  """
  from __future__ import annotations

  from dataclasses import dataclass
  from pathlib import Path
  from typing import Callable, Protocol

  APPROX_CHARS_PER_TOKEN = 4
  DOC_TOKEN_LIMIT = 100_000  # conservative vs the 120K-token voyage window
  SUPPORTED_EXTS = (".pdf", ".md", ".html", ".txt")


  def approx_token_count(text: str) -> int:
      return max(1, len(text) // APPROX_CHARS_PER_TOKEN)


  @dataclass(frozen=True)
  class LoadedDoc:
      doc_id: str
      text: str
      source_file: str
      split_index: int  # 0-based part number within the source document
      n_splits: int     # total parts the source document became (1 = not split)


  @dataclass(frozen=True)
  class LoadFailure:
      file: str
      error: str


  class IngestSink(Protocol):
      """Chunk, embed (both named-vector sets), and index docs; returns the manifest dict
      (contracts §Corpus manifest). Implemented by the Plan A adapter (production) and
      TestingIngestSink (TESTING mode)."""

      def write_corpus(self, *, corpus_id: str, docs: list[LoadedDoc],
                       emit: Callable[[str, float], None]) -> dict: ...


  def read_file(path: Path) -> str:
      """Dispatch by extension to the verified LlamaIndex readers; returns full text."""
      from llama_index.readers.file import (
          FlatReader,
          HTMLTagReader,
          MarkdownReader,
          PDFReader,
      )

      ext = path.suffix.lower()
      if ext == ".pdf":
          docs = PDFReader(return_full_document=True).load_data(path)
      elif ext == ".md":
          docs = MarkdownReader().load_data(str(path))  # docs: load_data(file: str)
      elif ext == ".html":
          docs = HTMLTagReader(tag="body").load_data(path)  # default tag is <section>
      elif ext == ".txt":
          docs = FlatReader().load_data(path)
      else:
          raise ValueError(f"unsupported extension: {ext} (supported: {SUPPORTED_EXTS})")
      return "\n\n".join(d.text for d in docs)


  def split_oversized(doc_id: str, text: str, *, source_file: str) -> list[LoadedDoc]:
      """Split at paragraph boundaries so no part exceeds DOC_TOKEN_LIMIT approx tokens."""
      limit_chars = DOC_TOKEN_LIMIT * APPROX_CHARS_PER_TOKEN
      if len(text) <= limit_chars:
          return [LoadedDoc(doc_id=doc_id, text=text, source_file=source_file,
                            split_index=0, n_splits=1)]
      paragraphs = text.split("\n\n")
      parts: list[str] = []
      current: list[str] = []
      size = 0
      for para in paragraphs:
          # A single paragraph longer than the limit is hard-cut.
          while len(para) > limit_chars:
              if current:
                  parts.append("\n\n".join(current))
                  current, size = [], 0
              parts.append(para[:limit_chars])
              para = para[limit_chars:]
          if size + len(para) > limit_chars and current:
              parts.append("\n\n".join(current))
              current, size = [], 0
          current.append(para)
          size += len(para) + 2
      if current:
          parts.append("\n\n".join(current))
      n = len(parts)
      return [
          LoadedDoc(doc_id=f"{doc_id}#part{i}", text=part, source_file=source_file,
                    split_index=i, n_splits=n)
          for i, part in enumerate(parts)
      ]


  def load_documents(files: list[Path]) -> tuple[list[LoadedDoc], list[LoadFailure]]:
      """Read every file; per-document failures collected, never batch-fatal (spec)."""
      docs: list[LoadedDoc] = []
      failures: list[LoadFailure] = []
      for path in files:
          try:
              text = read_file(path)
              docs.extend(split_oversized(path.stem, text, source_file=path.name))
          except Exception as exc:
              failures.append(LoadFailure(file=path.name, error=f"{type(exc).__name__}: {exc}"))
      return docs, failures


  def make_ingest_handler(sink: IngestSink, corpora_dir: Path):
      """Job handler: idempotent full rebuild from saved uploads (bm25s has no incremental
      indexing — spec accepts full rebuild), which is what makes resume() safe."""
      import json

      def handle(ctx) -> None:
          corpus_id = ctx.params["corpus_id"]
          files = [Path(p) for p in ctx.params["files"]]
          ctx.emit(f"loading {len(files)} files", 0.05)
          docs, failures = load_documents(files)
          ctx.emit(f"loaded {len(docs)} docs ({len(failures)} failed)", 0.3)
          if not docs:
              raise RuntimeError(
                  "no readable documents; failures: "
                  + "; ".join(f"{f.file}: {f.error}" for f in failures)
              )
          manifest = sink.write_corpus(corpus_id=corpus_id, docs=docs, emit=ctx.emit)
          manifest["byo"] = {
              # loaded files only; failed ones are disclosed under "failures"
              "source_files": sorted({d.source_file for d in docs}),
              "split_documents": [
                  {"doc_id": d.doc_id, "source_file": d.source_file, "n_splits": d.n_splits}
                  for d in docs if d.n_splits > 1
              ],
              "failures": [{"file": f.file, "error": f.error} for f in failures],
          }
          target = corpora_dir / corpus_id
          target.mkdir(parents=True, exist_ok=True)
          (target / "manifest.json").write_text(json.dumps(manifest, indent=2))
          ctx.emit("manifest written", 1.0)

      return handle
  ```

**Step 4 — endpoint + handler registration (Modify `app.py`):**

- [ ] Add imports `from typing import Annotated` and
  `from fastapi import File, Form, UploadFile`, plus
  `from ragreceipts.server.models import _SLUG` is NOT needed — reuse validation via
  Pydantic by validating manually:

  ```python
  @router.post("/corpora/ingest", response_model=m.IngestResponse)
  async def ingest_corpus(
      corpus_id: Annotated[str, Form()],
      files: Annotated[list[UploadFile], File()],
      deps: AppDeps = Depends(get_deps),
  ) -> m.IngestResponse:
      if deps.ingest_sink is None:
          missing = ", ".join(_missing_env_vars(deps))
          raise HTTPException(503, detail=f"ingest unavailable; missing env vars: {missing}")
      try:
          m._validate_corpus_id(corpus_id)
      except ValueError as exc:
          raise HTTPException(422, detail=str(exc))
      upload_dir = deps.paths.uploads_dir / corpus_id
      upload_dir.mkdir(parents=True, exist_ok=True)
      saved: list[str] = []
      for f in files:
          name = Path(f.filename or "unnamed").name  # strip any client-sent path
          dest = upload_dir / name
          dest.write_bytes(await f.read())
          saved.append(str(dest))
      job_id = deps.job_runner.submit("ingest", {"corpus_id": corpus_id, "files": saved})
      return m.IngestResponse(job_id=job_id, corpus_id=corpus_id)
  ```

- [ ] Register the handler in the lifespan, next to the eval registration:

  ```python
          if deps.ingest_sink is not None:
              from ragreceipts.server.ingest_byo import make_ingest_handler

              deps.job_runner.register(
                  "ingest", make_ingest_handler(deps.ingest_sink, deps.paths.corpora_dir)
              )
  ```

- [ ] Run: `cd api && uv run pytest tests/test_ingest_byo.py -q` — EXPECTED: `7 passed`.

**Step 5 — real sink (R9-pinned entry point) + regenerate the typed client:**

- [ ] Verify the pinned signature (drift guard — if it differs from R9, reconcile ONLY the
  adapter below; the `IngestSink` protocol and the endpoint MUST NOT change):

  ```bash
  cd api && uv run python -c "
  import inspect
  from ragreceipts.ingest.pipeline import run_ingest
  print(inspect.signature(run_ingest))"
  # EXPECTED (R9): (*, corpus_id: str, data_dir: Path, ingest_config: IngestConfig,
  #                 embed: EmbedTransport, qdrant: QdrantClient,
  #                 embed_model: str = 'voyage-context-3') -> dict
  ```

- [ ] Write the failing offline adapter test `api/tests/test_real_ingest_sink.py`
  (COMPLETE code — construction + marshalling against fakes, zero keys, zero network):

  ```python
  """RealIngestSink: writes the R1 raw/ layout, delegates to the R9-pinned run_ingest."""
  import json

  from ragreceipts.server.ingest_byo import LoadedDoc, RealIngestSink


  def test_write_corpus_marshals_r1_layout_and_pinned_kwargs(tmp_path):
      calls: dict = {}

      def fake_run_ingest(*, corpus_id, data_dir, ingest_config, embed, qdrant):
          calls.update(corpus_id=corpus_id, data_dir=data_dir,
                       chunk_size=ingest_config.chunk_size,
                       chunk_overlap=ingest_config.chunk_overlap,
                       embed=embed, qdrant=qdrant)
          return {"corpus_id": corpus_id, "n_docs": 2,
                  "index_hashes": {"sparse": "sha256:x"}}

      sink = RealIngestSink(data_dir=tmp_path, qdrant="qdrant-client",
                            embed="embed-transport", run_ingest_fn=fake_run_ingest)
      docs = [
          LoadedDoc(doc_id="a", text="alpha text", source_file="a.txt",
                    split_index=0, n_splits=1),
          LoadedDoc(doc_id="b#part0", text="beta", source_file="b.md",
                    split_index=0, n_splits=2),
      ]
      messages: list[str] = []
      manifest = sink.write_corpus(corpus_id="my-docs", docs=docs,
                                   emit=lambda msg, progress: messages.append(msg))

      raw = tmp_path / "corpora" / "my-docs" / "raw"
      rows = [json.loads(line) for line in (raw / "docs.jsonl").read_text().splitlines()]
      # R1 record shape; BYO docs are unsegmented so passage_id == doc_id
      assert rows[0] == {"doc_id": "a", "passage_id": "a", "title": "a.txt",
                         "text": "alpha text"}
      assert rows[1]["passage_id"] == "b#part0"
      meta = json.loads((raw / "download_meta.json").read_text())
      assert meta["dataset"]["name"] == "byo"  # the runner's multi-hop gate reads this (R10)
      assert calls == {"corpus_id": "my-docs", "data_dir": tmp_path, "chunk_size": 512,
                       "chunk_overlap": 64, "embed": "embed-transport",
                       "qdrant": "qdrant-client"}
      assert manifest["n_docs"] == 2
      assert len(messages) == 2


  def test_construction_resolves_pinned_entry_point(tmp_path):
      from ragreceipts.ingest.pipeline import run_ingest  # noqa: F401  (R9 drift guard)

      sink = RealIngestSink(data_dir=tmp_path, qdrant=None, embed=None)
      assert sink._run_ingest is run_ingest
  ```

  Run: `cd api && uv run pytest tests/test_real_ingest_sink.py -q`
  — EXPECTED: `ImportError: cannot import name 'RealIngestSink'`.

- [ ] Append to `api/ragreceipts/server/ingest_byo.py`:

  ```python
  class RealIngestSink:
      """IngestSink over Plan A's R9-pinned ingest entry point.

      Pin (contracts §Seam Resolutions R9):
        - ingest/pipeline.py::run_ingest(corpus_id=, data_dir=, ingest_config=,
          embed=, qdrant=) -> manifest dict
      write_corpus materializes the R1 raw/ layout — `raw/docs.jsonl` records
      {"doc_id","passage_id","title","text"} (BYO docs are unsegmented, so
      passage_id == doc_id) plus a BYO `raw/download_meta.json` whose dataset block
      carries {"name": "byo", ...} (the eval runner's multi-hop gate reads
      dataset.name, R10) — then delegates to run_ingest, which chunks, embeds BOTH
      named-vector sets, builds the bm25s index, writes manifest.json, and returns
      the manifest. Constructor seam defaults to the real entry point; tests inject
      a fake.
      """

      def __init__(self, *, data_dir: Path, qdrant, embed, run_ingest_fn=None) -> None:
          if run_ingest_fn is None:
              from ragreceipts.ingest.pipeline import run_ingest  # R9 ingest entry point

              run_ingest_fn = run_ingest
          self._data_dir = data_dir
          self._qdrant = qdrant
          self._embed = embed
          self._run_ingest = run_ingest_fn

      def write_corpus(self, *, corpus_id: str, docs: list[LoadedDoc],
                       emit: Callable[[str, float], None]) -> dict:
          import json
          from datetime import datetime, timezone

          from ragreceipts.config import IngestConfig

          raw_dir = self._data_dir / "corpora" / corpus_id / "raw"
          raw_dir.mkdir(parents=True, exist_ok=True)
          with (raw_dir / "docs.jsonl").open("w", encoding="utf-8") as fh:
              for d in docs:
                  fh.write(json.dumps({
                      "doc_id": d.doc_id, "passage_id": d.doc_id,
                      "title": d.source_file, "text": d.text,
                  }) + "\n")
          (raw_dir / "download_meta.json").write_text(json.dumps({
              "corpus_id": corpus_id,
              "dataset": {"name": "byo", "hf_id": None, "config": None,
                          "split": None, "revision": None},
              "created_at": datetime.now(timezone.utc).isoformat(),
          }, indent=2))
          emit(f"raw layout written ({len(docs)} docs)", 0.45)
          manifest = self._run_ingest(
              corpus_id=corpus_id, data_dir=self._data_dir,
              ingest_config=IngestConfig(), embed=self._embed, qdrant=self._qdrant,
          )
          emit("both dense vector sets + sparse index built", 0.85)
          return manifest


  def build_real_ingest_sink(*, paths, qdrant) -> RealIngestSink:
      """Production constructor — wired by deps.build_deps when all three vendor keys
      AND QDRANT_URL (R7) are present."""
      from ragreceipts.vendors.voyage_client import VoyageClient  # Plan A module name

      return RealIngestSink(data_dir=paths.data_dir, qdrant=qdrant, embed=VoyageClient())
  ```

- [ ] Run: `cd api && uv run pytest tests/test_real_ingest_sink.py -q`
  — EXPECTED: `2 passed`.

- [ ] In `deps.py` `build_deps()`, wire it inside the same key+`QDRANT_URL` check used for
  `query_runner`/`eval_runner`:

  ```python
      ingest_sink = None
      if qdrant is not None and all(v.configured for v in vendors):
          from ragreceipts.server.ingest_byo import build_real_ingest_sink

          ingest_sink = build_real_ingest_sink(paths=paths, qdrant=qdrant)
  ```

  (fold into the same `if` block; pass `ingest_sink=ingest_sink,`).
- [ ] Regenerate the typed client (endpoint set changed):

  ```bash
  cd api && uv run python -m ragreceipts.server.export_openapi > ../web/openapi.json
  cd ../web && pnpm gen:api && pnpm build
  ```

- [ ] Full api suite: `cd api && uv run pytest -q` — EXPECTED: all pass.
- [ ] Commit:

  ```bash
  git add api/ragreceipts/server api/tests/test_ingest_byo.py api/tests/test_real_ingest_sink.py api/pyproject.toml api/uv.lock web/openapi.json web/src/lib/api/schema.d.ts
  git commit -m "feat(server): BYO ingest with LlamaIndex readers, 120K-split disclosure, failure collection" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
  ```

---

### Task 14: Corpora upload UI with progress streaming + upload e2e

**Files:**
- Create: `web/src/components/UploadForm.tsx`
- Modify: `web/src/app/corpora/page.tsx` (mount the form)
- Modify: `api/tests/e2e_fixture.py` (TestingIngestSink so upload works offline)
- Test: `web/e2e/corpora-upload.spec.ts`

Progress "streaming" is 1s polling of `GET /jobs/{job_id}` job events — deliberate: the
job rows/events in SQLite are the durable source of truth (resumable), a poll loop renders
them faithfully, and no SSE infrastructure is needed under the single-worker constraint.

**Step 1 — TESTING sink (Modify `api/tests/e2e_fixture.py`):**

- [ ] Append (extend imports with
  `from ragreceipts.server.ingest_byo import LoadedDoc  # noqa: F401` only if needed for
  type hints — the sink is duck-typed):

  ```python
  class TestingIngestSink:
      """Offline IngestSink: counts chunks with the 512-token (~2048-char) window and
      writes deterministic fake index hashes — no Qdrant/bm25s writes. Exercises the BYO
      reader/split/manifest path end-to-end without vendors."""

      def write_corpus(self, *, corpus_id, docs, emit):
          n_chunks = sum(max(1, len(d.text) // 2048) for d in docs)
          emit(f"indexed {len(docs)} docs / {n_chunks} chunks (testing sink)", 0.9)
          return {
              "corpus_id": corpus_id,
              "dataset": {"name": "byo", "hf_id": None, "split": None, "revision": None},
              "chunking": {"chunk_size": 512, "chunk_overlap": 64},
              "embed_model": EMBED_MODEL,
              "index_hashes": {"dense_contextual": "sha256:testing",
                               "dense_isolated": "sha256:testing",
                               "sparse": "sha256:testing"},
              "tokenizer_artifact": "testing",
              "n_docs": len(docs), "n_chunks": n_chunks, "n_queries": 0,
              "created_at": datetime.now(timezone.utc).isoformat(),
          }
  ```

- [ ] In `build_testing_deps()`, replace
  `ingest_sink=None,   # TestingIngestSink wired in Task 14` with
  `ingest_sink=TestingIngestSink(),`.
- [ ] Add an offline check to `api/tests/test_testing_mode.py`:

  ```python
  def test_testing_ingest_round_trip(tmp_path, monkeypatch):
      import time

      with make_client(tmp_path, monkeypatch) as client:
          r = client.post(
              "/corpora/ingest",
              data={"corpus_id": "uploaded-docs"},
              files=[("files", ("note.txt", b"some text", "text/plain"))],
          )
          assert r.status_code == 200, r.text
          job_id = r.json()["job_id"]
          deadline = time.time() + 15
          while time.time() < deadline:
              status = client.get(f"/jobs/{job_id}").json()["status"]
              if status in ("succeeded", "failed"):
                  break
              time.sleep(0.05)
          assert status == "succeeded"
          ids = [c["corpus_id"] for c in client.get("/corpora").json()["corpora"]]
          assert "uploaded-docs" in ids
  ```

- [ ] Run: `cd api && uv run pytest tests/test_testing_mode.py -q` — EXPECTED: all pass.

**Step 2 — failing e2e:**

- [ ] Create `web/e2e/corpora-upload.spec.ts`:

  ```ts
  import { expect, test } from "@playwright/test";

  test("BYO upload streams job progress and the new corpus appears", async ({ page }) => {
    await page.goto("/corpora");
    await page.getByTestId("upload-corpus-id").fill("e2e-byo");
    await page.getByTestId("upload-files").setInputFiles([
      {
        name: "doc-one.txt",
        mimeType: "text/plain",
        buffer: Buffer.from("The first uploaded document, about rivers in France."),
      },
      {
        name: "doc-two.md",
        mimeType: "text/markdown",
        buffer: Buffer.from("# Second doc\n\nAbout capitals of Europe."),
      },
    ]);
    await page.getByTestId("upload-submit").click();
    await expect(page.getByTestId("job-progress")).toBeVisible();
    await expect(page.getByTestId("job-status")).toHaveText("succeeded", { timeout: 30_000 });
    await expect(
      page.getByTestId("corpus-card").filter({ hasText: "e2e-byo" })
    ).toBeVisible();
  });
  ```

- [ ] Run: `cd web && pnpm e2e e2e/corpora-upload.spec.ts` — EXPECTED: FAIL (no form yet).

**Step 3 — the form (COMPLETE code):**

- [ ] Create `web/src/components/UploadForm.tsx`:

  ```tsx
  "use client";
  import { useRef, useState } from "react";
  import { API_BASE, api } from "@/lib/api/client";
  import type { components } from "@/lib/api/schema";

  type JobResponse = components["schemas"]["JobResponse"];

  export default function UploadForm({ onDone }: { onDone: () => void }) {
    const [corpusId, setCorpusId] = useState("");
    const [job, setJob] = useState<JobResponse | null>(null);
    const [error, setError] = useState<string | null>(null);
    const fileRef = useRef<HTMLInputElement>(null);

    async function poll(jobId: string) {
      const { data } = await api.GET("/jobs/{job_id}", {
        params: { path: { job_id: jobId } },
      });
      if (data) setJob(data);
      if (data && (data.status === "succeeded" || data.status === "failed")) {
        onDone();
        return;
      }
      setTimeout(() => poll(jobId), 1000);
    }

    async function submit(e: React.FormEvent) {
      e.preventDefault();
      setError(null);
      const files = fileRef.current?.files;
      if (!files || files.length === 0 || !corpusId) return;
      const form = new FormData();
      form.append("corpus_id", corpusId);
      for (const f of Array.from(files)) form.append("files", f);
      // Multipart goes through raw fetch; the typed client covers the JSON endpoints.
      const res = await fetch(`${API_BASE}/corpora/ingest`, { method: "POST", body: form });
      if (!res.ok) {
        setError(`ingest failed: HTTP ${res.status} ${await res.text()}`);
        return;
      }
      const { job_id } = (await res.json()) as { job_id: string };
      poll(job_id);
    }

    const lastProgress = job?.events.length
      ? job.events[job.events.length - 1].progress
      : 0;

    return (
      <section className="card">
        <h2 style={{ marginTop: 0 }}>Bring your own documents</h2>
        <p className="muted">PDF, Markdown, HTML, or plain text. Runs as a background job.</p>
        <form onSubmit={submit} className="row">
          <input
            type="text"
            data-testid="upload-corpus-id"
            placeholder="corpus-id (lowercase slug)"
            value={corpusId}
            onChange={(e) => setCorpusId(e.target.value)}
          />
          <input
            type="file"
            data-testid="upload-files"
            ref={fileRef}
            multiple
            accept=".pdf,.md,.html,.txt"
          />
          <button className="primary" type="submit" data-testid="upload-submit">
            Ingest
          </button>
        </form>
        {error && <p className="error">{error}</p>}
        {job && (
          <div data-testid="job-progress" style={{ marginTop: 12 }}>
            <div className="row">
              <span className="badge" data-testid="job-status">
                {job.status}
              </span>
              <code>{job.job_id.slice(0, 8)}</code>
            </div>
            <div className="progress" style={{ margin: "8px 0" }}>
              <div style={{ width: `${Math.round(lastProgress * 100)}%` }} />
            </div>
            <ul className="muted" style={{ margin: 0, paddingLeft: 18 }}>
              {job.events.map((ev) => (
                <li key={ev.seq}>{ev.message}</li>
              ))}
            </ul>
            {job.error && <p className="error">{job.error}</p>}
          </div>
        )}
      </section>
    );
  }
  ```

- [ ] Mount it in `web/src/app/corpora/page.tsx`: add
  `import UploadForm from "@/components/UploadForm";` and render
  `<UploadForm onDone={reload} />` directly under the `<h1>`.
- [ ] Run: `cd web && pnpm e2e` — EXPECTED: all specs pass (nav, playground, ablation,
  corpora, corpora-upload).
- [ ] Commit:

  ```bash
  git add web/src api/tests web/e2e/corpora-upload.spec.ts
  git commit -m "feat(web): BYO upload form with polled job progress; offline testing ingest sink" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
  ```

---

### Task 15: Repository README

**Files:**
- Create: `README.md` (repo root)

- [ ] Create `README.md` with exactly this content (the two screenshot paths are created by
  Task 16 — Tasks 15 and 16 land together, README first so the image paths are anchored):

  ````markdown
  # rag-receipts

  **Every RAG technique, with receipts.** An adaptive RAG engine that routes each query
  between a fast path and a deliberative agentic loop — with a built-in Ablation Lab that
  measures each component's contribution on labeled data, compared honestly against
  published anchors.

  ## Why this is different

  Thousands of RAG demos exist; almost none measure whether their components help. Here,
  every pipeline component has a **receipt**: its measured contribution on labeled
  benchmark slices (Natural Questions, MuSiQue), shown next to the published number it is
  compared against — never conflated with it. Cross-domain anchors claim direction-match
  only; every receipt's `published_anchor.note` carries the comparability caveats
  machine-readably, and the UI renders them verbatim.

  ## What's inside (AI-native by construction)

  - **Adaptive routing** — Claude (Haiku 4.5) classifies query complexity; low confidence
    escalates to the System-2 agentic loop (LangGraph): decompose → retrieve per hop →
    CRAG-style grade → refine → synthesize, hard-bounded at 3 hops / 50K tokens.
  - **Strong retrieval first** — hybrid BM25 (bm25s) + dense (voyage-context-3, Qdrant
    named vectors) fused with RRF, then Cohere Rerank v4.0 Pro. Agency sits on top of
    strong retrieval, never replaces it.
  - **Receipts, not vibes** — an eval harness (RAGAS v0.4 + EM/F1 + Recall@5/MRR@3 with a
    pinned gold-alignment rule) that produces versioned `receipts.json`, with pre-run cost
    estimates, a confirmation gate, spend caps, and disclosed failures/abstentions.
  - **Honest degradation** — reranker down? You get RRF order *and* a visible
    `rerank-skipped` badge in the trace. Nothing degrades silently.

  ## Architecture

  ```mermaid
  flowchart LR
    subgraph web["web/ — Next.js (typed OpenAPI client)"]
      PG[Playground]
      AL[Ablation Lab]
      CO[Corpora]
    end
    subgraph api["api/ — FastAPI, single-worker uvicorn"]
      RT[router: Claude Haiku] --> S1[System-1 fast path]
      RT --> S2[System-2 agentic loop]
      S1 --> RC[RetrievalCore]
      S2 --> RC
      EV[ablation runner] --> RC
      RC --> QD[(Qdrant: named vectors)]
      RC --> BM[(bm25s sparse index)]
      TR[(SQLite: traces + jobs)]
    end
    web -->|OpenAPI 3.1| api
    S1 --> CL[Claude Sonnet]
    S2 --> CL
    RC --> CO2[Cohere Rerank v4.0 Pro]
    RC --> VO[voyage-context-3]
  ```

  ## Quickstart (docker compose)

  ```bash
  git clone <this-repo> && cd rag-receipts
  cp .env.example .env        # fill in ANTHROPIC_API_KEY, VOYAGE_API_KEY, COHERE_API_KEY
  docker compose up --build
  ```

  Open <http://localhost:3000>. `GET http://localhost:8000/health` reports per-vendor
  capability — missing keys are named, not stack-traced.

  | Playground | Ablation Lab |
  |---|---|
  | ![Playground: cited answer with hop-by-hop trace](docs/screenshots/playground.png) | ![Ablation Lab: receipts vs published anchors](docs/screenshots/ablation.png) |

  ## Using the eval plane (receipts)

  Estimate first (nothing runs without confirmation):

  ```bash
  curl -s localhost:8000/eval/runs -X POST -H 'content-type: application/json' \
    -d '{"corpus_id": "<corpus>", "preset": "rerank", "slice": "smoke"}'
  # -> {"status": "needs_confirmation", "estimate": {"n_queries": 15, "est_usd": ...}}
  ```

  Confirm to run (smoke slice = 15 queries, minutes not hours):

  ```bash
  curl -s localhost:8000/eval/runs -X POST -H 'content-type: application/json' \
    -d '{"corpus_id": "<corpus>", "preset": "rerank", "slice": "smoke", "confirm": true}'
  curl -s localhost:8000/eval/runs   # job status
  ```

  Results land in `data/receipts-local/` and render in the Ablation Lab next to the
  committed headline receipts from `receipts/` (read-only at runtime). Preset ladder:
  `bm25-only` → `dense-rrf` → `contextual` → `rerank` → `router-on`.

  ## Bring your own documents

  Corpora page → upload PDF/MD/HTML/TXT. Ingestion runs as a resumable background job;
  documents over the 120K-token contextualization window are split and disclosed in the
  corpus manifest; per-document failures are collected, never batch-fatal.

  ## Development

  ```bash
  # api (Python 3.12, uv)
  cd api && uv sync && uv run pytest          # all tests offline, zero keys
  uv run python -m uvicorn ragreceipts.server.app:app --port 8000 --workers 1

  # web (pnpm)
  cd web && pnpm install && pnpm dev

  # e2e (boots api in TESTING=1 fake-vendor mode + web, fully offline)
  cd web && pnpm e2e

  # regenerate the typed client after changing endpoints
  cd api && uv run python -m ragreceipts.server.export_openapi > ../web/openapi.json
  cd ../web && pnpm gen:api
  ```

  The api must run **single-worker**: background jobs are keyed by SQLite rows and executed
  by an in-process worker thread.

  ## Honesty notes

  - Published anchors come mostly from a single non-peer-reviewed financial-domain
    benchmark study; our corpora differ, so anchors claim direction-match only.
  - The `+contextual` receipt measures voyage-context-3 document-context value against the
    *same model* embedding isolated chunks — and its anchor note discloses that the
    independent +2–3pp figure is for LLM-prefix contextualization, a different technique.
  - LLM nondeterminism, model IDs, prompt versions, pricing-table version, and corpus
    manifest hashes are recorded in every receipt.

  Design spec: `docs/superpowers/specs/2026-06-10-rag-receipts-design.md` ·
  Research grounding: `docs/research/2026-06-10-deep-research-advanced-rag.json`
  ````

- [ ] Commit (together with Task 16's screenshots — see below; if committing separately is
  preferred, commit README now and amend nothing later, since the image paths are stable):

  ```bash
  git add README.md
  git commit -m "docs: repository README with architecture, quickstart, eval usage, honesty notes" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
  ```

---

### Task 16: Capture real screenshots from the running app and commit them

No placeholders: the README's image paths are made real here, in the same plan, by driving
the actual UI (TESTING mode renders the same components and the fixture receipts are
committed-format — the pixels are the real app).

**Files:**
- Create: `web/e2e/screenshots.spec.ts`, `docs/screenshots/playground.png`,
  `docs/screenshots/ablation.png`

- [ ] Create `web/e2e/screenshots.spec.ts`:

  ```ts
  import { test } from "@playwright/test";

  // Capture-only spec: skipped unless CAPTURE=1 so normal e2e runs don't rewrite images.
  test.skip(process.env.CAPTURE !== "1", "screenshot capture only");

  test("capture playground", async ({ page }) => {
    await page.setViewportSize({ width: 1280, height: 900 });
    await page.goto("/");
    await page.getByTestId("query-input").fill("What is the capital of France?");
    await page.getByTestId("preset-select").selectOption("rerank");
    await page.getByTestId("run-query").click();
    await page.getByTestId("trace-event").first().waitFor();
    await page.getByTestId("cite-1").click();
    await page.screenshot({ path: "../docs/screenshots/playground.png", fullPage: true });
  });

  test("capture ablation lab", async ({ page }) => {
    await page.setViewportSize({ width: 1280, height: 900 });
    await page.goto("/ablation");
    await page.getByTestId("anchor-note").first().waitFor();
    await page.screenshot({ path: "../docs/screenshots/ablation.png", fullPage: true });
  });
  ```

- [ ] Capture: `mkdir -p docs/screenshots && cd web && CAPTURE=1 pnpm exec playwright test e2e/screenshots.spec.ts`
  — EXPECTED: `2 passed`; two PNGs exist under `docs/screenshots/`.
- [ ] Eyeball both PNGs (open them) — the Playground one must show the route badge, cited
  answer with an open popover, and the trace; the Ablation one must show charts and the
  anchors panel. Re-capture if a loading state was caught.
- [ ] Verify the README image links resolve (paths match `docs/screenshots/*.png`).
- [ ] Commit:

  ```bash
  git add web/e2e/screenshots.spec.ts docs/screenshots
  git commit -m "docs: capture real app screenshots for README via Playwright" -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
  ```

---

### Task 17: Full-stack verification checklist

No new code — evidence before assertions. Every box requires running the command and
seeing the stated output.

**Offline gates (zero keys):**

- [ ] `cd api && uv run pytest -q` — all tests pass, no network, no keys set.
- [ ] `cd api && uv run ruff check .` — clean.
- [ ] `cd web && pnpm build` — type-safe production build.
- [ ] `cd web && pnpm e2e` — all Playwright specs pass against the TESTING=1 api.

**Compose stack (real keys in `.env`):**

- [ ] `cp .env.example .env`, fill the three keys, `docker compose up --build -d`.
- [ ] `docker compose ps` — qdrant healthy, api healthy, web running.
- [ ] `curl -s localhost:8000/health | python3 -m json.tool` — `"status": "ok"`,
  `"missing_env_vars": []`, `"qdrant_ok": true`, `"testing_mode": false`.

**Ingest a smoke corpus (BYO path):**

- [ ] Create two small files and ingest:

  ```bash
  printf 'Paris is the capital of France. The Seine flows through Paris.' > /tmp/a.txt
  printf '# Berlin\n\nBerlin is the capital of Germany.' > /tmp/b.md
  curl -s localhost:8000/corpora/ingest -F corpus_id=smoke-docs \
    -F files=@/tmp/a.txt -F files=@/tmp/b.md
  ```

- [ ] Poll `curl -s localhost:8000/jobs/<job_id>` until `"status": "succeeded"`; events show
  load → index → manifest progress.
- [ ] Corpora page at `localhost:3000/corpora` shows `smoke-docs` with manifest hashes.

**Query (live vendors — this is the 5-query class of manual smoke, never CI):**

- [ ] On the Playground: ask "What is the capital of France?" against `smoke-docs`, preset
  `rerank` — cited answer renders, trace shows route → s1_retrieve → s1_answer with model
  IDs and token counts; route badge shows System-1.

**Smoke eval + receipts in the UI:**

- [ ] `curl -s localhost:8000/eval/runs -X POST -H 'content-type: application/json' -d '{"corpus_id": "<a Plan A benchmark corpus id from /corpora>", "preset": "rerank", "slice": "smoke"}'`
  — returns `needs_confirmation` with a believable estimate and `pricing_table_version`.
- [ ] Re-POST with `"confirm": true` — job starts; `GET /eval/runs` shows it; it succeeds.
- [ ] Ablation Lab shows the new run under the **local** toggle next to the **committed**
  receipts from `receipts/`; anchor notes render verbatim; charts group committed vs local.
- [ ] `docker compose down` — clean shutdown; `docker compose up -d` again — corpora and
  local receipts survive (named volumes).

**Repo hygiene:**

- [ ] `git status` — clean tree; `data/` and `.env` untracked per `.gitignore`.
- [ ] README quickstart re-tested from the top of this checklist verbatim (a stranger can
  run it).
- [ ] Final commit if any checklist fixes were needed, then:

  ```bash
  git log --oneline | head -20   # one conventional commit per task, each with the trailer
  ```

---

## Execution order recap

Tasks 1–7 (server, offline-tested) → 8–11 (web pages + offline e2e) → 12 (compose) →
13–14 (BYO ingest, built last per spec) → 15–16 (README + real screenshots) → 17 (verify).
If v1 drags, Tasks 13–14 are the designated first cut (spec decision #1) — everything
before them ships without BYO.
