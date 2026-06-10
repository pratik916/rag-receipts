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
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class JobStatus(str, Enum):  # noqa: UP042 — str-keyed enum stored verbatim in SQLite
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

    def __init__(self, runner: JobRunner, job_id: str, params: dict) -> None:
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
        self._queue: queue.Queue[str] = queue.Queue()
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
                "SELECT seq, ts, message, progress FROM job_events WHERE job_id = ? ORDER BY seq",
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
