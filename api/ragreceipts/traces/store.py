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
                (
                    event.trace_id,
                    event.seq,
                    event.node,
                    json.dumps(event.payload),
                    event.model,
                    event.input_tokens,
                    event.output_tokens,
                    event.duration_ms,
                ),
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
            TraceEvent(
                trace_id=r[0],
                seq=r[1],
                node=r[2],
                payload=json.loads(r[3]),
                model=r[4],
                input_tokens=r[5],
                output_tokens=r[6],
                duration_ms=r[7],
            )
            for r in rows
        ]

    def close(self) -> None:
        with self._lock:
            self._conn.close()
