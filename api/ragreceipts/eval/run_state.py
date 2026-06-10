"""SQLite-backed run state: this is what makes eval runs resumable.

WAL mode per spec §Server runtime constraints. Primary key
(run_id, preset, query_id) means a resumed run skips completed queries and a
re-recorded query replaces its row (INSERT OR REPLACE).
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
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
  route TEXT,                        -- 's1' | 's2' | NULL (failed before routing)
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

    def start_run(
        self,
        *,
        run_id: str,
        corpus_id: str,
        slice_name: str,
        presets: list[str],
        spend_cap_usd: float,
    ) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO eval_runs VALUES (?, ?, ?, ?, ?, ?)",
            (
                run_id,
                corpus_id,
                slice_name,
                json.dumps(presets),
                spend_cap_usd,
                datetime.now(UTC).isoformat(),
            ),
        )
        self._conn.commit()

    def record_result(
        self,
        *,
        run_id: str,
        preset: str,
        query_id: str,
        status: str,
        retrieved: list[dict],
        answer: str | None,
        latency_ms: float,
        usd: float,
        input_tokens: int,
        output_tokens: int,
        error: str | None,
        route: str | None = None,
    ) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO eval_results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                preset,
                query_id,
                status,
                json.dumps(retrieved),
                answer,
                latency_ms,
                usd,
                input_tokens,
                output_tokens,
                error,
                route,
            ),
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
