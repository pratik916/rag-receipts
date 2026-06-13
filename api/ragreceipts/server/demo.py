"""DEMO_MODE cost-control layer.

DemoConfig   — frozen dataclass; from_env() returns None when DEMO_MODE is unset.
DemoLedger   — SQLite-backed rate + budget ledger; all checks raise HTTPException.
seed_demo_qdrant — seeds the Qdrant demo collection from committed corpus files on startup.
EST_DEMO_QUERY_USD — conservative per-query budget estimate used by post_query.
_get_client_ip — extracts client IP from X-Forwarded-For or request.client.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import HTTPException, Request

logger = logging.getLogger(__name__)

EST_DEMO_QUERY_USD: float = 0.02  # conservative per-query estimate (S2 worst-case)


@dataclass(frozen=True)
class DemoConfig:
    daily_budget_usd: float
    rate_per_min: int
    rate_per_day: int
    s2_token_ceiling: int
    demo_corpus_id: str

    @classmethod
    def from_env(cls) -> DemoConfig | None:
        if os.environ.get("DEMO_MODE") not in ("1", "true", "yes"):
            return None
        return cls(
            daily_budget_usd=float(os.environ.get("DEMO_DAILY_BUDGET_USD", "2.0")),
            rate_per_min=int(os.environ.get("DEMO_RATE_PER_MIN", "5")),
            rate_per_day=int(os.environ.get("DEMO_RATE_PER_DAY", "20")),
            s2_token_ceiling=int(os.environ.get("DEMO_S2_TOKEN_CEILING", "20000")),
            demo_corpus_id=os.environ.get("DEMO_CORPUS_ID", "demo"),
        )


class DemoLedger:
    """SQLite-backed rate + budget ledger. Single-worker safe (no cross-process locking)."""

    def __init__(self, config: DemoConfig, db_path: Path) -> None:
        self.config = config
        self._db_path = db_path
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS demo_query_log (
                    id       INTEGER PRIMARY KEY AUTOINCREMENT,
                    ip       TEXT    NOT NULL,
                    day      TEXT    NOT NULL,
                    ts_epoch REAL    NOT NULL,
                    usd_actual REAL  NOT NULL
                )"""
            )
            conn.commit()

    @staticmethod
    def _today() -> str:
        return datetime.now(UTC).strftime("%Y-%m-%d")

    def check_rate(self, ip: str) -> None:
        """Raise 429 if this IP exceeds per-minute or per-day limits."""
        now = time.time()
        today = self._today()
        with self._connect() as conn:
            per_min = conn.execute(
                "SELECT COUNT(*) FROM demo_query_log WHERE ip = ? AND ts_epoch > ?",
                (ip, now - 60),
            ).fetchone()[0]
            if per_min >= self.config.rate_per_min:
                raise HTTPException(429, detail={"reason": "rate", "retry_after_s": 60})
            per_day = conn.execute(
                "SELECT COUNT(*) FROM demo_query_log WHERE ip = ? AND day = ?",
                (ip, today),
            ).fetchone()[0]
            if per_day >= self.config.rate_per_day:
                raise HTTPException(429, detail={"reason": "rate", "retry_after_s": 86400})

    def check_budget(self, est_usd: float) -> None:
        """Raise 429 if today's spend + est_usd would exceed the daily budget."""
        today = self._today()
        with self._connect() as conn:
            spent: float = conn.execute(
                "SELECT COALESCE(SUM(usd_actual), 0.0) FROM demo_query_log WHERE day = ?",
                (today,),
            ).fetchone()[0]
        if spent + est_usd > self.config.daily_budget_usd:
            raise HTTPException(429, detail={"reason": "budget"})

    def record(self, ip: str, usd_actual: float) -> None:
        """Record a completed query. Call AFTER the query succeeds."""
        today = self._today()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO demo_query_log (ip, day, ts_epoch, usd_actual) VALUES (?, ?, ?, ?)",
                (ip, today, time.time(), usd_actual),
            )
            conn.commit()


def _get_client_ip(request: Request) -> str:
    """Extract the real client IP, respecting X-Forwarded-For (Railway / PaaS proxy)."""
    xff = request.headers.get("X-Forwarded-For")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def seed_demo_qdrant(qdrant: Any, demo_corpus_dir: Path, config: DemoConfig) -> None:
    """Seed the Qdrant demo collection from committed corpus artifacts.

    No-op if dense_vectors.npz is absent (pre-bootstrap) or the collection already
    has points (idempotent — safe to call on every startup).
    """
    import numpy as np
    from qdrant_client.models import Distance, PointStruct, VectorParams

    dense_path = demo_corpus_dir / "dense_vectors.npz"
    chunks_path = demo_corpus_dir / "chunks.jsonl"

    if not dense_path.exists():
        logger.warning(
            "demo/corpus/dense_vectors.npz not found — skipping seed. "
            "Run docs/runbooks/demo-bootstrap.md first."
        )
        return

    collection_name = config.demo_corpus_id

    # Idempotency: skip if collection is non-empty
    try:
        info = qdrant.get_collection(collection_name)
        if getattr(info, "points_count", 0) and info.points_count > 0:
            logger.info(
                "Demo collection %r already has %d points — skipping seed",
                collection_name,
                info.points_count,
            )
            return
    except Exception as e:
        logger.debug("Demo collection %r not found, will create: %s", collection_name, e)

    data = np.load(dense_path)
    contextual_vecs = data["contextual"]  # shape (n_chunks, embed_dim)
    isolated_vecs = data["isolated"]  # shape (n_chunks, embed_dim)
    embed_dim = int(contextual_vecs.shape[1])

    chunks: list[dict] = [
        json.loads(line) for line in chunks_path.read_text().splitlines() if line.strip()
    ]

    # Recreate the collection (delete if stale empty collection exists)
    try:
        qdrant.delete_collection(collection_name)
    except Exception as e:
        logger.debug("Could not delete stale collection %r (may not exist): %s", collection_name, e)
    qdrant.create_collection(
        collection_name=collection_name,
        vectors_config={
            "contextual": VectorParams(size=embed_dim, distance=Distance.COSINE),
            "isolated": VectorParams(size=embed_dim, distance=Distance.COSINE),
        },
    )

    points = [
        PointStruct(
            id=i,
            vector={
                "contextual": contextual_vecs[i].tolist(),
                "isolated": isolated_vecs[i].tolist(),
            },
            payload=chunk,
        )
        for i, chunk in enumerate(chunks)
    ]
    qdrant.upsert(collection_name=collection_name, points=points, wait=True)
    logger.info("Seeded %d points into demo collection %r", len(points), collection_name)


def materialize_demo_corpus(demo_corpus_dir: Path, corpora_dir: Path, corpus_id: str) -> None:
    """Copy committed demo corpus artifacts into the runtime corpora dir.

    The query path reads chunks/sparse/graph/manifest from {data_dir}/corpora/<id>/,
    but the committed artifacts live in demo_corpus_dir. Copy the query-time artifacts
    across on startup. dense_vectors.npz is intentionally NOT copied (it feeds the Qdrant
    seed only). No-op pre-bootstrap (source manifest absent) and idempotent (skips when the
    target manifest already exists).
    """
    import shutil

    src_manifest = demo_corpus_dir / "manifest.json"
    if not src_manifest.exists():
        logger.warning(
            "demo/corpus/manifest.json not found — skipping corpus materialization. "
            "Run docs/runbooks/demo-bootstrap.md first."
        )
        return

    target = corpora_dir / corpus_id
    if (target / "manifest.json").exists():
        logger.info("Demo corpus already materialized at %s — skipping", target)
        return

    target.mkdir(parents=True, exist_ok=True)
    # Copy the query-time artifacts only (not dense_vectors.npz). manifest.json is the
    # idempotency sentinel, so it is copied LAST — an interrupted startup then leaves no
    # sentinel and the next start re-materializes rather than serving a partial corpus.
    if (demo_corpus_dir / "chunks.jsonl").exists():
        shutil.copy2(demo_corpus_dir / "chunks.jsonl", target / "chunks.jsonl")
    for name in ("sparse", "graph"):
        src_sub = demo_corpus_dir / name
        if src_sub.is_dir():
            shutil.copytree(src_sub, target / name, dirs_exist_ok=True)
    shutil.copy2(src_manifest, target / "manifest.json")

    logger.info("Materialized demo corpus into %s", target)
