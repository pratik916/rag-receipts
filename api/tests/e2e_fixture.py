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
from datetime import UTC, datetime
from pathlib import Path

from qdrant_client import QdrantClient

from ragreceipts.constants import EMBED_MODEL
from ragreceipts.server.deps import VENDOR_ENV_VARS, AppDeps, AppPaths, VendorCapability
from ragreceipts.server.jobs import JobRunner
from tests.fakes import InMemoryTraceStore

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
        "n_docs": 6,
        "n_chunks": 7,
        "n_queries": 0,
        "created_at": datetime.now(UTC).isoformat(),
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
        query_runner=None,  # FixtureQueryRunner wired in Task 5
        eval_runner=None,  # FixtureEvalRunner wired in Task 7
        ingest_sink=None,  # TestingIngestSink wired in Task 14
        testing_mode=True,
    )
