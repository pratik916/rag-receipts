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
