"""Application dependency container.

Everything the endpoints need arrives through AppDeps — never module globals — so unit
tests construct it directly with fakes, and TESTING=1 swaps the whole container.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from ragreceipts.server.demo import DemoLedger

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
    demo_corpus_dir: Path
    demo_examples_dir: Path

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

    @property
    def demo_db(self) -> Path:
        return self.data_dir / "demo.sqlite"

    def ensure(self) -> None:
        for d in (self.data_dir, self.corpora_dir, self.receipts_local_dir, self.uploads_dir):
            d.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls) -> AppPaths:
        return cls(
            data_dir=Path(os.environ.get("RAGRECEIPTS_DATA_DIR", "../data")).resolve(),
            receipts_committed_dir=Path(
                os.environ.get("RAGRECEIPTS_RECEIPTS_DIR", "../receipts")
            ).resolve(),
            demo_corpus_dir=Path(
                os.environ.get("RAGRECEIPTS_DEMO_CORPUS_DIR", "../demo/corpus")
            ).resolve(),
            demo_examples_dir=Path(
                os.environ.get("RAGRECEIPTS_DEMO_EXAMPLES_DIR", "../demo/examples")
            ).resolve(),
        )


@dataclass
class AppDeps:
    paths: AppPaths
    vendors: list[VendorCapability]
    qdrant: object | None  # qdrant_client.QdrantClient when wired
    trace_store: TraceReadWrite
    job_runner: JobRunner
    query_runner: object | None  # server.pipeline.QueryRunner (Task 5)
    eval_runner: object | None  # server.evalruns.EvalRunner (Task 7)
    ingest_sink: object | None  # server.ingest_byo.IngestSink (Task 13, built last)
    testing_mode: bool
    demo_ledger: DemoLedger | None = None


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
    query_runner = None
    eval_runner = None
    ingest_sink = None
    if qdrant is not None and all(v.configured for v in vendors):
        from ragreceipts.server.evalruns import RealEvalRunner
        from ragreceipts.server.ingest_byo import build_real_ingest_sink
        from ragreceipts.server.pipeline import build_real_query_runner

        query_runner = build_real_query_runner(paths=paths, qdrant=qdrant, trace_store=trace_store)
        eval_runner = RealEvalRunner(data_dir=paths.data_dir)
        ingest_sink = build_real_ingest_sink(paths=paths, qdrant=qdrant)
    from ragreceipts.server.demo import DemoConfig, DemoLedger

    demo_cfg = DemoConfig.from_env()
    demo_ledger = DemoLedger(demo_cfg, paths.demo_db) if demo_cfg is not None else None
    return AppDeps(
        paths=paths,
        vendors=vendors,
        qdrant=qdrant,
        trace_store=trace_store,
        job_runner=job_runner,
        query_runner=query_runner,  # wired when all keys + QDRANT_URL are present (R7)
        eval_runner=eval_runner,  # wired when all keys + QDRANT_URL are present (R7)
        ingest_sink=ingest_sink,  # wired when all keys + QDRANT_URL are present (R7)
        testing_mode=False,
        demo_ledger=demo_ledger,
    )
