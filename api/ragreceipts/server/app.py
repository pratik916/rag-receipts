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
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import asdict

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
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
        allow_origins=os.environ.get("RAGRECEIPTS_CORS_ORIGINS", "http://localhost:3000").split(
            ","
        ),
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)
    return app


app = create_app()
