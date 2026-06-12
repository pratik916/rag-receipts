"""FastAPI app (contracts §Server). Run SINGLE-WORKER only:

    cd api && uv run python -m uvicorn ragreceipts.server.app:app \
        --host 0.0.0.0 --port 8000 --workers 1

Single worker is load-bearing: the JobRunner worker thread and its dispatch queue are
in-process state; with more workers, jobs visible in SQLite would belong to a process
that will never execute them. `python -m uvicorn` (not bare `uvicorn`) puts api/ on
sys.path, which the TESTING=1 seam needs to import the tests package.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError

from ragreceipts.server import models as m
from ragreceipts.server.demo import EST_DEMO_QUERY_USD, _get_client_ip
from ragreceipts.server.deps import AppDeps, build_deps

logger = logging.getLogger(__name__)

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
def post_query(
    req: m.QueryRequest, request: Request, deps: AppDeps = Depends(get_deps)
) -> m.QueryResponse:
    ip = "unknown"
    if deps.demo_ledger is not None:
        ip = _get_client_ip(request)
        if req.corpus_id != deps.demo_ledger.config.demo_corpus_id:
            raise HTTPException(
                403, detail="query is limited to the demo corpus in the public demo"
            )
        deps.demo_ledger.check_rate(ip)
        deps.demo_ledger.check_budget(EST_DEMO_QUERY_USD)
    if deps.query_runner is None:
        missing = ", ".join(_missing_env_vars(deps))
        raise HTTPException(503, detail=f"query unavailable; missing env vars: {missing}")
    if not (deps.paths.corpora_dir / req.corpus_id / "manifest.json").exists():
        raise HTTPException(404, detail=f"unknown corpus: {req.corpus_id}")
    result = deps.query_runner.run(
        query=req.query,
        corpus_id=req.corpus_id,
        preset=req.preset,
        token_ceiling=deps.demo_ledger.config.s2_token_ceiling if deps.demo_ledger else None,
    )
    if deps.demo_ledger is not None:
        deps.demo_ledger.record(ip, EST_DEMO_QUERY_USD)
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


@router.post("/corpora/ingest", response_model=m.IngestResponse)
async def ingest_corpus(
    corpus_id: Annotated[str, Form()],
    files: Annotated[list[UploadFile], File()],
    deps: AppDeps = Depends(get_deps),
) -> m.IngestResponse:
    if deps.demo_ledger is not None:
        raise HTTPException(403, detail="ingest is read-only in the public demo")
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


def _load_receipts(directory: Path, source: str, errors: list[str]) -> list[m.ReceiptEntryModel]:
    entries: list[m.ReceiptEntryModel] = []
    if not directory.exists():
        return entries
    for path in sorted(directory.glob("*.json")):
        try:
            data = json.loads(path.read_text())
            entries.append(
                m.ReceiptEntryModel(
                    source=source,
                    path=str(path),
                    schema_version=int(data["schema_version"]),
                    receipt=data["receipt"],
                )
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            errors.append(f"{path}: {exc.__class__.__name__}: {exc}")
    return entries


@router.get("/receipts", response_model=m.ReceiptsResponse)
def list_receipts(deps: AppDeps = Depends(get_deps)) -> m.ReceiptsResponse:
    errors: list[str] = []
    receipts = _load_receipts(deps.paths.receipts_committed_dir, "committed", errors)
    receipts += _load_receipts(deps.paths.receipts_local_dir, "local", errors)
    return m.ReceiptsResponse(receipts=receipts, errors=errors)


@router.get("/demo/examples", response_model=m.DemoExamplesResponse)
def list_demo_examples(deps: AppDeps = Depends(get_deps)) -> m.DemoExamplesResponse:
    examples: list[m.DemoExampleItem] = []
    if deps.paths.demo_examples_dir.exists():
        for path in sorted(deps.paths.demo_examples_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text())
                examples.append(m.DemoExampleItem(**data))
            except (json.JSONDecodeError, TypeError, ValueError, KeyError, ValidationError):
                logger.warning("Skipping malformed demo example: %s", path)
    return m.DemoExamplesResponse(examples=examples)


@router.get("/eval/runs", response_model=m.EvalRunsResponse)
def list_eval_runs(deps: AppDeps = Depends(get_deps)) -> m.EvalRunsResponse:
    return m.EvalRunsResponse(
        runs=[
            m.EvalRunListItem(
                job_id=r.job_id,
                status=r.status.value,
                corpus_id=r.params.get("corpus_id", "?"),
                preset=r.params.get("preset", "?"),
                slice=r.params.get("slice", "?"),
                created_at=r.created_at,
            )
            for r in deps.job_runner.list(kind="eval")
        ]
    )


@router.post("/eval/runs", response_model=m.EvalRunResponse)
def create_eval_run(req: m.EvalRunRequest, deps: AppDeps = Depends(get_deps)) -> m.EvalRunResponse:
    if deps.demo_ledger is not None:
        raise HTTPException(403, detail="eval is read-only in the public demo")
    if deps.eval_runner is None:
        missing = ", ".join(_missing_env_vars(deps))
        raise HTTPException(503, detail=f"eval unavailable; missing env vars: {missing}")
    est = deps.eval_runner.estimate(
        corpus_id=req.corpus_id,
        preset=req.preset,
        slice_name=req.slice,
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
        job_id=row.job_id,
        kind=row.kind,
        status=row.status.value,
        params=row.params,
        error=row.error,
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


def create_app(deps_factory: Callable[[], AppDeps] = build_deps) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        deps = deps_factory()
        app.state.deps = deps
        if deps.eval_runner is not None:
            from ragreceipts.server.evalruns import EvalRunner  # noqa: F401  (protocol doc)

            eval_runner = deps.eval_runner

            def eval_handler(ctx):
                p = ctx.params
                ctx.emit(f"eval start preset={p['preset']} slice={p['slice']}", 0.0)
                run_id = eval_runner.run(
                    corpus_id=p["corpus_id"],
                    preset=p["preset"],
                    slice_name=p["slice"],
                    spend_cap_usd=p["spend_cap_usd"],
                    emit=ctx.emit,
                )
                ctx.emit(f"eval complete run_id={run_id}", 1.0)

            deps.job_runner.register("eval", eval_handler)
        if deps.ingest_sink is not None:
            from ragreceipts.server.ingest_byo import make_ingest_handler

            deps.job_runner.register(
                "ingest", make_ingest_handler(deps.ingest_sink, deps.paths.corpora_dir)
            )
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
