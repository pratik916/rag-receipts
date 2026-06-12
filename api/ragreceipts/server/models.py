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
    route: Literal["s1", "s2", "graph"]
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


# -- demo examples ------------------------------------------------------------


class DemoExampleItem(BaseModel):
    label: str
    query: str
    answer: str
    route: str
    citations: list[CitationModel]
    trace_events: list[TraceEventModel]


class DemoExamplesResponse(BaseModel):
    examples: list[DemoExampleItem]
