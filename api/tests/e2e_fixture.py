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
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel
from qdrant_client import QdrantClient

from ragreceipts.constants import EMBED_MODEL, RERANK_MODEL, ROUTER_MODEL, SYNTH_MODEL
from ragreceipts.eval.pricing import PRICING_VERSION
from ragreceipts.server.deps import VENDOR_ENV_VARS, AppDeps, AppPaths, VendorCapability
from ragreceipts.server.evalruns import CostEstimate
from ragreceipts.server.jobs import JobRunner
from ragreceipts.server.pipeline import Citation, QueryResult
from ragreceipts.traces.models import TraceEvent
from ragreceipts.types import Chunk
from tests.fakes import InMemoryTraceStore, ScriptedTransport

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


class RouteDecision(BaseModel):
    """Local stand-in for parse() output in TESTING mode; field names live in
    ROUTE_PAYLOAD so this never has to match Plan C's internal route model."""

    complexity: str
    confidence: float


def load_fixture_chunks() -> list[Chunk]:
    raw = json.loads((Path(__file__).parent / "fixtures" / "e2e_corpus.json").read_text())
    # R3: Chunk carries whitespace-token offsets within its parent passage. Fixture
    # chunks are laid out back-to-back per doc, so the offsets are the running sum.
    offsets: dict[str, int] = {}
    chunks: list[Chunk] = []
    for c in raw["chunks"]:
        start = offsets.get(c["doc_id"], 0)
        end = start + len(c["text"].split())
        offsets[c["doc_id"]] = end
        chunks.append(
            Chunk(
                chunk_id=f"{c['doc_id']}:{c['position']}",
                corpus_id=raw["corpus_id"],
                doc_id=c["doc_id"],
                passage_id=c["passage_id"],
                text=c["text"],
                position=c["position"],
                start_token=start,
                end_token=end,
            )
        )
    return chunks


class FixtureQueryRunner:
    """QueryRunner over the fixture corpus, FakeClaude-backed.

    Routing and the answer text come from a ClaudeTransport fake (ScriptedTransport);
    retrieval is deterministic lexical word-overlap so trace scores are stable. A query
    containing the word "degrade" triggers the rerank-skipped degraded path so e2e can
    assert the badge. This exercises the HTTP+UI contract; the real agent graph has its
    own offline tests in Plan C.
    """

    def __init__(self, transport: ScriptedTransport, chunks: list[Chunk], trace_store) -> None:
        self._transport = transport
        self._chunks = chunks
        self._traces = trace_store

    def run(self, *, query: str, corpus_id: str, preset: str) -> QueryResult:
        trace_id = uuid.uuid4().hex
        t0 = time.perf_counter()
        parsed = self._transport.parse(
            model=ROUTER_MODEL,
            system="route",
            user=query,
            max_tokens=1024,
            output_format=RouteDecision,
        )
        decision = parsed.parsed
        self._traces.append(
            TraceEvent(
                trace_id=trace_id,
                seq=0,
                node="route",
                payload={
                    "complexity": decision.complexity,
                    "confidence": decision.confidence,
                    "route": "s1",
                },
                model=ROUTER_MODEL,
                input_tokens=parsed.input_tokens,
                output_tokens=parsed.output_tokens,
                duration_ms=(time.perf_counter() - t0) * 1000,
            )
        )

        degraded = ["rerank-skipped"] if "degrade" in query.lower() else []
        words = set(query.lower().split())
        scored = sorted(
            ((len(words & set(c.text.lower().split())), c) for c in self._chunks),
            key=lambda pair: (-pair[0], pair[1].chunk_id),
        )[:5]
        top = [(score / max(len(words), 1), c) for score, c in scored]
        self._traces.append(
            TraceEvent(
                trace_id=trace_id,
                seq=1,
                node="s1_retrieve",
                payload={
                    "k": 5,
                    "degraded": degraded,
                    "rerank_model": None if degraded else RERANK_MODEL,
                    "chunks": [
                        {
                            "chunk_id": c.chunk_id,
                            "passage_id": c.passage_id,
                            "score": round(s, 4),
                            "text": c.text,
                        }
                        for s, c in top
                    ],
                },
                model=None,
                input_tokens=0,
                output_tokens=0,
                duration_ms=2.0,
            )
        )

        completion = self._transport.complete(
            model=SYNTH_MODEL,
            system="answer",
            user=query,
            max_tokens=4096,
        )
        self._traces.append(
            TraceEvent(
                trace_id=trace_id,
                seq=2,
                node="s1_answer",
                payload={"answer": completion.text, "abstained": False, "degraded": degraded},
                model=SYNTH_MODEL,
                input_tokens=completion.input_tokens,
                output_tokens=completion.output_tokens,
                duration_ms=8.0,
            )
        )
        citations = [
            Citation(
                n=i + 1,
                chunk_id=c.chunk_id,
                passage_id=c.passage_id,
                text=c.text,
                score=round(s, 4),
            )
            for i, (s, c) in enumerate(top)
        ]
        return QueryResult(
            answer=completion.text,
            abstained=False,
            route="s1",
            degraded=degraded,
            citations=citations,
            trace_id=trace_id,
        )


_PRESET_RECALL = {
    "bm25-only": 0.55,
    "dense-rrf": 0.63,
    "contextual": 0.66,
    "rerank": 0.78,
    "router-on": 0.80,
}
# Per-preset index hashes mirror the contracts: IngestConfig.contextual selects the
# named vector AND the matching manifest hash. dense-rrf queries dense_isolated;
# contextual/rerank/router-on query dense_contextual — the differing dense hash is
# what drives the Ablation Lab's cell-level cross-index marker (R11).
_PRESET_INDEX_HASHES = {
    "bm25-only": {"sparse": "sha256:fixture-sparse"},
    "dense-rrf": {"dense_isolated": "sha256:fixture-iso", "sparse": "sha256:fixture-sparse"},
    "contextual": {"dense_contextual": "sha256:fixture-ctx", "sparse": "sha256:fixture-sparse"},
    "rerank": {"dense_contextual": "sha256:fixture-ctx", "sparse": "sha256:fixture-sparse"},
    "router-on": {"dense_contextual": "sha256:fixture-ctx", "sparse": "sha256:fixture-sparse"},
}


def _fixture_receipt(preset: str, run_id: str) -> dict:
    r5 = _PRESET_RECALL.get(preset, 0.6)
    return {
        "run_id": run_id,
        "corpus_id": FIXTURE_CORPUS_ID,
        "preset": preset,
        "config": {"name": preset},
        "index_hashes": _PRESET_INDEX_HASHES.get(preset, {"sparse": "sha256:fixture-sparse"}),
        "models": {
            "router": "claude-haiku-4-5-20251001",
            "synth": "claude-sonnet-4-6",
            "judge": "claude-sonnet-4-6",
            "rerank": "rerank-v4.0-pro",
            "embed": "voyage-context-3",
        },
        "pricing_table_version": PRICING_VERSION,
        "prompts_version": "n/a",
        "n_total": 15,
        "n_failed": 0,
        "n_abstained": 1,
        "metrics": {
            "recall_at_5": r5,
            "mrr_at_3": round(r5 - 0.14, 2),
            "em": 0.33,
            "f1": 0.46,
            "ragas_faithfulness": 0.79,
            "ragas_answer_relevancy": 0.74,
            "latency_p50_ms": 820,
            "latency_p95_ms": 1900,
            "usd_per_query": 0.011,
        },
        # R11: per_query rows use the committed schema exactly —
        # {query_id, retrieved_chunk_ids, latency_ms, usd, flags: {...}}
        "per_query": [
            {
                "query_id": "q-001",
                "retrieved_chunk_ids": ["geo-001:0"],
                "latency_ms": 800,
                "usd": 0.01,
                "flags": {},
            }
        ],
        "anchors": [],
    }


class FixtureEvalRunner:
    """Deterministic EvalRunner for TESTING mode: fixed estimate; writes a complete
    schema_version-1 receipt to data/receipts-local/ so the Ablation Lab local toggle
    has real files to render."""

    def __init__(self, paths: AppPaths) -> None:
        self._paths = paths

    def estimate(self, *, corpus_id: str, preset: str, slice_name: str) -> CostEstimate:
        n = 15 if slice_name == "smoke" else 300
        return CostEstimate(
            n_queries=n,
            est_tokens=n * 4700,
            est_usd=round(n * 0.018, 2),
            pricing_table_version=PRICING_VERSION,
        )

    def run(self, *, corpus_id, preset, slice_name, spend_cap_usd, emit) -> str:
        import uuid as _uuid

        run_id = f"local-{preset}-{_uuid.uuid4().hex[:8]}"
        emit("scoring fixture queries", 0.5)
        path = self._paths.receipts_local_dir / f"{run_id}.json"
        # R11 committed-envelope schema: nondeterminism_note is Plan B's fixed
        # constant — import it rather than duplicating the literal.
        from ragreceipts.eval.receipts import NONDETERMINISM_NOTE

        path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "nondeterminism_note": NONDETERMINISM_NOTE,
                    "receipt": _fixture_receipt(preset, run_id),
                },
                indent=2,
            )
        )
        emit("receipt written", 1.0)
        return run_id


def build_testing_deps() -> AppDeps:
    data_dir = Path(
        os.environ.get("RAGRECEIPTS_DATA_DIR", tempfile.mkdtemp(prefix="ragreceipts-testing-"))
    )
    receipts_dir = Path(os.environ.get("RAGRECEIPTS_RECEIPTS_DIR", "../receipts")).resolve()
    paths = AppPaths(data_dir=data_dir, receipts_committed_dir=receipts_dir)
    paths.ensure()
    _write_fixture_manifest(paths)
    trace_store = InMemoryTraceStore()
    # Seed a local run so the Ablation Lab committed/local toggle has both sources.
    FixtureEvalRunner(paths).run(
        corpus_id=FIXTURE_CORPUS_ID, preset="dense-rrf", slice_name="smoke",
        spend_cap_usd=1.0, emit=lambda message, progress: None,
    )
    return AppDeps(
        paths=paths,
        vendors=[VendorCapability(name, True, env) for name, env in VENDOR_ENV_VARS.items()],
        qdrant=QdrantClient(":memory:"),  # local mode supports named vectors (verified)
        trace_store=trace_store,
        job_runner=JobRunner(paths.jobs_db),
        query_runner=FixtureQueryRunner(
            ScriptedTransport(completions=[ANSWER_TEXT], parse_payloads=[ROUTE_PAYLOAD]),
            load_fixture_chunks(),
            trace_store,
        ),
        eval_runner=FixtureEvalRunner(paths),
        ingest_sink=None,  # TestingIngestSink wired in Task 14
        testing_mode=True,
    )
