"""RetrievalCore: the single composed retrieval entry point (contracts).

System-1, System-2 hops, and the eval harness all execute THIS code, parameterized only
by PipelineConfig (shared-retrieval-core invariant). Honors config.query flags, returns
top_k_final chunks, emits one TraceEvent per call through the injected TraceCallback.
Degrade visibly, never silently: dense failure -> BM25-only + "dense-skipped";
rerank failure -> RRF order + "rerank-skipped"; no fallback available -> raise.
route_mode is deliberately ignored here — it is consumed by the Plan C router.
"""

import time
import uuid

from ragreceipts.config import PipelineConfig
from ragreceipts.retrieval.base import Retriever
from ragreceipts.retrieval.rerank import RerankStage
from ragreceipts.traces.models import TraceCallback, TraceEvent
from ragreceipts.types import ScoredChunk
from ragreceipts.vendors.base import VendorUnavailable


class RetrievalCore:
    def __init__(
        self,
        config: PipelineConfig,
        dense: Retriever | None,
        sparse: Retriever | None,
        rerank_stage: RerankStage | None,
        on_trace: TraceCallback | None = None,
    ):
        query = config.query
        if not query.bm25 and not query.dense:
            raise ValueError("config must enable at least one of bm25/dense")
        if query.bm25 and sparse is None:
            raise ValueError("config enables bm25 but no sparse retriever was provided")
        if query.dense and dense is None:
            raise ValueError("config enables dense but no dense retriever was provided")
        if query.rerank and rerank_stage is None:
            raise ValueError("config enables rerank but no rerank stage was provided")
        self._config = config
        self._dense = dense
        self._sparse = sparse
        self._rerank_stage = rerank_stage
        self._on_trace = on_trace

    def retrieve(
        self,
        query: str,
        *,
        trace_id: str | None = None,
        node: str = "s1_retrieve",
        seq_start: int = 0,
    ) -> list[ScoredChunk]:
        q = self._config.query
        started = time.perf_counter()
        degraded: list[str] = []

        retrievers: list[Retriever] = []
        if q.bm25:
            retrievers.append(self._sparse)
        if q.dense:
            retrievers.append(self._dense)

        try:
            candidates = self._fused_search(retrievers, query, q.top_k_fuse)
        except VendorUnavailable:
            if not q.bm25:
                raise  # nothing to fall back to: fail visibly
            degraded.append("dense-skipped")
            candidates = self._sparse.search(query, q.top_k_fuse)

        if q.rerank:
            try:
                final = self._rerank_stage.rerank(query, candidates, q.top_k_final)
            except VendorUnavailable:
                degraded.append("rerank-skipped")
                final = candidates[: q.top_k_final]
        else:
            final = candidates[: q.top_k_final]

        self._emit(
            query,
            candidates,
            final,
            degraded,
            trace_id=trace_id or uuid.uuid4().hex,
            node=node,
            seq=seq_start,
            duration_ms=(time.perf_counter() - started) * 1000.0,
        )
        return final

    @staticmethod
    def _fused_search(retrievers: list[Retriever], query: str, k: int) -> list[ScoredChunk]:
        if len(retrievers) == 1:  # passthrough keeps source labels honest
            return retrievers[0].search(query, k)
        # top_k_fuse is the PER-RETRIEVER candidate depth; the fused UNION becomes the
        # candidate pool and top_k_final does the final cut downstream. HybridRRF couples
        # per-retriever depth and post-truncation to a single k (and would drop union
        # members below that depth), so apply the binding RRF formula here over each
        # retriever's top-k, keeping the full union (rrf_k=60, deterministic id tie-break).
        scores: dict[str, float] = {}
        chunk_by_id = {}
        for retriever in retrievers:
            for rank, scored in enumerate(retriever.search(query, k), start=1):
                chunk_id = scored.chunk.chunk_id
                chunk_by_id[chunk_id] = scored.chunk
                scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (60 + rank)
        ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return [
            ScoredChunk(chunk=chunk_by_id[cid], score=score, source="rrf") for cid, score in ordered
        ]

    def _emit(
        self,
        query: str,
        candidates: list[ScoredChunk],
        final: list[ScoredChunk],
        degraded: list[str],
        *,
        trace_id: str,
        node: str,
        seq: int,
        duration_ms: float,
    ) -> None:
        if self._on_trace is None:
            return
        q = self._config.query
        self._on_trace(
            TraceEvent(
                trace_id=trace_id,
                seq=seq,
                node=node,
                payload={
                    "query": query,
                    "config": {
                        "bm25": q.bm25,
                        "dense": q.dense,
                        "rerank": q.rerank,
                        "route_mode": q.route_mode.value,
                        "top_k_fuse": q.top_k_fuse,
                        "top_k_final": q.top_k_final,
                    },
                    "candidates": [
                        {"chunk_id": c.chunk.chunk_id, "score": c.score, "source": c.source}
                        for c in candidates
                    ],
                    "results": [
                        {"chunk_id": c.chunk.chunk_id, "score": c.score, "source": c.source}
                        for c in final
                    ],
                    "degraded": degraded,
                },
                model=None,
                input_tokens=0,
                output_tokens=0,
                duration_ms=duration_ms,
            )
        )
