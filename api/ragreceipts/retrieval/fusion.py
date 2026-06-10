"""Reciprocal Rank Fusion over any set of Retrievers.

RRF score for a chunk = sum(1 / (rrf_k + rank_i)) over the rank lists containing it,
rank 1-based (binding definition from contracts). Ties break by chunk_id ascending so
fusion is fully deterministic — receipts must be reproducible.
"""

from ragreceipts.retrieval.base import Retriever
from ragreceipts.types import ScoredChunk


def rrf_fuse(
    rank_lists: list[list[ScoredChunk]], rrf_k: int = 60, *, limit: int | None = None
) -> list[ScoredChunk]:
    """Fuse N ranked lists into one deduped RRF ranking (the binding fusion primitive).

    RRF score for a chunk = sum(1 / (rrf_k + rank)) over the lists containing it, rank
    1-based. Ties break by chunk_id ascending for determinism. Returns the full deduped
    union unless `limit` truncates it. Fused chunks carry source="rrf".
    """
    scores: dict[str, float] = {}
    chunk_by_id = {}
    for ranked in rank_lists:
        for rank, scored in enumerate(ranked, start=1):
            chunk_id = scored.chunk.chunk_id
            chunk_by_id[chunk_id] = scored.chunk
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    if limit is not None:
        ordered = ordered[:limit]
    return [
        ScoredChunk(chunk=chunk_by_id[chunk_id], score=score, source="rrf")
        for chunk_id, score in ordered
    ]


class HybridRRF:
    def __init__(self, retrievers: list[Retriever], rrf_k: int = 60):
        if not retrievers:
            raise ValueError("HybridRRF needs at least one retriever")
        self._retrievers = retrievers
        self._rrf_k = rrf_k

    def search(self, query: str, k: int) -> list[ScoredChunk]:
        rank_lists = [retriever.search(query, k) for retriever in self._retrievers]
        return rrf_fuse(rank_lists, self._rrf_k, limit=k)
