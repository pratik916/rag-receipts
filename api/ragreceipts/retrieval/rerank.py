"""Rerank is a stage, not a Retriever (contracts). Degradation on VendorUnavailable is
RetrievalCore's job — this stage propagates the exception."""

from ragreceipts.types import ScoredChunk
from ragreceipts.vendors.base import RerankTransport


class RerankStage:
    def __init__(self, transport: RerankTransport):
        self._transport = transport

    def rerank(self, query: str, candidates: list[ScoredChunk], top_n: int) -> list[ScoredChunk]:
        if not candidates:
            return []
        ranked = self._transport.rerank(query, [c.chunk.text for c in candidates], top_n)
        return [
            ScoredChunk(chunk=candidates[index].chunk, score=float(score), source="rerank")
            for index, score in ranked[:top_n]
        ]
