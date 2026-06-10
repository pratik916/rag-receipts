"""Hand-computed golden tests for reciprocal rank fusion (rrf_k=60, 1-based ranks)."""

import pytest

from ragreceipts.retrieval.fusion import HybridRRF
from ragreceipts.types import ScoredChunk
from tests.corpus_fixtures import make_chunk

A = make_chunk("a:0", "text a")
B = make_chunk("b:0", "text b")
C = make_chunk("c:0", "text c")
D = make_chunk("d:0", "text d")


class ListRetriever:
    """Test stand-in for the Retriever protocol: returns a fixed ranked list."""

    def __init__(self, ranked, source="bm25"):
        self._ranked = [
            ScoredChunk(chunk=ch, score=10.0 - i, source=source) for i, ch in enumerate(ranked)
        ]

    def search(self, query: str, k: int):
        return self._ranked[:k]


def test_golden_rrf_scores_and_order():
    fused = HybridRRF([ListRetriever([A, B, C]), ListRetriever([B, D])]).search("q", k=4)
    assert [s.chunk.chunk_id for s in fused] == ["b:0", "a:0", "d:0", "c:0"]
    assert fused[0].score == pytest.approx(1 / 62 + 1 / 61)
    assert fused[1].score == pytest.approx(1 / 61)
    assert fused[2].score == pytest.approx(1 / 62)
    assert fused[3].score == pytest.approx(1 / 63)
    assert all(s.source == "rrf" for s in fused)


def test_truncates_to_k():
    fused = HybridRRF([ListRetriever([A, B, C]), ListRetriever([B, D])]).search("q", k=2)
    assert [s.chunk.chunk_id for s in fused] == ["b:0", "a:0"]


def test_each_retriever_consulted_with_k():
    calls = []

    class Spy(ListRetriever):
        def search(self, query, k):
            calls.append(k)
            return super().search(query, k)

    HybridRRF([Spy([A]), Spy([B])]).search("q", k=7)
    assert calls == [7, 7]


def test_ties_break_deterministically_by_chunk_id():
    fused = HybridRRF([ListRetriever([B]), ListRetriever([A])]).search("q", k=2)
    assert [s.chunk.chunk_id for s in fused] == ["a:0", "b:0"]  # equal 1/61, id ascending
    assert fused[0].score == pytest.approx(fused[1].score)


def test_custom_rrf_k():
    fused = HybridRRF([ListRetriever([A])], rrf_k=10).search("q", k=1)
    assert fused[0].score == pytest.approx(1 / 11)


def test_requires_at_least_one_retriever():
    with pytest.raises(ValueError):
        HybridRRF([])
