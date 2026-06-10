"""RerankStage maps transport (index, score) pairs back onto candidate chunks."""

import pytest

from ragreceipts.retrieval.rerank import RerankStage
from ragreceipts.types import ScoredChunk
from ragreceipts.vendors.base import VendorUnavailable
from tests.corpus_fixtures import make_chunk
from tests.fakes import FakeRerank

CANDIDATES = [
    ScoredChunk(chunk=make_chunk("a:0", "text a"), score=0.03, source="rrf"),
    ScoredChunk(chunk=make_chunk("b:0", "text b"), score=0.02, source="rrf"),
    ScoredChunk(chunk=make_chunk("c:0", "text c"), score=0.01, source="rrf"),
]


def test_reorders_per_transport_and_relabels_source():
    stage = RerankStage(FakeRerank(script={"q": [2, 0, 1]}))
    got = stage.rerank("q", CANDIDATES, top_n=3)
    assert [s.chunk.chunk_id for s in got] == ["c:0", "a:0", "b:0"]
    assert [s.score for s in got] == pytest.approx([1.0, 0.99, 0.98])
    assert all(s.source == "rerank" for s in got)


def test_passes_chunk_texts_and_top_n_to_transport():
    fake = FakeRerank()
    RerankStage(fake).rerank("q", CANDIDATES, top_n=2)
    query, texts, top_n = fake.calls[0]
    assert (query, texts, top_n) == ("q", ["text a", "text b", "text c"], 2)


def test_truncates_to_top_n():
    got = RerankStage(FakeRerank()).rerank("q", CANDIDATES, top_n=2)
    assert len(got) == 2


def test_empty_candidates_short_circuits():
    fake = FakeRerank()
    assert RerankStage(fake).rerank("q", [], top_n=5) == []
    assert fake.calls == []


def test_transport_failure_propagates():
    with pytest.raises(VendorUnavailable):
        RerankStage(FakeRerank(fail=True)).rerank("q", CANDIDATES, top_n=2)
