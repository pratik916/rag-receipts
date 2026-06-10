"""GraphRetriever: both recognition modes, dense blend, source='graph', and the
failure paths (embed down, llm-without-claude, empty index). Fully offline."""

import pytest

from ragreceipts.ingest.graph_index import build_graph_index
from ragreceipts.retrieval.graph import GraphRetriever, SeedSelection
from ragreceipts.vendors.base import VendorUnavailable
from tests.fakes import FakeClaude, FakeEmbed, FakeOpenIE
from tests.graph_fixtures import FIXTURE_CHUNKS, build_fixture_graph


def _retriever(*, recognition="embedding", claude=None, embed=None, result=None):
    result = result or build_fixture_graph()
    return GraphRetriever(
        result.index,
        chunks=FIXTURE_CHUNKS,
        embed=embed or FakeEmbed(),
        claude=claude,
        recognition=recognition,
    )


class TestConstruction:
    def test_llm_recognition_requires_claude(self):
        with pytest.raises(ValueError):
            _retriever(recognition="llm", claude=None)

    def test_invalid_recognition_mode_rejected(self):
        with pytest.raises(ValueError):
            GraphRetriever(
                build_fixture_graph().index,
                chunks=FIXTURE_CHUNKS,
                embed=FakeEmbed(),
                recognition="banana",
            )

    def test_empty_index_rejected_at_query_time(self):
        empty = build_graph_index(corpus_id="x", chunks=[], openie=FakeOpenIE(), embed=FakeEmbed())
        retr = GraphRetriever(empty.index, chunks=[], embed=FakeEmbed(), recognition="embedding")
        with pytest.raises(VendorUnavailable):
            retr.search("anything", k=3)


class TestEmbeddingMode:
    def test_returns_graph_sourced_chunks(self):
        results = _retriever(recognition="embedding").search("Eiffel Tower Paris", k=4)
        assert results
        assert all(r.source == "graph" for r in results)
        assert all(isinstance(r.score, float) for r in results)
        ids = [r.chunk.chunk_id for r in results]
        assert len(ids) == len(set(ids))  # deduped passages
        assert len(results) <= 4

    def test_query_about_eiffel_ranks_its_passage_high(self):
        # FakeEmbed aliases let us steer the query vector onto c0's text.
        embed = FakeEmbed(query_aliases={"q": FIXTURE_CHUNKS[0].text})
        results = _retriever(recognition="embedding", embed=embed).search("q", k=4)
        assert results[0].chunk.chunk_id == "d1:0"

    def test_no_claude_call_in_embedding_mode(self):
        claude = FakeClaude(script=[])  # would AssertionError if called
        retr = GraphRetriever(
            build_fixture_graph().index,
            chunks=FIXTURE_CHUNKS,
            embed=FakeEmbed(),
            claude=claude,
            recognition="embedding",
        )
        retr.search("Paris", k=3)  # must not consume the (empty) script
        assert claude.parse_calls == []


class TestLlmMode:
    def test_llm_recognition_filters_seeds_and_returns_chunks(self):
        claude = FakeClaude(script=[SeedSelection(phrases=["paris", "eiffel tower"])])
        results = _retriever(recognition="llm", claude=claude).search(
            "Where is the Eiffel Tower?", k=4
        )
        assert results
        assert all(r.source == "graph" for r in results)
        assert len(claude.parse_calls) == 1
        assert claude.parse_calls[0]["output_format"] is SeedSelection

    def test_llm_returning_no_phrases_falls_back_to_embedding_seeds(self):
        claude = FakeClaude(script=[SeedSelection(phrases=[])])
        results = _retriever(recognition="llm", claude=claude).search("Paris", k=4)
        assert results  # never empties the seed set


class TestFailurePaths:
    def test_embed_failure_raises_vendor_unavailable(self):
        retr = _retriever(recognition="embedding", embed=FakeEmbed(fail_query=True))
        with pytest.raises(VendorUnavailable):
            retr.search("Paris", k=3)

    def test_llm_failure_raises_vendor_unavailable(self):
        # FakeClaude empty-script -> AssertionError; the retriever wraps recognition
        # failures as VendorUnavailable so RetrievalCore can degrade visibly.
        retr = _retriever(recognition="llm", claude=FakeClaude(script=[]))
        with pytest.raises(VendorUnavailable):
            retr.search("Paris", k=3)


class TestBlend:
    def test_blend_weights_ppr_and_dense(self):
        result = build_fixture_graph()
        embed = FakeEmbed(query_aliases={"q": FIXTURE_CHUNKS[3].text})  # solar panels
        pure_dense = GraphRetriever(
            result.index, chunks=FIXTURE_CHUNKS, embed=embed, recognition="embedding", blend=0.0
        ).search("q", k=4)
        # blend=0 -> pure dense -> the aliased passage tops the ranking
        assert pure_dense[0].chunk.chunk_id == "d3:0"
