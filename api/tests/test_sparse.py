"""SparseRetriever: bm25s build/serialize/load with tokenizer artifact, plus edge guards."""

import pytest

from ragreceipts.ingest.chunk_store import read_chunks, write_chunks
from ragreceipts.retrieval.sparse import SparseRetriever, build_sparse_index
from tests.corpus_fixtures import make_chunk

CHUNKS = [
    make_chunk("d1:0", "the eiffel tower is a lattice tower in paris", passage_id="d1-p0"),
    make_chunk("d2:0", "domestic cats often hunt mice and birds", passage_id="d2-p0"),
    make_chunk("d3:0", "solar panels convert sunlight into electricity", passage_id="d3-p0"),
]


@pytest.fixture()
def built(tmp_path):
    retriever = build_sparse_index(CHUNKS, tmp_path / "sparse")
    return retriever, tmp_path / "sparse"


def test_search_returns_lexical_top_hit(built):
    retriever, _ = built
    results = retriever.search("eiffel tower paris", k=3)
    assert results[0].chunk.chunk_id == "d1:0"
    assert results[0].source == "bm25"
    assert results[0].score > 0


def test_persisted_index_round_trips_with_tokenizer_artifact(built, tmp_path):
    retriever, index_dir = built
    assert (index_dir / "vocab.tokenizer.json").exists()  # the tokenizer artifact
    assert (index_dir / "stopwords.tokenizer.json").exists()
    chunks_path = tmp_path / "chunks.jsonl"
    write_chunks(chunks_path, CHUNKS)
    reloaded = SparseRetriever.load(index_dir, read_chunks(chunks_path))
    live = retriever.search("eiffel tower paris", k=3)
    loaded = reloaded.search("eiffel tower paris", k=3)
    assert [r.chunk.chunk_id for r in loaded] == [r.chunk.chunk_id for r in live]
    assert [r.score for r in loaded] == pytest.approx([r.score for r in live])


def test_k_larger_than_corpus_is_clamped(built):
    retriever, _ = built
    results = retriever.search("eiffel tower paris", k=50)  # bm25s would raise unclamped
    assert len(results) <= len(CHUNKS)


def test_zero_score_results_filtered(built):
    retriever, _ = built
    assert retriever.search("the of and", k=3) == []  # all stopwords -> 0.0 scores


def test_empty_chunk_list_searches_empty(tmp_path):
    retriever = build_sparse_index(
        [make_chunk("d1:0", "only one document here")], tmp_path / "sparse"
    )
    assert retriever.search("unrelated query terms", k=5) == []
