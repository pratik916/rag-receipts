"""Loaders read Spike 0's raw/ corpus layout (R1); chunk_store round-trips Chunk rows."""

import pytest

from ragreceipts.ingest.chunk_store import read_chunks, write_chunks
from ragreceipts.ingest.loaders import (
    count_queries,
    dataset_name,
    group_documents,
    load_dataset_info,
    load_passages,
)
from tests.corpus_fixtures import TINY_PASSAGES, make_chunk, write_tiny_corpus


def test_load_passages_preserves_order_and_fields(tmp_path):
    corpus_dir = write_tiny_corpus(tmp_path)
    passages = load_passages(corpus_dir)
    assert [p.passage_id for p in passages] == [row["passage_id"] for row in TINY_PASSAGES]
    assert passages[0].doc_id == "d1"
    assert passages[0].title == "Eiffel Tower"
    assert "wrought iron lattice" in passages[0].text


def test_load_passages_missing_corpus_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_passages(tmp_path / "corpora" / "nope")


def test_group_documents_by_doc_id_stable_order(tmp_path):
    passages = load_passages(write_tiny_corpus(tmp_path))
    docs = group_documents(passages)
    assert [doc[0].doc_id for doc in docs] == ["d1", "d2", "d3"]
    assert [p.passage_id for p in docs[0]] == ["d1-p0", "d1-p1"]


def test_dataset_name_strips_dev_slice_suffix():
    assert dataset_name("musique-dev-300") == "musique"
    assert dataset_name("nq-dev-300") == "nq"
    assert dataset_name("tiny") == "tiny"


def test_dataset_info_built_from_download_meta_and_query_count(tmp_path):
    corpus_dir = write_tiny_corpus(tmp_path)
    info = load_dataset_info(corpus_dir)
    # contracts manifest shape, incl. the "name" key Plan B's multi-hop gate reads (R1)
    assert info == {
        "name": "tiny",
        "hf_id": "local/tiny-fixture",
        "split": "test",
        "revision": "fixture-v1",
    }
    assert count_queries(corpus_dir) == 2
    assert count_queries(tmp_path / "corpora" / "nope") == 0


def test_chunk_store_round_trip(tmp_path):
    chunks = [make_chunk("d1:0", "alpha"), make_chunk("d1:1", "beta", passage_id="d1-p1")]
    path = tmp_path / "chunks.jsonl"
    write_chunks(path, chunks)
    assert read_chunks(path) == chunks
