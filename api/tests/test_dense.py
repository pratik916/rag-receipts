"""DenseRetriever over Qdrant named vectors ("contextual"/"isolated") in :memory: mode."""

import pytest
from qdrant_client import QdrantClient

from ragreceipts.ingest.indexer import write_dense_index
from ragreceipts.retrieval.dense import (
    VECTOR_CONTEXTUAL,
    VECTOR_ISOLATED,
    DenseRetriever,
    point_id_for_chunk,
    vector_name_for,
)
from ragreceipts.vendors.base import VendorUnavailable
from tests.corpus_fixtures import make_chunk
from tests.fakes import FakeEmbed

# m1 is a two-chunk document; m2 repeats m1's first chunk text as a single-chunk doc.
# Isolated vectors for the repeated text are identical; contextual vectors differ —
# that asymmetry is what proves the named-vector selection actually changes behavior.
C_M1A = make_chunk("m1:0", "alpha beta gamma", corpus_id="dense-test", passage_id="m1-p0")
C_M1B = make_chunk("m1:1", "delta epsilon zeta", corpus_id="dense-test", passage_id="m1-p0")
C_M2 = make_chunk("m2:0", "alpha beta gamma", corpus_id="dense-test", passage_id="m2-p0")
CHUNKS = [C_M1A, C_M1B, C_M2]
DOC_CHUNK_TEXTS = [[C_M1A.text, C_M1B.text], [C_M2.text]]


@pytest.fixture()
def indexed():
    fake = FakeEmbed()
    contextual = [vec for doc in fake.embed_documents(DOC_CHUNK_TEXTS) for vec in doc]
    isolated = [
        doc[0] for doc in fake.embed_documents([[t] for doc in DOC_CHUNK_TEXTS for t in doc])
    ]
    client = QdrantClient(":memory:")
    write_dense_index(client, "dense-test", CHUNKS, contextual, isolated)
    return client, fake


def test_vector_name_for():
    assert vector_name_for(True) == VECTOR_CONTEXTUAL == "contextual"
    assert vector_name_for(False) == VECTOR_ISOLATED == "isolated"


def test_point_ids_deterministic_and_distinct():
    assert point_id_for_chunk("m1:0") == point_id_for_chunk("m1:0")
    assert point_id_for_chunk("m1:0") != point_id_for_chunk("m2:0")


def test_isolated_vector_ties_on_identical_text(indexed):
    client, fake = indexed
    retriever = DenseRetriever(client, "dense-test", VECTOR_ISOLATED, fake)
    results = retriever.search("alpha beta gamma", k=2)
    assert {r.chunk.chunk_id for r in results} == {"m1:0", "m2:0"}
    assert results[0].score == pytest.approx(1.0)
    assert results[1].score == pytest.approx(1.0)
    assert all(r.source == "dense" for r in results)


def test_contextual_vector_separates_same_text_in_different_docs(indexed):
    client, fake = indexed
    retriever = DenseRetriever(client, "dense-test", VECTOR_CONTEXTUAL, fake)
    results = retriever.search("alpha beta gamma", k=2)
    assert results[0].chunk.chunk_id == "m2:0"  # single-chunk doc: context == chunk
    assert results[0].score == pytest.approx(1.0)
    assert results[1].score < 0.999  # doc context shifted m1:0's vector


def test_payload_round_trips_full_chunk(indexed):
    client, fake = indexed
    retriever = DenseRetriever(client, "dense-test", VECTOR_ISOLATED, fake)
    top = retriever.search("delta epsilon zeta", k=1)[0]
    assert top.chunk == C_M1B


def test_k_larger_than_point_count(indexed):
    client, fake = indexed
    retriever = DenseRetriever(client, "dense-test", VECTOR_ISOLATED, fake)
    assert len(retriever.search("alpha beta gamma", k=10)) == 3


def test_embed_failure_propagates_vendor_unavailable(indexed):
    client, _ = indexed
    retriever = DenseRetriever(client, "dense-test", VECTOR_ISOLATED, FakeEmbed(fail_query=True))
    with pytest.raises(VendorUnavailable):
        retriever.search("anything", k=1)


def test_write_dense_index_rejects_empty_or_mismatched():
    client = QdrantClient(":memory:")
    with pytest.raises(ValueError):
        write_dense_index(client, "x", [], [], [])
    with pytest.raises(ValueError):
        write_dense_index(client, "x", CHUNKS, [[0.0] * 8] * 2, [[0.0] * 8] * 3)
