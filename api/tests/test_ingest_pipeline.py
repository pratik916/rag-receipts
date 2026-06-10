"""End-to-end ingest (offline): both vector sets, sparse rebuild, manifest with hashes."""

import json

import pytest
from qdrant_client import QdrantClient

from ragreceipts.config import IngestConfig
from ragreceipts.ingest.chunk_store import read_chunks
from ragreceipts.ingest.contextualizer import embed_corpus
from ragreceipts.ingest.hashing import hash_files, hash_vectors
from ragreceipts.ingest.pipeline import run_ingest
from ragreceipts.retrieval.dense import VECTOR_CONTEXTUAL, VECTOR_ISOLATED, DenseRetriever
from ragreceipts.retrieval.sparse import SparseRetriever
from tests.corpus_fixtures import TINY_PASSAGES, write_tiny_corpus
from tests.fakes import FakeEmbed

CONFIG = IngestConfig(chunk_size=40, chunk_overlap=10)


def ingest(tmp_path, qdrant=None):
    write_tiny_corpus(tmp_path)
    return run_ingest(
        corpus_id="tiny",
        data_dir=tmp_path,
        ingest_config=CONFIG,
        embed=FakeEmbed(),
        qdrant=qdrant or QdrantClient(":memory:"),
    )


class TestEmbedCorpus:
    def test_both_vector_sets_chunk_aligned(self):
        docs = [["a b c", "d e f"], ["g h i"]]
        contextual, isolated = embed_corpus(docs, FakeEmbed())
        assert len(contextual) == len(isolated) == 3
        assert contextual[0] != pytest.approx(isolated[0])  # multi-chunk doc: context shifts
        assert contextual[2] == pytest.approx(isolated[2])  # single-chunk doc: identical


class TestHashing:
    def test_hash_vectors_deterministic_and_order_sensitive(self):
        a, b = [1.0, 2.0], [3.0, 4.0]
        assert hash_vectors([a, b]) == hash_vectors([a, b])
        assert hash_vectors([a, b]) != hash_vectors([b, a])
        assert hash_vectors([a]).startswith("sha256:")

    def test_hash_files_content_sensitive(self, tmp_path):
        (tmp_path / "x.json").write_text("one")
        first = hash_files([tmp_path / "x.json"])
        (tmp_path / "x.json").write_text("two")
        assert first != hash_files([tmp_path / "x.json"])


class TestRunIngest:
    def test_manifest_schema_and_counts(self, tmp_path):
        manifest = ingest(tmp_path)
        assert set(manifest) == {
            "corpus_id",
            "dataset",
            "chunking",
            "embed_model",
            "index_hashes",
            "tokenizer_artifact",
            "n_docs",
            "n_chunks",
            "n_queries",
            "created_at",
        }
        assert manifest["corpus_id"] == "tiny"
        # dataset block constructed from raw/download_meta.json, incl. "name" (R1)
        assert manifest["dataset"] == {
            "name": "tiny",
            "hf_id": "local/tiny-fixture",
            "split": "test",
            "revision": "fixture-v1",
        }
        assert manifest["chunking"] == {"chunk_size": 40, "chunk_overlap": 10}
        assert manifest["embed_model"] == "voyage-context-3"
        assert manifest["n_docs"] == 3
        assert manifest["n_queries"] == 2
        assert manifest["tokenizer_artifact"] == "sparse/vocab.tokenizer.json"
        on_disk = json.loads((tmp_path / "corpora" / "tiny" / "manifest.json").read_text())
        assert on_disk == manifest

    def test_index_hashes_present_and_distinct(self, tmp_path):
        hashes = ingest(tmp_path)["index_hashes"]
        assert set(hashes) == {"dense_contextual", "dense_isolated", "sparse"}
        assert all(v.startswith("sha256:") for v in hashes.values())
        # d1 has two passages -> a multi-chunk doc -> contextual must differ from isolated
        assert hashes["dense_contextual"] != hashes["dense_isolated"]

    def test_hashes_reproducible_across_runs(self, tmp_path):
        first = ingest(tmp_path / "run1")["index_hashes"]
        second = ingest(tmp_path / "run2")["index_hashes"]
        assert first == second

    def test_chunks_jsonl_written_with_alignment_metadata(self, tmp_path):
        manifest = ingest(tmp_path)
        chunks = read_chunks(tmp_path / "corpora" / "tiny" / "chunks.jsonl")
        assert len(chunks) == manifest["n_chunks"] > 0
        assert {c.passage_id for c in chunks} == {"d1-p0", "d1-p1", "d2-p0", "d3-p0"}
        assert all(c.chunk_id == f"{c.doc_id}:{c.position}" for c in chunks)
        # R3: persisted token ranges are exact slices of the parent passage's tokens
        passage_text = {row["passage_id"]: row["text"] for row in TINY_PASSAGES}
        for c in chunks:
            tokens = passage_text[c.passage_id].split()
            assert 0 <= c.start_token < c.end_token <= len(tokens)
            assert c.text == " ".join(tokens[c.start_token : c.end_token])

    def test_both_named_vectors_queryable(self, tmp_path):
        client = QdrantClient(":memory:")
        ingest(tmp_path, qdrant=client)
        fake = FakeEmbed()
        for name in (VECTOR_CONTEXTUAL, VECTOR_ISOLATED):
            hits = DenseRetriever(client, "tiny", name, fake).search("anything", k=3)
            assert len(hits) == 3

    def test_sparse_index_loadable_and_searches(self, tmp_path):
        ingest(tmp_path)
        corpus_dir = tmp_path / "corpora" / "tiny"
        retriever = SparseRetriever.load(
            corpus_dir / "sparse", read_chunks(corpus_dir / "chunks.jsonl")
        )
        top = retriever.search("eiffel tower paris", k=3)[0]
        assert top.chunk.doc_id == "d1"

    def test_missing_corpus_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            run_ingest(
                corpus_id="nope",
                data_dir=tmp_path,
                ingest_config=CONFIG,
                embed=FakeEmbed(),
                qdrant=QdrantClient(":memory:"),
            )
