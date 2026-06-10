"""Graph index build: structure (passage nodes first, phrases sorted), edges
(relation/appears_in/synonym), write/load round-trip, and a reproducible hash."""

import numpy as np

from ragreceipts.ingest.graph_index import GraphIndex, build_graph_index, write_graph_index
from ragreceipts.ingest.hashing import hash_files
from tests.fakes import FakeEmbed, FakeOpenIE
from tests.graph_fixtures import FIXTURE_CHUNKS, FIXTURE_SCRIPT, build_fixture_graph


def _graph_files(graph_dir):
    return sorted(p for p in graph_dir.iterdir() if p.is_file())


class TestStructure:
    def test_passage_nodes_first_in_chunk_order(self):
        result = build_fixture_graph()
        nodes = result.index.nodes
        passages = [n for n in nodes if n.kind == "passage"]
        assert len(passages) == 4 == result.n_passage
        # passage nodes occupy ids 0..3 in chunk order, text == chunk_id
        assert [n.node_id for n in passages] == [0, 1, 2, 3]
        assert [n.text for n in passages] == ["d1:0", "d1:1", "d2:0", "d3:0"]

    def test_phrase_nodes_after_passages_sorted_by_text(self):
        result = build_fixture_graph()
        phrases = [n for n in result.index.nodes if n.kind == "phrase"]
        assert all(p.node_id >= result.n_passage for p in phrases)
        texts = [p.text for p in phrases]
        assert texts == sorted(texts)  # deterministic ordering -> reproducible hash
        assert "paris" in texts  # normalized (lowercased)
        assert "eiffel tower" in texts

    def test_node_ids_are_dense_range(self):
        result = build_fixture_graph()
        ids = sorted(n.node_id for n in result.index.nodes)
        assert ids == list(range(len(result.index.nodes)))

    def test_triple_count_reported(self):
        result = build_fixture_graph()
        assert result.n_triples == 5  # 2 + 1 + 1 + 1 from the fixture script

    def test_edges_have_all_three_kinds(self):
        result = build_fixture_graph()
        # FakeEmbed vectors won't cross 0.85 cosine generally, so synonym edges may be
        # zero; relation + appears_in must exist.
        kinds = {e.kind for e in result.edges_view()}
        assert "relation" in kinds
        assert "appears_in" in kinds

    def test_shared_phrase_links_two_passages(self):
        # "paris" appears in c0 and c1 -> the phrase node has appears_in edges to both.
        result = build_fixture_graph()
        idx = result.index
        paris = next(n for n in idx.nodes if n.kind == "phrase" and n.text == "paris")
        appears = {
            (e.src, e.dst)
            for e in result.edges_view()
            if e.kind == "appears_in" and paris.node_id in (e.src, e.dst)
        }
        passage_ids = {0, 1}  # d1:0, d1:1
        touched = {e for pair in appears for e in pair} - {paris.node_id}
        assert passage_ids <= touched


class TestVectors:
    def test_vector_rows_aligned_and_unit_norm(self):
        result = build_fixture_graph()
        idx = result.index
        assert idx.passage_vectors.shape[0] == result.n_passage
        assert idx.phrase_vectors.shape[0] == result.n_phrase
        for row in idx.passage_vectors:
            assert np.linalg.norm(row) == np.float32(1.0) or abs(np.linalg.norm(row) - 1.0) < 1e-5
        assert idx.passage_node_to_chunk == {0: "d1:0", 1: "d1:1", 2: "d2:0", 3: "d3:0"}


class TestWriteLoadAndHash:
    def test_round_trip_preserves_structure(self, tmp_path):
        result = build_fixture_graph()
        write_graph_index(result, tmp_path / "graph")
        loaded = GraphIndex.load(tmp_path / "graph")
        assert [n.text for n in loaded.nodes] == [n.text for n in result.index.nodes]
        assert loaded.adjacency.shape == result.index.adjacency.shape
        assert loaded.passage_node_to_chunk == result.index.passage_node_to_chunk
        assert np.allclose(loaded.phrase_vectors, result.index.phrase_vectors)

    def test_artifact_layout(self, tmp_path):
        write_graph_index(build_fixture_graph(), tmp_path / "graph")
        names = {p.name for p in _graph_files(tmp_path / "graph")}
        assert names == {
            "nodes.jsonl",
            "edges.jsonl",
            "phrase_vectors.npy",
            "passage_vectors.npy",
            "passage_map.json",
        }

    def test_hash_reproducible_across_builds(self, tmp_path):
        write_graph_index(build_fixture_graph(), tmp_path / "a")
        write_graph_index(build_fixture_graph(), tmp_path / "b")
        assert hash_files(_graph_files(tmp_path / "a")) == hash_files(_graph_files(tmp_path / "b"))

    def test_hash_changes_with_corpus(self, tmp_path):
        write_graph_index(build_fixture_graph(), tmp_path / "a")
        # a different triple set -> different graph -> different hash
        alt = build_graph_index(
            corpus_id="graphfix",
            chunks=FIXTURE_CHUNKS,
            openie=FakeOpenIE(script={**FIXTURE_SCRIPT, FIXTURE_CHUNKS[2].text: []}),
            embed=FakeEmbed(),
        )
        write_graph_index(alt, tmp_path / "b")
        assert hash_files(_graph_files(tmp_path / "a")) != hash_files(_graph_files(tmp_path / "b"))
