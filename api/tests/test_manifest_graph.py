"""G6: the graph index hash over the artifact files (sorted), omitted when absent."""

from ragreceipts.ingest.graph_index import write_graph_index
from ragreceipts.ingest.manifest import graph_index_hash
from tests.graph_fixtures import build_fixture_graph


def test_hash_over_artifact_files(tmp_path):
    graph_dir = tmp_path / "graph"
    write_graph_index(build_fixture_graph(), graph_dir)
    h = graph_index_hash(graph_dir)
    assert h is not None
    assert h.startswith("sha256:")


def test_hash_reproducible(tmp_path):
    write_graph_index(build_fixture_graph(), tmp_path / "a")
    write_graph_index(build_fixture_graph(), tmp_path / "b")
    assert graph_index_hash(tmp_path / "a") == graph_index_hash(tmp_path / "b")


def test_absent_artifact_returns_none(tmp_path):
    assert graph_index_hash(tmp_path / "missing") is None


def test_hash_matches_manual_hash_files(tmp_path):
    from ragreceipts.ingest.hashing import hash_files

    graph_dir = tmp_path / "graph"
    write_graph_index(build_fixture_graph(), graph_dir)
    files = sorted(p for p in graph_dir.iterdir() if p.is_file())
    assert graph_index_hash(graph_dir) == hash_files(files)
