"""CLI wiring: factories are monkeypatched so the test stays offline and keyless."""

import json

from qdrant_client import QdrantClient

import ragreceipts.cli as cli
from tests.corpus_fixtures import write_tiny_corpus
from tests.fakes import FakeEmbed


def test_ingest_command_writes_manifest_and_prints_it(tmp_path, monkeypatch, capsys):
    write_tiny_corpus(tmp_path)
    monkeypatch.setattr(cli, "build_embed_transport", lambda: FakeEmbed())
    monkeypatch.setattr(cli, "build_qdrant", lambda data_dir: QdrantClient(":memory:"))
    code = cli.main(
        [
            "ingest",
            "--corpus",
            "tiny",
            "--data-dir",
            str(tmp_path),
            "--chunk-size",
            "40",
            "--chunk-overlap",
            "10",
        ]
    )
    assert code == 0
    assert (tmp_path / "corpora" / "tiny" / "manifest.json").exists()
    printed = json.loads(capsys.readouterr().out)
    assert printed["corpus_id"] == "tiny"
    assert printed["chunking"] == {"chunk_size": 40, "chunk_overlap": 10}


def test_missing_corpus_exits_nonzero_with_named_message(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "build_embed_transport", lambda: FakeEmbed())
    monkeypatch.setattr(cli, "build_qdrant", lambda data_dir: QdrantClient(":memory:"))
    code = cli.main(["ingest", "--corpus", "nope", "--data-dir", str(tmp_path)])
    assert code == 1
    assert "nope" in capsys.readouterr().err


def test_missing_voyage_key_is_a_named_error(monkeypatch):
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    try:
        cli.build_embed_transport()
        raised = False
    except SystemExit as err:
        raised = "VOYAGE_API_KEY" in str(err)
    assert raised
