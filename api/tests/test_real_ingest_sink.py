"""RealIngestSink: writes the R1 raw/ layout, delegates to the R9-pinned run_ingest."""
import json

from ragreceipts.server.ingest_byo import LoadedDoc, RealIngestSink


def test_write_corpus_marshals_r1_layout_and_pinned_kwargs(tmp_path):
    calls: dict = {}

    def fake_run_ingest(*, corpus_id, data_dir, ingest_config, embed, qdrant):
        calls.update(corpus_id=corpus_id, data_dir=data_dir,
                     chunk_size=ingest_config.chunk_size,
                     chunk_overlap=ingest_config.chunk_overlap,
                     embed=embed, qdrant=qdrant)
        return {"corpus_id": corpus_id, "n_docs": 2,
                "index_hashes": {"sparse": "sha256:x"}}

    sink = RealIngestSink(data_dir=tmp_path, qdrant="qdrant-client",
                          embed="embed-transport", run_ingest_fn=fake_run_ingest)
    docs = [
        LoadedDoc(doc_id="a", text="alpha text", source_file="a.txt",
                  split_index=0, n_splits=1),
        LoadedDoc(doc_id="b#part0", text="beta", source_file="b.md",
                  split_index=0, n_splits=2),
    ]
    messages: list[str] = []
    manifest = sink.write_corpus(corpus_id="my-docs", docs=docs,
                                 emit=lambda msg, progress: messages.append(msg))

    raw = tmp_path / "corpora" / "my-docs" / "raw"
    rows = [json.loads(line) for line in (raw / "docs.jsonl").read_text().splitlines()]
    # R1 record shape; BYO docs are unsegmented so passage_id == doc_id
    assert rows[0] == {"doc_id": "a", "passage_id": "a", "title": "a.txt",
                       "text": "alpha text"}
    assert rows[1]["passage_id"] == "b#part0"
    meta = json.loads((raw / "download_meta.json").read_text())
    assert meta["dataset"]["name"] == "byo"  # the runner's multi-hop gate reads this (R10)
    assert calls == {"corpus_id": "my-docs", "data_dir": tmp_path, "chunk_size": 512,
                     "chunk_overlap": 64, "embed": "embed-transport",
                     "qdrant": "qdrant-client"}
    assert manifest["n_docs"] == 2
    assert len(messages) == 2


def test_construction_resolves_pinned_entry_point(tmp_path):
    from ragreceipts.ingest.pipeline import run_ingest  # noqa: F401  (R9 drift guard)

    sink = RealIngestSink(data_dir=tmp_path, qdrant=None, embed=None)
    assert sink._run_ingest is run_ingest
