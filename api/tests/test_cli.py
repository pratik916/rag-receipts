"""CLI wiring tests.

Plan A: ingest wiring with monkeypatched factories (offline, keyless).
Plan B (appended): eval arg validation, named missing-key errors, the cost
confirm gate, the offline composition-root construction test, and promote.
The eval happy path against real vendors is the keyed manual step (Task 10);
the offline end-to-end eval path is covered by the harness self-test.
"""

import json
from pathlib import Path

import pytest
from qdrant_client import QdrantClient

import ragreceipts.cli as cli
from ragreceipts.cli import main
from ragreceipts.config import PRESETS
from ragreceipts.eval.receipts import Receipt, make_run_doc, write_run_doc
from ragreceipts.ingest.chunk_store import write_chunks
from ragreceipts.retrieval.core import RetrievalCore
from ragreceipts.retrieval.sparse import build_sparse_index
from ragreceipts.types import Chunk
from tests.corpus_fixtures import write_tiny_corpus
from tests.fakes import FakeEmbed, FakeRerank


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


# =====================================================================
# Plan B (R6): eval + receipts promote - appended after Plan A's tests
# =====================================================================

KEYS = ("ANTHROPIC_API_KEY", "VOYAGE_API_KEY", "COHERE_API_KEY")


def minimal_receipt() -> Receipt:
    return Receipt(
        run_id="r1",
        corpus_id="c1",
        preset="bm25-only",
        config={"name": "bm25-only"},
        index_hashes={"sparse": "sha256:s"},
        models={
            "router": "claude-haiku-4-5-20251001",
            "synth": "claude-sonnet-4-6",
            "judge": "claude-sonnet-4-6",
            "rerank": "rerank-v4.0-pro",
            "embed": "voyage-context-3",
        },
        pricing_table_version="2026-06-10",
        prompts_version="n/a",
        n_total=1,
        n_failed=0,
        n_abstained=0,
        metrics={
            "recall_at_5": 1.0,
            "mrr_at_3": 1.0,
            "em": 1.0,
            "f1": 1.0,
            "ragas_faithfulness": None,
            "ragas_answer_relevancy": None,
            "latency_p50_ms": 1.0,
            "latency_p95_ms": 1.0,
            "usd_per_query": 0.001,
        },
        per_query=[
            {
                "query_id": "q0",
                "retrieved_chunk_ids": ["d:0"],
                "answer": "secret model text",
                "latency_ms": 1.0,
                "usd": 0.001,
                "flags": {"status": "ok", "em": 1.0, "f1": 1.0},
            }
        ],
        anchors=[],
    )


def write_min_corpus(data_dir: Path, corpus_id: str = "c1") -> None:
    """Spike 0 raw layout (R1) + Plan A manifest."""
    raw = data_dir / "corpora" / corpus_id / "raw"
    raw.mkdir(parents=True)
    (raw / "queries.jsonl").write_text(
        json.dumps(
            {
                "query_id": "q0",
                "question": "q?",
                "answer": "a",
                "answer_aliases": [],
                "gold": {"type": "passage", "passage_ids": ["p0"]},
            }
        )
        + "\n"
    )
    (raw / "slice-full.json").write_text(json.dumps(["q0"]))
    (raw / "slice-smoke.json").write_text(json.dumps(["q0"]))
    (data_dir / "corpora" / corpus_id / "manifest.json").write_text(
        json.dumps(
            {
                "corpus_id": corpus_id,
                "dataset": {"name": "nq", "hf_id": "x", "split": "dev", "revision": "r"},
                "index_hashes": {"dense_contextual": "c", "dense_isolated": "i", "sparse": "s"},
                "n_queries": 1,
            }
        )
    )


def test_unknown_preset_rejected_with_valid_list(capsys) -> None:
    rc = main(["eval", "--corpus", "c1", "--presets", "bm25-only,nope", "--yes"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "nope" in err and "bm25-only" in err


def test_missing_keys_produce_named_env_var_errors(monkeypatch, capsys) -> None:
    for key in KEYS:
        monkeypatch.delenv(key, raising=False)
    rc = main(["eval", "--corpus", "c1", "--presets", "rerank", "--yes"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "ANTHROPIC_API_KEY" in err
    assert "VOYAGE_API_KEY" in err
    assert "COHERE_API_KEY" in err


def test_bm25_only_needs_no_voyage_or_cohere_key(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    # R6: --data-dir defaults to RAGRECEIPTS_DATA_DIR (hermetic here)
    monkeypatch.setenv("RAGRECEIPTS_DATA_DIR", str(tmp_path))
    # passes key validation, then fails on the missing corpus - proving the
    # voyage/cohere keys were not demanded for a sparse-only run
    with pytest.raises(FileNotFoundError):
        main(["eval", "--corpus", "missing-corpus", "--presets", "bm25-only", "--yes"])


def test_data_dir_default_honors_env_var(monkeypatch, capsys, tmp_path) -> None:
    # R6: data dir resolution everywhere is RAGRECEIPTS_DATA_DIR env var,
    # default ../data relative to api/ - promote shares the same default.
    monkeypatch.setenv("RAGRECEIPTS_DATA_DIR", str(tmp_path))
    rc = main(["receipts", "promote", "ghost"])
    assert rc == 2
    assert str(tmp_path) in capsys.readouterr().err


def test_confirm_gate_aborts_before_any_spend(monkeypatch, capsys, tmp_path) -> None:
    for key in KEYS:
        monkeypatch.setenv(key, "k")
    write_min_corpus(tmp_path)
    monkeypatch.setattr("builtins.input", lambda prompt: "n")
    rc = main(
        [
            "eval",
            "--corpus",
            "c1",
            "--slice",
            "smoke",
            "--presets",
            "rerank",
            "--data-dir",
            str(tmp_path),
        ]
    )
    assert rc == 1
    out = capsys.readouterr().out
    assert "Estimated cost" in out
    assert not (tmp_path / "receipts-local").exists()  # nothing ran, nothing spent


def test_build_core_real_composes_offline_with_fakes(tmp_path, monkeypatch) -> None:
    """Offline construction test for the R9-pinned composition root
    cli._build_core_real(config, corpus_id, data_dir): Plan A's real
    SparseRetriever.load / DenseRetriever / RerankStage assembled with fakes
    monkeypatched into the module-level factory seams - zero keys, zero
    network (bm25s builds locally; Qdrant runs in :memory: mode)."""
    corpus_dir = tmp_path / "corpora" / "c1"
    corpus_dir.mkdir(parents=True)
    chunks = [
        Chunk(
            chunk_id="d1:0",
            corpus_id="c1",
            doc_id="d1",
            passage_id="d1",
            text="alpha bravo charlie",
            position=0,
            start_token=0,
            end_token=3,
        )
    ]
    write_chunks(corpus_dir / "chunks.jsonl", chunks)
    build_sparse_index(chunks, corpus_dir / "sparse")
    monkeypatch.setattr(cli, "build_embed_transport", lambda: FakeEmbed())
    monkeypatch.setattr(cli, "build_qdrant", lambda data_dir: QdrantClient(":memory:"))
    monkeypatch.setattr(cli, "build_rerank_transport", lambda: FakeRerank())
    for preset in ("bm25-only", "dense-rrf", "contextual", "rerank"):
        core = cli._build_core_real(PRESETS[preset], "c1", tmp_path)
        assert isinstance(core, RetrievalCore)


def test_build_core_real_builds_graph_for_graph_preset(tmp_path, monkeypatch) -> None:
    """The serving/eval composition root wires GraphRetriever for graph presets.

    Offline: a real byte-reproducible graph artifact built by Plan E's build_graph_index
    (FakeOpenIE/FakeEmbed) lives at {corpus_dir}/graph/; chunks.jsonl holds the SAME
    chunks so the retriever's chunk-by-id map resolves. The graph 'preset' (graph-only,
    FORCE_S1) needs no sparse/dense index. Asserts the core's graph retriever is set
    (not None) and construction does not raise the 'no graph retriever' error."""
    from tests.graph_fixtures import fixture_chunks, write_graph_artifact

    corpus_dir = tmp_path / "corpora" / "graph-harness"
    corpus_dir.mkdir(parents=True)
    write_chunks(corpus_dir / "chunks.jsonl", fixture_chunks())
    write_graph_artifact(corpus_dir)  # builds {corpus_dir}/graph/ with the real builder
    monkeypatch.setattr(cli, "build_embed_transport", lambda: FakeEmbed())
    monkeypatch.setattr(cli, "_make_claude", lambda: object())  # recognition='llm' ctor only

    core = cli._build_core_real(PRESETS["graph"], "graph-harness", tmp_path)
    assert isinstance(core, RetrievalCore)
    assert core._graph is not None  # graph retriever wired (was the unwired bug)


def test_build_core_real_builds_graph_rrf_with_sparse_dense_and_graph(
    tmp_path, monkeypatch
) -> None:
    """graph-rrf fuses bm25 + dense + graph: all three retrievers must be present."""
    from tests.graph_fixtures import fixture_chunks, write_graph_artifact

    corpus_dir = tmp_path / "corpora" / "graph-harness"
    corpus_dir.mkdir(parents=True)
    chunks = fixture_chunks()
    write_chunks(corpus_dir / "chunks.jsonl", chunks)
    build_sparse_index(chunks, corpus_dir / "sparse")
    write_graph_artifact(corpus_dir)
    monkeypatch.setattr(cli, "build_embed_transport", lambda: FakeEmbed())
    monkeypatch.setattr(cli, "build_qdrant", lambda data_dir: QdrantClient(":memory:"))
    monkeypatch.setattr(cli, "_make_claude", lambda: object())

    core = cli._build_core_real(PRESETS["graph-rrf"], "graph-harness", tmp_path)
    assert isinstance(core, RetrievalCore)
    assert core._graph is not None and core._sparse is not None and core._dense is not None


def test_build_core_real_raises_when_graph_preset_lacks_artifact(tmp_path, monkeypatch) -> None:
    """Honest failure: a graph preset on a corpus ingested WITHOUT a graph artifact
    surfaces RetrievalCore's clear 'no graph retriever' error — never a silent disable."""
    from tests.graph_fixtures import fixture_chunks

    corpus_dir = tmp_path / "corpora" / "no-graph"
    corpus_dir.mkdir(parents=True)
    write_chunks(corpus_dir / "chunks.jsonl", fixture_chunks())  # no graph/ dir written
    monkeypatch.setattr(cli, "build_embed_transport", lambda: FakeEmbed())
    monkeypatch.setattr(cli, "_make_claude", lambda: object())

    with pytest.raises(ValueError, match="no graph retriever"):
        cli._build_core_real(PRESETS["graph"], "no-graph", tmp_path)


def test_build_graph_route_core_wraps_graph_in_retrieve_shaped_core(tmp_path, monkeypatch) -> None:
    """The agent-route helper returns a `.retrieve`-shaped graph-only RetrievalCore
    (the SupportsRetrieve object the router's graph route consumes), or None when the
    corpus has no graph artifact."""
    from tests.graph_fixtures import fixture_chunks, write_graph_artifact

    corpus_dir = tmp_path / "corpora" / "graph-harness"
    corpus_dir.mkdir(parents=True)
    chunks = fixture_chunks()
    write_chunks(corpus_dir / "chunks.jsonl", chunks)
    write_graph_artifact(corpus_dir)
    monkeypatch.setattr(cli, "build_embed_transport", lambda: FakeEmbed())
    monkeypatch.setattr(cli, "_make_claude", lambda: object())

    # router-on's recognition is 'llm' by default; the wrapping core is graph-only.
    route_core = cli.build_graph_route_core(corpus_dir, PRESETS["router-on"], chunks)
    assert isinstance(route_core, RetrievalCore)
    assert hasattr(route_core, "retrieve")  # the protocol the agent route needs
    assert route_core._graph is not None
    assert route_core._sparse is None and route_core._dense is None

    # absent artifact -> None (route falls back to s1, honest)
    empty_dir = tmp_path / "corpora" / "empty"
    empty_dir.mkdir(parents=True)
    assert cli.build_graph_route_core(empty_dir, PRESETS["router-on"], chunks) is None


def test_promote_strips_text_and_writes_to_receipts_dir(tmp_path, capsys) -> None:
    data_dir = tmp_path / "data"
    receipts_dir = tmp_path / "receipts"
    doc = make_run_doc(
        run_id="r1", corpus_id="c1", slice_name="smoke", receipts=[minimal_receipt()], skipped=[]
    )
    write_run_doc(doc, data_dir)
    rc = main(
        [
            "receipts",
            "promote",
            "r1",
            "--data-dir",
            str(data_dir),
            "--receipts-dir",
            str(receipts_dir),
        ]
    )
    assert rc == 0
    committed = json.loads((receipts_dir / "r1.json").read_text())
    pq = committed["receipts"][0]["receipt"]["per_query"][0]
    assert "answer" not in pq  # IDs + metrics only
    assert pq["retrieved_chunk_ids"] == ["d:0"]
    assert pq["flags"]["f1"] == 1.0


def test_promote_missing_run_is_actionable(tmp_path, capsys) -> None:
    rc = main(["receipts", "promote", "ghost", "--data-dir", str(tmp_path)])
    assert rc == 2
    assert "ghost" in capsys.readouterr().err


def test_eval_accepts_graph_recognition_flag(monkeypatch, tmp_path):
    """--graph-recognition is parsed and threaded to the runner (recognition sweep)."""
    captured: dict = {}

    class _Runner:
        def __init__(self, **kw):
            pass

        def run(self, **kw):
            captured.update(kw)
            return {"skipped": [], "receipts": []}

    monkeypatch.setattr(cli, "AblationRunner", _Runner)
    monkeypatch.setattr(cli, "_make_claude", lambda: object())
    monkeypatch.setattr(cli, "_build_core_real", lambda *a, **k: object())
    monkeypatch.setattr(cli, "estimate_run_cost", lambda *a, **k: 0.01)
    monkeypatch.setattr(cli, "load_queries", lambda *a, **k: [])
    monkeypatch.setattr(cli, "slice_query_ids", lambda *a, **k: [])
    monkeypatch.setattr(cli, "slice_queries", lambda *a, **k: [])
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    rc = cli.main(
        [
            "eval",
            "--corpus",
            "graph-harness",
            "--slice",
            "smoke",
            "--presets",
            "graph",
            "--graph-recognition",
            "embedding",
            "--yes",
        ]
    )
    assert rc == 0
    assert captured["graph_recognition"] == "embedding"
