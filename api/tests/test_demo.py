"""Tests for DEMO_MODE config, rate/budget ledger."""

from __future__ import annotations

import json
import json as _json
import shutil
import sqlite3

import numpy as np
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from ragreceipts.server.app import create_app
from ragreceipts.server.demo import DemoConfig, DemoLedger
from ragreceipts.server.deps import AppDeps, AppPaths
from ragreceipts.server.jobs import JobRunner

# ── DemoConfig ────────────────────────────────────────────────────────────────


def test_demo_config_from_env_returns_none_when_unset(monkeypatch):
    monkeypatch.delenv("DEMO_MODE", raising=False)
    assert DemoConfig.from_env() is None


def test_demo_config_from_env_returns_none_when_zero(monkeypatch):
    monkeypatch.setenv("DEMO_MODE", "0")
    assert DemoConfig.from_env() is None


def test_demo_config_from_env_returns_config_when_one(monkeypatch):
    monkeypatch.setenv("DEMO_MODE", "1")
    for k in (
        "DEMO_DAILY_BUDGET_USD",
        "DEMO_RATE_PER_MIN",
        "DEMO_RATE_PER_DAY",
        "DEMO_S2_TOKEN_CEILING",
        "DEMO_CORPUS_ID",
    ):
        monkeypatch.delenv(k, raising=False)
    config = DemoConfig.from_env()
    assert config is not None
    assert config.daily_budget_usd == 2.0
    assert config.rate_per_min == 5
    assert config.rate_per_day == 20
    assert config.s2_token_ceiling == 20_000
    assert config.demo_corpus_id == "demo"


def test_demo_config_from_env_reads_overrides(monkeypatch):
    monkeypatch.setenv("DEMO_MODE", "1")
    monkeypatch.setenv("DEMO_DAILY_BUDGET_USD", "5.5")
    monkeypatch.setenv("DEMO_RATE_PER_MIN", "3")
    monkeypatch.setenv("DEMO_CORPUS_ID", "my-demo")
    config = DemoConfig.from_env()
    assert config is not None
    assert config.daily_budget_usd == 5.5
    assert config.rate_per_min == 3
    assert config.demo_corpus_id == "my-demo"


# ── DemoLedger helpers ────────────────────────────────────────────────────────


def _make_ledger(tmp_path, *, rate_per_min=5, rate_per_day=20, daily_budget_usd=2.0):
    config = DemoConfig(
        daily_budget_usd=daily_budget_usd,
        rate_per_min=rate_per_min,
        rate_per_day=rate_per_day,
        s2_token_ceiling=20_000,
        demo_corpus_id="demo",
    )
    return DemoLedger(config, tmp_path / "demo.sqlite")


# ── DemoLedger: init + record ─────────────────────────────────────────────────


def test_demo_ledger_init_creates_table(tmp_path):
    _make_ledger(tmp_path)
    conn = sqlite3.connect(tmp_path / "demo.sqlite")
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "demo_query_log" in tables


def test_demo_ledger_record_stores_row(tmp_path):
    ledger = _make_ledger(tmp_path)
    ledger.record("1.2.3.4", 0.05)
    conn = sqlite3.connect(tmp_path / "demo.sqlite")
    rows = conn.execute("SELECT ip, usd_actual FROM demo_query_log").fetchall()
    assert len(rows) == 1
    assert rows[0][0] == "1.2.3.4"
    assert abs(rows[0][1] - 0.05) < 1e-9


# ── DemoLedger: check_rate ────────────────────────────────────────────────────


def test_demo_ledger_check_rate_allows_under_per_min(tmp_path):
    ledger = _make_ledger(tmp_path, rate_per_min=3)
    ledger.record("1.2.3.4", 0.01)
    ledger.record("1.2.3.4", 0.01)
    ledger.check_rate("1.2.3.4")  # 2 < 3 → should not raise


def test_demo_ledger_check_rate_raises_at_per_min_limit(tmp_path):
    ledger = _make_ledger(tmp_path, rate_per_min=2)
    ledger.record("1.2.3.4", 0.01)
    ledger.record("1.2.3.4", 0.01)
    with pytest.raises(HTTPException) as exc:
        ledger.check_rate("1.2.3.4")
    assert exc.value.status_code == 429
    assert exc.value.detail["reason"] == "rate"
    assert exc.value.detail["retry_after_s"] == 60


def test_demo_ledger_check_rate_different_ips_are_isolated(tmp_path):
    ledger = _make_ledger(tmp_path, rate_per_min=1)
    ledger.record("1.2.3.4", 0.01)
    ledger.check_rate("9.9.9.9")  # different IP — must not raise


def test_demo_ledger_check_rate_raises_at_per_day_limit(tmp_path):
    ledger = _make_ledger(tmp_path, rate_per_min=1000, rate_per_day=2)
    ledger.record("1.2.3.4", 0.01)
    ledger.record("1.2.3.4", 0.01)
    with pytest.raises(HTTPException) as exc:
        ledger.check_rate("1.2.3.4")
    assert exc.value.status_code == 429
    assert exc.value.detail["reason"] == "rate"
    assert exc.value.detail["retry_after_s"] == 86400


# ── DemoLedger: check_budget ──────────────────────────────────────────────────


def test_demo_ledger_check_budget_passes_when_under(tmp_path):
    ledger = _make_ledger(tmp_path, daily_budget_usd=1.0)
    ledger.record("1.2.3.4", 0.50)
    ledger.check_budget(0.49)  # 0.50 + 0.49 = 0.99 < 1.0 → OK


def test_demo_ledger_check_budget_raises_when_over(tmp_path):
    ledger = _make_ledger(tmp_path, daily_budget_usd=0.01)
    ledger.record("1.2.3.4", 0.01)
    with pytest.raises(HTTPException) as exc:
        ledger.check_budget(0.001)
    assert exc.value.status_code == 429
    assert exc.value.detail["reason"] == "budget"


def test_demo_ledger_check_budget_passes_when_no_spend(tmp_path):
    ledger = _make_ledger(tmp_path, daily_budget_usd=2.0)
    ledger.check_budget(0.02)  # no prior spend → 0.0 + 0.02 < 2.0 → OK


# ── AppPaths + AppDeps wiring ─────────────────────────────────────────────────


def test_app_paths_from_env_has_demo_dirs(tmp_path, monkeypatch):
    monkeypatch.setenv("RAGRECEIPTS_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("RAGRECEIPTS_RECEIPTS_DIR", str(tmp_path / "receipts"))
    monkeypatch.setenv("RAGRECEIPTS_DEMO_CORPUS_DIR", str(tmp_path / "demo" / "corpus"))
    monkeypatch.setenv("RAGRECEIPTS_DEMO_EXAMPLES_DIR", str(tmp_path / "demo" / "examples"))
    from ragreceipts.server.deps import AppPaths

    paths = AppPaths.from_env()
    assert paths.demo_corpus_dir == (tmp_path / "demo" / "corpus").resolve()
    assert paths.demo_examples_dir == (tmp_path / "demo" / "examples").resolve()
    assert paths.demo_db == paths.data_dir / "demo.sqlite"


def test_build_deps_wires_demo_ledger_when_demo_mode_set(tmp_path, monkeypatch):
    monkeypatch.setenv("DEMO_MODE", "1")
    monkeypatch.setenv("RAGRECEIPTS_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("RAGRECEIPTS_RECEIPTS_DIR", str(tmp_path / "receipts"))
    monkeypatch.setenv("RAGRECEIPTS_DEMO_CORPUS_DIR", str(tmp_path / "demo" / "corpus"))
    monkeypatch.setenv("RAGRECEIPTS_DEMO_EXAMPLES_DIR", str(tmp_path / "demo" / "examples"))
    monkeypatch.delenv("QDRANT_URL", raising=False)
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("TESTING", raising=False)
    from ragreceipts.server.deps import build_deps

    deps = build_deps()
    assert deps.demo_ledger is not None
    assert deps.demo_ledger.config.demo_corpus_id == "demo"


def test_build_deps_demo_ledger_is_none_when_demo_mode_unset(tmp_path, monkeypatch):
    monkeypatch.delenv("DEMO_MODE", raising=False)
    monkeypatch.setenv("RAGRECEIPTS_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("RAGRECEIPTS_RECEIPTS_DIR", str(tmp_path / "receipts"))
    monkeypatch.setenv("RAGRECEIPTS_DEMO_CORPUS_DIR", str(tmp_path / "demo" / "corpus"))
    monkeypatch.setenv("RAGRECEIPTS_DEMO_EXAMPLES_DIR", str(tmp_path / "demo" / "examples"))
    monkeypatch.delenv("QDRANT_URL", raising=False)
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("TESTING", raising=False)
    from ragreceipts.server.deps import build_deps

    deps = build_deps()
    assert deps.demo_ledger is None


# ── Endpoint 403 guards ────────────────────────────────────────────────────────


def _make_test_app(tmp_path, *, with_demo: bool = True):
    """Return a FastAPI app with minimal fake AppDeps for endpoint tests."""
    paths = AppPaths(
        data_dir=tmp_path / "data",
        receipts_committed_dir=tmp_path / "receipts",
        demo_corpus_dir=tmp_path / "demo" / "corpus",
        demo_examples_dir=tmp_path / "demo" / "examples",
    )
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)

    if with_demo:
        demo_cfg = DemoConfig(
            daily_budget_usd=2.0,
            rate_per_min=5,
            rate_per_day=20,
            s2_token_ceiling=20_000,
            demo_corpus_id="demo",
        )
        demo_ledger = DemoLedger(demo_cfg, paths.demo_db)
    else:
        demo_ledger = None

    from tests.fakes import InMemoryTraceStore

    deps = AppDeps(
        paths=paths,
        vendors=[],
        qdrant=None,
        trace_store=InMemoryTraceStore(),
        job_runner=JobRunner(paths.jobs_db),
        query_runner=None,
        eval_runner=None,
        ingest_sink=None,
        testing_mode=True,
        demo_ledger=demo_ledger,
    )
    return create_app(deps_factory=lambda: deps)


def test_ingest_returns_403_in_demo_mode(tmp_path):
    app = _make_test_app(tmp_path, with_demo=True)
    with TestClient(app) as client:
        response = client.post(
            "/corpora/ingest",
            data={"corpus_id": "test"},
            files={"files": ("a.txt", b"hello", "text/plain")},
        )
    assert response.status_code == 403
    assert "read-only" in response.json()["detail"]


def test_ingest_works_normally_without_demo_mode(tmp_path):
    app = _make_test_app(tmp_path, with_demo=False)
    with TestClient(app) as client:
        response = client.post(
            "/corpora/ingest",
            data={"corpus_id": "test"},
            files={"files": ("a.txt", b"hello", "text/plain")},
        )
    assert response.status_code != 403  # 503 because ingest_sink is None


def test_eval_runs_post_returns_403_in_demo_mode(tmp_path):
    app = _make_test_app(tmp_path, with_demo=True)
    with TestClient(app) as client:
        response = client.post(
            "/eval/runs",
            json={
                "corpus_id": "x",
                "preset": "bm25-only",
                "slice": "smoke",
                "spend_cap_usd": 1.0,
                "confirm": False,
            },
        )
    assert response.status_code == 403
    assert "read-only" in response.json()["detail"]


def test_eval_runs_post_works_without_demo_mode(tmp_path):
    app = _make_test_app(tmp_path, with_demo=False)
    with TestClient(app) as client:
        response = client.post(
            "/eval/runs",
            json={
                "corpus_id": "x",
                "preset": "bm25-only",
                "slice": "smoke",
                "spend_cap_usd": 1.0,
                "confirm": False,
            },
        )
    assert response.status_code != 403  # 503 because eval_runner is None


# ── /query guardrails ─────────────────────────────────────────────────────────


def test_query_corpus_allow_list_returns_403(tmp_path):
    app = _make_test_app(tmp_path, with_demo=True)
    with TestClient(app) as client:
        response = client.post(
            "/query",
            json={"query": "hello", "corpus_id": "not-demo", "preset": "bm25-only"},
        )
    assert response.status_code == 403
    assert "demo corpus" in response.json()["detail"]


def test_query_correct_corpus_passes_allow_list(tmp_path):
    app = _make_test_app(tmp_path, with_demo=True)
    with TestClient(app) as client:
        response = client.post(
            "/query",
            json={"query": "hello", "corpus_id": "demo", "preset": "bm25-only"},
        )
    assert response.status_code != 403  # 503 because query_runner is None


def test_query_rate_limit_raises_429(tmp_path):
    paths = AppPaths(
        data_dir=tmp_path / "data",
        receipts_committed_dir=tmp_path / "receipts",
        demo_corpus_dir=tmp_path / "demo" / "corpus",
        demo_examples_dir=tmp_path / "demo" / "examples",
    )
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    config = DemoConfig(
        daily_budget_usd=99.0,
        rate_per_min=0,
        rate_per_day=100,
        s2_token_ceiling=20_000,
        demo_corpus_id="demo",
    )
    ledger = DemoLedger(config, paths.demo_db)
    from tests.fakes import InMemoryTraceStore

    deps = AppDeps(
        paths=paths,
        vendors=[],
        qdrant=None,
        trace_store=InMemoryTraceStore(),
        job_runner=JobRunner(paths.jobs_db),
        query_runner=None,
        eval_runner=None,
        ingest_sink=None,
        testing_mode=True,
        demo_ledger=ledger,
    )
    app = create_app(deps_factory=lambda: deps)
    with TestClient(app) as client:
        response = client.post(
            "/query",
            json={"query": "hello", "corpus_id": "demo", "preset": "bm25-only"},
        )
    assert response.status_code == 429
    assert response.json()["detail"]["reason"] == "rate"


def test_query_budget_exhausted_raises_429(tmp_path):
    paths = AppPaths(
        data_dir=tmp_path / "data",
        receipts_committed_dir=tmp_path / "receipts",
        demo_corpus_dir=tmp_path / "demo" / "corpus",
        demo_examples_dir=tmp_path / "demo" / "examples",
    )
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    config = DemoConfig(
        daily_budget_usd=0.0,
        rate_per_min=100,
        rate_per_day=100,
        s2_token_ceiling=20_000,
        demo_corpus_id="demo",
    )
    ledger = DemoLedger(config, paths.demo_db)
    from tests.fakes import InMemoryTraceStore

    deps = AppDeps(
        paths=paths,
        vendors=[],
        qdrant=None,
        trace_store=InMemoryTraceStore(),
        job_runner=JobRunner(paths.jobs_db),
        query_runner=None,
        eval_runner=None,
        ingest_sink=None,
        testing_mode=True,
        demo_ledger=ledger,
    )
    app = create_app(deps_factory=lambda: deps)
    with TestClient(app) as client:
        response = client.post(
            "/query",
            json={"query": "hello", "corpus_id": "demo", "preset": "bm25-only"},
        )
    assert response.status_code == 429
    assert response.json()["detail"]["reason"] == "budget"


# ── GET /demo/examples ────────────────────────────────────────────────────────


def test_demo_examples_returns_empty_list_when_dir_absent(tmp_path):
    app = _make_test_app(tmp_path, with_demo=True)
    with TestClient(app) as client:
        response = client.get("/demo/examples")
    assert response.status_code == 200
    assert response.json()["examples"] == []


def test_demo_examples_returns_empty_list_when_dir_is_empty(tmp_path):
    (tmp_path / "demo" / "examples").mkdir(parents=True)
    app = _make_test_app(tmp_path, with_demo=True)
    with TestClient(app) as client:
        response = client.get("/demo/examples")
    assert response.status_code == 200
    assert response.json()["examples"] == []


def test_demo_examples_returns_examples_from_json_files(tmp_path):
    ex_dir = tmp_path / "demo" / "examples"
    ex_dir.mkdir(parents=True)
    example = {
        "label": "s1",
        "query": "Who walked on the Moon?",
        "answer": "Neil Armstrong",
        "route": "s1",
        "citations": [
            {
                "n": 1,
                "chunk_id": "c1",
                "passage_id": "p1",
                "text": "Neil Armstrong walked on the Moon.",
                "score": 0.9,
            }
        ],
        "trace_events": [],
    }
    (ex_dir / "example_s1.json").write_text(_json.dumps(example))
    app = _make_test_app(tmp_path, with_demo=True)
    with TestClient(app) as client:
        response = client.get("/demo/examples")
    assert response.status_code == 200
    data = response.json()
    assert len(data["examples"]) == 1
    assert data["examples"][0]["label"] == "s1"
    assert data["examples"][0]["query"] == "Who walked on the Moon?"


def test_demo_examples_works_without_demo_mode(tmp_path):
    ex_dir = tmp_path / "demo" / "examples"
    ex_dir.mkdir(parents=True)
    app = _make_test_app(tmp_path, with_demo=False)
    with TestClient(app) as client:
        response = client.get("/demo/examples")
    assert response.status_code == 200


# ── seed_demo_qdrant unit tests ───────────────────────────────────────────────


def _write_fake_corpus(corpus_dir, n_chunks: int = 3, embed_dim: int = 4) -> None:
    corpus_dir.mkdir(parents=True, exist_ok=True)
    chunks = [
        {"chunk_id": f"c{i}", "passage_id": f"p{i}", "text": f"Text {i}"} for i in range(n_chunks)
    ]
    import json as _json2

    (corpus_dir / "chunks.jsonl").write_text("\n".join(_json2.dumps(c) for c in chunks))
    vecs = np.random.rand(n_chunks, embed_dim).astype("float32")
    np.savez(corpus_dir / "dense_vectors.npz", contextual=vecs, isolated=vecs)


def _make_demo_config(corpus_id: str = "demo") -> DemoConfig:
    return DemoConfig(
        daily_budget_usd=2.0,
        rate_per_min=5,
        rate_per_day=20,
        s2_token_ceiling=20_000,
        demo_corpus_id=corpus_id,
    )


def test_seed_demo_qdrant_noop_when_vectors_absent(tmp_path):
    from ragreceipts.server.demo import seed_demo_qdrant

    qdrant = QdrantClient(":memory:")
    seed_demo_qdrant(qdrant, tmp_path / "corpus", _make_demo_config())
    with pytest.raises(Exception):
        qdrant.get_collection("demo")


def test_seed_demo_qdrant_creates_collection(tmp_path):
    from ragreceipts.server.demo import seed_demo_qdrant

    corpus_dir = tmp_path / "corpus"
    _write_fake_corpus(corpus_dir, n_chunks=3, embed_dim=4)
    qdrant = QdrantClient(":memory:")
    seed_demo_qdrant(qdrant, corpus_dir, _make_demo_config())
    info = qdrant.get_collection("demo")
    assert info.points_count == 3


def test_seed_demo_qdrant_is_idempotent(tmp_path):
    from ragreceipts.server.demo import seed_demo_qdrant

    corpus_dir = tmp_path / "corpus"
    _write_fake_corpus(corpus_dir, n_chunks=3, embed_dim=4)
    qdrant = QdrantClient(":memory:")
    seed_demo_qdrant(qdrant, corpus_dir, _make_demo_config())
    seed_demo_qdrant(qdrant, corpus_dir, _make_demo_config())
    info = qdrant.get_collection("demo")
    assert info.points_count == 3  # still 3, not doubled


# ── Corpus structure validation ───────────────────────────────────────────────


def test_demo_corpus_docs_jsonl_has_12_docs():
    """Validate demo/corpus/docs.jsonl exists and has exactly 12 well-formed entries."""
    import json as _json
    from pathlib import Path

    docs_path = Path(__file__).resolve().parents[2] / "demo" / "corpus" / "docs.jsonl"
    assert docs_path.exists(), f"docs.jsonl not found at {docs_path}"
    lines = [ln for ln in docs_path.read_text().splitlines() if ln.strip()]
    assert len(lines) == 12, f"Expected 12 docs, got {len(lines)}"
    for i, line in enumerate(lines):
        doc = _json.loads(line)
        for key in ("id", "title", "text"):
            assert key in doc, f"doc {i} missing key {key!r}"
        assert len(doc["text"]) >= 100, f"doc {i} text too short"


# ── materialize_demo_corpus ───────────────────────────────────────────────────


def _write_fake_committed_corpus(src_dir):
    """Write a minimal committed demo/corpus/ tree (manifest + chunks + sparse + graph)."""
    src_dir.mkdir(parents=True, exist_ok=True)
    (src_dir / "manifest.json").write_text(json.dumps({"corpus_id": "demo"}))
    (src_dir / "chunks.jsonl").write_text(json.dumps({"chunk_id": "c0"}) + "\n")
    (src_dir / "sparse").mkdir(exist_ok=True)
    (src_dir / "sparse" / "index.bin").write_text("bm25")
    (src_dir / "graph").mkdir(exist_ok=True)
    (src_dir / "graph" / "nodes.jsonl").write_text(json.dumps({"id": "n0"}) + "\n")
    # dense_vectors.npz exists in the committed tree but must NOT be copied to the corpora dir
    (src_dir / "dense_vectors.npz").write_text("not-really-npz")


def test_materialize_demo_corpus_copies_artifacts(tmp_path):
    from ragreceipts.server.demo import materialize_demo_corpus

    src = tmp_path / "demo" / "corpus"
    _write_fake_committed_corpus(src)
    corpora_dir = tmp_path / "data" / "corpora"
    materialize_demo_corpus(src, corpora_dir, "demo")
    target = corpora_dir / "demo"
    assert (target / "manifest.json").exists()
    assert (target / "chunks.jsonl").exists()
    assert (target / "sparse" / "index.bin").exists()
    assert (target / "graph" / "nodes.jsonl").exists()


def test_materialize_demo_corpus_skips_dense_vectors(tmp_path):
    from ragreceipts.server.demo import materialize_demo_corpus

    src = tmp_path / "demo" / "corpus"
    _write_fake_committed_corpus(src)
    corpora_dir = tmp_path / "data" / "corpora"
    materialize_demo_corpus(src, corpora_dir, "demo")
    # dense_vectors.npz is for Qdrant seeding only — must not land in the corpora dir
    assert not (corpora_dir / "demo" / "dense_vectors.npz").exists()


def test_materialize_demo_corpus_is_idempotent(tmp_path):
    from ragreceipts.server.demo import materialize_demo_corpus

    src = tmp_path / "demo" / "corpus"
    _write_fake_committed_corpus(src)
    corpora_dir = tmp_path / "data" / "corpora"
    materialize_demo_corpus(src, corpora_dir, "demo")
    # mutate the target to prove a second call does NOT overwrite (sentinel = manifest present)
    (corpora_dir / "demo" / "manifest.json").write_text(
        json.dumps({"corpus_id": "demo", "touched": True})
    )
    materialize_demo_corpus(src, corpora_dir, "demo")
    data = json.loads((corpora_dir / "demo" / "manifest.json").read_text())
    assert data.get("touched") is True  # not clobbered


def test_materialize_demo_corpus_noop_when_src_absent(tmp_path):
    from ragreceipts.server.demo import materialize_demo_corpus

    src = tmp_path / "demo" / "corpus"  # never created (pre-bootstrap)
    corpora_dir = tmp_path / "data" / "corpora"
    materialize_demo_corpus(src, corpora_dir, "demo")  # must not raise
    assert not (corpora_dir / "demo").exists()


def test_materialize_demo_corpus_crash_recovery_cleans_stale_subdir(tmp_path):
    """Crash-recovery: an interrupted prior run left a partial target subdir but no
    sentinel manifest. The re-materialize must produce a CLEAN copy — stale files that
    are not in the committed source must not survive (copytree merge would keep them)."""
    from ragreceipts.server.demo import materialize_demo_corpus

    src = tmp_path / "demo" / "corpus"
    _write_fake_committed_corpus(src)
    corpora_dir = tmp_path / "data" / "corpora"
    materialize_demo_corpus(src, corpora_dir, "demo")

    # Simulate an interrupted run: drop the sentinel manifest (so the next call
    # re-materializes) and leave an orphaned file the committed source does not contain.
    (corpora_dir / "demo" / "manifest.json").unlink()
    (corpora_dir / "demo" / "sparse" / "orphaned.bin").write_text("stale")

    materialize_demo_corpus(src, corpora_dir, "demo")

    # The orphaned file is gone (clean re-copy) and the real artifacts are present.
    assert not (corpora_dir / "demo" / "sparse" / "orphaned.bin").exists()
    assert (corpora_dir / "demo" / "sparse" / "index.bin").exists()
    assert (corpora_dir / "demo" / "manifest.json").exists()


def test_materialize_demo_corpus_handles_missing_optional_artifacts(tmp_path):
    """A source without a graph/ (corpus ingested without a graph) materializes cleanly:
    the present artifacts copy across and the absent one is simply not created."""
    from ragreceipts.server.demo import materialize_demo_corpus

    src = tmp_path / "demo" / "corpus"
    _write_fake_committed_corpus(src)
    shutil.rmtree(src / "graph")  # no graph artifact in the committed source
    corpora_dir = tmp_path / "data" / "corpora"
    materialize_demo_corpus(src, corpora_dir, "demo")
    target = corpora_dir / "demo"
    assert (target / "manifest.json").exists()
    assert (target / "chunks.jsonl").exists()
    assert (target / "sparse" / "index.bin").exists()
    assert not (target / "graph").exists()  # absent in source -> absent in target
