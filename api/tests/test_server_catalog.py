"""Read-only catalog endpoints: corpora, receipts (committed+local), eval runs."""

import json

from fastapi.testclient import TestClient

from ragreceipts.server.app import create_app
from tests.helpers_server import make_test_deps


def seed(deps):
    c = deps.paths.corpora_dir / "musique-dev-300"
    c.mkdir(parents=True)
    (c / "manifest.json").write_text(
        json.dumps(
            {
                "corpus_id": "musique-dev-300",
                "n_docs": 10,
                "n_chunks": 42,
            }
        )
    )
    deps.paths.receipts_committed_dir.mkdir(parents=True, exist_ok=True)
    (deps.paths.receipts_committed_dir / "headline.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "receipt": {"run_id": "r1", "preset": "rerank", "metrics": {"recall_at_5": 0.78}},
            }
        )
    )
    (deps.paths.receipts_committed_dir / "corrupt.json").write_text("{not json")
    (deps.paths.receipts_local_dir / "local.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "receipt": {
                    "run_id": "r2",
                    "preset": "bm25-only",
                    "metrics": {"recall_at_5": 0.55},
                },
            }
        )
    )


def test_corpora_lists_manifests(tmp_path):
    deps = make_test_deps(tmp_path)
    seed(deps)
    with TestClient(create_app(deps_factory=lambda: deps)) as client:
        body = client.get("/corpora").json()
    assert [c["corpus_id"] for c in body["corpora"]] == ["musique-dev-300"]
    assert body["corpora"][0]["manifest"]["n_chunks"] == 42


def test_receipts_merges_committed_and_local_and_discloses_corrupt_files(tmp_path):
    deps = make_test_deps(tmp_path)
    seed(deps)
    with TestClient(create_app(deps_factory=lambda: deps)) as client:
        body = client.get("/receipts").json()
    by_source = {(r["source"], r["receipt"]["run_id"]) for r in body["receipts"]}
    assert by_source == {("committed", "r1"), ("local", "r2")}
    assert len(body["errors"]) == 1 and "corrupt.json" in body["errors"][0]


def test_eval_runs_empty_initially(tmp_path):
    deps = make_test_deps(tmp_path)
    with TestClient(create_app(deps_factory=lambda: deps)) as client:
        assert client.get("/eval/runs").json() == {"runs": []}
