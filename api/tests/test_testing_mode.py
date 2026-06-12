"""TESTING=1 end-to-end (in-process): the exact stack Playwright runs against."""

from fastapi.testclient import TestClient

from ragreceipts.server.app import create_app
from ragreceipts.server.deps import build_deps


def make_client(tmp_path, monkeypatch) -> TestClient:
    monkeypatch.setenv("TESTING", "1")
    monkeypatch.setenv("RAGRECEIPTS_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("RAGRECEIPTS_RECEIPTS_DIR", str(tmp_path / "receipts"))
    return TestClient(create_app(deps_factory=build_deps))


def test_testing_query_round_trip(tmp_path, monkeypatch):
    with make_client(tmp_path, monkeypatch) as client:
        assert client.get("/health").json()["testing_mode"] is True
        r = client.post(
            "/query",
            json={
                "query": "What is the capital of France?",
                "corpus_id": "fixture-corpus",
                "preset": "rerank",
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert "Paris" in body["answer"]
        assert body["route"] == "s1"
        assert body["citations"][0]["passage_id"] == "geo-001"
        nodes = [e["node"] for e in client.get(f"/traces/{body['trace_id']}").json()["events"]]
        assert nodes == ["route", "s1_retrieve", "s1_answer"]


def test_testing_degraded_path(tmp_path, monkeypatch):
    with make_client(tmp_path, monkeypatch) as client:
        body = client.post(
            "/query",
            json={
                "query": "degrade: capital of France?",
                "corpus_id": "fixture-corpus",
                "preset": "rerank",
            },
        ).json()
        assert body["degraded"] == ["rerank-skipped"]


def test_testing_ingest_blocked_in_demo_mode(tmp_path, monkeypatch):
    """TESTING=1 activates demo_ledger, which blocks ingest with 403."""
    with make_client(tmp_path, monkeypatch) as client:
        r = client.post(
            "/corpora/ingest",
            data={"corpus_id": "uploaded-docs"},
            files=[("files", ("note.txt", b"some text", "text/plain"))],
        )
        assert r.status_code == 403, r.text
        assert "read-only" in r.json()["detail"]
