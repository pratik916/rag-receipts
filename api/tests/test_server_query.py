"""/query and /traces/{trace_id} against a stub QueryRunner (offline)."""

import json

from fastapi.testclient import TestClient

from ragreceipts.constants import ROUTER_MODEL
from ragreceipts.server.app import create_app
from ragreceipts.server.pipeline import Citation, QueryResult
from ragreceipts.traces.models import TraceEvent
from tests.helpers_server import make_test_deps


class StubQueryRunner:
    def __init__(self, trace_store) -> None:
        self._ts = trace_store

    def run(self, *, query: str, corpus_id: str, preset: str) -> QueryResult:
        trace_id = "t-123"
        self._ts.append(
            TraceEvent(
                trace_id=trace_id,
                seq=0,
                node="route",
                payload={"route": "s1"},
                model=ROUTER_MODEL,
                input_tokens=10,
                output_tokens=2,
                duration_ms=5.0,
            )
        )
        return QueryResult(
            answer="Paris [1].",
            abstained=False,
            route="s1",
            degraded=["rerank-skipped"],
            citations=[
                Citation(
                    n=1,
                    chunk_id="geo-001:0",
                    passage_id="geo-001",
                    text="Paris is the capital of France.",
                    score=0.91,
                )
            ],
            trace_id=trace_id,
        )


def make_app(tmp_path, *, with_runner: bool = True):
    deps = make_test_deps(tmp_path, configured=with_runner)
    if with_runner:
        deps.query_runner = StubQueryRunner(deps.trace_store)
    corpus = deps.paths.corpora_dir / "fixture-corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    (corpus / "manifest.json").write_text(json.dumps({"corpus_id": "fixture-corpus"}))
    return create_app(deps_factory=lambda: deps)


def test_query_round_trip_returns_answer_trace_and_degraded_flags(tmp_path):
    with TestClient(make_app(tmp_path)) as client:
        r = client.post(
            "/query",
            json={
                "query": "capital of France?",
                "corpus_id": "fixture-corpus",
                "preset": "rerank",
            },
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["answer"] == "Paris [1]."
        assert body["route"] == "s1"
        assert body["degraded"] == ["rerank-skipped"]
        assert body["citations"][0]["chunk_id"] == "geo-001:0"
        t = client.get(f"/traces/{body['trace_id']}")
        assert t.status_code == 200
        assert t.json()["events"][0]["node"] == "route"


def test_unknown_corpus_404(tmp_path):
    with TestClient(make_app(tmp_path)) as client:
        r = client.post(
            "/query",
            json={
                "query": "q",
                "corpus_id": "nope",
                "preset": "rerank",
            },
        )
    assert r.status_code == 404
    assert "unknown corpus" in r.json()["detail"]


def test_unknown_preset_422(tmp_path):
    with TestClient(make_app(tmp_path)) as client:
        r = client.post(
            "/query",
            json={
                "query": "q",
                "corpus_id": "fixture-corpus",
                "preset": "bad",
            },
        )
    assert r.status_code == 422


def test_query_unavailable_names_missing_env_vars(tmp_path):
    with TestClient(make_app(tmp_path, with_runner=False)) as client:
        r = client.post(
            "/query",
            json={
                "query": "q",
                "corpus_id": "fixture-corpus",
                "preset": "rerank",
            },
        )
    assert r.status_code == 503
    assert "VOYAGE_API_KEY" in r.json()["detail"]


def test_unknown_trace_404(tmp_path):
    with TestClient(make_app(tmp_path)) as client:
        assert client.get("/traces/missing").status_code == 404
