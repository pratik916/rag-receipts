"""Eval runs: pre-run cost estimate, confirmation gate, job execution, resume API."""

import time

from fastapi.testclient import TestClient

from ragreceipts.server.app import create_app
from ragreceipts.server.evalruns import CostEstimate
from tests.helpers_server import make_test_deps


class StubEvalRunner:
    def __init__(self) -> None:
        self.ran: list[dict] = []

    def estimate(self, *, corpus_id: str, preset: str, slice_name: str) -> CostEstimate:
        n = 15 if slice_name == "smoke" else 300
        return CostEstimate(
            n_queries=n,
            est_tokens=n * 4700,
            est_usd=round(n * 0.02, 2),
            pricing_table_version="2026-06-10",
        )

    def run(self, *, corpus_id, preset, slice_name, spend_cap_usd, emit) -> str:
        emit("scoring", 0.5)
        self.ran.append({"corpus_id": corpus_id, "preset": preset, "slice": slice_name})
        return "run-001"


def make_client(tmp_path):
    deps = make_test_deps(tmp_path, configured=True)
    deps.eval_runner = StubEvalRunner()
    return TestClient(create_app(deps_factory=lambda: deps)), deps


def wait_status(client, job_id, want, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/jobs/{job_id}").json()
        if body["status"] == want:
            return body
        time.sleep(0.05)
    raise AssertionError(f"job never reached {want}")


def test_estimate_without_confirm_creates_no_job(tmp_path):
    client, deps = make_client(tmp_path)
    with client:
        r = client.post("/eval/runs", json={"corpus_id": "c1", "preset": "rerank"})
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "needs_confirmation"
        assert body["job_id"] is None
        assert body["estimate"]["n_queries"] == 15
        assert client.get("/eval/runs").json()["runs"] == []
    assert deps.eval_runner.ran == []


def test_confirm_starts_job_that_runs_and_lists(tmp_path):
    client, deps = make_client(tmp_path)
    with client:
        r = client.post(
            "/eval/runs",
            json={
                "corpus_id": "c1",
                "preset": "rerank",
                "confirm": True,
            },
        )
        body = r.json()
        assert body["status"] == "started" and body["job_id"]
        done = wait_status(client, body["job_id"], "succeeded")
        assert any("run-001" in e["message"] for e in done["events"])
        runs = client.get("/eval/runs").json()["runs"]
        assert runs[0]["preset"] == "rerank" and runs[0]["status"] == "succeeded"
    assert deps.eval_runner.ran[0]["slice"] == "smoke"


def test_estimate_over_spend_cap_refused(tmp_path):
    client, _ = make_client(tmp_path)
    with client:
        r = client.post(
            "/eval/runs",
            json={
                "corpus_id": "c1",
                "preset": "rerank",
                "slice": "full",
                "confirm": True,
                "spend_cap_usd": 1.0,
            },
        )
    assert r.status_code == 400
    assert "exceeds spend cap" in r.json()["detail"]


def test_eval_unavailable_names_missing_env_vars(tmp_path):
    deps = make_test_deps(tmp_path, configured=False)
    with TestClient(create_app(deps_factory=lambda: deps)) as client:
        r = client.post("/eval/runs", json={"corpus_id": "c1", "preset": "rerank"})
    assert r.status_code == 503
    assert "ANTHROPIC_API_KEY" in r.json()["detail"]


def test_jobs_404_and_resume_409(tmp_path):
    client, _ = make_client(tmp_path)
    with client:
        assert client.get("/jobs/none").status_code == 404
        r = client.post(
            "/eval/runs",
            json={
                "corpus_id": "c1",
                "preset": "rerank",
                "confirm": True,
            },
        )
        job_id = r.json()["job_id"]
        wait_status(client, job_id, "succeeded")
        assert client.post(f"/jobs/{job_id}/resume").status_code == 409
