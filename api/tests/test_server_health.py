"""/health: per-vendor capability with NAMED env vars (spec §Error handling)."""

from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from ragreceipts.server.app import create_app
from tests.helpers_server import make_test_deps


def test_health_names_missing_env_vars(tmp_path):
    app = create_app(deps_factory=lambda: make_test_deps(tmp_path, configured=False))
    with TestClient(app) as client:  # context manager runs lifespan
        r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "degraded"
    # R7: deps.qdrant is None (no QDRANT_URL) -> the healthcheck fails NAMING the env var
    assert body["missing_env_vars"] == [
        "VOYAGE_API_KEY",
        "COHERE_API_KEY",
        "ANTHROPIC_API_KEY",
        "QDRANT_URL",
    ]
    assert {v["name"]: v["configured"] for v in body["vendors"]} == {
        "voyage": False,
        "cohere": False,
        "anthropic": False,
    }
    assert body["qdrant_ok"] is False
    assert body["testing_mode"] is False
    assert body["demo_mode"] is False  # no demo_ledger wired in make_test_deps


def test_health_reports_demo_mode_when_ledger_wired(tmp_path):
    """The Corpora page reads /health.demo_mode to swap the BYO form for a read-only
    note, so the flag must be True exactly when a demo ledger is wired (mirrors the
    demo_mode=False assertion above)."""
    from ragreceipts.server.demo import DemoConfig, DemoLedger

    deps = make_test_deps(tmp_path, configured=True)
    deps.qdrant = QdrantClient(":memory:")
    deps.demo_ledger = DemoLedger(
        DemoConfig(
            daily_budget_usd=2.0,
            rate_per_min=5,
            rate_per_day=20,
            s2_token_ceiling=20_000,
            demo_corpus_id="demo",
        ),
        tmp_path / "demo.sqlite",
    )
    app = create_app(deps_factory=lambda: deps)
    with TestClient(app) as client:  # context manager runs lifespan (materialize/seed no-op)
        body = client.get("/health").json()
    assert body["demo_mode"] is True


def test_health_ok_when_configured_and_qdrant_reachable(tmp_path):
    deps = make_test_deps(tmp_path, configured=True)
    deps.qdrant = QdrantClient(":memory:")  # in-process local mode (named vectors verified)
    app = create_app(deps_factory=lambda: deps)
    with TestClient(app) as client:
        body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["missing_env_vars"] == []
    assert body["qdrant_ok"] is True


def test_openapi_is_31_and_lists_health():
    app = create_app(deps_factory=lambda: None)  # schema generation never builds deps
    schema = app.openapi()
    assert schema["openapi"].startswith("3.1")  # FastAPI default (verified, see plan table)
    assert "/health" in schema["paths"]
