"""build_deps(): env-driven vendor capability + the TESTING=1 seam."""

from ragreceipts.server.deps import VENDOR_ENV_VARS, AppPaths, build_deps


def test_vendor_env_var_names_are_the_official_sdk_names():
    assert VENDOR_ENV_VARS == {
        "voyage": "VOYAGE_API_KEY",
        "cohere": "COHERE_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
    }


def test_app_paths_layout(tmp_path):
    paths = AppPaths(data_dir=tmp_path / "data", receipts_committed_dir=tmp_path / "receipts")
    paths.ensure()
    assert paths.corpora_dir == tmp_path / "data" / "corpora"
    assert paths.receipts_local_dir == tmp_path / "data" / "receipts-local"
    assert paths.uploads_dir == tmp_path / "data" / "uploads"
    assert paths.jobs_db == tmp_path / "data" / "server-jobs.sqlite"
    assert paths.corpora_dir.is_dir() and paths.uploads_dir.is_dir()


def test_build_deps_reports_missing_keys_without_touching_network(tmp_path, monkeypatch):
    for env in VENDOR_ENV_VARS.values():
        monkeypatch.delenv(env, raising=False)
    monkeypatch.delenv("TESTING", raising=False)
    monkeypatch.delenv("QDRANT_URL", raising=False)
    monkeypatch.setenv("RAGRECEIPTS_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("RAGRECEIPTS_RECEIPTS_DIR", str(tmp_path / "receipts"))
    deps = build_deps()
    assert deps.testing_mode is False
    assert [v.configured for v in deps.vendors] == [False, False, False]
    assert deps.qdrant is None  # R7: QDRANT_URL unset -> NO silent localhost default
    assert deps.query_runner is None  # no keys -> endpoint will 503 with named env vars
    deps.job_runner.stop()


def test_testing_env_wires_fixture_deps(tmp_path, monkeypatch):
    monkeypatch.setenv("TESTING", "1")
    monkeypatch.setenv("RAGRECEIPTS_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("RAGRECEIPTS_RECEIPTS_DIR", str(tmp_path / "receipts"))
    deps = build_deps()
    assert deps.testing_mode is True
    assert all(v.configured for v in deps.vendors)  # fakes count as configured
    assert deps.qdrant is not None  # QdrantClient(":memory:") — named vectors verified
    manifest = deps.paths.corpora_dir / "fixture-corpus" / "manifest.json"
    assert manifest.exists()
    deps.job_runner.stop()
