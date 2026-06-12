"""Tests for DEMO_MODE config, rate/budget ledger."""

from __future__ import annotations

import sqlite3

import pytest
from fastapi import HTTPException

from ragreceipts.server.demo import DemoConfig, DemoLedger

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
