"""Golden tests for the versioned pricing table. All values hand-computed."""

import pytest

from ragreceipts.constants import EMBED_MODEL, RERANK_MODEL, ROUTER_MODEL, SYNTH_MODEL
from ragreceipts.eval.pricing import (
    PRICING,
    PRICING_VERSION,
    usd_for_rerank,
    usd_for_tokens,
)


def test_pricing_version_is_dated() -> None:
    assert PRICING_VERSION == "2026-06-10"


def test_all_contract_models_priced() -> None:
    for model in (ROUTER_MODEL, SYNTH_MODEL, EMBED_MODEL, RERANK_MODEL):
        assert model in PRICING


def test_haiku_one_mtok_input() -> None:
    # 1,000,000 input tokens x $1.00/MTok = $1.00
    assert usd_for_tokens(ROUTER_MODEL, 1_000_000, 0) == pytest.approx(1.00)


def test_sonnet_mixed() -> None:
    # 100k in x $3/MTok = $0.30; 10k out x $15/MTok = $0.15; total $0.45
    assert usd_for_tokens(SYNTH_MODEL, 100_000, 10_000) == pytest.approx(0.45)


def test_voyage_embed_input_only() -> None:
    # 1M tokens x $0.18/MTok = $0.18; output side is 0 for embeddings
    assert usd_for_tokens(EMBED_MODEL, 1_000_000, 0) == pytest.approx(0.18)


def test_rerank_per_search_unit() -> None:
    # 1,000 search units x $0.0025 = $2.50
    assert usd_for_rerank(1_000) == pytest.approx(2.50)


def test_unpriced_model_raises_keyerror() -> None:
    with pytest.raises(KeyError):
        usd_for_tokens("gpt-4o", 1, 1)
