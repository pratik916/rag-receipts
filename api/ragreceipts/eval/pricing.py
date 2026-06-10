"""Versioned pricing table. PRICING_VERSION is recorded in every receipt.

Prices verified 2026-06-10:
- claude-haiku-4-5-20251001 $1.00/$5.00 per MTok and claude-sonnet-4-6
  $3.00/$15.00 per MTok: claude-api skill model table /
  https://platform.claude.com/docs/en/pricing.md
- voyage-context-3 $0.18 per 1M tokens: https://docs.voyageai.com/docs/pricing
- rerank-v4.0-pro $0.0025 per search unit (1 query + up to 100 docs; docs
  >500 tokens auto-chunk, each chunk counts): search-unit definition from
  https://cohere.com/pricing; the per-search price is not published there
  (sales-gated) and is corroborated by https://openrouter.ai/cohere/rerank-4-pro
  and https://vercel.com/ai-gateway/models/rerank-v4-pro - RE-VERIFY on the
  Cohere billing dashboard during the first keyed run (see
  docs/runbooks/first-keyed-run.md) and bump PRICING_VERSION if it differs.

Lookups raise KeyError for unknown models: an unpriced call must never be
silently billed at $0.
"""

from __future__ import annotations

from ragreceipts.constants import EMBED_MODEL, RERANK_MODEL, ROUTER_MODEL, SYNTH_MODEL

PRICING_VERSION = "2026-06-10"

PRICING: dict[str, dict] = {
    ROUTER_MODEL: {"usd_per_mtok_input": 1.00, "usd_per_mtok_output": 5.00},
    SYNTH_MODEL: {"usd_per_mtok_input": 3.00, "usd_per_mtok_output": 15.00},
    # JUDGE_MODEL == SYNTH_MODEL ("claude-sonnet-4-6"): same key, priced once.
    EMBED_MODEL: {"usd_per_mtok_input": 0.18, "usd_per_mtok_output": 0.0},
    RERANK_MODEL: {"usd_per_search_unit": 0.0025},
}


def usd_for_tokens(model: str, input_tokens: int, output_tokens: int) -> float:
    """Cost in USD for a token-billed model. KeyError if the model is unpriced."""
    entry = PRICING[model]
    return (
        input_tokens * entry["usd_per_mtok_input"] + output_tokens * entry["usd_per_mtok_output"]
    ) / 1_000_000


def usd_for_rerank(n_search_units: int, model: str = RERANK_MODEL) -> float:
    """Cost in USD for search-unit-billed rerank calls."""
    return n_search_units * PRICING[model]["usd_per_search_unit"]
