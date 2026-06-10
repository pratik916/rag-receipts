"""Live System-2 smoke — MANUAL ONLY, never CI.

Sends ONE real multi-hop query through the LangGraph System-2 path against an
ingested corpus (real Anthropic/Voyage/Cohere calls), then prints the answer and
the full trace with per-node timings and token counts.

Prerequisites:
  - an ingested corpus; Qdrant running (docker compose up qdrant) with
    QDRANT_URL set, or QDRANT_URL unset to use the CLI's local-file fallback
    at {data_dir}/qdrant-local (R7)
  - ANTHROPIC_API_KEY, VOYAGE_API_KEY, COHERE_API_KEY set (one .env, spec)

Usage (from rag-receipts/api/):
  uv run python scripts/smoke_s2.py --corpus musique-dev-300 \\
      --query "Who is the spouse of the director of the film Parasite?"
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import uuid
from pathlib import Path

from ragreceipts.agents.service import run_query
from ragreceipts.cli import _build_core_real  # Plan B's composition root (R9)
from ragreceipts.config import PRESETS
from ragreceipts.traces.store import TraceStore
from ragreceipts.types import RouteMode
from ragreceipts.vendors.anthropic_client import AnthropicClient

# R6 data-dir resolution: RAGRECEIPTS_DATA_DIR env var, default ../data from api/.
DATA_DIR = Path(
    os.environ.get("RAGRECEIPTS_DATA_DIR") or Path(__file__).resolve().parents[2] / "data"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Live System-2 smoke (manual only)")
    parser.add_argument("--corpus", required=True, help="ingested corpus_id")
    parser.add_argument("--query", required=True, help="a multi-hop question")
    args = parser.parse_args()

    preset = PRESETS["router-on"]
    config = dataclasses.replace(
        preset, query=dataclasses.replace(preset.query, route_mode=RouteMode.FORCE_S2)
    )
    core = _build_core_real(config, args.corpus, DATA_DIR)
    claude = AnthropicClient()  # reads ANTHROPIC_API_KEY
    store = TraceStore(DATA_DIR / "traces-smoke.sqlite3")
    trace_id = f"smoke-{uuid.uuid4().hex[:8]}"

    result = run_query(
        query=args.query,
        core=core,
        claude=claude,
        store=store,
        config=config,
        trace_id=trace_id,
    )

    print(
        f"\nsystem={result.system}  hops={result.hops_used}  "
        f"tokens={result.tokens_used}  trace={trace_id}"
    )
    print(f"abstained={result.final.abstained}  contradiction={result.final.contradiction_flag}")
    print(f"unresolved={result.final.unresolved_subqueries}")
    print(f"citations={result.final.citations}")
    print(f"\nANSWER:\n{result.final.text}\n\nTRACE:")
    for e in store.get(trace_id):
        print(
            f"  [{e.seq:02d}] {e.node:<12} {e.duration_ms:7.1f}ms "
            f"in={e.input_tokens:<6} out={e.output_tokens:<5} "
            f"model={e.model or '-'}"
        )
        if e.node == "grade":
            print(f"        verdict={e.payload['verdict']} -> {e.payload['next_action']}")


if __name__ == "__main__":
    main()
