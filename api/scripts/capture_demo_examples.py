#!/usr/bin/env python3
"""Capture three representative query traces for the demo examples.

MANUAL ONLY, never CI. Run inside the one-time keyed bootstrap session after the
demo corpus is ingested and its graph is built (see docs/runbooks/demo-bootstrap.md).
Each captured JSON file is written so that `DemoExampleItem(**json.loads(...))`
round-trips cleanly — the server's `/demo/examples` endpoint loads each file that
way (server/app.py::list_demo_examples).

Serialization is by-construction aligned with the Pydantic models in
server/models.py:
  - `Citation` and `TraceEvent` are frozen dataclasses, so `dataclasses.asdict`
    yields exactly the fields `CitationModel` / `TraceEventModel` expect.
  - `deps.trace_store.get(trace_id)` returns `list[TraceEvent]` (TraceReadWrite
    protocol in server/deps.py).

Usage (from rag-receipts/api/):
  VOYAGE_API_KEY=... COHERE_API_KEY=... ANTHROPIC_API_KEY=... \
    QDRANT_URL=http://localhost:6333 \
    uv run python scripts/capture_demo_examples.py --corpus-id demo \
        --output-dir ../demo/examples/
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ragreceipts.server.deps import build_deps

DEMO_QUERIES = [
    {
        "label": "s1",
        "query": "Who was the first American to walk on the Moon?",
        "preset": "router-on",
    },
    {
        "label": "s2",
        "query": (
            "Which astronaut flew on both a Mercury mission and later commanded "
            "an Apollo mission to land on the Moon?"
        ),
        "preset": "router-on",
    },
    {
        "label": "graph",
        "query": (
            "What spacecraft served as the lifeboat during the Apollo 13 mission, "
            "and what was the name of the Apollo program that produced it?"
        ),
        "preset": "graph-rrf",
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-id", default="demo")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    os.environ.setdefault("RAGRECEIPTS_DATA_DIR", "../data")
    os.environ.setdefault("RAGRECEIPTS_RECEIPTS_DIR", "../receipts")
    os.environ.setdefault("DEMO_MODE", "0")  # no budget limits during capture

    deps = build_deps()
    assert deps.query_runner is not None, "All vendor keys must be set"
    assert deps.qdrant is not None, "QDRANT_URL must be set"

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for q in DEMO_QUERIES:
        print(f"Capturing {q['label']}: {q['query'][:60]}...")
        result = deps.query_runner.run(
            query=q["query"],
            corpus_id=args.corpus_id,
            preset=q["preset"],
        )
        events = deps.trace_store.get(result.trace_id)
        example = {
            "label": q["label"],
            "query": q["query"],
            "answer": result.answer,
            "route": result.route,
            "citations": [dataclasses.asdict(c) for c in result.citations],
            "trace_events": [dataclasses.asdict(e) for e in events],
        }
        out_path = args.output_dir / f"example_{q['label']}.json"
        out_path.write_text(json.dumps(example, indent=2))
        print(f"  -> {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
