"""Build the knowledge-graph artifact for an ingested corpus — MANUAL ONLY, never CI.

Reads {data_dir}/corpora/{corpus}/chunks.jsonl (written by `ragreceipts ingest`),
runs REAL OpenIE (Haiku) + REAL Voyage embeddings, builds the graph index, writes
data/corpora/{corpus}/graph/, and updates the manifest's index_hashes["graph"] (G6).
Cost: ~$10-30 for ~5k passages (Haiku OpenIE, one parse per chunk). NEVER wired into
CI — the whole graph plane is offline-tested with FakeOpenIE (see test_graph_*.py).

Prerequisites:
  - `ragreceipts ingest --corpus <id>` has run (chunks.jsonl + manifest.json exist).
  - ANTHROPIC_API_KEY and VOYAGE_API_KEY exported (or in api/.env, loaded).

Usage (from rag-receipts/api/):
  ANTHROPIC_API_KEY=... VOYAGE_API_KEY=... \
    uv run python scripts/build_graph.py --corpus musique-dev-300
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from ragreceipts.agents.openie import OpenIEExtractor
from ragreceipts.ingest.chunk_store import read_chunks
from ragreceipts.ingest.graph_index import build_graph_index, write_graph_index
from ragreceipts.ingest.manifest import graph_index_hash, read_manifest, write_manifest
from ragreceipts.vendors.anthropic_client import AnthropicClient
from ragreceipts.vendors.voyage_client import VoyageClient

# R6 data-dir resolution: RAGRECEIPTS_DATA_DIR env var, default ../data from api/.
DATA_DIR = Path(
    os.environ.get("RAGRECEIPTS_DATA_DIR") or Path(__file__).resolve().parents[2] / "data"
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the graph artifact (manual, keyed)")
    parser.add_argument("--corpus", required=True, help="ingested corpus_id, e.g. musique-dev-300")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("ANTHROPIC_API_KEY is not set — graph OpenIE needs Haiku (set it in .env)")
    if not os.environ.get("VOYAGE_API_KEY"):
        raise SystemExit("VOYAGE_API_KEY is not set — graph build needs Voyage embeddings (.env)")

    corpus_dir = args.data_dir / "corpora" / args.corpus
    chunks_path = corpus_dir / "chunks.jsonl"
    if not chunks_path.exists():
        raise SystemExit(
            f"{chunks_path} not found — run `ragreceipts ingest --corpus {args.corpus}` first"
        )

    chunks = read_chunks(chunks_path)
    print(f"loaded {len(chunks)} chunks from {chunks_path}")

    openie = OpenIEExtractor(AnthropicClient(api_key=os.environ["ANTHROPIC_API_KEY"]))
    embed = VoyageClient(api_key=os.environ["VOYAGE_API_KEY"])

    result = build_graph_index(corpus_id=args.corpus, chunks=chunks, openie=openie, embed=embed)
    print(
        f"graph: {result.n_passage} passage nodes, {result.n_phrase} phrase nodes, "
        f"{result.n_edges} edges from {result.n_triples} triples"
    )

    graph_dir = corpus_dir / "graph"
    write_graph_index(result, graph_dir)
    print(f"wrote artifact to {graph_dir}")

    manifest = read_manifest(corpus_dir)
    manifest["index_hashes"]["graph"] = graph_index_hash(graph_dir)
    write_manifest(corpus_dir, manifest)
    print(f"manifest index_hashes['graph'] = {manifest['index_hashes']['graph']}")
    print(json.dumps({"corpus": args.corpus, "graph_dir": str(graph_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
