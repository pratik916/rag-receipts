"""ragreceipts CLI. `ragreceipts ingest --corpus <id>` rebuilds every index variant for a
Spike 0 corpus. Missing keys produce a named env-var message, never a stack trace.

Factories build_embed_transport/build_qdrant are module-level seams: tests monkeypatch
them; Plan D's server reuses them. Plan B MODIFIES this file in place (R6), adding
`eval` and `receipts` subparsers plus _build_core_real(config, corpus_id, data_dir),
and keeps these seams and main(argv) unchanged."""

import argparse
import json
import os
import sys
from pathlib import Path

from qdrant_client import QdrantClient

from ragreceipts.config import IngestConfig
from ragreceipts.ingest.pipeline import run_ingest
from ragreceipts.vendors.voyage_client import VoyageClient


def build_embed_transport() -> VoyageClient:
    api_key = os.environ.get("VOYAGE_API_KEY")
    if not api_key:
        raise SystemExit(
            "VOYAGE_API_KEY is not set — ingest needs Voyage embeddings (set it in .env)"
        )
    return VoyageClient(api_key=api_key)


def build_qdrant(data_dir: Path) -> QdrantClient:
    url = os.environ.get("QDRANT_URL")
    if url:
        return QdrantClient(url=url)
    # CLI-scoped fallback ONLY (R7): local file mode, no server. The FastAPI server
    # (Plan D) REQUIRES QDRANT_URL and fails its healthcheck with a named-env-var
    # message when it is missing.
    return QdrantClient(path=str(data_dir / "qdrant-local"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ragreceipts")
    subparsers = parser.add_subparsers(dest="command", required=True)
    ingest = subparsers.add_parser("ingest", help="(re)build all index variants for a corpus")
    ingest.add_argument("--corpus", required=True, help="corpus id, e.g. nq-dev-300")
    ingest.add_argument(
        "--data-dir",
        type=Path,
        default=Path(os.environ.get("RAGRECEIPTS_DATA_DIR", "../data")),
        help="data dir holding corpora/ (default ../data, run from api/)",
    )
    ingest.add_argument("--chunk-size", type=int, default=IngestConfig().chunk_size)
    ingest.add_argument("--chunk-overlap", type=int, default=IngestConfig().chunk_overlap)
    args = parser.parse_args(argv)

    if args.command == "ingest":
        try:
            manifest = run_ingest(
                corpus_id=args.corpus,
                data_dir=args.data_dir,
                ingest_config=IngestConfig(
                    chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap
                ),
                embed=build_embed_transport(),
                qdrant=build_qdrant(args.data_dir),
            )
        except FileNotFoundError:
            print(
                f"error: corpus '{args.corpus}' not found under "
                f"{args.data_dir / 'corpora'} — run the Spike 0 download script first",
                file=sys.stderr,
            )
            return 1
        print(json.dumps(manifest, indent=2))
        return 0
    return 2
