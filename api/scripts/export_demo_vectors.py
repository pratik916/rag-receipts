#!/usr/bin/env python3
"""Export Qdrant named vectors for a corpus to a .npz file for demo seeding.

MANUAL ONLY, never CI. Run inside the one-time keyed bootstrap session after
`ragreceipts ingest --corpus demo` has populated the Qdrant collection (see
docs/runbooks/demo-bootstrap.md). The dense vectors live only in Qdrant after
ingest, so this script pulls them out into a committed `dense_vectors.npz` that
`ragreceipts.server.demo.seed_demo_qdrant` reloads on startup.

The two named-vector keys MUST stay `contextual` / `isolated` — they match the
collection written by ingest (ragreceipts.retrieval.dense.write_dense_index) and
the keys `seed_demo_qdrant` reads back (`data["contextual"]`, `data["isolated"]`).

Usage (from rag-receipts/api/):
  QDRANT_URL=http://localhost:6333 \
    uv run python scripts/export_demo_vectors.py --corpus-id demo \
        --output ../demo/corpus/dense_vectors.npz
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    qdrant_url = os.environ["QDRANT_URL"]
    client = QdrantClient(url=qdrant_url)

    # Scroll every point with its named vectors, in id order (matches chunk order
    # 0..n-1, which is how seed_demo_qdrant re-associates vectors with chunks).
    offset = None
    contextual, isolated = [], []
    while True:
        results, offset = client.scroll(
            collection_name=args.corpus_id,
            with_vectors=True,
            limit=100,
            offset=offset,
        )
        for point in results:
            contextual.append(point.vector["contextual"])
            isolated.append(point.vector["isolated"])
        if offset is None:
            break

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        contextual=np.array(contextual, dtype="float32"),
        isolated=np.array(isolated, dtype="float32"),
    )
    print(f"Exported {len(contextual)} vectors to {args.output}")


if __name__ == "__main__":
    main()
