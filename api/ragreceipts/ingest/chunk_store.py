"""chunks.jsonl persistence — the canonical chunk order shared by ALL index variants.

SparseRetriever maps bm25s row indices into this list; Qdrant payloads duplicate the
fields for dense lookups. Rows carry the FULL Chunk via asdict — including the R3
start_token/end_token token-range fields. Row order is load-bearing: never reorder."""

import json
from dataclasses import asdict
from pathlib import Path

from ragreceipts.types import Chunk


def write_chunks(path: Path, chunks: list[Chunk]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            fh.write(json.dumps(asdict(chunk), ensure_ascii=False) + "\n")


def read_chunks(path: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                chunks.append(Chunk(**json.loads(line)))
    return chunks
