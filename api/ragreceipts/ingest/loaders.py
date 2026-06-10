"""Readers for Spike 0's raw benchmark-slice corpus layout (binding per R1):

    data/corpora/{corpus_id}/raw/{docs.jsonl, queries.jsonl,
                                  slice-full.json, slice-smoke.json, download_meta.json}

docs.jsonl record: {"doc_id", "passage_id", "title", "text"}.
queries.jsonl record: {"query_id", "question", "answer"/"answer_texts",
"answer_aliases", "gold": {"type": "passage", "passage_ids": [...]} or
{"type": "span", "doc_id", "start_token", "end_token"}} (+ top-level "gold_text").
Plan A only COUNTS query records — per R2 no intermediate eval-queries file is ever
materialized; Plan B's load_queries reads raw/queries.jsonl directly.
SourcePassage is the seam everything downstream depends on.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SourcePassage:
    passage_id: str
    doc_id: str
    title: str
    text: str


def load_passages(corpus_dir: Path) -> list[SourcePassage]:
    path = corpus_dir / "raw" / "docs.jsonl"
    passages: list[SourcePassage] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            passages.append(
                SourcePassage(
                    passage_id=str(row["passage_id"]),
                    doc_id=str(row["doc_id"]),
                    title=str(row.get("title", "")),
                    text=str(row["text"]),
                )
            )
    return passages


def dataset_name(corpus_id: str) -> str:
    """'musique-dev-300' -> 'musique', 'nq-dev-300' -> 'nq'; anything without the
    -dev-N suffix is its own name. Load-bearing: the manifest's dataset.name feeds
    Plan B's MULTI_HOP_DATASETS gate (router-on runs only on multi-hop corpora)."""
    return re.sub(r"-dev-\d+$", "", corpus_id)


def load_dataset_info(corpus_dir: Path) -> dict:
    """Constructs the manifest's dataset block (contracts shape, incl. "name")
    from Spike 0's download_meta.json (R1)."""
    meta = json.loads((corpus_dir / "raw" / "download_meta.json").read_text(encoding="utf-8"))
    dataset = meta["dataset"]
    return {
        "name": dataset_name(str(meta["corpus_id"])),
        "hf_id": dataset["hf_id"],
        "split": dataset["split"],
        "revision": dataset["revision"],
    }


def count_queries(corpus_dir: Path) -> int:
    path = corpus_dir / "raw" / "queries.jsonl"
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


def group_documents(passages: list[SourcePassage]) -> list[list[SourcePassage]]:
    """Groups passages by doc_id, preserving first-seen document order and
    within-document passage order (dict preserves insertion order)."""
    by_doc: dict[str, list[SourcePassage]] = {}
    for passage in passages:
        by_doc.setdefault(passage.doc_id, []).append(passage)
    return list(by_doc.values())
