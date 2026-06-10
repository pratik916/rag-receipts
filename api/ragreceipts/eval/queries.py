"""Query/gold loading + slices.

Reads Spike 0's raw slice layout DIRECTLY and normalizes in memory (R1/R2 -
there is no intermediate eval queries file):

  data/corpora/{corpus_id}/raw/queries.jsonl - one JSON object per line:
    MuSiQue: {"query_id","question","answer","answer_aliases",
              "gold":{"type":"passage","passage_ids":[...]}}
    NQ:      {"query_id","question","answer_texts",
              "gold":{"type":"span","doc_id","start_token","end_token"},
              "gold_text"}
  data/corpora/{corpus_id}/raw/slice-{smoke,full}.json - JSON arrays of
    query_id strings (slice-smoke.json is the first 15 of slice-full.json,
    Spike 0 decisions D4 - the size lives in the data, not in this module).

Normalization: gold_answers = [answer] + answer_aliases (MuSiQue) or
answer_texts (NQ); golds become Spike 0's typed GoldPassage / GoldSpan
from eval/alignment.py. Span golds are token ranges, never text.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ragreceipts.eval.alignment import Gold, GoldPassage, GoldSpan


@dataclass(frozen=True)
class QueryRecord:
    query_id: str
    question: str
    gold_answers: list[str]
    golds: list[Gold]


def _normalize(raw: dict) -> QueryRecord:
    gold = raw["gold"]
    if gold["type"] == "passage":
        golds: list[Gold] = [
            GoldPassage(query_id=raw["query_id"], passage_id=pid) for pid in gold["passage_ids"]
        ]
        gold_answers = [raw["answer"], *raw.get("answer_aliases", [])]
    elif gold["type"] == "span":
        golds = [
            GoldSpan(
                query_id=raw["query_id"],
                doc_id=gold["doc_id"],
                start_token=gold["start_token"],
                end_token=gold["end_token"],
            )
        ]
        gold_answers = list(raw["answer_texts"])
    else:
        raise ValueError(f"{raw['query_id']}: unknown gold type {gold['type']!r}")
    return QueryRecord(
        query_id=raw["query_id"], question=raw["question"], gold_answers=gold_answers, golds=golds
    )


def load_queries(data_dir: Path, corpus_id: str) -> list[QueryRecord]:
    path = data_dir / "corpora" / corpus_id / "raw" / "queries.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found - run Spike 0's download script first: "
            f"`uv run --project api python scripts/download_data.py --corpus all` "
            f"(from the repo root) materializes data/corpora/{corpus_id}/raw/"
        )
    with path.open() as f:
        return [_normalize(json.loads(line)) for line in f if line.strip()]


def slice_query_ids(data_dir: Path, corpus_id: str, slice_name: str) -> list[str]:
    """Read Spike 0's slice file: a JSON array of query_id strings."""
    if slice_name not in ("smoke", "full"):
        raise ValueError(f"unknown slice {slice_name!r}; expected 'smoke' or 'full'")
    path = data_dir / "corpora" / corpus_id / "raw" / f"slice-{slice_name}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found - Spike 0's scripts/download_data.py writes the "
            f"slice files next to queries.jsonl"
        )
    return list(json.loads(path.read_text()))


def slice_queries(queries: list[QueryRecord], slice_ids: list[str]) -> list[QueryRecord]:
    """Project queries onto a slice (query-id list), preserving the slice's order."""
    by_id = {q.query_id: q for q in queries}
    missing = [qid for qid in slice_ids if qid not in by_id]
    if missing:
        raise ValueError(f"slice references query_ids absent from queries.jsonl: {missing}")
    return [by_id[qid] for qid in slice_ids]


def load_manifest(data_dir: Path, corpus_id: str) -> dict:
    return json.loads((data_dir / "corpora" / corpus_id / "manifest.json").read_text())
