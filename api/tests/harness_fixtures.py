"""Tiny in-repo labeled corpus for the harness self-test (spec §Testing).

Engineered so flipping the rerank flag PROVABLY changes Recall@5:
- q0's gold chunk sits at RRF rank 7 (outside top_k_final=5, inside
  top_k_fuse=50); the scripted reranker scores it 0.99 and lifts it to rank 1.
- q1-q3 have their gold at rank 1 regardless.
Therefore recall@5(rerank off) = 3/4 = 0.75 and recall@5(rerank on) = 4/4 = 1.0.
If the ablation runner ever stops detecting that delta, CI fails:
receipts that can't fail aren't receipts.

Corpus files use Spike 0's raw layout (R1): raw/queries.jsonl with typed
passage golds plus slice-full.json / slice-smoke.json query-id lists.
"""

from __future__ import annotations

import json
from pathlib import Path

from ragreceipts.types import Chunk, ScoredChunk

N_QUERIES = 4


def _chunk(passage_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=f"{passage_id}:0",
        corpus_id="harness",
        doc_id=passage_id,
        passage_id=passage_id,
        text=text,
        position=0,
        start_token=0,
        end_token=len(text.split()),
    )


class ListRetriever:
    """Retriever-protocol fake returning a fixed ranking per question text."""

    def __init__(self, rankings: dict[str, list[Chunk]], source: str) -> None:
        self._rankings = rankings
        self._source = source

    def search(self, query: str, k: int) -> list[ScoredChunk]:
        chunks = self._rankings[query][:k]
        n = len(chunks)
        return [
            ScoredChunk(chunk=c, score=float(n - i), source=self._source)
            for i, c in enumerate(chunks)
        ]


def build_harness_fixture(*, misaligned: bool = False) -> dict:
    """Rankings + scripted rerank scores + raw-layout query records.

    misaligned=True deliberately breaks every gold passage_id ("WRONG-..."):
    the self-test asserts this scores recall 0.0, proving the alignment rule
    is load-bearing (an is_hit that always matched would fail that test).
    """
    rankings: dict[str, list[Chunk]] = {}
    rerank_scores: dict[str, float] = {}
    queries: list[dict] = []
    for i in range(N_QUERIES):
        gold = _chunk(f"g{i}", f"gold passage text {i}")
        fillers = [_chunk(f"f{i}-{j}", f"filler text {i}-{j}") for j in range(6)]
        ranking = fillers + [gold] if i == 0 else [gold] + fillers
        question = f"harness question {i}?"
        rankings[question] = ranking
        rerank_scores[gold.text] = 0.99
        for j, filler in enumerate(fillers):
            rerank_scores[filler.text] = 0.5 - 0.01 * j
        gold_pid = f"WRONG-g{i}" if misaligned else f"g{i}"
        queries.append(
            {
                "query_id": f"q{i}",
                "question": question,
                "answer": f"gold answer {i}",
                "answer_aliases": [],
                "gold": {"type": "passage", "passage_ids": [gold_pid]},
            }
        )
    return {"rankings": rankings, "rerank_scores": rerank_scores, "queries": queries}


def write_harness_corpus(data_dir: Path, corpus_id: str, queries: list[dict]) -> None:
    raw = data_dir / "corpora" / corpus_id / "raw"
    raw.mkdir(parents=True)
    (raw / "queries.jsonl").write_text("\n".join(json.dumps(q) for q in queries) + "\n")
    query_ids = [q["query_id"] for q in queries]
    (raw / "slice-full.json").write_text(json.dumps(query_ids))
    (raw / "slice-smoke.json").write_text(json.dumps(query_ids[:15]))
    (data_dir / "corpora" / corpus_id / "manifest.json").write_text(
        json.dumps(
            {
                "corpus_id": corpus_id,
                "dataset": {
                    "name": "musique",
                    "hf_id": "in-repo-fixture",
                    "split": "fixture",
                    "revision": "0",
                },
                "index_hashes": {
                    "dense_contextual": "sha256:c",
                    "dense_isolated": "sha256:i",
                    "sparse": "sha256:s",
                },
                "n_queries": len(queries),
            }
        )
    )


# --- Plan F: graph self-test helper -------------------------------------------


def build_misaligned_graph_queries() -> list[dict]:
    """Graph fixture queries with every gold passage_id broken ('WRONG-...').

    The graph self-test asserts this scores recall 0.0 even though PPR still lands
    on the (correctly-built) gold passage NODE — proving the alignment rule, not the
    graph, is what the metric trusts."""
    from tests.graph_fixtures import fixture_queries

    out: list[dict] = []
    for q in fixture_queries():
        broken = dict(q)
        broken["gold"] = {
            "type": "passage",
            "passage_ids": [f"WRONG-{pid}" for pid in q["gold"]["passage_ids"]],
        }
        out.append(broken)
    return out
