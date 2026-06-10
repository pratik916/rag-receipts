"""Graph harness self-test (spec §Testing, graph half). CI-enforced.

1. On the fixture graph, the `graph` preset's Recall@5 PROVABLY differs from a
   deliberately weak `rerank` cell — the graph receipt cell can fail.
2. A misaligned-gold graph run scores Recall@5 0.0 while the aligned run scores
   1.0 — the alignment rule is load-bearing for the graph path too.

Real-code reconciliations (RG1/RG8, mirroring F2's test_eval_graph.py): the
GraphRetriever uses the keyword-only RG1 ctor (chunks required, embed keyword);
recognition goes through claude.parse(output_format=SeedSelection); every
RetrievalCore passes rerank_stage explicitly. The graph cell deterministically
lands Recall@5==1.0 because the retriever's FakeEmbed is seeded with
fixture_query_aliases() (query vector == gold passage vector). The weak rerank
cell ranks every gold OUTSIDE top_k_final=5 (gold given the lowest score), so
its Recall@5 is a deterministic 0.0 and the signed graph-vs-rerank delta is real.
"""

import json
from pathlib import Path

import pytest

from ragreceipts.agents.schemas import FinalAnswer, RouteDecision
from ragreceipts.eval.run_state import RunStore
from ragreceipts.eval.runner import AblationRunner
from ragreceipts.ingest.graph_index import GraphIndex
from ragreceipts.retrieval.core import RetrievalCore
from ragreceipts.retrieval.graph import GraphRetriever, SeedSelection
from ragreceipts.retrieval.rerank import RerankStage
from ragreceipts.traces.store import TraceStore
from ragreceipts.types import Chunk
from ragreceipts.vendors.base import ParsedResult
from tests.fakes import FakeEmbed, FakeRerank
from tests.graph_fixtures import (
    fixture_chunks,
    fixture_queries,
    fixture_query_aliases,
    write_graph_artifact,
    write_graph_corpus,
)
from tests.harness_fixtures import ListRetriever, build_misaligned_graph_queries


class GraphEchoClaude:
    """Tiny ClaudeTransport echo (mirrors F2). Under FORCE_S1 the router never
    runs, so the only live parse is FinalAnswer (synthesis). Recognition
    (recognition='llm') calls parse(output_format=SeedSelection); returning an
    empty selection is a safe no-op — GraphRetriever's never-empty fallback keeps
    all seeds, so the fixture's gold-seeded PPR is untouched. RouteDecision is
    dead code under FORCE_S1 but answered for completeness."""

    def parse(self, *, model, system, user, max_tokens, output_format, temperature=0.0):
        if output_format is SeedSelection:
            return ParsedResult(
                parsed=SeedSelection(phrases=[]), input_tokens=900, output_tokens=80
            )
        if output_format is RouteDecision:
            return ParsedResult(
                parsed=RouteDecision(route="simple", confidence=0.95),
                input_tokens=50,
                output_tokens=10,
            )
        return ParsedResult(
            parsed=FinalAnswer(text="answer [1]", citations=[1]),
            input_tokens=400,
            output_tokens=40,
        )


def _filler_chunk(j: int) -> Chunk:
    """A dedicated non-gold filler chunk (never a graph node, never a gold).

    Distinct from the fixture's g*/f* passages so its text NEVER collides with a
    gold text — the collision (a query's gold doubling as another query's filler)
    is what made a naive text-keyed weak rerank score a deceptive 0.75."""
    text = f"Distractor passage number {j} about an unrelated subject."
    return Chunk(
        chunk_id=f"x{j}:0",
        corpus_id="graph-harness",
        doc_id=f"x{j}",
        passage_id=f"x{j}",
        text=text,
        position=0,
        start_token=0,
        end_token=len(text.split()),
    )


def _weak_rerank_rankings() -> tuple[dict[str, list], dict[str, float]]:
    """For each query, rank six dedicated distractors BEFORE the gold and score the
    gold the LOWEST, so the rerank cell's Recall@5 is a deterministic 0.0.

    Returns (rankings keyed by question text, rerank scores keyed by chunk text).
    The candidate pool comes from RRF over the bm25+dense ListRetrievers; the gold
    is appended last and given a score below every distractor, so after rerank's
    top_n=5 cut the gold is dropped for every query (six distractors > five slots).
    The distractors are dedicated x* chunks whose text never collides with a gold,
    so every gold reliably keeps its lowest score."""
    chunks = {c.passage_id: c for c in fixture_chunks()}
    fillers = [_filler_chunk(j) for j in range(6)]
    rankings: dict[str, list] = {}
    scores: dict[str, float] = {}
    for rank, c in enumerate(fillers):
        scores[c.text] = 0.9 - 0.01 * rank  # six distractors, all high, descending
    for q in fixture_queries():
        i = q["query_id"][1:]
        gold = chunks[f"g{i}"]
        rankings[q["question"]] = fillers + [gold]
        scores[gold.text] = 0.01  # gold lowest -> dropped from top-5 after rerank
    return rankings, scores


def make_graph_runner(tmp_path: Path, *, misaligned: bool = False) -> AblationRunner:
    if misaligned:
        data_dir = _misaligned_corpus(tmp_path)
    else:
        data_dir = write_graph_corpus(tmp_path)
    graph_dir = data_dir / "corpora" / "graph-harness" / "graph"
    rankings, rerank_scores = _weak_rerank_rankings()
    aliases = fixture_query_aliases()

    def core_factory(cfg) -> RetrievalCore:
        if cfg.query.graph:
            index = GraphIndex.load(graph_dir)
            mode = cfg.query.graph_recognition
            graph = GraphRetriever(
                index,
                chunks=fixture_chunks(),
                embed=FakeEmbed(query_aliases=aliases),
                claude=GraphEchoClaude() if mode == "llm" else None,
                recognition=mode,
            )
            return RetrievalCore(
                config=cfg, dense=None, sparse=None, rerank_stage=None, graph=graph
            )
        sparse = ListRetriever(rankings, source="bm25") if cfg.query.bm25 else None
        dense = ListRetriever(rankings, source="dense") if cfg.query.dense else None
        stage = RerankStage(FakeRerank(scores=rerank_scores)) if cfg.query.rerank else None
        return RetrievalCore(config=cfg, dense=dense, sparse=sparse, rerank_stage=stage)

    return AblationRunner(
        core_factory=core_factory,
        claude=GraphEchoClaude(),
        store=RunStore(tmp_path / "runs.db"),
        data_dir=data_dir,
        trace_store=TraceStore(tmp_path / "traces.sqlite3"),
    )


def _misaligned_corpus(tmp_path: Path) -> Path:
    """write_graph_corpus, but with broken gold passage_ids (the graph artifact is
    still built from the correct chunks, so PPR still lands on the gold NODE)."""
    corpus_dir = tmp_path / "corpora" / "graph-harness"
    raw = corpus_dir / "raw"
    raw.mkdir(parents=True)
    queries = build_misaligned_graph_queries()
    (raw / "queries.jsonl").write_text("\n".join(json.dumps(q) for q in queries) + "\n")
    ids = [q["query_id"] for q in queries]
    (raw / "slice-full.json").write_text(json.dumps(ids))
    (raw / "slice-smoke.json").write_text(json.dumps(ids[:15]))
    write_graph_artifact(corpus_dir)
    (corpus_dir / "manifest.json").write_text(
        json.dumps(
            {
                "corpus_id": "graph-harness",
                "dataset": {"name": "musique", "hf_id": "fixture", "split": "x", "revision": "0"},
                "index_hashes": {
                    "dense_contextual": "sha256:c",
                    "dense_isolated": "sha256:i",
                    "sparse": "sha256:s",
                    "graph": "sha256:g",
                },
                "n_queries": len(queries),
            }
        )
    )
    return tmp_path


def _metrics(doc: dict, preset: str) -> dict:
    for env in doc["receipts"]:
        if env["receipt"]["preset"] == preset:
            return env["receipt"]["metrics"]
    raise AssertionError(f"no receipt for {preset!r}")


def test_graph_recall_moves_vs_rerank(tmp_path: Path) -> None:
    runner = make_graph_runner(tmp_path)
    doc = runner.run(
        run_id="gs",
        corpus_id="graph-harness",
        slice_name="smoke",
        presets=["rerank", "graph"],
        spend_cap_usd=5.0,
    )
    rerank_r5 = _metrics(doc, "rerank")["recall_at_5"]
    graph_r5 = _metrics(doc, "graph")["recall_at_5"]
    assert rerank_r5 == pytest.approx(0.0)  # weak rerank cell: gold outside top-5
    assert graph_r5 == pytest.approx(1.0)  # PPR + dense land on the gold passages
    assert graph_r5 > rerank_r5  # the graph receipt CAN fail


def test_misaligned_graph_golds_score_zero(tmp_path: Path) -> None:
    ok = make_graph_runner(tmp_path / "ok")
    doc_ok = ok.run(
        run_id="a",
        corpus_id="graph-harness",
        slice_name="smoke",
        presets=["graph"],
        spend_cap_usd=5.0,
    )
    bad = make_graph_runner(tmp_path / "bad", misaligned=True)
    doc_bad = bad.run(
        run_id="b",
        corpus_id="graph-harness",
        slice_name="smoke",
        presets=["graph"],
        spend_cap_usd=5.0,
    )
    assert _metrics(doc_ok, "graph")["recall_at_5"] == pytest.approx(1.0)
    assert _metrics(doc_bad, "graph")["recall_at_5"] == pytest.approx(0.0)
