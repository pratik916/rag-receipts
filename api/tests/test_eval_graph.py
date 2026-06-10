"""G4: the runner produces a graph receipt; the recognition mini-ablation sweeps
graph_recognition in {llm, embedding}. Offline: FakeOpenIE/FakeEmbed + a real
fixture graph built by Plan E's build_graph_index + a tiny echo ClaudeTransport.

Real-code reconciliations (RG1/RG8): GraphRetriever takes the keyword-only RG1
ctor (chunks required); recognition goes through claude.parse(output_format=
SeedSelection); RetrievalCore takes rerank_stage=None explicitly. Graph USD is
synthesis-only (RG8): recognition Haiku tokens are NOT traced, so usd_per_query
captures the FinalAnswer (synthesis) spend.
"""

from pathlib import Path

import pytest

from ragreceipts.agents.schemas import FinalAnswer, RouteDecision
from ragreceipts.eval.run_state import RunStore
from ragreceipts.eval.runner import AblationRunner
from ragreceipts.ingest.graph_index import GraphIndex
from ragreceipts.retrieval.core import RetrievalCore
from ragreceipts.retrieval.graph import GraphRetriever, SeedSelection
from ragreceipts.traces.store import TraceStore
from ragreceipts.vendors.base import ParsedResult
from tests.fakes import FakeEmbed
from tests.graph_fixtures import (
    fixture_chunks,
    fixture_query_aliases,
    write_graph_corpus,
)


class GraphEchoClaude:
    """Tiny ClaudeTransport echo. Under FORCE_S1 the router never runs, so the
    only live parse is FinalAnswer (synthesis). Recognition (recognition='llm')
    calls parse(output_format=SeedSelection); returning an empty selection is a
    safe no-op — GraphRetriever's never-empty fallback keeps all seeds, so the
    fixture's gold-seeded PPR is untouched. RouteDecision is dead code under
    FORCE_S1 but answered for completeness."""

    def parse(self, *, model, system, user, max_tokens, output_format, temperature=0.0):
        if output_format is SeedSelection:
            # No-op recognition: empty -> fallback keeps the embedding seeds.
            return ParsedResult(
                parsed=SeedSelection(phrases=[]), input_tokens=900, output_tokens=80
            )
        if output_format is RouteDecision:
            return ParsedResult(
                parsed=RouteDecision(route="simple", confidence=0.95),
                input_tokens=50,
                output_tokens=10,
            )
        # FinalAnswer (synthesis): non-zero output tokens -> usd_per_query > 0.
        return ParsedResult(
            parsed=FinalAnswer(text="answer [1]", citations=[1]),
            input_tokens=400,
            output_tokens=40,
        )


def make_runner(tmp_path: Path, recognition: str) -> AblationRunner:
    data_dir = write_graph_corpus(tmp_path)
    graph_dir = data_dir / "corpora" / "graph-harness" / "graph"
    aliases = fixture_query_aliases()

    def core_factory(cfg) -> RetrievalCore:
        index = GraphIndex.load(graph_dir)
        mode = cfg.query.graph_recognition  # runner override flows through cfg
        graph = GraphRetriever(
            index,
            chunks=fixture_chunks(),
            embed=FakeEmbed(query_aliases=aliases),
            claude=GraphEchoClaude() if mode == "llm" else None,
            recognition=mode,
        )
        return RetrievalCore(config=cfg, dense=None, sparse=None, rerank_stage=None, graph=graph)

    return AblationRunner(
        core_factory=core_factory,
        claude=GraphEchoClaude(),
        store=RunStore(tmp_path / "runs.db"),
        data_dir=data_dir,
        trace_store=TraceStore(tmp_path / "traces.sqlite3"),
    )


def _metrics(doc: dict, preset: str) -> dict:
    for env in doc["receipts"]:
        if env["receipt"]["preset"] == preset:
            return env["receipt"]["metrics"]
    raise AssertionError(f"no receipt for {preset!r}")


def test_graph_preset_produces_receipt_with_populated_recall(tmp_path: Path) -> None:
    runner = make_runner(tmp_path, recognition="llm")
    doc = runner.run(
        run_id="g1",
        corpus_id="graph-harness",
        slice_name="smoke",
        presets=["graph"],
        spend_cap_usd=5.0,
    )
    assert doc["skipped"] == []
    m = _metrics(doc, "graph")
    # FORCE_S1 single retrieval -> recall/mrr stay populated (no union-of-hops nulling)
    assert m["recall_at_5"] is not None
    assert "union_of_hops" not in m
    assert m["recall_at_5"] == pytest.approx(1.0)  # PPR + dense land on the gold passages
    assert m["usd_per_query"] > 0  # synthesis-only USD (RG8: recognition is out-of-band)


def test_recognition_mini_ablation_runs_both_modes(tmp_path: Path) -> None:
    # embedding mode: no LLM recognition call -> still produces a graph receipt
    runner = make_runner(tmp_path, recognition="embedding")
    doc = runner.run(
        run_id="g2",
        corpus_id="graph-harness",
        slice_name="smoke",
        presets=["graph"],
        spend_cap_usd=5.0,
        graph_recognition="embedding",
    )
    m = _metrics(doc, "graph")
    assert m["recall_at_5"] is not None
    cfg = doc["receipts"][0]["receipt"]["config"]
    assert cfg["query"]["graph_recognition"] == "embedding"


def test_graph_rrf_preset_fuses_graph_with_hybrid(tmp_path: Path) -> None:
    data_dir = write_graph_corpus(tmp_path)
    # graph-rrf needs bm25+dense too; this offline test only asserts the runner
    # SKIPS nothing and writes a graph-rrf receipt when given a fused core.
    from tests.harness_fixtures import ListRetriever

    graph_dir = data_dir / "corpora" / "graph-harness" / "graph"
    aliases = fixture_query_aliases()

    rankings = {
        f"Where is Entity{i} located?": [c for c in fixture_chunks() if c.passage_id == f"g{i}"]
        for i in range(4)
    }

    def core_factory(cfg) -> RetrievalCore:
        index = GraphIndex.load(graph_dir)
        graph = GraphRetriever(
            index,
            chunks=fixture_chunks(),
            embed=FakeEmbed(query_aliases=aliases),
            claude=GraphEchoClaude() if cfg.query.graph_recognition == "llm" else None,
            recognition=cfg.query.graph_recognition,
        )
        sparse = ListRetriever(rankings, source="bm25") if cfg.query.bm25 else None
        dense = ListRetriever(rankings, source="dense") if cfg.query.dense else None
        return RetrievalCore(config=cfg, dense=dense, sparse=sparse, rerank_stage=None, graph=graph)

    runner = AblationRunner(
        core_factory=core_factory,
        claude=GraphEchoClaude(),
        store=RunStore(tmp_path / "runs.db"),
        data_dir=data_dir,
        trace_store=TraceStore(tmp_path / "traces.sqlite3"),
    )
    doc = runner.run(
        run_id="g3",
        corpus_id="graph-harness",
        slice_name="smoke",
        presets=["graph-rrf"],
        spend_cap_usd=5.0,
    )
    assert [e["receipt"]["preset"] for e in doc["receipts"]] == ["graph-rrf"]
    assert _metrics(doc, "graph-rrf")["recall_at_5"] is not None
