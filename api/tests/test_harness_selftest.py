"""Harness self-test (spec §Testing, 'the on-brand one'). CI-enforced.

Runs Plan A's REAL RetrievalCore + RerankStage with FakeRerank through the
Plan B runner on the in-repo fixture corpus. Two properties must hold forever:
  1. flipping the rerank flag changes Recall@5 (0.75 -> 1.0 here);
  2. a deliberately misaligned gold mapping scores 0.0 while the aligned
     mapping scores 1.0 - the alignment rule is load-bearing.
"""

from pathlib import Path

import pytest

from ragreceipts.eval.run_state import RunStore
from ragreceipts.eval.runner import AblationRunner, S1Answer
from ragreceipts.retrieval.core import RetrievalCore
from ragreceipts.retrieval.rerank import RerankStage
from ragreceipts.vendors.base import ParsedResult
from tests.fakes import FakeRerank  # tests/ is a package (R8)
from tests.harness_fixtures import (
    ListRetriever,
    build_harness_fixture,
    write_harness_corpus,
)


class EchoClaude:
    """ClaudeTransport stub answering each fixture question with its gold answer."""

    def complete(self, *, model, system, user, max_tokens, temperature=0.0):
        raise AssertionError("self-test synthesis uses parse(), not complete()")

    def parse(self, *, model, system, user, max_tokens, output_format, temperature=0.0):
        question = user.rsplit("Question: ", 1)[1]  # "harness question {i}?"
        i = question.split()[-1].rstrip("?")
        return ParsedResult(
            parsed=S1Answer(answer=f"gold answer {i}", abstained=False),
            input_tokens=500,
            output_tokens=50,
        )


def make_runner(tmp_path: Path, *, misaligned: bool = False) -> AblationRunner:
    fixture = build_harness_fixture(misaligned=misaligned)
    write_harness_corpus(tmp_path, "harness", fixture["queries"])

    def core_factory(cfg) -> RetrievalCore:
        sparse = ListRetriever(fixture["rankings"], source="bm25")
        dense = ListRetriever(fixture["rankings"], source="dense")
        # R5 final constructor: FakeRerank(script=None, scores=None, fail=False);
        # the text-keyed scores mode exists exactly for this fixture.
        stage = RerankStage(FakeRerank(scores=fixture["rerank_scores"]))
        return RetrievalCore(
            config=cfg,
            dense=dense if cfg.query.dense else None,
            sparse=sparse if cfg.query.bm25 else None,
            rerank_stage=stage if cfg.query.rerank else None,
        )

    return AblationRunner(
        core_factory=core_factory,
        claude=EchoClaude(),
        store=RunStore(tmp_path / "runs.db"),
        data_dir=tmp_path,
    )


def metrics_for(doc: dict, preset: str) -> dict:
    for env in doc["receipts"]:
        if env["receipt"]["preset"] == preset:
            return env["receipt"]["metrics"]
    raise AssertionError(f"no receipt for preset {preset!r} in run doc")


def test_rerank_flip_provably_changes_recall_at_5(tmp_path: Path) -> None:
    runner = make_runner(tmp_path)
    doc = runner.run(
        run_id="selftest",
        corpus_id="harness",
        slice_name="smoke",
        presets=["contextual", "rerank"],
        spend_cap_usd=5.0,
    )
    off = metrics_for(doc, "contextual")  # bm25+dense+RRF, rerank OFF
    on = metrics_for(doc, "rerank")  # same + rerank ON
    assert off["recall_at_5"] == pytest.approx(0.75)
    assert on["recall_at_5"] == pytest.approx(1.0)
    assert on["recall_at_5"] > off["recall_at_5"]  # the receipt CAN fail
    # MRR@3 moves too: q0 has no top-3 hit without rerank
    assert off["mrr_at_3"] == pytest.approx(0.75)
    assert on["mrr_at_3"] == pytest.approx(1.0)
    # and the rerank cell carries its anchors with direction computed
    rerank_receipt = doc["receipts"][1]["receipt"]
    assert rerank_receipt["preset"] == "rerank"
    assert len(rerank_receipt["anchors"]) == 2
    assert all(a["direction_match"] is True for a in rerank_receipt["anchors"])


def test_full_ladder_runs_offline_with_disclosed_skip(tmp_path: Path) -> None:
    runner = make_runner(tmp_path)
    doc = runner.run(
        run_id="ladder",
        corpus_id="harness",
        slice_name="smoke",
        presets=["bm25-only", "dense-rrf", "contextual", "rerank", "router-on"],
        spend_cap_usd=5.0,
    )
    assert [e["receipt"]["preset"] for e in doc["receipts"]] == [
        "bm25-only",
        "dense-rrf",
        "contextual",
        "rerank",
    ]
    assert doc["skipped"] == [
        {
            "preset": "router-on",
            "reason": (
                "skipped: requires Plan C (LangGraph System-2 is not built "
                "yet; only route_mode=force_s1 cells are runnable)"
            ),
        }
    ]
    assert (tmp_path / "receipts-local" / "ladder.json").exists()


def test_misaligned_golds_provably_score_zero(tmp_path: Path) -> None:
    aligned = make_runner(tmp_path / "aligned")
    doc_ok = aligned.run(
        run_id="a",
        corpus_id="harness",
        slice_name="smoke",
        presets=["rerank"],
        spend_cap_usd=5.0,
    )
    misaligned = make_runner(tmp_path / "broken", misaligned=True)
    doc_bad = misaligned.run(
        run_id="b",
        corpus_id="harness",
        slice_name="smoke",
        presets=["rerank"],
        spend_cap_usd=5.0,
    )
    assert metrics_for(doc_ok, "rerank")["recall_at_5"] == pytest.approx(1.0)
    assert metrics_for(doc_bad, "rerank")["recall_at_5"] == pytest.approx(0.0)
    assert metrics_for(doc_bad, "rerank")["mrr_at_3"] == pytest.approx(0.0)
    # answer-level metrics are alignment-independent and stay perfect - the
    # zero comes from the alignment rule alone, not from a broken pipeline
    assert metrics_for(doc_bad, "rerank")["em"] == pytest.approx(1.0)
