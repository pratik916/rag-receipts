"""Router-on eval integration: the graph-driven runner produces route stats.

Offline: FakeClaude scripts every Claude call; FakeCore replaces retrieval;
the corpus fixture is written in Spike 0's raw/ layout (R1/R2).
"""

import json
from pathlib import Path

import pytest

from ragreceipts.agents.prompts import PROMPTS_VERSION
from ragreceipts.agents.schemas import (
    FinalAnswer,
    GradeResult,
    RouteDecision,
    SubQueries,
)
from ragreceipts.eval.run_state import RunStore
from ragreceipts.eval.runner import AblationRunner
from ragreceipts.traces.store import TraceStore
from tests.fakes import FakeClaude, FakeCore, make_chunk


def write_corpus(tmp_path: Path, dataset_name: str = "musique") -> Path:
    """R1 raw layout: raw/{queries.jsonl, slice-*.json} + manifest.json."""
    raw = tmp_path / "corpora" / "c1" / "raw"
    raw.mkdir(parents=True)
    records = [
        {
            "query_id": "q0",
            "question": "question 0?",
            "answer": "answer zero",
            "answer_aliases": [],
            "gold": {"type": "passage", "passage_ids": ["p0"]},
        },
        {
            "query_id": "q1",
            "question": "question 1?",
            "answer": "answer two",
            "answer_aliases": [],
            "gold": {"type": "passage", "passage_ids": ["p1"]},
        },
    ]
    (raw / "queries.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n")
    (raw / "slice-full.json").write_text(json.dumps(["q0", "q1"]))
    (raw / "slice-smoke.json").write_text(json.dumps(["q0", "q1"]))
    (tmp_path / "corpora" / "c1" / "manifest.json").write_text(
        json.dumps(
            {
                "corpus_id": "c1",
                "dataset": {"name": dataset_name, "hf_id": "x", "split": "dev", "revision": "r"},
                "index_hashes": {
                    "dense_contextual": "sha256:c",
                    "dense_isolated": "sha256:i",
                    "sparse": "sha256:s",
                },
                "n_queries": 2,
            }
        )
    )
    return tmp_path


def make_runner(
    tmp_path: Path, claude: FakeClaude, dataset_name: str = "musique"
) -> AblationRunner:
    data_dir = write_corpus(tmp_path, dataset_name)
    core = FakeCore(
        by_query={
            "question 0?": [make_chunk(0, doc="p0")],  # S1: gold p0 at rank 1
            "hop one": [make_chunk(0, doc="p1"), make_chunk(1, doc="f1")],
            "hop two": [make_chunk(0, doc="f2")],  # union still holds gold p1
        }
    )
    return AblationRunner(
        core_factory=lambda cfg: core,
        claude=claude,
        store=RunStore(tmp_path / "runs.db"),
        data_dir=data_dir,
        trace_store=TraceStore(tmp_path / "traces.sqlite3"),
    )


def router_on_script() -> list:
    return [
        # q0 -> S1
        RouteDecision(route="simple", confidence=0.95),
        FinalAnswer(text="answer one [1]", citations=[1]),
        # q1 -> S2, two hops, both sufficient
        RouteDecision(route="complex", confidence=0.9),
        SubQueries(items=["hop one", "hop two"]),
        GradeResult(verdict="sufficient"),
        GradeResult(verdict="sufficient"),
        FinalAnswer(text="answer two [1][3]", citations=[1, 3]),
    ]


def test_router_on_preset_produces_route_stats(tmp_path: Path) -> None:
    runner = make_runner(tmp_path, FakeClaude(script=router_on_script()))
    doc = runner.run(
        run_id="r1",
        corpus_id="c1",
        slice_name="smoke",
        presets=["router-on"],
        spend_cap_usd=5.0,
    )
    assert doc["skipped"] == []  # the temporary skip is gone
    receipt = doc["receipts"][0]["receipt"]
    assert receipt["preset"] == "router-on"
    m = receipt["metrics"]
    assert m["n_s1"] + m["n_s2"] == receipt["n_total"]
    assert m["n_s2"] >= 1
    assert m.get("union_of_hops") is True
    assert m["recall_at_5"] is None  # ill-defined across hops
    assert m["mrr_at_3"] is None
    assert m["recall_union_of_hops"] == pytest.approx(1.0)
    assert m["usd_per_query"] > 0  # R10: priced from TraceEvents
    assert all("route" in pq for pq in receipt["per_query"])
    assert {pq["route"] for pq in receipt["per_query"]} == {"s1", "s2"}
    assert receipt["prompts_version"] == PROMPTS_VERSION  # R11


def test_router_on_still_skips_on_single_hop_corpus(tmp_path: Path) -> None:
    # R10: the MULTI_HOP_DATASETS gate is permanent — nq corpora never run AUTO.
    runner = make_runner(tmp_path, FakeClaude(script=[]), dataset_name="nq")
    doc = runner.run(
        run_id="r1",
        corpus_id="c1",
        slice_name="smoke",
        presets=["router-on"],
        spend_cap_usd=5.0,
    )
    assert doc["receipts"] == []
    assert doc["skipped"][0]["preset"] == "router-on"
    assert "multi-hop" in doc["skipped"][0]["reason"]
