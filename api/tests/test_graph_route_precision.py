"""G4: graph_route_precision is a router-on-only diagnostic, kept separate from the
headline retrieval metrics. Offline: FakeClaude scripts a graph route for one query;
a FakeCore graph double serves a gold hit for it."""

import json
from pathlib import Path

import pytest

from ragreceipts.agents.schemas import FinalAnswer, RouteDecision
from ragreceipts.eval.run_state import RunStore
from ragreceipts.eval.runner import AblationRunner
from ragreceipts.traces.store import TraceStore
from tests.fakes import FakeClaude, FakeCore, make_chunk


def write_corpus(tmp_path: Path) -> Path:
    raw = tmp_path / "corpora" / "c1" / "raw"
    raw.mkdir(parents=True)
    records = [
        {
            "query_id": "q0",
            "question": "simple q?",
            "answer": "a0",
            "answer_aliases": [],
            "gold": {"type": "passage", "passage_ids": ["p0"]},
        },
        {
            "query_id": "q1",
            "question": "graph entity chain?",
            "answer": "a1",
            "answer_aliases": [],
            "gold": {"type": "passage", "passage_ids": ["pg"]},
        },
    ]
    (raw / "queries.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n")
    (raw / "slice-full.json").write_text(json.dumps(["q0", "q1"]))
    (raw / "slice-smoke.json").write_text(json.dumps(["q0", "q1"]))
    (tmp_path / "corpora" / "c1" / "manifest.json").write_text(
        json.dumps(
            {
                "corpus_id": "c1",
                "dataset": {"name": "musique", "hf_id": "x", "split": "d", "revision": "r"},
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


def make_runner(tmp_path: Path) -> AblationRunner:
    data_dir = write_corpus(tmp_path)
    s1_core = FakeCore(by_query={"simple q?": [make_chunk(0, doc="p0")]})
    graph_core = FakeCore(by_query={"graph entity chain?": [make_chunk(0, doc="pg")]})
    claude = FakeClaude(
        script=[
            # q0 -> simple -> s1
            RouteDecision(route="simple", confidence=0.95),
            FinalAnswer(text="a0 [1]", citations=[1]),
            # q1 -> graph -> graph_retrieve (gold pg in top-k)
            RouteDecision(route="graph", confidence=0.93),
            FinalAnswer(text="a1 [1]", citations=[1]),
        ]
    )
    return AblationRunner(
        core_factory=lambda cfg: s1_core,
        claude=claude,
        store=RunStore(tmp_path / "runs.db"),
        data_dir=data_dir,
        trace_store=TraceStore(tmp_path / "traces.sqlite3"),
        graph_factory=lambda cfg: graph_core,
    )


def test_graph_route_precision_is_reported_separately(tmp_path: Path) -> None:
    runner = make_runner(tmp_path)
    doc = runner.run(
        run_id="r1",
        corpus_id="c1",
        slice_name="smoke",
        presets=["router-on"],
        spend_cap_usd=5.0,
    )
    receipt = doc["receipts"][0]["receipt"]
    m = receipt["metrics"]
    # the graph route fired for exactly one query, and it hit its gold
    assert m["n_graph"] == 1
    assert m["n_graph_routed"] == 1
    assert m["graph_route_precision"] == pytest.approx(1.0)
    # headline retrieval metrics stay null (router-on has an s1 + a graph row,
    # union-of-hops nulling applies because the receipt mixes routes) — the
    # diagnostic is NOT one of them
    assert "graph_route_precision" in m and "recall_at_5" in m
    assert m["recall_at_5"] is None
    # per-query route is persisted as "graph" for q1
    routes = {pq["query_id"]: pq["route"] for pq in receipt["per_query"]}
    assert routes["q1"] == "graph"
    assert routes["q0"] == "s1"


def test_graph_route_precision_absent_when_no_graph_route(tmp_path: Path) -> None:
    data_dir = write_corpus(tmp_path)
    s1_core = FakeCore(
        by_query={
            "simple q?": [make_chunk(0, doc="p0")],
            "graph entity chain?": [make_chunk(0, doc="f")],
        }
    )
    claude = FakeClaude(
        script=[
            RouteDecision(route="simple", confidence=0.95),
            FinalAnswer(text="a0 [1]", citations=[1]),
            RouteDecision(route="simple", confidence=0.95),
            FinalAnswer(text="a1 [1]", citations=[1]),
        ]
    )
    runner = AblationRunner(
        core_factory=lambda cfg: s1_core,
        claude=claude,
        store=RunStore(tmp_path / "runs.db"),
        data_dir=data_dir,
        trace_store=TraceStore(tmp_path / "traces.sqlite3"),
    )
    doc = runner.run(
        run_id="r2",
        corpus_id="c1",
        slice_name="smoke",
        presets=["router-on"],
        spend_cap_usd=5.0,
    )
    m = doc["receipts"][0]["receipt"]["metrics"]
    assert m["n_graph"] == 0
    assert m["n_graph_routed"] == 0
    assert m["graph_route_precision"] is None  # nothing routed to graph
