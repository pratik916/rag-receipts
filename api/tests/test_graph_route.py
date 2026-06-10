"""G2: the router's third route reaches graph_retrieve, gated to multi-hop corpora.

Offline: FakeClaude scripts route='graph'; a FakeCore-style graph double records the
retrieval. When no graph retriever is injected (off-corpus), a graph decision falls
back to s1."""

from ragreceipts.agents.prompts import PROMPTS_VERSION
from ragreceipts.agents.schemas import FinalAnswer, RouteDecision
from ragreceipts.agents.service import run_query
from ragreceipts.config import PRESETS
from ragreceipts.traces.store import TraceStore
from tests.fakes import FakeClaude, FakeCore, make_chunk


def test_prompts_version_bumped_to_p2():
    assert PROMPTS_VERSION == "2026-06-11.p2"


def test_route_decision_accepts_graph():
    d = RouteDecision(route="graph", confidence=0.9)
    assert d.route == "graph"


def test_graph_route_reaches_graph_retrieve_node(tmp_path):
    store = TraceStore(tmp_path / "t.sqlite3")
    core = FakeCore()  # S1/S2 retrieval double (unused on the graph path)
    graph_core = FakeCore(by_query={"who founded the company X?": [make_chunk(0, doc="gp")]})
    claude = FakeClaude(
        script=[
            RouteDecision(route="graph", confidence=0.92),
            FinalAnswer(text="Answer [1]", citations=[1]),
        ]
    )
    result = run_query(
        query="who founded the company X?",
        core=core,
        claude=claude,
        store=store,
        config=PRESETS["router-on"],
        graph_retriever=graph_core,
    )
    assert result.system == "graph"
    # the graph retriever, not the S1/S2 core, served the query
    assert graph_core.queries == ["who founded the company X?"]
    assert core.queries == []
    nodes = [e.node for e in store.get(result.trace_id)]
    assert "graph_retrieve" in nodes
    assert "synthesize" in nodes


def test_graph_decision_without_graph_retriever_falls_back_to_s1(tmp_path):
    store = TraceStore(tmp_path / "t.sqlite3")
    core = FakeCore(by_query={"who founded X?": [make_chunk(0, doc="p0")]})
    claude = FakeClaude(
        script=[
            RouteDecision(route="graph", confidence=0.92),
            FinalAnswer(text="Answer [1]", citations=[1]),
        ]
    )
    # no graph_retriever -> off-corpus gating: graph decision falls back to s1
    result = run_query(
        query="who founded X?",
        core=core,
        claude=claude,
        store=store,
        config=PRESETS["router-on"],
    )
    assert result.system == "s1"
    assert core.queries == ["who founded X?"]
    nodes = [e.node for e in store.get(result.trace_id)]
    assert "graph_retrieve" not in nodes
    assert "s1_retrieve" in nodes
