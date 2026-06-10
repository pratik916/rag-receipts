from ragreceipts.agents.graph import build_graph, initial_state
from ragreceipts.agents.schemas import FinalAnswer, RouteDecision
from ragreceipts.traces.recorder import TraceRecorder
from ragreceipts.traces.store import TraceStore
from ragreceipts.types import RouteMode
from tests.fakes import FakeClaude, FakeCore


def run(
    tmp_path, script, route_mode, query="what is the capital of France?", core=None, **graph_kwargs
):
    """Build graph with fakes, invoke once, return (final_state, store, core, claude)."""
    store = TraceStore(tmp_path / "t.sqlite3")
    recorder = TraceRecorder(store, "t-1")
    core = core or FakeCore()
    claude = FakeClaude(script=script)
    graph = build_graph(
        core=core, claude=claude, recorder=recorder, route_mode=route_mode, **graph_kwargs
    )
    out = graph.invoke(initial_state(query), config={"recursion_limit": 50})
    return out, store, core, claude


def test_simple_routes_to_s1(tmp_path):
    out, store, core, claude = run(
        tmp_path,
        [
            RouteDecision(route="simple", confidence=0.95),
            FinalAnswer(text="Paris [1]", citations=[1]),
        ],
        RouteMode.AUTO,
    )
    assert out["chosen_system"] == "s1"
    assert out["final"].text == "Paris [1]"
    assert core.queries == ["what is the capital of France?"]
    assert [e.node for e in store.get("t-1")] == ["route", "s1_retrieve", "s1_answer"]
    # route used Haiku, answer used Sonnet (contract model split)
    assert claude.calls[0]["model"] == "claude-haiku-4-5-20251001"
    assert claude.calls[1]["model"] == "claude-sonnet-4-6"


def test_force_s1_skips_route_node(tmp_path):
    out, store, _, claude = run(
        tmp_path, [FinalAnswer(text="Paris [1]", citations=[1])], RouteMode.FORCE_S1
    )
    assert [e.node for e in store.get("t-1")] == ["s1_retrieve", "s1_answer"]
    assert claude.calls[0]["output_format"] == "FinalAnswer"  # no route call happened


def test_s1_abstention_is_structured(tmp_path):
    out, store, _, _ = run(
        tmp_path,
        [
            RouteDecision(route="simple", confidence=0.9),
            FinalAnswer(text="The context does not mention this.", abstained=True),
        ],
        RouteMode.AUTO,
    )
    assert out["final"].abstained is True
    assert out["final"].citations == []
    assert store.get("t-1")[-1].payload["abstained"] is True  # surfaced in trace too


def test_tokens_accumulate_across_calls(tmp_path):
    out, _, _, _ = run(
        tmp_path,
        [
            (RouteDecision(route="simple", confidence=0.9), 100, 20),
            (FinalAnswer(text="Paris [1]", citations=[1]), 300, 50),
        ],
        RouteMode.AUTO,
    )
    assert out["tokens_used"] == 470


def test_trace_events_carry_chunk_scores(tmp_path):
    _, store, _, _ = run(
        tmp_path, [FinalAnswer(text="Paris [1]", citations=[1])], RouteMode.FORCE_S1
    )
    retrieve_event = store.get("t-1")[0]
    assert retrieve_event.node == "s1_retrieve"
    assert retrieve_event.payload["chunks"][0]["chunk_id"] == "d1:0"
    assert "score" in retrieve_event.payload["chunks"][0]
