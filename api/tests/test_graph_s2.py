from ragreceipts.agents.schemas import FinalAnswer, GradeResult, RouteDecision, SubQueries
from ragreceipts.types import RouteMode
from tests.test_graph_s1 import run


def test_complex_routes_to_s2_two_hops(tmp_path):
    script = [
        RouteDecision(route="complex", confidence=0.9),
        SubQueries(items=["who directed Film X", "what else did that director direct"]),
        GradeResult(verdict="sufficient"),
        GradeResult(verdict="sufficient"),
        FinalAnswer(text="Director Y [1]; also Film Z [3]", citations=[1, 3]),
    ]
    out, store, core, _ = run(tmp_path, script, RouteMode.AUTO, query="multi-hop?")
    assert out["chosen_system"] == "s2"
    assert out["hops_used"] == 2
    assert core.queries == ["who directed Film X", "what else did that director direct"]
    assert [e.node for e in store.get("t-1")] == [
        "route",
        "decompose",
        "retrieve_hop",
        "grade",
        "retrieve_hop",
        "grade",
        "synthesize",
    ]
    assert out["final"].unresolved_subqueries == []
    assert out["budget_exhausted"] is False


def test_low_confidence_escalates_to_s2(tmp_path):
    script = [
        RouteDecision(route="simple", confidence=0.4),  # below 0.7 threshold
        SubQueries(items=["sq1"]),
        GradeResult(verdict="sufficient"),
        FinalAnswer(text="answer [1]", citations=[1]),
    ]
    out, store, _, _ = run(tmp_path, script, RouteMode.AUTO)
    assert out["chosen_system"] == "s2"
    assert store.get("t-1")[1].node == "decompose"  # route -> decompose


def test_force_s2_skips_route(tmp_path):
    script = [
        SubQueries(items=["sq1"]),
        GradeResult(verdict="sufficient"),
        FinalAnswer(text="a [1]", citations=[1]),
    ]
    out, store, _, claude = run(tmp_path, script, RouteMode.FORCE_S2)
    assert store.get("t-1")[0].node == "decompose"
    assert claude.calls[0]["output_format"] == "SubQueries"


def test_decompose_truncated_to_max_hops(tmp_path):
    script = [
        SubQueries(items=["a", "b", "c", "d", "e"]),
        GradeResult(verdict="sufficient"),
        GradeResult(verdict="sufficient"),
        GradeResult(verdict="sufficient"),
        FinalAnswer(text="x [1]", citations=[1]),
    ]
    out, store, _, _ = run(tmp_path, script, RouteMode.FORCE_S2)
    assert out["subqueries"] == ["a", "b", "c"]  # S2_MAX_HOPS = 3
    assert out["hops_used"] == 3
    assert store.get("t-1")[0].payload["truncated"] is True


def test_empty_decomposition_goes_straight_to_synthesize(tmp_path):
    script = [
        SubQueries(items=[]),
        FinalAnswer(text="No retrievable sub-questions.", abstained=True),
    ]
    out, store, core, _ = run(tmp_path, script, RouteMode.FORCE_S2)
    assert core.queries == []  # nothing retrieved
    assert out["final"].abstained is True
    assert [e.node for e in store.get("t-1")] == ["decompose", "synthesize"]
