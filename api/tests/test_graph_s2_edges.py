from ragreceipts.agents.schemas import FinalAnswer, GradeResult, SubQueries
from ragreceipts.types import RouteMode
from tests.test_graph_s1 import run


def test_insufficient_triggers_refine_loop(tmp_path):
    script = [
        SubQueries(items=["sq1"]),
        GradeResult(verdict="insufficient"),
        "sq1 rewritten with entities",  # refine goes through complete()
        GradeResult(verdict="sufficient"),
        FinalAnswer(text="a [1]", citations=[1]),
    ]
    out, store, core, _ = run(tmp_path, script, RouteMode.FORCE_S2)
    assert core.queries == ["sq1", "sq1 rewritten with entities"]
    assert [e.node for e in store.get("t-1")] == [
        "decompose",
        "retrieve_hop",
        "grade",
        "refine",
        "retrieve_hop",
        "grade",
        "synthesize",
    ]
    assert out["final"].unresolved_subqueries == []
    refine_event = store.get("t-1")[3]
    assert refine_event.payload == {"original": "sq1", "refined": "sq1 rewritten with entities"}


def test_contradictory_re_retrieves_once_then_flags(tmp_path):
    script = [
        SubQueries(items=["sq1"]),
        GradeResult(verdict="contradictory"),
        GradeResult(verdict="contradictory"),
        # Model "forgets" to set the flag — the graph must enforce it from state.
        FinalAnswer(text="Source A says 1990 [1]; source B says 1992 [2]", citations=[1, 2]),
    ]
    out, store, core, _ = run(tmp_path, script, RouteMode.FORCE_S2)
    assert core.queries == ["sq1", "sq1"]  # exactly ONE re-retrieve
    assert out["final"].contradiction_flag is True  # state-enforced
    assert [e.node for e in store.get("t-1")] == [
        "decompose",
        "retrieve_hop",
        "grade",
        "retrieve_hop",
        "grade",
        "synthesize",
    ]
    synth = store.get("t-1")[-1]
    assert synth.payload["contradiction_flag"] is True  # flagged in trace too


def test_hop_budget_exhaustion_yields_caveated_synthesis(tmp_path):
    script = [
        SubQueries(items=["sq1", "sq2", "sq3"]),
        GradeResult(verdict="sufficient"),  # hop 1 (hops_used=1) ok
        GradeResult(verdict="insufficient"),  # hop 2 (hops_used=2) weak -> refine
        "sq2 rewritten",
        GradeResult(verdict="insufficient"),  # hops_used=3 == max -> stop
        FinalAnswer(text="partial answer [1]", citations=[1]),
    ]
    out, store, _, _ = run(tmp_path, script, RouteMode.FORCE_S2)
    assert out["hops_used"] == 3
    assert out["budget_exhausted"] is True
    # sq2 (original phrasing, not the refined one) and never-reached sq3 disclosed.
    assert out["final"].unresolved_subqueries == ["sq2", "sq3"]
    assert store.get("t-1")[-1].payload["budget_exhausted"] is True


def test_token_ceiling_exhaustion(tmp_path):
    script = [
        (SubQueries(items=["sq1", "sq2"]), 50, 10),  # tokens 60 < 100 -> proceed
        (GradeResult(verdict="sufficient"), 80, 20),  # tokens 160 >= 100 -> stop
        (FinalAnswer(text="partial [1]", citations=[1]), 10, 10),
    ]
    out, _, core, _ = run(tmp_path, script, RouteMode.FORCE_S2, token_ceiling=100)
    assert core.queries == ["sq1"]  # sq2 never retrieved
    assert out["budget_exhausted"] is True
    assert out["final"].unresolved_subqueries == ["sq2"]


def test_token_ceiling_before_first_hop(tmp_path):
    script = [
        (SubQueries(items=["sq1", "sq2"]), 90, 20),  # tokens 110 >= 100 at decompose
        (FinalAnswer(text="cannot pursue sub-queries", abstained=True), 5, 5),
    ]
    out, store, core, _ = run(tmp_path, script, RouteMode.FORCE_S2, token_ceiling=100)
    assert core.queries == []
    assert out["budget_exhausted"] is True
    assert out["final"].unresolved_subqueries == ["sq1", "sq2"]
    assert [e.node for e in store.get("t-1")] == ["decompose", "synthesize"]
