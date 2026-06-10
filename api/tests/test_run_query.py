import dataclasses

from ragreceipts.agents.schemas import (
    FinalAnswer,
    GradeResult,
    RouteDecision,
    SubQueries,
)
from ragreceipts.agents.service import GraphResult, route_counts, run_query
from ragreceipts.config import PRESETS
from ragreceipts.traces.store import TraceStore
from ragreceipts.types import RouteMode
from tests.fakes import FakeClaude, FakeCore, make_chunk


def force_s2(config):
    return dataclasses.replace(
        config, query=dataclasses.replace(config.query, route_mode=RouteMode.FORCE_S2)
    )


def test_run_query_s1_result_and_trace(tmp_path):
    store = TraceStore(tmp_path / "t.sqlite3")
    core = FakeCore()
    claude = FakeClaude(
        script=[
            RouteDecision(route="simple", confidence=0.9),
            FinalAnswer(text="Paris [1]", citations=[1]),
        ]
    )
    result = run_query(
        query="capital of France?",
        core=core,
        claude=claude,
        store=store,
        config=PRESETS["router-on"],
    )
    assert result.system == "s1"
    assert result.final.text == "Paris [1]"
    assert [s.chunk.chunk_id for s in result.retrieved] == ["d1:0", "d1:1"]
    events = store.get(result.trace_id)
    assert [e.node for e in events] == ["route", "s1_retrieve", "s1_answer"]
    assert [e.seq for e in events] == [0, 1, 2]
    assert result.tokens_used == 30  # 2 calls x FakeClaude default 10 in / 5 out tokens (R5)


def test_run_query_s2_union_of_hops_dedupes(tmp_path):
    store = TraceStore(tmp_path / "t.sqlite3")
    a, b, c = make_chunk(0), make_chunk(1), make_chunk(2)
    core = FakeCore(by_query={"sq1": [a, b], "sq2": [b, c]})
    claude = FakeClaude(
        script=[
            SubQueries(items=["sq1", "sq2"]),
            GradeResult(verdict="sufficient"),
            GradeResult(verdict="sufficient"),
            FinalAnswer(text="x [1][3]", citations=[1, 3]),
        ]
    )
    result = run_query(
        query="multi?",
        core=core,
        claude=claude,
        store=store,
        config=force_s2(PRESETS["router-on"]),
    )
    assert result.system == "s2"
    # union of per-hop top-k, deduped (b appears once), first-seen order
    assert [s.chunk.chunk_id for s in result.retrieved] == ["d1:0", "d1:1", "d1:2"]
    assert result.hops_used == 2


def test_trace_ids_are_distinct_per_query(tmp_path):
    store = TraceStore(tmp_path / "t.sqlite3")
    script = [
        RouteDecision(route="simple", confidence=0.9),
        FinalAnswer(text="a [1]", citations=[1]),
        RouteDecision(route="simple", confidence=0.9),
        FinalAnswer(text="b [1]", citations=[1]),
    ]
    claude = FakeClaude(script=script)
    r1 = run_query(
        query="q1", core=FakeCore(), claude=claude, store=store, config=PRESETS["router-on"]
    )
    r2 = run_query(
        query="q2", core=FakeCore(), claude=claude, store=store, config=PRESETS["router-on"]
    )
    assert r1.trace_id != r2.trace_id
    assert len(store.get(r1.trace_id)) == 3 and len(store.get(r2.trace_id)) == 3


def test_core_factory_receives_per_query_recorder(tmp_path):
    # R9: on_trace is wired at RetrievalCore CONSTRUCTION. run_query accepts a
    # per-query factory, calls it with this query's TraceRecorder, and the core
    # built with on_trace=recorder lands intra-retrieval events on the same trace.
    store = TraceStore(tmp_path / "t.sqlite3")
    captured = []

    class TracingCore(FakeCore):
        """Stands in for RetrievalCore(config, dense, sparse, stage, on_trace=...)."""

        def __init__(self, *, on_trace):
            super().__init__()
            self._on_trace = on_trace

        def retrieve(self, query):
            self._on_trace({"node": "s1_retrieve", "payload": {"stage": "inner", "query": query}})
            return super().retrieve(query)

    def per_query_factory(recorder):
        captured.append(recorder)
        return TracingCore(on_trace=recorder)

    claude = FakeClaude(script=[FinalAnswer(text="a [1]", citations=[1])])
    result = run_query(
        query="q",
        core=per_query_factory,
        claude=claude,
        store=store,
        config=PRESETS["rerank"],
    )  # force_s1 preset
    assert captured[0].trace_id == result.trace_id
    events = store.get(result.trace_id)
    assert [(e.seq, e.node) for e in events] == [
        (0, "s1_retrieve"),  # the core's inner event, stamped by the recorder
        (1, "s1_retrieve"),  # the graph node's own event
        (2, "s1_answer"),
    ]
    assert events[0].payload == {"stage": "inner", "query": "q"}


def test_route_counts():
    def fake_result(system: str) -> GraphResult:
        return GraphResult(
            final=FinalAnswer(text="x"),
            system=system,
            trace_id="t",
            tokens_used=0,
            hops_used=0,
            retrieved=[],
        )

    assert route_counts([fake_result("s1"), fake_result("s2"), fake_result("s2")]) == {
        "n_s1": 1,
        "n_s2": 2,
    }
    assert route_counts([]) == {"n_s1": 0, "n_s2": 0}
