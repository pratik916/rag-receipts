"""RealQueryRunner: GraphResult -> QueryResult marshalling against the R9 pins."""

from ragreceipts.agents.schemas import FinalAnswer
from ragreceipts.agents.service import GraphResult
from ragreceipts.server.pipeline import RealQueryRunner
from ragreceipts.traces.models import TraceEvent
from ragreceipts.types import Chunk, ScoredChunk
from tests.fakes import InMemoryTraceStore


def _scored(doc_id: str, position: int, text: str, score: float) -> ScoredChunk:
    n_tokens = len(text.split())
    chunk = Chunk(
        chunk_id=f"{doc_id}:{position}",
        corpus_id="c1",
        doc_id=doc_id,
        passage_id=doc_id,
        text=text,
        position=position,
        start_token=0,
        end_token=n_tokens,
    )
    return ScoredChunk(chunk=chunk, score=score, source="rerank")


def test_run_marshals_graph_result_and_trace_degraded_flags(tmp_path):
    store = InMemoryTraceStore()
    store.append(
        TraceEvent(
            trace_id="t-1",
            seq=0,
            node="s1_retrieve",
            payload={"degraded": ["rerank-skipped"]},
            model=None,
            input_tokens=0,
            output_tokens=0,
            duration_ms=1.0,
        )
    )
    retrieved = [_scored("geo-001", 0, "Paris is the capital of France.", 0.91)]
    graph_result = GraphResult(
        final=FinalAnswer(text="Paris [1].", citations=[1], abstained=False),
        system="s1",
        trace_id="t-1",
        tokens_used=160,
        hops_used=0,
        retrieved=retrieved,
    )
    seen: dict = {}

    def fake_run_query(
        *, query, core, claude, store, config, graph_retriever=None, token_ceiling=None
    ):
        seen.update(query=query, core=core, preset=config.name, graph_retriever=graph_retriever)
        return graph_result

    runner = RealQueryRunner(
        data_dir=tmp_path,
        trace_store=store,
        claude="claude-transport",
        core_factory=lambda config, corpus_id, data_dir: f"core:{corpus_id}:{config.name}",
        run_query_fn=fake_run_query,
    )
    result = runner.run(query="capital of France?", corpus_id="c1", preset="rerank")
    # rerank is FORCE_S1: the agent graph route is unreachable, so no graph retriever
    # is built (graph stays None — FORCE_S1 serving is unchanged).
    assert seen == {
        "query": "capital of France?",
        "core": "core:c1:rerank",
        "preset": "rerank",
        "graph_retriever": None,
    }
    assert result.answer == "Paris [1]." and result.route == "s1"
    assert result.abstained is False
    assert result.degraded == ["rerank-skipped"]  # collected from the trace, not invented
    assert result.citations[0].n == 1
    assert result.citations[0].chunk_id == "geo-001:0"
    assert result.trace_id == "t-1"


def test_out_of_range_citation_indices_are_dropped(tmp_path):
    store = InMemoryTraceStore()
    graph_result = GraphResult(
        final=FinalAnswer(text="x [9].", citations=[9], abstained=True),
        system="s2",
        trace_id="t-2",
        tokens_used=10,
        hops_used=2,
        retrieved=[],
    )
    runner = RealQueryRunner(
        data_dir=tmp_path,
        trace_store=store,
        claude=None,
        core_factory=lambda config, corpus_id, data_dir: None,
        run_query_fn=lambda **kwargs: graph_result,
    )
    result = runner.run(query="q", corpus_id="c1", preset="router-on")
    assert result.citations == [] and result.route == "s2"
    assert result.abstained is True and result.degraded == []


def test_construction_resolves_pinned_entry_points(tmp_path):
    from ragreceipts.agents.service import run_query  # noqa: F401  (R9 drift guard)
    from ragreceipts.cli import _build_core_real  # noqa: F401

    runner = RealQueryRunner(data_dir=tmp_path, trace_store=InMemoryTraceStore(), claude=None)
    assert runner._run_query is run_query
    assert runner._core_factory is _build_core_real
    # the agent-route graph retriever is built via the module's default factory seam
    from ragreceipts.server.pipeline import _default_graph_retriever_factory

    assert runner._graph_retriever_factory is _default_graph_retriever_factory


def test_router_on_threads_graph_retriever_into_run_query(tmp_path):
    """AUTO preset (router-on): RealQueryRunner builds a graph retriever via the seam
    and threads it into run_query so the agent-layer graph route is reachable.
    Vendor-free — both the core_factory and the graph_retriever_factory are injected."""
    store = InMemoryTraceStore()
    graph_result = GraphResult(
        final=FinalAnswer(text="g [1].", citations=[1], abstained=False),
        system="graph",
        trace_id="t-g",
        tokens_used=1,
        hops_used=1,
        retrieved=[_scored("gp", 0, "graph passage", 0.5)],
    )
    seen: dict = {}

    def fake_run_query(
        *, query, core, claude, store, config, graph_retriever=None, token_ceiling=None
    ):
        seen.update(preset=config.name, graph_retriever=graph_retriever)
        return graph_result

    sentinel = object()  # stands in for the graph-only RetrievalCore
    factory_calls: list = []

    def fake_graph_factory(config, corpus_id, data_dir):
        factory_calls.append((config.name, corpus_id, data_dir))
        return sentinel

    runner = RealQueryRunner(
        data_dir=tmp_path,
        trace_store=store,
        claude=None,
        core_factory=lambda config, corpus_id, data_dir: object(),
        run_query_fn=fake_run_query,
        graph_retriever_factory=fake_graph_factory,
    )
    result = runner.run(query="entity chain?", corpus_id="c1", preset="router-on")
    # the graph retriever the factory produced is exactly what run_query received
    assert seen["preset"] == "router-on"
    assert seen["graph_retriever"] is sentinel
    assert factory_calls == [("router-on", "c1", tmp_path)]
    assert result.route == "graph"


def test_graph_retriever_factory_skipped_for_force_s1_presets(tmp_path):
    """FORCE_S1 presets never run the router, so the graph route is unreachable: the
    runner must NOT even call the graph factory (no wasted artifact load / embed key)."""
    store = InMemoryTraceStore()
    graph_result = GraphResult(
        final=FinalAnswer(text="a [1].", citations=[1], abstained=False),
        system="s1",
        trace_id="t-s1",
        tokens_used=1,
        hops_used=0,
        retrieved=[_scored("p0", 0, "p", 0.5)],
    )
    factory_calls: list = []

    runner = RealQueryRunner(
        data_dir=tmp_path,
        trace_store=store,
        claude=None,
        core_factory=lambda config, corpus_id, data_dir: object(),
        run_query_fn=lambda **kwargs: graph_result,
        graph_retriever_factory=lambda *a, **k: factory_calls.append(a) or object(),
    )
    for preset in ("bm25-only", "dense-rrf", "rerank"):
        runner.run(query="q", corpus_id="c1", preset=preset)
    assert factory_calls == []  # FORCE_S1: graph factory never invoked


def test_default_graph_factory_returns_none_without_artifact(tmp_path):
    """The default factory degrades honestly: no chunks.jsonl -> None (graph route
    falls back to s1), and it never touches a vendor seam in the process."""
    from ragreceipts.config import PRESETS
    from ragreceipts.server.pipeline import _default_graph_retriever_factory

    assert _default_graph_retriever_factory(PRESETS["router-on"], "missing", tmp_path) is None
