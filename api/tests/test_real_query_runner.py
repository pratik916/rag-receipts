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

    def fake_run_query(*, query, core, claude, store, config):
        seen.update(query=query, core=core, preset=config.name)
        return graph_result

    runner = RealQueryRunner(
        data_dir=tmp_path,
        trace_store=store,
        claude="claude-transport",
        core_factory=lambda config, corpus_id, data_dir: f"core:{corpus_id}:{config.name}",
        run_query_fn=fake_run_query,
    )
    result = runner.run(query="capital of France?", corpus_id="c1", preset="rerank")
    assert seen == {"query": "capital of France?", "core": "core:c1:rerank", "preset": "rerank"}
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
