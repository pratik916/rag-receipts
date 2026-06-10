"""Flag-flip tests: every QueryConfig flag provably changes RetrievalCore behavior
(route_mode provably does NOT — it belongs to the Plan C router). Plus degraded paths
and TraceEvent emission."""

import pytest
from qdrant_client import QdrantClient

from ragreceipts.config import IngestConfig, PipelineConfig, QueryConfig
from ragreceipts.ingest.indexer import write_dense_index
from ragreceipts.retrieval.core import RetrievalCore
from ragreceipts.retrieval.dense import VECTOR_ISOLATED, DenseRetriever
from ragreceipts.retrieval.rerank import RerankStage
from ragreceipts.retrieval.sparse import build_sparse_index
from ragreceipts.traces.models import TraceEvent
from ragreceipts.types import RouteMode
from ragreceipts.vendors.base import VendorUnavailable
from tests.corpus_fixtures import make_chunk
from tests.fakes import FakeEmbed, FakeRerank

QUERY = "capital of france"
C_LEX = make_chunk(
    "d1:0",
    "france capital city facts about the capital of france",
    corpus_id="flagflip",
    passage_id="d1-p0",
)
C_SEM = make_chunk(
    "d2:0",
    "the eiffel tower attracts millions of visitors",
    corpus_id="flagflip",
    passage_id="d2-p0",
)
C_MID = make_chunk(
    "d3:0", "paris is the capital of france", corpus_id="flagflip", passage_id="d3-p0"
)
CHUNKS = [C_LEX, C_SEM, C_MID]


def qc(**overrides) -> QueryConfig:
    base = dict(
        bm25=True,
        dense=True,
        rerank=False,
        route_mode=RouteMode.FORCE_S1,
        top_k_fuse=3,
        top_k_final=3,
    )
    base.update(overrides)
    return QueryConfig(**base)


@pytest.fixture()
def stack(tmp_path):
    fake = FakeEmbed(query_aliases={QUERY: C_SEM.text})
    sparse = build_sparse_index(CHUNKS, tmp_path / "sparse")
    client = QdrantClient(":memory:")
    vectors = [doc[0] for doc in fake.embed_documents([[c.text] for c in CHUNKS])]
    write_dense_index(client, "flagflip", CHUNKS, vectors, vectors)
    dense = DenseRetriever(client, "flagflip", VECTOR_ISOLATED, fake)
    return {"sparse": sparse, "dense": dense, "client": client}


def make_core(
    stack, query: QueryConfig, *, dense=None, rerank_fail=False, on_trace=None
) -> RetrievalCore:
    config = PipelineConfig(name="test", ingest=IngestConfig(contextual=False), query=query)
    return RetrievalCore(
        config,
        dense or stack["dense"],
        stack["sparse"],
        RerankStage(FakeRerank(fail=rerank_fail)),
        on_trace=on_trace,
    )


def ids(results):
    return [r.chunk.chunk_id for r in results]


def test_bm25_only_flag(stack):
    results = make_core(stack, qc(dense=False)).retrieve(QUERY)
    assert ids(results) == ["d1:0", "d3:0"]
    assert all(r.source == "bm25" for r in results)
    assert "d2:0" not in ids(results)


def test_dense_only_flag(stack):
    results = make_core(stack, qc(bm25=False)).retrieve(QUERY)
    assert ids(results)[0] == "d2:0"
    assert results[0].source == "dense"
    assert results[0].score > 0.99


def test_hybrid_fuses_both_flags(stack):
    results = make_core(stack, qc()).retrieve(QUERY)
    assert ids(results) == ["d1:0", "d3:0", "d2:0"]
    assert all(r.source == "rrf" for r in results)


def test_rerank_flag_reorders(stack):
    base = make_core(stack, qc()).retrieve(QUERY)
    reranked = make_core(stack, qc(rerank=True)).retrieve(QUERY)
    assert ids(reranked) == list(reversed(ids(base)))  # FakeRerank default reverses
    assert all(r.source == "rerank" for r in reranked)


def test_top_k_final_flag(stack):
    assert len(make_core(stack, qc(top_k_final=2)).retrieve(QUERY)) == 2
    assert len(make_core(stack, qc(top_k_final=3)).retrieve(QUERY)) == 3


def test_top_k_fuse_flag(stack):
    narrow = make_core(stack, qc(top_k_fuse=1)).retrieve(QUERY)
    wide = make_core(stack, qc(top_k_fuse=3)).retrieve(QUERY)
    assert ids(narrow) == ["d1:0", "d2:0"]  # one candidate per retriever, id tie-break
    assert "d3:0" not in ids(narrow)
    assert "d3:0" in ids(wide)


def test_route_mode_does_not_change_retrieval_core(stack):
    s1 = make_core(stack, qc(route_mode=RouteMode.FORCE_S1)).retrieve(QUERY)
    auto = make_core(stack, qc(route_mode=RouteMode.AUTO)).retrieve(QUERY)
    assert ids(s1) == ids(auto)  # route_mode is consumed by the Plan C router only


def test_dense_failure_degrades_to_bm25_with_flag(stack):
    events: list[TraceEvent] = []
    failing = DenseRetriever(
        stack["client"], "flagflip", VECTOR_ISOLATED, FakeEmbed(fail_query=True)
    )
    results = make_core(stack, qc(), dense=failing, on_trace=events.append).retrieve(QUERY)
    assert ids(results) == ["d1:0", "d3:0"]
    assert all(r.source == "bm25" for r in results)
    assert events[0].payload["degraded"] == ["dense-skipped"]


def test_rerank_failure_degrades_to_rrf_order(stack):
    events: list[TraceEvent] = []
    core = make_core(stack, qc(rerank=True), rerank_fail=True, on_trace=events.append)
    results = core.retrieve(QUERY)
    assert ids(results) == ["d1:0", "d3:0", "d2:0"]  # RRF order preserved
    assert all(r.source == "rrf" for r in results)
    assert events[0].payload["degraded"] == ["rerank-skipped"]


def test_dense_only_failure_raises(stack):
    failing = DenseRetriever(
        stack["client"], "flagflip", VECTOR_ISOLATED, FakeEmbed(fail_query=True)
    )
    core = make_core(stack, qc(bm25=False), dense=failing)
    with pytest.raises(VendorUnavailable):
        core.retrieve(QUERY)  # no bm25 to fall back to


def test_invalid_configs_rejected(stack):
    with pytest.raises(ValueError):
        make_core(stack, qc(bm25=False, dense=False))
    config = PipelineConfig(name="t", ingest=IngestConfig(), query=qc())
    with pytest.raises(ValueError):
        RetrievalCore(config, None, stack["sparse"], None)  # dense flag on, none given


def test_trace_event_payload_and_threading(stack):
    events: list[TraceEvent] = []
    core = make_core(stack, qc(rerank=True), on_trace=events.append)
    results = core.retrieve(QUERY, trace_id="t-123", node="retrieve_hop", seq_start=7)
    assert len(events) == 1
    event = events[0]
    assert (event.trace_id, event.node, event.seq) == ("t-123", "retrieve_hop", 7)
    assert event.model is None and event.input_tokens == 0 and event.output_tokens == 0
    assert event.duration_ms >= 0.0
    assert event.payload["query"] == QUERY
    assert event.payload["config"]["rerank"] is True
    assert [r["chunk_id"] for r in event.payload["results"]] == ids(results)
    assert len(event.payload["candidates"]) == 3
    assert event.payload["degraded"] == []


def test_trace_id_generated_when_absent(stack):
    events: list[TraceEvent] = []
    make_core(stack, qc(), on_trace=events.append).retrieve(QUERY)
    assert events[0].node == "s1_retrieve"
    assert len(events[0].trace_id) == 32  # uuid4().hex


# --- Plan E: graph-mode wiring -------------------------------------------------

from ragreceipts.retrieval.graph import GraphRetriever  # noqa: E402
from tests.graph_fixtures import FIXTURE_CHUNKS, build_fixture_graph  # noqa: E402


class _AlwaysFailGraph:
    """A Retriever that always raises VendorUnavailable (graph artifact missing)."""

    def search(self, query: str, k: int):
        raise VendorUnavailable("graph down")


def _graph_retriever(recognition="embedding", embed=None):
    return GraphRetriever(
        build_fixture_graph().index,
        chunks=FIXTURE_CHUNKS,
        embed=embed or FakeEmbed(),
        recognition=recognition,
    )


def _graph_core(query: QueryConfig, *, graph=None, dense=None, sparse=None, on_trace=None):
    config = PipelineConfig(name="g", ingest=IngestConfig(contextual=False), query=query)
    return RetrievalCore(
        config,
        dense,
        sparse,
        rerank_stage=None,
        graph=graph,
        on_trace=on_trace,
    )


def gqc(**overrides) -> QueryConfig:
    base = dict(
        bm25=False,
        dense=False,
        rerank=False,
        graph=True,
        route_mode=RouteMode.FORCE_S1,
        top_k_fuse=4,
        top_k_final=4,
    )
    base.update(overrides)
    return QueryConfig(**base)


def test_graph_only_passthrough_keeps_source(stack):
    core = _graph_core(gqc(), graph=_graph_retriever())
    results = core.retrieve("Eiffel Tower Paris")
    assert results
    assert all(r.source == "graph" for r in results)


def test_graph_enabled_without_retriever_raises(stack):
    with pytest.raises(ValueError):
        _graph_core(gqc(), graph=None)


def test_at_least_one_retriever_guard_allows_graph_only(stack):
    # bm25=dense=False but graph=True must be valid (widened guard).
    core = _graph_core(gqc(), graph=_graph_retriever())
    assert core.retrieve("Paris")  # no ValueError


def test_graph_rrf_fuses_sparse_dense_graph(stack):
    core = _graph_core(
        gqc(bm25=True, dense=True),
        graph=_graph_retriever(),
        dense=stack["dense"],
        sparse=stack["sparse"],
    )
    results = core.retrieve(QUERY)
    assert results
    assert all(r.source == "rrf" for r in results)  # ≥2 lists -> RRF fusion


def test_graph_skipped_degrades_when_fallback_exists(stack):
    events: list[TraceEvent] = []
    core = _graph_core(
        gqc(bm25=True, dense=True),
        graph=_AlwaysFailGraph(),
        dense=stack["dense"],
        sparse=stack["sparse"],
        on_trace=events.append,
    )
    results = core.retrieve(QUERY)
    assert results
    assert "graph-skipped" in events[0].payload["degraded"]
    assert all(r.source == "rrf" for r in results)  # fused over the surviving sparse+dense


def test_graph_only_failure_raises(stack):
    core = _graph_core(gqc(), graph=_AlwaysFailGraph())
    with pytest.raises(VendorUnavailable):
        core.retrieve(QUERY)  # no bm25/dense to fall back to


def test_graph_run_trace_pins_graph_config(stack):
    # RG7: the emitted trace event's config payload pins graph + graph_recognition
    # straight from the QueryConfig (the spec's thesis: receipts pin the config).
    events: list[TraceEvent] = []
    query = gqc(graph_recognition="embedding")
    core = _graph_core(query, graph=_graph_retriever(), on_trace=events.append)
    core.retrieve("Eiffel Tower Paris")
    config = events[0].payload["config"]
    assert config["graph"] is True
    assert config["graph_recognition"] == query.graph_recognition == "embedding"
