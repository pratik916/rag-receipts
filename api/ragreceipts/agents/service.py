"""run_query: the one entry point that drives the graph for a single query.

Used by the eval runner (Plan C Task 12) and the FastAPI POST /query (Plan D),
so both execute the identical retrieval+agent code path (spec invariant).
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from ragreceipts.agents.graph import SupportsRetrieve, build_graph, initial_state
from ragreceipts.agents.schemas import FinalAnswer
from ragreceipts.config import PipelineConfig
from ragreceipts.traces.recorder import TraceRecorder
from ragreceipts.traces.store import TraceStore
from ragreceipts.types import ScoredChunk
from ragreceipts.vendors.base import ClaudeTransport

# R9: RetrievalCore's trace wiring is the constructor kwarg `on_trace` — never a
# private-attribute assignment. A caller that wants the core's intra-retrieval
# events (per-retriever timings, degraded flags) on a query's trace passes a
# per-query factory: run_query invokes it with this query's TraceRecorder and
# the factory constructs RetrievalCore(..., on_trace=recorder) itself.
type CoreOrFactory = SupportsRetrieve | Callable[[TraceRecorder], SupportsRetrieve]


@dataclass(frozen=True)
class GraphResult:
    final: FinalAnswer
    system: str  # "s1" | "s2" | "graph"
    trace_id: str
    tokens_used: int
    hops_used: int  # 0 on the S1 path
    retrieved: list[ScoredChunk]  # S1 top-k, or S2 union-of-hops (deduped,
    # first-seen order) for the eval diagnostic


def union_of_hops(hop_records: list[dict]) -> list[ScoredChunk]:
    seen: set[str] = set()
    out: list[ScoredChunk] = []
    for rec in hop_records:
        for sc in rec["chunks"]:
            if sc.chunk.chunk_id not in seen:
                seen.add(sc.chunk.chunk_id)
                out.append(sc)
    return out


def run_query(
    *,
    query: str,
    core: CoreOrFactory,
    claude: ClaudeTransport,
    store: TraceStore,
    config: PipelineConfig,
    trace_id: str | None = None,
    graph_retriever: SupportsRetrieve | None = None,
) -> GraphResult:
    trace_id = trace_id or uuid.uuid4().hex
    recorder = TraceRecorder(store, trace_id)
    if not hasattr(core, "retrieve"):
        core = core(recorder)  # per-query factory (see CoreOrFactory above)
    graph = build_graph(
        core=core,
        claude=claude,
        recorder=recorder,
        route_mode=config.query.route_mode,
        graph_retriever=graph_retriever,
    )
    out = graph.invoke(initial_state(query), config={"recursion_limit": 50})
    system = out.get("chosen_system", "s1")
    if system == "s2":
        retrieved = union_of_hops(out["hop_records"])
    else:  # "s1" or "graph": top-k is the single retrieval
        retrieved = out["retrieved"]
    return GraphResult(
        final=out["final"],
        system=system,
        trace_id=trace_id,
        tokens_used=out["tokens_used"],
        hops_used=out["hops_used"],
        retrieved=retrieved,
    )


def route_counts(results: Iterable[GraphResult]) -> dict[str, int]:
    """Receipt route-distribution stats: {"n_s1": ..., "n_s2": ..., "n_graph": ...}."""
    rs = list(results)
    n_s2 = sum(1 for r in rs if r.system == "s2")
    n_graph = sum(1 for r in rs if r.system == "graph")
    return {"n_s1": len(rs) - n_s2 - n_graph, "n_s2": n_s2, "n_graph": n_graph}
