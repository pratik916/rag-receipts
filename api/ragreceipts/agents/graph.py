"""LangGraph state machine: route -> System-1 fast path | System-2 agentic loop.

Pure orchestration (spec boundary rule): retrieval happens only through the
injected core's .retrieve(); every Claude call goes through ClaudeTransport.

LangGraph API verified 2026-06-10 against
https://docs.langchain.com/oss/python/langgraph/graph-api (langgraph 1.2.x):
nodes return partial state updates; add_conditional_edges(source, fn, path_map);
recursion_limit is a standalone top-level config key on invoke().
"""

from __future__ import annotations

import time
from typing import Protocol, TypedDict

from langgraph.graph import END, START, StateGraph

from ragreceipts.agents import prompts
from ragreceipts.agents.schemas import (
    FinalAnswer,
    GradeResult,
    RouteDecision,
    SubQueries,
)
from ragreceipts.constants import (
    ROUTE_CONFIDENCE_THRESHOLD,
    ROUTER_MODEL,
    S2_MAX_HOPS,
    S2_TOKEN_CEILING,
    SYNTH_MODEL,
)
from ragreceipts.traces.recorder import TraceRecorder
from ragreceipts.types import RouteMode, ScoredChunk
from ragreceipts.vendors.base import ClaudeTransport


class SupportsRetrieve(Protocol):
    """Structural match for retrieval.core.RetrievalCore — tests inject fakes."""

    def retrieve(self, query: str) -> list[ScoredChunk]: ...


class GraphState(TypedDict, total=False):
    query: str
    route: str  # "simple" | "complex" | "graph" (set by route node)
    confidence: float
    chosen_system: str  # "s1" | "s2" | "graph"
    retrieved: list  # list[ScoredChunk] — S1 top-k
    subqueries: list[str]
    hop_index: int  # index of the sub-query currently being retrieved
    hop_records: list[dict]  # {"subquery","original","chunks","verdict"}
    hops_used: int  # every retrieve_hop execution counts (incl. retries)
    tokens_used: int  # input+output summed over every Claude call
    refined_query: str | None  # set by refine, consumed by next retrieve_hop
    contradiction_retried: bool  # the one re-retrieve attempt was used
    contradiction_flag: bool
    unresolved: list[str]
    budget_exhausted: bool
    next_action: str  # set by decompose/grade, read by conditional edges
    final: FinalAnswer | None


def initial_state(query: str) -> GraphState:
    return GraphState(
        query=query,
        retrieved=[],
        subqueries=[],
        hop_index=0,
        hop_records=[],
        hops_used=0,
        tokens_used=0,
        refined_query=None,
        contradiction_retried=False,
        contradiction_flag=False,
        unresolved=[],
        budget_exhausted=False,
        final=None,
    )


def _chunk_payload(chunks: list[ScoredChunk]) -> list[dict]:
    return [{"chunk_id": c.chunk.chunk_id, "score": c.score, "source": c.source} for c in chunks]


def build_graph(
    *,
    core: SupportsRetrieve,
    claude: ClaudeTransport,
    recorder: TraceRecorder,
    route_mode: RouteMode = RouteMode.AUTO,
    confidence_threshold: float = ROUTE_CONFIDENCE_THRESHOLD,
    max_hops: int = S2_MAX_HOPS,
    token_ceiling: int = S2_TOKEN_CEILING,
    graph_retriever: SupportsRetrieve | None = None,
):
    """Compile the query graph. Dependencies are closed over; state holds data only."""

    # ---------------------------------------------------------------- route
    def route_node(state: GraphState) -> dict:
        t0 = time.perf_counter()
        res = claude.parse(
            model=ROUTER_MODEL,
            system=prompts.ROUTE_SYSTEM,
            user=prompts.ROUTE_USER.format(query=state["query"]),
            max_tokens=1024,
            output_format=RouteDecision,
            temperature=0.0,
        )
        decision: RouteDecision = res.parsed
        recorder.emit(
            "route",
            {"query": state["query"], "route": decision.route, "confidence": decision.confidence},
            model=ROUTER_MODEL,
            input_tokens=res.input_tokens,
            output_tokens=res.output_tokens,
            duration_ms=(time.perf_counter() - t0) * 1000,
        )
        return {
            "route": decision.route,
            "confidence": decision.confidence,
            "tokens_used": state["tokens_used"] + res.input_tokens + res.output_tokens,
        }

    def after_route(state: GraphState) -> str:
        # The graph route is reachable only when a graph retriever was injected
        # (gated to multi-hop corpora by the caller). Off-corpus graph decisions
        # fall back to s1. Confidence still escalates complex/low-confidence to S2.
        if state["route"] == "graph" and graph_retriever is not None:
            return "graph_retrieve"
        # Confidence is consumed, not decorative (spec): low confidence escalates.
        if state["route"] == "complex" or state["confidence"] < confidence_threshold:
            return "decompose"
        return "s1_retrieve"

    # ---------------------------------------------------------------- System-1
    def s1_retrieve_node(state: GraphState) -> dict:
        t0 = time.perf_counter()
        chunks = core.retrieve(state["query"])
        recorder.emit(
            "s1_retrieve",
            {"query": state["query"], "chunks": _chunk_payload(chunks)},
            duration_ms=(time.perf_counter() - t0) * 1000,
        )
        return {"retrieved": chunks, "chosen_system": "s1"}

    def s1_answer_node(state: GraphState) -> dict:
        t0 = time.perf_counter()
        context = prompts.format_numbered_context(state["retrieved"])
        res = claude.parse(
            model=SYNTH_MODEL,
            system=prompts.S1_ANSWER_SYSTEM,
            user=prompts.S1_ANSWER_USER.format(query=state["query"], context=context),
            max_tokens=4096,
            output_format=FinalAnswer,
            temperature=0.0,
        )
        final: FinalAnswer = res.parsed
        recorder.emit(
            "s1_answer",
            {"text": final.text, "citations": final.citations, "abstained": final.abstained},
            model=SYNTH_MODEL,
            input_tokens=res.input_tokens,
            output_tokens=res.output_tokens,
            duration_ms=(time.perf_counter() - t0) * 1000,
        )
        return {
            "final": final,
            "tokens_used": state["tokens_used"] + res.input_tokens + res.output_tokens,
        }

    # ---------------------------------------------------------------- graph route
    def graph_retrieve_node(state: GraphState) -> dict:
        t0 = time.perf_counter()
        chunks = graph_retriever.retrieve(state["query"])
        recorder.emit(
            "graph_retrieve",
            {"query": state["query"], "chunks": _chunk_payload(chunks)},
            duration_ms=(time.perf_counter() - t0) * 1000,
        )
        # Reuse the S2 synthesis path: one hop record, then synthesize.
        record = {
            "subquery": state["query"],
            "original": state["query"],
            "chunks": chunks,
            "verdict": "sufficient",
        }
        return {
            "retrieved": chunks,
            "hop_records": [record],
            "chosen_system": "graph",
        }

    # ---------------------------------------------------------------- System-2
    def decompose_node(state: GraphState) -> dict:
        t0 = time.perf_counter()
        res = claude.parse(
            model=ROUTER_MODEL,
            system=prompts.DECOMPOSE_SYSTEM,
            user=prompts.DECOMPOSE_USER.format(query=state["query"], max_hops=max_hops),
            max_tokens=1024,
            output_format=SubQueries,
            temperature=0.0,
        )
        items = list(res.parsed.items)[:max_hops]  # hard cap (spec S2 bound)
        tokens = state["tokens_used"] + res.input_tokens + res.output_tokens
        recorder.emit(
            "decompose",
            {
                "query": state["query"],
                "subqueries": items,
                "truncated": len(res.parsed.items) > max_hops,
            },
            model=ROUTER_MODEL,
            input_tokens=res.input_tokens,
            output_tokens=res.output_tokens,
            duration_ms=(time.perf_counter() - t0) * 1000,
        )
        update: dict = {
            "subqueries": items,
            "hop_index": 0,
            "chosen_system": "s2",
            "tokens_used": tokens,
            "next_action": "retrieve_hop",
        }
        if not items:
            # Degenerate decomposition: nothing to retrieve; synthesize will abstain.
            update["next_action"] = "synthesize"
        elif tokens >= token_ceiling:
            # Ceiling crossed before any retrieval: caveated synthesis, all unresolved.
            update.update(next_action="synthesize", budget_exhausted=True, unresolved=items)
        return update

    def after_decompose(state: GraphState) -> str:
        return state["next_action"]

    def retrieve_hop_node(state: GraphState) -> dict:
        t0 = time.perf_counter()
        original = state["subqueries"][state["hop_index"]]
        sub = state["refined_query"] or original
        chunks = core.retrieve(sub)
        record = {"subquery": sub, "original": original, "chunks": chunks, "verdict": None}
        recorder.emit(
            "retrieve_hop",
            {"hop_index": state["hop_index"], "subquery": sub, "chunks": _chunk_payload(chunks)},
            duration_ms=(time.perf_counter() - t0) * 1000,
        )
        return {
            "hop_records": state["hop_records"] + [record],
            "hops_used": state["hops_used"] + 1,
            "refined_query": None,
        }

    def grade_node(state: GraphState) -> dict:
        t0 = time.perf_counter()
        record = state["hop_records"][-1]
        res = claude.parse(
            model=ROUTER_MODEL,
            system=prompts.GRADE_SYSTEM,
            user=prompts.GRADE_USER.format(
                subquery=record["subquery"],
                context=prompts.format_numbered_context(record["chunks"]),
            ),
            max_tokens=1024,
            output_format=GradeResult,
            temperature=0.0,
        )
        verdict = res.parsed.verdict
        tokens = state["tokens_used"] + res.input_tokens + res.output_tokens
        budget_ok = state["hops_used"] < max_hops and tokens < token_ceiling
        remaining = state["subqueries"][state["hop_index"] + 1 :]
        update: dict = {
            "hop_records": state["hop_records"][:-1] + [dict(record, verdict=verdict)],
            "tokens_used": tokens,
        }

        def advance() -> None:
            if remaining and budget_ok:
                update.update(
                    next_action="retrieve_hop",
                    hop_index=state["hop_index"] + 1,
                    contradiction_retried=False,
                )
            elif remaining:  # budget exhausted mid-plan: disclose the rest
                update.update(
                    next_action="synthesize",
                    budget_exhausted=True,
                    unresolved=state["unresolved"] + remaining,
                )
            else:
                update["next_action"] = "synthesize"

        if verdict == "sufficient":
            advance()
        elif verdict == "insufficient":
            if budget_ok:
                update["next_action"] = "refine"
            else:
                update.update(
                    next_action="synthesize",
                    budget_exhausted=True,
                    unresolved=state["unresolved"] + [record["original"]] + remaining,
                )
        else:  # contradictory — one re-retrieve attempt, then flagged synthesis
            if not state["contradiction_retried"] and budget_ok:
                update.update(next_action="retrieve_hop", contradiction_retried=True)
            else:
                update["contradiction_flag"] = True
                advance()

        recorder.emit(
            "grade",
            {
                "subquery": record["subquery"],
                "verdict": verdict,
                "next_action": update["next_action"],
                "budget_ok": budget_ok,
            },
            model=ROUTER_MODEL,
            input_tokens=res.input_tokens,
            output_tokens=res.output_tokens,
            duration_ms=(time.perf_counter() - t0) * 1000,
        )
        return update

    def after_grade(state: GraphState) -> str:
        return state["next_action"]

    def refine_node(state: GraphState) -> dict:
        t0 = time.perf_counter()
        record = state["hop_records"][-1]
        res = claude.complete(
            model=ROUTER_MODEL,
            system=prompts.REFINE_SYSTEM,
            user=prompts.REFINE_USER.format(
                subquery=record["subquery"],
                context=prompts.format_numbered_context(record["chunks"]),
            ),
            max_tokens=1024,
            temperature=0.0,
        )
        refined = res.text.strip()
        recorder.emit(
            "refine",
            {"original": record["subquery"], "refined": refined},
            model=ROUTER_MODEL,
            input_tokens=res.input_tokens,
            output_tokens=res.output_tokens,
            duration_ms=(time.perf_counter() - t0) * 1000,
        )
        return {
            "refined_query": refined,
            "tokens_used": state["tokens_used"] + res.input_tokens + res.output_tokens,
        }

    def synthesize_node(state: GraphState) -> dict:
        t0 = time.perf_counter()
        context, _ordered = prompts.format_hop_context(state["hop_records"])
        unresolved = list(state["unresolved"])
        res = claude.parse(
            model=SYNTH_MODEL,
            system=prompts.SYNTHESIZE_SYSTEM,
            user=prompts.SYNTHESIZE_USER.format(
                query=state["query"],
                context=context or "(no evidence retrieved)",
                unresolved=", ".join(unresolved) or "(none)",
                contradiction="yes" if state["contradiction_flag"] else "no",
            ),
            max_tokens=4096,
            output_format=FinalAnswer,
            temperature=0.0,
        )
        final: FinalAnswer = res.parsed
        # State-enforced disclosure: never trust the model alone for budget or
        # contradiction flags (spec: flagged, never papered over).
        final = final.model_copy(
            update={
                "contradiction_flag": final.contradiction_flag or state["contradiction_flag"],
                "unresolved_subqueries": sorted(set(final.unresolved_subqueries) | set(unresolved)),
            }
        )
        recorder.emit(
            "synthesize",
            {
                "text": final.text,
                "citations": final.citations,
                "abstained": final.abstained,
                "contradiction_flag": final.contradiction_flag,
                "unresolved_subqueries": final.unresolved_subqueries,
                "budget_exhausted": state["budget_exhausted"],
            },
            model=SYNTH_MODEL,
            input_tokens=res.input_tokens,
            output_tokens=res.output_tokens,
            duration_ms=(time.perf_counter() - t0) * 1000,
        )
        return {
            "final": final,
            "tokens_used": state["tokens_used"] + res.input_tokens + res.output_tokens,
        }

    # ---------------------------------------------------------------- wiring
    def select_entry(state: GraphState) -> str:
        if route_mode is RouteMode.FORCE_S1:
            return "s1_retrieve"
        if route_mode is RouteMode.FORCE_S2:
            return "decompose"
        return "route"

    builder = StateGraph(GraphState)
    builder.add_node("route", route_node)
    builder.add_node("s1_retrieve", s1_retrieve_node)
    builder.add_node("s1_answer", s1_answer_node)
    builder.add_node("graph_retrieve", graph_retrieve_node)
    builder.add_node("decompose", decompose_node)
    builder.add_node("retrieve_hop", retrieve_hop_node)
    builder.add_node("grade", grade_node)
    builder.add_node("refine", refine_node)
    builder.add_node("synthesize", synthesize_node)

    # Conditional entry keeps all nodes statically reachable in every mode.
    builder.add_conditional_edges(
        START,
        select_entry,
        {"route": "route", "s1_retrieve": "s1_retrieve", "decompose": "decompose"},
    )
    builder.add_conditional_edges(
        "route",
        after_route,
        {
            "s1_retrieve": "s1_retrieve",
            "decompose": "decompose",
            "graph_retrieve": "graph_retrieve",
        },
    )
    builder.add_edge("s1_retrieve", "s1_answer")
    builder.add_edge("s1_answer", END)
    builder.add_edge("graph_retrieve", "synthesize")
    builder.add_conditional_edges(
        "decompose", after_decompose, {"retrieve_hop": "retrieve_hop", "synthesize": "synthesize"}
    )
    builder.add_edge("retrieve_hop", "grade")
    builder.add_conditional_edges(
        "grade",
        after_grade,
        {"retrieve_hop": "retrieve_hop", "refine": "refine", "synthesize": "synthesize"},
    )
    builder.add_edge("refine", "retrieve_hop")
    builder.add_edge("synthesize", END)
    return builder.compile()
