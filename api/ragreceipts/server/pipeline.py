"""Query execution seam between HTTP and the Plan C agent graph."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ragreceipts.config import PRESETS


@dataclass(frozen=True)
class Citation:
    n: int  # matches the [n] markers in the answer text
    chunk_id: str
    passage_id: str
    text: str
    score: float


@dataclass(frozen=True)
class QueryResult:
    answer: str
    abstained: bool  # structured field per spec — never prose-only
    route: str  # "s1" | "s2"
    degraded: list[str]  # e.g. ["rerank-skipped"] (contracts §Vendor protocols)
    citations: list[Citation]
    trace_id: str


class QueryRunner(Protocol):
    def run(
        self, *, query: str, corpus_id: str, preset: str, token_ceiling: int | None = None
    ) -> QueryResult: ...


def _collect_degraded(events) -> list[str]:
    """Union of degraded flags recorded in the query's TraceEvents, first-seen order
    (degrade visibly, never silently — the flags live in the trace, not invented here)."""
    out: list[str] = []
    for ev in events:
        for flag in ev.payload.get("degraded") or []:
            if flag not in out:
                out.append(flag)
    return out


class RealQueryRunner:
    """QueryRunner over the R9-pinned production entry points.

    Pins (contracts §Seam Resolutions R9):
      - agents/service.py::run_query(query=, core=, claude=, store=, config=) -> GraphResult
      - cli.py::_build_core_real(config, corpus_id, data_dir) -> RetrievalCore
    GraphResult: final (FinalAnswer: text/citations/abstained), system ("s1"|"s2"),
    trace_id, tokens_used, hops_used, retrieved (list[ScoredChunk]).
    Constructor seams default to the real entry points; tests inject fakes.
    """

    def __init__(
        self,
        *,
        data_dir: Path,
        trace_store,
        claude,
        core_factory: Callable | None = None,
        run_query_fn: Callable | None = None,
    ) -> None:
        if core_factory is None:
            from ragreceipts.cli import _build_core_real  # R9 composition root

            core_factory = _build_core_real
        if run_query_fn is None:
            from ragreceipts.agents.service import run_query  # R9 graph entry point

            run_query_fn = run_query
        self._data_dir = data_dir
        self._trace_store = trace_store
        self._claude = claude
        self._core_factory = core_factory
        self._run_query = run_query_fn

    def run(
        self, *, query: str, corpus_id: str, preset: str, token_ceiling: int | None = None
    ) -> QueryResult:
        config = PRESETS[preset]
        core = self._core_factory(config, corpus_id, self._data_dir)
        result = self._run_query(
            query=query,
            core=core,
            claude=self._claude,
            store=self._trace_store,
            config=config,
            token_ceiling=token_ceiling,
        )
        citations: list[Citation] = []
        for n in result.final.citations:
            if 1 <= n <= len(result.retrieved):  # out-of-range [n] markers are dropped
                sc = result.retrieved[n - 1]
                citations.append(
                    Citation(
                        n=n,
                        chunk_id=sc.chunk.chunk_id,
                        passage_id=sc.chunk.passage_id,
                        text=sc.chunk.text,
                        score=sc.score,
                    )
                )
        return QueryResult(
            answer=result.final.text,
            abstained=result.final.abstained,
            route=result.system,
            degraded=_collect_degraded(self._trace_store.get(result.trace_id)),
            citations=citations,
            trace_id=result.trace_id,
        )


def build_real_query_runner(*, paths, qdrant, trace_store) -> RealQueryRunner:
    """Production constructor — wired by deps.build_deps when all three vendor keys AND
    QDRANT_URL are present (R7). `qdrant` is accepted for parity with the deps container
    (it backs /health); core construction goes through Plan B's composition root
    `_build_core_real`, which reads QDRANT_URL itself — guaranteed set on this path.
    """
    from ragreceipts.cli import _make_claude  # Plan B's real ClaudeTransport factory

    return RealQueryRunner(data_dir=paths.data_dir, trace_store=trace_store, claude=_make_claude())
