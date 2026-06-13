"""Eval-run seam (spec: estimate -> confirm -> hard cap).

Plan B owns BOTH the authoritative runner and the authoritative cost estimator:
`eval/runner.py::estimate_run_cost(preset_names, n_queries) -> float` (R9-pinned).
The server never re-implements a pricing formula — `RealEvalRunner.estimate` (Step 5)
delegates to `estimate_run_cost`, which already prices Claude synthesis + voyage query
embeddings + cohere rerank per query (and, after Plan C, the System-2 estimate for AUTO
presets — R10). The mid-run hard spend cap also lives in Plan B's runner
(`SpendCapExceeded`); a raise fails the job with the named error.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ragreceipts.config import PRESETS
from ragreceipts.eval.pricing import PRICING_VERSION


@dataclass(frozen=True)
class CostEstimate:
    n_queries: int
    est_tokens: int
    est_usd: float
    pricing_table_version: str


class EvalRunner(Protocol):
    def estimate(self, *, corpus_id: str, preset: str, slice_name: str) -> CostEstimate: ...

    def run(
        self,
        *,
        corpus_id: str,
        preset: str,
        slice_name: str,
        spend_cap_usd: float,
        emit: Callable[[str, float], None],
    ) -> str:
        """Execute the run; emit(message, progress) streams job events; returns run_id."""
        ...


class RealEvalRunner:
    """EvalRunner over Plan B's R9-pinned entry points.

    Pins (contracts §Seam Resolutions R9/R10):
      - eval/runner.py::estimate_run_cost(preset_names, n_queries) -> float (USD) —
        the ONLY estimator; it prices Claude + voyage + cohere and (post-Plan C) the
        System-2 estimate for AUTO presets. No server-side pricing formula exists.
      - eval/runner.py::AblationRunner(core_factory=, claude=, store=, data_dir=,
        ragas=None) with .run(run_id=, corpus_id=, slice_name=, presets=, spend_cap_usd=)
      - cli.py::_build_core_real(config, corpus_id, data_dir) (composition root)
    Plan B's hard mid-run spend cap (SpendCapExceeded) propagates out of run() and
    fails the job with the named error. Constructor seams default to the real entry
    points; tests inject fakes.
    """

    def __init__(
        self,
        *,
        data_dir: Path,
        n_queries_fn: Callable[[str, str], int] | None = None,
        estimate_fn: Callable[[list[str], int], float] | None = None,
        runner_factory: Callable[[str], object] | None = None,
        run_id_fn: Callable[[str, str], str] | None = None,
    ) -> None:
        self._data_dir = data_dir
        self._n_queries_fn = n_queries_fn
        self._estimate_fn = estimate_fn
        self._runner_factory = runner_factory
        self._run_id_fn = run_id_fn

    def estimate(self, *, corpus_id: str, preset: str, slice_name: str) -> CostEstimate:
        from ragreceipts.eval.runner import (
            EST_QUERY_EMBED_TOKENS,
            EST_SYNTH_INPUT_TOKENS,
            EST_SYNTH_OUTPUT_TOKENS,
            estimate_run_cost,
        )

        n_queries_fn = self._n_queries_fn or self._count_slice_queries
        estimate_fn = self._estimate_fn or estimate_run_cost
        n = n_queries_fn(corpus_id, slice_name)
        per_q_tokens = EST_SYNTH_INPUT_TOKENS + EST_SYNTH_OUTPUT_TOKENS
        if PRESETS[preset].query.dense:
            per_q_tokens += EST_QUERY_EMBED_TOKENS
        return CostEstimate(
            n_queries=n,
            est_tokens=n * per_q_tokens,
            est_usd=round(estimate_fn([preset], n), 4),
            pricing_table_version=PRICING_VERSION,
        )

    def run(
        self,
        *,
        corpus_id: str,
        preset: str,
        slice_name: str,
        spend_cap_usd: float,
        emit: Callable[[str, float], None],
    ) -> str:
        if self._run_id_fn is None:
            from ragreceipts.eval.runner import new_run_id as run_id_fn
        else:
            run_id_fn = self._run_id_fn
        runner_factory = self._runner_factory or self._build_runner
        run_id = run_id_fn(corpus_id, slice_name)
        emit(f"eval run {run_id}: preset={preset} slice={slice_name}", 0.05)
        runner_factory(corpus_id).run(
            run_id=run_id,
            corpus_id=corpus_id,
            slice_name=slice_name,
            presets=[preset],
            spend_cap_usd=spend_cap_usd,
        )
        emit(f"receipt written: receipts-local/{run_id}.json", 0.95)
        return run_id

    # -- real-entry-point defaults (overridden by fakes in tests) --------------------

    def _count_slice_queries(self, corpus_id: str, slice_name: str) -> int:
        from ragreceipts.eval.queries import load_queries, slice_queries, slice_query_ids

        # slice_queries takes query IDs, not a slice NAME — resolve the name first
        # via the slice files (Plan B's slice_query_ids).
        return len(
            slice_queries(
                load_queries(self._data_dir, corpus_id),
                slice_query_ids(self._data_dir, corpus_id, slice_name),
            )
        )

    def _build_runner(self, corpus_id: str):
        from ragreceipts.cli import _build_core_real, _make_claude
        from ragreceipts.eval.run_state import RunStore
        from ragreceipts.eval.runner import AblationRunner
        from ragreceipts.server.pipeline import _default_graph_retriever_factory

        return AblationRunner(
            core_factory=lambda cfg: _build_core_real(cfg, corpus_id, self._data_dir),
            claude=_make_claude(),
            store=RunStore(self._data_dir / "eval-runs.db"),
            data_dir=self._data_dir,
            # Serving/eval symmetry: the eval router-on graph route reaches the SAME
            # graph-only RetrievalCore RealQueryRunner serves. AblationRunner calls
            # graph_factory(cfg); we bind corpus/data_dir into the serving factory, which
            # returns None when the corpus has no graph artifact (honest s1 fallback).
            graph_factory=lambda cfg: _default_graph_retriever_factory(
                cfg, corpus_id, self._data_dir
            ),
        )
