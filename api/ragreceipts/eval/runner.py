"""Ablation runner: preset ladder x corpus slice -> one Receipt per runnable cell.

Degrade visibly, never silently:
- presets with route_mode != FORCE_S1 face the PERMANENT MULTI_HOP_DATASETS
  gate (R10): router-on runs on multi-hop corpora only, producing a disclosed
  SkippedCell on a single-hop corpus rather than a fake run. (Plan C deleted
  the temporary "requires Plan C" skip; generation now drives the agent graph
  via agents.service.run_query, so System-2 cells run live.)
- per-query failures -> status 'failed', excluded from metrics, counted in
  n_failed; abstentions -> 'abstained', excluded from RAGAS, counted in
  n_abstained; both visible in per_query flags.
- hard spend cap: checked BEFORE every query against actual spent USD plus
  the per-query estimate; on breach the run aborts with saved state and the
  exact resume instructions.
Cost notes: query-embedding cost uses the EST_QUERY_EMBED_TOKENS heuristic
(EmbedTransport does not report usage); RAGAS judge cost is not metered in
Plan B - the pre-run ESTIMATE includes a per-ok-query judge heuristic when
requested (ragas=True), but actual judge spend is NOT counted against the
hard cap and is disclosed via the ragas_judge_usd_untracked flag.
"""

from __future__ import annotations

import dataclasses
import math
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from ragreceipts.agents.prompts import PROMPTS_VERSION
from ragreceipts.agents.service import run_query
from ragreceipts.config import PRESETS, PipelineConfig
from ragreceipts.constants import (
    EMBED_MODEL,
    JUDGE_MODEL,
    RERANK_MODEL,
    ROUTER_MODEL,
    S2_MAX_HOPS,
    SYNTH_MODEL,
)
from ragreceipts.eval.metrics import exact_match, f1, mrr_at_k, recall_at_k
from ragreceipts.eval.pricing import PRICING_VERSION, usd_for_rerank, usd_for_tokens
from ragreceipts.eval.queries import (
    QueryRecord,
    load_manifest,
    load_queries,
    slice_queries,
    slice_query_ids,
)
from ragreceipts.eval.ragas_adapter import RagasJudge
from ragreceipts.eval.receipts import (
    ANCHOR_SPECS,
    Receipt,
    build_anchor,
    make_run_doc,
    write_run_doc,
)
from ragreceipts.eval.run_state import RunStore
from ragreceipts.retrieval.core import RetrievalCore
from ragreceipts.traces.store import TraceStore
from ragreceipts.types import Chunk, RouteMode
from ragreceipts.vendors.base import ClaudeTransport

# PERMANENT gate data (R10): router-on cells run on multi-hop corpora only.
# Plan C must keep and test this gate.
MULTI_HOP_DATASETS = {"musique", "2wikimultihopqa"}

# Pre-run estimate inputs: ~5 chunks x 512 tokens + ~740 tokens prompt/question.
EST_SYNTH_INPUT_TOKENS = 3_300
EST_SYNTH_OUTPUT_TOKENS = 300
EST_QUERY_EMBED_TOKENS = 40
# Per-ok-query RAGAS judge heuristic: faithfulness reads ~5 chunks of context,
# answer-relevancy generates ~3 reverse questions -> ~4k in / ~500 out sonnet.
EST_RAGAS_INPUT_TOKENS = 4_000
EST_RAGAS_OUTPUT_TOKENS = 500

# System-2 estimate inputs (R10): route + decompose + one grade per hop, all on
# Haiku; synthesis is the Sonnet base already counted below.
EST_S2_HAIKU_CALLS = 2 + S2_MAX_HOPS  # route + decompose + S2_MAX_HOPS grades
EST_S2_HAIKU_INPUT_TOKENS = 1_200  # ~5 chunks x 200 tokens + prompt
EST_S2_HAIKU_OUTPUT_TOKENS = 100

# Graph-recognition estimate inputs (G4): one Haiku call filtering ~GRAPH_SEED_TOP_N
# seed phrases/triples down to the query-relevant set.
EST_GRAPH_RECOGNITION_INPUT_TOKENS = 900
EST_GRAPH_RECOGNITION_OUTPUT_TOKENS = 80


@dataclass(frozen=True)
class SkippedCell:
    preset: str
    reason: str


class SpendCapExceeded(RuntimeError):
    pass


def new_run_id(corpus_id: str, slice_name: str) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return f"{corpus_id}-{slice_name}-{stamp}-{uuid.uuid4().hex[:6]}"


def estimate_run_cost(preset_names: list[str], n_queries: int, *, ragas: bool = False) -> float:
    """Pre-run cost estimate (spec: estimate + confirmation gate + hard cap).

    AUTO presets are priced as a System-2 upper bound (R10): every query
    escalates and spends all S2_MAX_HOPS hops. ragas=True keeps Plan B's
    per-ok-query judge heuristic (conservative; the HARD CAP still meters
    only tracked spend — RAGAS judge usage stays untracked and disclosed).
    """
    total = 0.0
    for name in preset_names:
        cfg = PRESETS[name]
        per_q = usd_for_tokens(SYNTH_MODEL, EST_SYNTH_INPUT_TOKENS, EST_SYNTH_OUTPUT_TOKENS)
        if cfg.query.dense:
            per_q += usd_for_tokens(EMBED_MODEL, EST_QUERY_EMBED_TOKENS, 0)
        if cfg.query.rerank:
            per_q += usd_for_rerank(1)
        if cfg.query.graph and cfg.query.graph_recognition == "llm":
            # Graph recognition is one Haiku call per query (priced like the router);
            # embedding-mode graph makes no LLM recognition call.
            per_q += usd_for_tokens(
                ROUTER_MODEL,
                EST_GRAPH_RECOGNITION_INPUT_TOKENS,
                EST_GRAPH_RECOGNITION_OUTPUT_TOKENS,
            )
        if cfg.query.route_mode is not RouteMode.FORCE_S1:
            per_q += EST_S2_HAIKU_CALLS * usd_for_tokens(
                ROUTER_MODEL, EST_S2_HAIKU_INPUT_TOKENS, EST_S2_HAIKU_OUTPUT_TOKENS
            )
            extra_hops = S2_MAX_HOPS - 1  # the first hop's retrieval is the base
            if cfg.query.dense:
                per_q += extra_hops * usd_for_tokens(EMBED_MODEL, EST_QUERY_EMBED_TOKENS, 0)
            if cfg.query.rerank:
                per_q += extra_hops * usd_for_rerank(1)
        if ragas:
            per_q += usd_for_tokens(JUDGE_MODEL, EST_RAGAS_INPUT_TOKENS, EST_RAGAS_OUTPUT_TOKENS)
        total += per_q * n_queries
    return total


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _percentile(values: list[float], pct: float) -> float:
    vals = sorted(values)
    idx = max(0, math.ceil(pct * len(vals)) - 1)
    return vals[idx]


def _chunk_from_stored(d: dict, corpus_id: str) -> Chunk:
    doc_id, _, position = d["chunk_id"].rpartition(":")  # chunk_id = f"{doc_id}:{position}"
    return Chunk(
        chunk_id=d["chunk_id"],
        corpus_id=corpus_id,
        doc_id=doc_id,
        passage_id=d["passage_id"],
        text=d["text"],
        position=int(position),
        start_token=int(d["start_token"]),  # R3: span hits need token offsets
        end_token=int(d["end_token"]),
    )


def _used_index_hashes(cfg: PipelineConfig, manifest: dict) -> dict:
    """Record only the index-variant hashes this cell actually used."""
    hashes = manifest["index_hashes"]
    used: dict = {}
    if cfg.query.bm25:
        used["sparse"] = hashes["sparse"]
    if cfg.query.dense:
        key = "dense_contextual" if cfg.ingest.contextual else "dense_isolated"
        used[key] = hashes[key]
    return used


def _config_as_dict(cfg: PipelineConfig) -> dict:
    query = dataclasses.asdict(cfg.query)
    query["route_mode"] = cfg.query.route_mode.value
    return {
        "name": cfg.name,
        "ingest": dataclasses.asdict(cfg.ingest),
        "query": query,
    }


class AblationRunner:
    def __init__(
        self,
        *,
        core_factory: Callable[[PipelineConfig], RetrievalCore],
        claude: ClaudeTransport,
        store: RunStore,
        data_dir: Path,
        ragas: RagasJudge | None = None,
        clock: Callable[[], float] = time.perf_counter,
        trace_store: TraceStore | None = None,
    ) -> None:
        self._core_factory = core_factory
        self._claude = claude
        self._store = store
        self._data_dir = data_dir
        self._ragas = ragas
        self._clock = clock
        # Default resolves next to the corpora dirs (the same data_dir the
        # runner already uses); tests pass a tmp_path-backed store.
        self._trace_store = trace_store or TraceStore(data_dir / "traces-eval.sqlite3")

    def run(
        self,
        *,
        run_id: str,
        corpus_id: str,
        slice_name: str,
        presets: list[str],
        spend_cap_usd: float,
        graph_recognition: str | None = None,
    ) -> dict:
        manifest = load_manifest(self._data_dir, corpus_id)
        queries = slice_queries(
            load_queries(self._data_dir, corpus_id),
            slice_query_ids(self._data_dir, corpus_id, slice_name),
        )
        self._store.start_run(
            run_id=run_id,
            corpus_id=corpus_id,
            slice_name=slice_name,
            presets=presets,
            spend_cap_usd=spend_cap_usd,
        )
        receipts: list[Receipt] = []
        skipped: list[SkippedCell] = []
        results_by_preset: dict[str, dict] = {}
        for name in presets:
            cfg = self._apply_graph_recognition(PRESETS[name], graph_recognition)
            if cfg.query.route_mode is not RouteMode.FORCE_S1:
                # R10: permanent gate — AUTO presets run on multi-hop corpora only.
                # (The temporary "requires Plan C" skip that used to follow is gone.)
                dataset = manifest.get("dataset", {}).get("name", "")
                if dataset not in MULTI_HOP_DATASETS:
                    skipped.append(
                        SkippedCell(
                            preset=name,
                            reason=(
                                f"skipped: '{name}' runs on the multi-hop corpus only; "
                                f"corpus dataset is '{dataset}'"
                            ),
                        )
                    )
                    continue
            self._run_preset(run_id=run_id, cfg=cfg, queries=queries, spend_cap_usd=spend_cap_usd)
            receipt = self._build_receipt(
                run_id=run_id,
                corpus_id=corpus_id,
                cfg=cfg,
                manifest=manifest,
                queries=queries,
                results_by_preset=results_by_preset,
            )
            results_by_preset[name] = receipt.metrics
            receipts.append(receipt)
        doc = make_run_doc(
            run_id=run_id,
            corpus_id=corpus_id,
            slice_name=slice_name,
            receipts=receipts,
            skipped=skipped,
        )
        write_run_doc(doc, self._data_dir)
        return doc

    @staticmethod
    def _apply_graph_recognition(
        cfg: PipelineConfig, graph_recognition: str | None
    ) -> PipelineConfig:
        """G4 recognition mini-ablation: override graph_recognition on graph presets.

        Returns the preset unchanged unless it is a graph preset AND an override was
        requested; the override flows into the cell's QueryConfig so the core_factory
        (which reads cfg.query.graph_recognition) builds the GraphRetriever in the
        requested mode, and the receipt's config records it.
        """
        if graph_recognition is None or not cfg.query.graph:
            return cfg
        if graph_recognition not in ("llm", "embedding"):
            raise ValueError(
                f"graph_recognition must be 'llm' or 'embedding', got {graph_recognition!r}"
            )
        new_query = dataclasses.replace(cfg.query, graph_recognition=graph_recognition)
        return dataclasses.replace(cfg, query=new_query)

    def _run_preset(
        self,
        *,
        run_id: str,
        cfg: PipelineConfig,
        queries: list[QueryRecord],
        spend_cap_usd: float,
    ) -> None:
        done = self._store.completed_query_ids(run_id, cfg.name)
        core = self._core_factory(cfg)
        per_query_estimate = estimate_run_cost([cfg.name], 1)
        for q in queries:
            if q.query_id in done:
                continue  # resumable: completed queries are never re-billed
            spent = self._store.spent_usd(run_id)
            if spent + per_query_estimate > spend_cap_usd:
                raise SpendCapExceeded(
                    f"hard spend cap hit mid-run: spent ${spent:.4f} of cap "
                    f"${spend_cap_usd:.2f} before query {q.query_id!r} in preset "
                    f"{cfg.name!r}. State is saved; resume with --run-id {run_id} "
                    f"and a higher --spend-cap-usd."
                )
            t0 = self._clock()
            try:
                result = run_query(
                    query=q.question,
                    core=core,
                    claude=self._claude,
                    store=self._trace_store,
                    config=cfg,
                )
                latency_ms = (self._clock() - t0) * 1000.0
                events = self._trace_store.get(result.trace_id)
                tin = sum(e.input_tokens for e in events)
                tout = sum(e.output_tokens for e in events)
                # R10: actual per-query usd from traced (model, in, out) tokens.
                usd = sum(
                    usd_for_tokens(e.model, e.input_tokens, e.output_tokens)
                    for e in events
                    if e.model is not None
                )
                n_retrievals = result.hops_used if result.system == "s2" else 1
                if cfg.query.rerank:
                    usd += usd_for_rerank(n_retrievals)
                if cfg.query.dense:
                    usd += usd_for_tokens(EMBED_MODEL, n_retrievals * EST_QUERY_EMBED_TOKENS, 0)
                self._store.record_result(
                    run_id=run_id,
                    preset=cfg.name,
                    query_id=q.query_id,
                    status="abstained" if result.final.abstained else "ok",
                    retrieved=[
                        {
                            "chunk_id": sc.chunk.chunk_id,
                            "passage_id": sc.chunk.passage_id,
                            "start_token": sc.chunk.start_token,  # R3: span-gold
                            "end_token": sc.chunk.end_token,  # hits stay computable
                            "text": sc.chunk.text,
                        }
                        for sc in result.retrieved
                    ],
                    answer=result.final.text,
                    latency_ms=latency_ms,
                    usd=usd,
                    input_tokens=tin,
                    output_tokens=tout,
                    error=None,
                    route=result.system,
                )
            except Exception as exc:  # disclosed, never batch-fatal
                latency_ms = (self._clock() - t0) * 1000.0
                self._store.record_result(
                    run_id=run_id,
                    preset=cfg.name,
                    query_id=q.query_id,
                    status="failed",
                    retrieved=[],
                    answer=None,
                    latency_ms=latency_ms,
                    usd=0.0,
                    input_tokens=0,
                    output_tokens=0,
                    error=repr(exc),
                    route=None,
                )

    def _build_receipt(
        self,
        *,
        run_id: str,
        corpus_id: str,
        cfg: PipelineConfig,
        manifest: dict,
        queries: list[QueryRecord],
        results_by_preset: dict[str, dict],
    ) -> Receipt:
        by_id = {q.query_id: q for q in queries}
        rows = [r for r in self._store.results_for(run_id, cfg.name) if r["query_id"] in by_id]
        scored = [r for r in rows if r["status"] in ("ok", "abstained")]
        failed = [r for r in rows if r["status"] == "failed"]

        recalls: list[float] = []
        mrrs: list[float] = []
        ems: list[float] = []
        f1s: list[float] = []
        for r in scored:
            q = by_id[r["query_id"]]
            chunks = [_chunk_from_stored(d, corpus_id) for d in r["retrieved"]]
            recalls.append(recall_at_k(chunks, q.golds, k=5))
            mrrs.append(mrr_at_k(chunks, q.golds, k=3))
            ems.append(exact_match(r["answer"] or "", q.gold_answers))
            f1s.append(f1(r["answer"] or "", q.gold_answers))

        ragas_faith = ragas_rel = None
        ragas_flags: dict = {}
        if self._ragas is not None:
            ok_rows = [r for r in scored if r["status"] == "ok"]  # no abstentions
            faiths: list[float] = []
            rels: list[float] = []
            for r in ok_rows:
                q = by_id[r["query_id"]]
                s = self._ragas.score(
                    question=q.question,
                    answer=r["answer"] or "",
                    contexts=[d["text"] for d in r["retrieved"]],
                )
                faiths.append(s.faithfulness)
                rels.append(s.answer_relevancy)
            ragas_faith = _mean(faiths)
            ragas_rel = _mean(rels)
            ragas_flags = {"ragas_judge_usd_untracked": True}

        latencies = [r["latency_ms"] for r in rows]
        metrics = {
            "recall_at_5": _mean(recalls),
            "mrr_at_3": _mean(mrrs),
            "em": _mean(ems),
            "f1": _mean(f1s),
            "ragas_faithfulness": ragas_faith,
            "ragas_answer_relevancy": ragas_rel,
            "latency_p50_ms": _percentile(latencies, 0.50) if latencies else None,
            "latency_p95_ms": _percentile(latencies, 0.95) if latencies else None,
            "usd_per_query": (sum(r["usd"] for r in scored) / len(scored) if scored else None),
        }

        n_s1 = sum(1 for r in rows if r.get("route") == "s1")
        n_s2 = sum(1 for r in rows if r.get("route") == "s2")
        metrics["n_s1"] = n_s1
        metrics["n_s2"] = n_s2
        if n_s2 > 0:
            # Contract: router-on retrieval recall is a secondary diagnostic over
            # the union of per-hop top-5, flagged union_of_hops; primary metrics
            # are EM/F1 + RAGAS.
            metrics["union_of_hops"] = True
            metrics["recall_union_of_hops"] = metrics["recall_at_5"]
            metrics["recall_at_5"] = None
            metrics["mrr_at_3"] = None

        anchors = []
        for spec in ANCHOR_SPECS.get(cfg.name, []):
            baseline = results_by_preset.get(spec.baseline_preset or "")
            if (
                baseline is None
                or metrics.get(spec.metric) is None
                or baseline.get(spec.metric) is None
            ):
                continue  # baseline cell absent from this run; CLI discloses
            anchors.append(
                build_anchor(
                    spec,
                    metrics[spec.metric] - baseline[spec.metric],
                    corpus_id=corpus_id,  # R11: nq-dev-300 appends the scale caveat
                )
            )

        per_query = []
        for r in rows:
            flags: dict = {"status": r["status"]}
            if r["status"] in ("ok", "abstained"):
                q = by_id[r["query_id"]]
                flags["em"] = exact_match(r["answer"] or "", q.gold_answers)
                flags["f1"] = f1(r["answer"] or "", q.gold_answers)
            if r["error"]:
                flags["error"] = r["error"]
            flags.update(ragas_flags)
            per_query.append(
                {
                    "query_id": r["query_id"],
                    "retrieved_chunk_ids": [d["chunk_id"] for d in r["retrieved"]],
                    "answer": r["answer"],
                    "latency_ms": r["latency_ms"],
                    "usd": r["usd"],
                    "route": r.get("route"),
                    "flags": flags,
                }
            )

        return Receipt(
            run_id=run_id,
            corpus_id=corpus_id,
            preset=cfg.name,
            config=_config_as_dict(cfg),
            index_hashes=_used_index_hashes(cfg, manifest),
            models={
                "router": ROUTER_MODEL,
                "synth": SYNTH_MODEL,
                "judge": JUDGE_MODEL,
                "rerank": RERANK_MODEL,
                "embed": EMBED_MODEL,
            },
            pricing_table_version=PRICING_VERSION,
            prompts_version=PROMPTS_VERSION,  # R11: was "n/a"
            n_total=len(rows),
            n_failed=len(failed),
            n_abstained=sum(1 for r in rows if r["status"] == "abstained"),
            metrics=metrics,
            per_query=per_query,
            anchors=anchors,
        )
