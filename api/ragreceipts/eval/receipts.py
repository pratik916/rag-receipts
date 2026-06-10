"""receipts.json schema (binding: contracts §receipts.json schema + R11).

- Receipt/PublishedAnchor dataclasses exactly as in the contracts; Receipt
  carries prompts_version ("n/a" until Plan C populates it from
  agents.prompts.PROMPTS_VERSION - R11).
- Versioned envelope {"schema_version": 1, "nondeterminism_note": ...,
  "receipt": {...}} per receipt; the note is one fixed string (R11).
- ANCHOR_SPECS: the published anchors for each preset cell, with REQUIRED
  machine-readable comparability caveats in `note` (spec §Eval plane and
  §Research grounding). Cross-domain anchors claim direction-match only.
  nq-dev-300 runs append the corpus-scale caveat at build_anchor time (R11).
- strip_for_commit: committed per-query records are IDs + metrics only -
  never passage text, never model answers (benchmark redistribution terms).
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

SCHEMA_VERSION = 1

# R11: fixed disclosure attached to every envelope.
NONDETERMINISM_NOTE = (
    "LLM calls are nondeterministic even at temperature=0: answer-dependent "
    "metrics (em, f1, ragas_*) can shift slightly between identical runs. "
    "Retrieval metrics (recall_at_5, mrr_at_3) are deterministic for a fixed "
    "index. Treat small answer-metric deltas as noise, not findings."
)

# R11: corpus-scale caveat from Spike 0's decisions doc (D1), appended to every
# anchor note on nq-dev-300 runs.
NQ_CORPUS_SCALE_NOTE = (
    " | nq-dev-300 corpus-scale caveat (Spike 0 decisions D1): the corpus is "
    "query-derived (~300 content-deduped Wikipedia pages), so retrieval "
    "difficulty is 'find the right chunk among ~300 pages' - easier than the "
    "open-corpus retrieval behind published numbers."
)


@dataclass(frozen=True)
class PublishedAnchor:
    source: str  # e.g. "arXiv 2604.01733 Table I"
    published_value: float
    measured_value: float
    direction_match: bool
    note: str  # REQUIRED - comparability caveats (domain/technique mismatch)


@dataclass(frozen=True)
class Receipt:
    run_id: str
    corpus_id: str
    preset: str
    config: dict  # full PipelineConfig as dict
    index_hashes: dict  # the variant hashes actually used
    models: dict  # router/synth/judge/rerank/embed model IDs
    pricing_table_version: str
    prompts_version: str  # "n/a" in Plan B; agents.prompts.PROMPTS_VERSION in Plan C (R11)
    n_total: int
    n_failed: int
    n_abstained: int
    metrics: dict  # recall_at_5, mrr_at_3, em, f1, ragas_faithfulness,
    # ragas_answer_relevancy, latency_p50_ms, latency_p95_ms,
    # usd_per_query
    per_query: list[dict]  # query_id, retrieved chunk_ids, answer, latency_ms, usd, flags
    anchors: list[PublishedAnchor]


def to_envelope(receipt: Receipt) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "nondeterminism_note": NONDETERMINISM_NOTE,
        "receipt": dataclasses.asdict(receipt),
    }


def from_envelope(data: dict) -> Receipt:
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported receipt schema_version: {data.get('schema_version')!r} "
            f"(this build reads version {SCHEMA_VERSION})"
        )
    raw = dict(data["receipt"])
    raw["anchors"] = [PublishedAnchor(**a) for a in raw["anchors"]]
    return Receipt(**raw)


# ---------------------------------------------------------------------------
# Published anchors per preset cell.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnchorSpec:
    """An anchor template: published delta on `metric` vs `baseline_preset`.

    measured_value for the receipt is computed by the runner as
    metrics[metric](preset) - metrics[metric](baseline_preset); direction_match
    compares the SIGN of that delta against the sign of published_value.
    """

    source: str
    published_value: float
    metric: str
    baseline_preset: str | None
    note: str


ANCHOR_SPECS: dict[str, list[AnchorSpec]] = {
    # Ladder base: nothing to compare against; honest empty list.
    "bm25-only": [],
    "dense-rrf": [
        AnchorSpec(
            source="arXiv 2604.01733 (T2-RAGBench: BM25 0.644 vs dense 0.587 R@5)",
            published_value=0.057,
            metric="recall_at_5",
            baseline_preset="bm25-only",
            note=(
                "Comparison mismatch disclosed: the published figure is BM25-alone beating "
                "SOTA-dense-alone by +5.7pp Recall@5 on a financial-domain benchmark (single "
                "non-peer-reviewed Apr 2026 preprint); this cell measures "
                "hybrid(BM25+dense+RRF) minus BM25-alone on NQ/MuSiQue - an adjacent but "
                "distinct comparison, included to ground the design rule that sparse "
                "retrieval must be kept. Cross-domain: direction-match only, never magnitude."
            ),
        )
    ],
    "contextual": [
        AnchorSpec(
            source=(
                "arXiv 2604.01733 (independent eval: +2-3pp for LLM-prefix contextual "
                "retrieval) vs anthropic.com/news/contextual-retrieval"
            ),
            published_value=0.025,  # midpoint of the independent +2-3pp range
            metric="recall_at_5",
            baseline_preset="dense-rrf",
            note=(
                "Technique mismatch (REQUIRED caveat): the independent +2-3pp figure is for "
                "LLM-prefix-style contextual retrieval, while this cell uses voyage-context-3 "
                "whole-document contextualized embeddings - a different technique. Voyage's "
                "own deltas and Anthropic's 35-67% failure-reduction headline are vendor "
                "self-benchmarks, verified-as-stated but not independently replicated. This "
                "cell is also a cross-index comparison (contextual vs isolated named vectors, "
                "different manifest hashes), not a query-time flag flip. Direction-match only."
            ),
        )
    ],
    "rerank": [
        AnchorSpec(
            source="arXiv 2604.01733 Table I (Cohere Rerank v4.0 Pro on T2-RAGBench)",
            published_value=0.121,
            metric="recall_at_5",
            baseline_preset="contextual",
            note=(
                "+12.1pp Recall@5 over hybrid RRF was measured on T2-RAGBench, a "
                "financial-domain benchmark, in a single non-peer-reviewed Apr 2026 preprint; "
                "our corpora are NQ/MuSiQue. Cross-domain: direction-match only, never "
                "magnitude reproduction. The rerank model matches the anchor variant exactly "
                "(rerank-v4.0-pro). Domain transfer is itself a finding, not a failure."
            ),
        ),
        AnchorSpec(
            source="arXiv 2604.01733 Table I (Cohere Rerank v4.0 Pro on T2-RAGBench)",
            published_value=0.172,
            metric="mrr_at_3",
            baseline_preset="contextual",
            note=(
                "+17.2pp MRR@3 over hybrid RRF; same financial-domain, single-preprint "
                "caveat as the Recall@5 anchor - direction-match only, never magnitude."
            ),
        ),
    ],
    "router-on": [
        AnchorSpec(
            source="arXiv 2604.01733 (CRAG 0.658 < plain hybrid fusion 0.695)",
            published_value=-0.037,
            metric="f1",
            baseline_preset="rerank",
            note=(
                "Architecture mismatch disclosed: the published number shows agentic "
                "rewriting REPLACING strong retrieval (CRAG) underperforming plain hybrid "
                "fusion, while router-on layers a System-2 loop ON TOP OF the same retrieval "
                "core - the closest independent agentic-vs-static datapoint, not the same "
                "design. Primary metrics for this cell are answer-level EM/F1 + RAGAS; "
                "retrieval recall over the union of per-hop top-5 is a secondary diagnostic "
                "flagged union_of_hops:true. Runs on the multi-hop corpus only."
            ),
        )
    ],
}


def build_anchor(
    spec: AnchorSpec, measured_delta: float, *, corpus_id: str = ""
) -> PublishedAnchor:
    note = spec.note
    if corpus_id == "nq-dev-300":
        note += NQ_CORPUS_SCALE_NOTE  # R11: corpus-scale caveat, machine-appended
    return PublishedAnchor(
        source=spec.source,
        published_value=spec.published_value,
        measured_value=round(measured_delta, 4),
        direction_match=(measured_delta > 0) == (spec.published_value > 0),
        note=note,
    )


# ---------------------------------------------------------------------------
# Run documents: data/receipts-local/<run_id>.json
# ---------------------------------------------------------------------------


def make_run_doc(
    *,
    run_id: str,
    corpus_id: str,
    slice_name: str,
    receipts: list[Receipt],
    skipped: list,  # list[SkippedCell] - kept untyped to avoid a runner import cycle
) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "corpus_id": corpus_id,
        "slice": slice_name,
        "created_at": datetime.now(UTC).isoformat(),
        "receipts": [to_envelope(r) for r in receipts],
        "skipped": [{"preset": s.preset, "reason": s.reason} for s in skipped],
    }


def write_run_doc(doc: dict, data_dir: Path) -> Path:
    path = data_dir / "receipts-local" / f"{doc['run_id']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2) + "\n")
    return path


def read_run_doc(path: Path) -> dict:
    doc = json.loads(path.read_text())
    if doc.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported run-doc schema_version in {path}")
    return doc


# Committed receipts: IDs + metrics ONLY (no passage text, no model answers).
_COMMITTED_PER_QUERY_KEYS = {
    "query_id",
    "retrieved_chunk_ids",
    "latency_ms",
    "usd",
    "flags",
}


def strip_for_commit(run_doc: dict) -> dict:
    """Deep-copy a run doc and strip per-query fields not safe to commit."""
    out = json.loads(json.dumps(run_doc))
    for env in out["receipts"]:
        env["receipt"]["per_query"] = [
            {k: v for k, v in pq.items() if k in _COMMITTED_PER_QUERY_KEYS}
            for pq in env["receipt"]["per_query"]
        ]
    return out
