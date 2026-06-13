"""ragreceipts CLI.

  ragreceipts ingest --corpus <id>                  (Plan A) rebuild all index variants
  ragreceipts eval --corpus <id> --slice smoke|full \
      --presets bm25-only,dense-rrf,contextual,rerank,router-on \
      [--spend-cap-usd 5.0] [--run-id <resume-id>] [--ragas] [--yes]
  ragreceipts receipts promote <run_id>

Plan A created this module with the ingest subcommand; Plan B (R6) adds the
eval/receipts subparsers. Factories build_embed_transport/build_qdrant
(Plan A) and build_rerank_transport (Plan B) are module-level seams: tests
monkeypatch them; Plan D's server reuses them. Missing keys produce a named
env-var message, never a stack trace.
Data dir resolution (R6): RAGRECEIPTS_DATA_DIR env var, default ../data
relative to api/; `receipts promote` defaults --receipts-dir to ../receipts.

eval writes {data_dir}/receipts-local/<run_id>.json; promote copies a run to
receipts/<run_id>.json with passage text and answers stripped (IDs + metrics
only - benchmark redistribution terms).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from qdrant_client import QdrantClient

from ragreceipts.config import PRESETS, IngestConfig, PipelineConfig
from ragreceipts.eval.queries import load_queries, slice_queries, slice_query_ids
from ragreceipts.eval.receipts import read_run_doc, strip_for_commit
from ragreceipts.eval.run_state import RunStore
from ragreceipts.eval.runner import (
    AblationRunner,
    SpendCapExceeded,
    estimate_run_cost,
    new_run_id,
)
from ragreceipts.ingest.pipeline import run_ingest
from ragreceipts.retrieval.core import RetrievalCore
from ragreceipts.types import RouteMode
from ragreceipts.vendors.cohere_client import CohereClient
from ragreceipts.vendors.voyage_client import VoyageClient

DEFAULT_PRESETS = "bm25-only,dense-rrf,contextual,rerank,router-on"


def _default_data_dir() -> Path:
    """R6: RAGRECEIPTS_DATA_DIR env var, default ../data relative to api/."""
    return Path(os.environ.get("RAGRECEIPTS_DATA_DIR", "../data"))


# --- factory seams (Plan A's two kept verbatim; Plan B adds the third) ------


def build_embed_transport() -> VoyageClient:
    api_key = os.environ.get("VOYAGE_API_KEY")
    if not api_key:
        raise SystemExit(
            "VOYAGE_API_KEY is not set — ingest needs Voyage embeddings (set it in .env)"
        )
    return VoyageClient(api_key=api_key)


def build_qdrant(data_dir: Path) -> QdrantClient:
    url = os.environ.get("QDRANT_URL")
    if url:
        return QdrantClient(url=url)
    return QdrantClient(path=str(data_dir / "qdrant-local"))  # local file mode, no server


def build_rerank_transport() -> CohereClient:
    api_key = os.environ.get("COHERE_API_KEY")
    if not api_key:
        raise SystemExit(
            "COHERE_API_KEY is not set — rerank cells need Cohere rerank-v4.0-pro (set it in .env)"
        )
    return CohereClient(api_key=api_key)


def build_graph_retriever(corpus_dir: Path, config: PipelineConfig, chunks):
    """Load the graph artifact from {corpus_dir}/graph/ and build a GraphRetriever.

    Mirrors the keyed eval construction (tests/test_eval_graph.py): GraphIndex.load
    reads the byte-reproducible artifact `ragreceipts ingest`/build_graph.py wrote,
    and the retriever is constructed with the RG1 keyword-only ctor (chunks required,
    embed via the shared seam, claude only when recognition='llm'). Returns None if
    the artifact dir is absent — the corpus was ingested without a graph, so callers
    degrade honestly rather than fabricate one.
    """
    from ragreceipts.ingest.graph_index import GraphIndex
    from ragreceipts.retrieval.graph import GraphRetriever

    graph_dir = corpus_dir / "graph"
    if not graph_dir.exists():
        return None
    index = GraphIndex.load(graph_dir)
    recognition = config.query.graph_recognition
    return GraphRetriever(
        index,
        chunks=chunks,
        embed=build_embed_transport(),
        claude=_make_claude() if recognition == "llm" else None,
        recognition=recognition,
    )


def build_graph_route_core(
    corpus_dir: Path, config: PipelineConfig, chunks
) -> RetrievalCore | None:
    """A graph-only RetrievalCore for the agent-layer graph route (router-on).

    The agent graph (agents/graph.py::graph_retrieve_node) calls
    `graph_retriever.retrieve(query)` — the SupportsRetrieve protocol — so the route
    needs a `.retrieve` object, not a raw GraphRetriever (which exposes `.search`).
    Wrapping the GraphRetriever in a graph-only RetrievalCore satisfies that protocol
    AND keeps the shared-core guarantee: the graph route runs the same RetrievalCore
    the eval graph preset does. The wrapping core inherits the serving preset's
    graph_recognition and top_k_final so the route returns the same depth. Returns
    None when the corpus has no graph artifact (route stays unreachable -> s1 fallback).
    """
    import dataclasses

    graph_retriever = build_graph_retriever(corpus_dir, config, chunks)
    if graph_retriever is None:
        return None
    # Graph-only query config: only the graph retriever fires, with the serving
    # preset's recognition mode and final depth (route_mode is irrelevant here —
    # RetrievalCore.retrieve is route-agnostic).
    graph_query = dataclasses.replace(
        config.query, bm25=False, dense=False, rerank=False, graph=True
    )
    graph_config = dataclasses.replace(config, query=graph_query)
    return RetrievalCore(
        config=graph_config, dense=None, sparse=None, rerank_stage=None, graph=graph_retriever
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ragreceipts")
    sub = parser.add_subparsers(dest="command", required=True)

    # --- ingest (Plan A, preserved) ---
    ingest_p = sub.add_parser("ingest", help="(re)build all index variants for a corpus")
    ingest_p.add_argument("--corpus", required=True, help="corpus id, e.g. nq-dev-300")
    ingest_p.add_argument(
        "--data-dir",
        type=Path,
        default=_default_data_dir(),
        help="data dir holding corpora/ (default ../data, run from api/)",
    )
    ingest_p.add_argument("--chunk-size", type=int, default=IngestConfig().chunk_size)
    ingest_p.add_argument("--chunk-overlap", type=int, default=IngestConfig().chunk_overlap)

    # --- eval (Plan B) ---
    eval_p = sub.add_parser("eval", help="run the ablation ladder on a corpus slice")
    eval_p.add_argument("--corpus", required=True, help="corpus_id under {data_dir}/corpora/")
    eval_p.add_argument("--slice", choices=["smoke", "full"], default="smoke")
    eval_p.add_argument(
        "--presets", default=DEFAULT_PRESETS, help="comma-separated preset ladder subset"
    )
    eval_p.add_argument(
        "--spend-cap-usd",
        type=float,
        default=5.0,
        help="hard cap; the run aborts (resumably) when reached",
    )
    eval_p.add_argument("--run-id", default=None, help="resume an aborted run by its run_id")
    eval_p.add_argument(
        "--data-dir",
        type=Path,
        default=_default_data_dir(),
        help="data dir holding corpora/ (default ../data, run from api/)",
    )
    eval_p.add_argument(
        "--ragas",
        action="store_true",
        help="score RAGAS faithfulness/relevancy (extra Claude spend)",
    )
    eval_p.add_argument(
        "--graph-recognition",
        choices=["llm", "embedding"],
        default=None,
        help="override graph_recognition for graph presets (recognition mini-ablation)",
    )
    eval_p.add_argument(
        "--yes", action="store_true", help="skip the interactive cost confirmation gate"
    )

    # --- receipts (Plan B) ---
    receipts_p = sub.add_parser("receipts", help="manage committed receipts")
    rsub = receipts_p.add_subparsers(dest="receipts_command", required=True)
    promote_p = rsub.add_parser(
        "promote", help="copy a local run to receipts/ stripped to IDs + metrics"
    )
    promote_p.add_argument("run_id")
    promote_p.add_argument("--data-dir", type=Path, default=_default_data_dir())
    promote_p.add_argument("--receipts-dir", type=Path, default=Path("../receipts"))

    args = parser.parse_args(argv)
    if args.command == "ingest":
        return _cmd_ingest(args)
    if args.command == "eval":
        return _cmd_eval(args)
    return _cmd_promote(args)


def _cmd_ingest(args: argparse.Namespace) -> int:
    """Plan A's ingest behavior, moved out of main() verbatim."""
    try:
        manifest = run_ingest(
            corpus_id=args.corpus,
            data_dir=args.data_dir,
            ingest_config=IngestConfig(
                chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap
            ),
            embed=build_embed_transport(),
            qdrant=build_qdrant(args.data_dir),
        )
    except FileNotFoundError:
        print(
            f"error: corpus '{args.corpus}' not found under "
            f"{args.data_dir / 'corpora'} — run the Spike 0 download script first",
            file=sys.stderr,
        )
        return 1
    print(json.dumps(manifest, indent=2))
    return 0


def _missing_keys(preset_names: list[str]) -> list[str]:
    """Named env-var messages, never stack traces (spec §Error handling)."""
    runnable = [
        PRESETS[n] for n in preset_names if PRESETS[n].query.route_mode is RouteMode.FORCE_S1
    ]
    needed = {"ANTHROPIC_API_KEY": "Claude answer synthesis"}
    if any(cfg.query.dense for cfg in runnable):
        needed["VOYAGE_API_KEY"] = "voyage-context-3 query embeddings"
    if any(cfg.query.rerank for cfg in runnable):
        needed["COHERE_API_KEY"] = "Cohere rerank-v4.0-pro"
    return [
        f"missing env var {key} (needed for {why})"
        for key, why in needed.items()
        if not os.environ.get(key)
    ]


def _build_core_real(config: PipelineConfig, corpus_id: str, data_dir: Path) -> RetrievalCore:
    """Composition root (name + signature pinned by R9).

    Assembles Plan A's real retrieval stack for one preset from the
    artifacts `ragreceipts ingest` wrote (surfaces pinned by Plan A):
      - chunks: ingest/chunk_store.read_chunks({corpus_dir}/chunks.jsonl)
      - sparse: SparseRetriever.load({corpus_dir}/sparse, chunks)
      - dense:  DenseRetriever(client, collection=corpus_id,
                vector_name_for(config.ingest.contextual), embed)
      - rerank: RerankStage(transport)
    All vendor/Qdrant access flows through the module-level factory seams so
    tests can monkeypatch them (offline construction test in test_cli.py).
    """
    from ragreceipts.ingest.chunk_store import read_chunks
    from ragreceipts.retrieval.dense import DenseRetriever, vector_name_for
    from ragreceipts.retrieval.rerank import RerankStage
    from ragreceipts.retrieval.sparse import SparseRetriever

    corpus_dir = data_dir / "corpora" / corpus_id
    chunks = read_chunks(corpus_dir / "chunks.jsonl")
    sparse = SparseRetriever.load(corpus_dir / "sparse", chunks) if config.query.bm25 else None
    dense = (
        DenseRetriever(
            build_qdrant(data_dir),
            corpus_id,  # collection name == corpus_id (Plan A's ingest)
            vector_name_for(config.ingest.contextual),
            build_embed_transport(),
        )
        if config.query.dense
        else None
    )
    rerank_stage = RerankStage(build_rerank_transport()) if config.query.rerank else None
    # graph presets (graph, graph-rrf) enable query.graph: load the artifact and pass it
    # in. If config.query.graph is True but the artifact is absent, graph stays None and
    # RetrievalCore raises the existing clear "no graph retriever" error (honest failure —
    # the corpus must be (re)ingested with a graph, never a silently disabled retriever).
    graph = build_graph_retriever(corpus_dir, config, chunks) if config.query.graph else None
    return RetrievalCore(
        config=config, dense=dense, sparse=sparse, rerank_stage=rerank_stage, graph=graph
    )


def _make_claude():
    """The real ClaudeTransport (vendors/ seam, vendors/anthropic_client.py).

    Lazy import: offline tests never construct it, and the module follows
    Plan A's vendor naming convention (voyage_client.py / cohere_client.py).
    """
    from ragreceipts.vendors.anthropic_client import AnthropicClient

    return AnthropicClient()


def _cmd_eval(args: argparse.Namespace) -> int:
    preset_names = [p.strip() for p in args.presets.split(",") if p.strip()]
    unknown = [p for p in preset_names if p not in PRESETS]
    if unknown:
        print(f"error: unknown presets {unknown}; valid presets: {list(PRESETS)}", file=sys.stderr)
        return 2
    missing = _missing_keys(preset_names)
    if missing:
        for line in missing:
            print(f"error: {line}", file=sys.stderr)
        print("Set the keys in api/.env or the environment, then re-run.", file=sys.stderr)
        return 2

    queries = slice_queries(
        load_queries(args.data_dir, args.corpus),
        slice_query_ids(args.data_dir, args.corpus, args.slice),
    )
    estimate = estimate_run_cost(preset_names, len(queries), ragas=args.ragas)
    print(
        f"Run plan: corpus={args.corpus} slice={args.slice} "
        f"({len(queries)} queries) presets={preset_names}"
    )
    print(f"Estimated cost: ${estimate:.2f}  |  hard spend cap: ${args.spend_cap_usd:.2f}")
    if args.ragas:
        print(
            "Estimate includes a per-query RAGAS judge heuristic; actual "
            "judge spend is untracked in Plan B and NOT counted against "
            "the hard cap (disclosed per receipt)."
        )
    if not args.yes:
        reply = input("Proceed? [y/N] ").strip().lower()
        if reply != "y":
            print("Aborted before any spend.")
            return 1

    run_id = args.run_id or new_run_id(args.corpus, args.slice)
    ragas = None
    if args.ragas:
        from ragreceipts.eval.ragas_adapter import RagasV04Judge
        from ragreceipts.vendors.ragas_clients import make_anthropic_client

        ragas = RagasV04Judge(make_anthropic_client())
    runner = AblationRunner(
        core_factory=lambda cfg: _build_core_real(cfg, args.corpus, args.data_dir),
        claude=_make_claude(),
        store=RunStore(args.data_dir / "eval-runs.db"),
        data_dir=args.data_dir,
        ragas=ragas,
    )
    try:
        doc = runner.run(
            run_id=run_id,
            corpus_id=args.corpus,
            slice_name=args.slice,
            presets=preset_names,
            spend_cap_usd=args.spend_cap_usd,
            graph_recognition=args.graph_recognition,
        )
    except SpendCapExceeded as exc:
        print(f"ABORTED: {exc}", file=sys.stderr)
        return 3

    print(f"Wrote {args.data_dir / 'receipts-local' / (run_id + '.json')}")
    for skip in doc["skipped"]:
        print(f"  SKIPPED {skip['preset']}: {skip['reason']}")
    for env in doc["receipts"]:
        receipt = env["receipt"]
        m = receipt["metrics"]
        print(
            f"  {receipt['preset']}: recall@5={m['recall_at_5']} "
            f"mrr@3={m['mrr_at_3']} em={m['em']} f1={m['f1']} "
            f"n={receipt['n_total']} failed={receipt['n_failed']} "
            f"abstained={receipt['n_abstained']}"
        )
        if not receipt["anchors"]:
            print("    (no anchors: ladder base, or baseline cell absent from this run)")
    return 0


def _cmd_promote(args: argparse.Namespace) -> int:
    src = args.data_dir / "receipts-local" / f"{args.run_id}.json"
    if not src.exists():
        print(
            f"error: {src} not found - run `ragreceipts eval` first (run_id {args.run_id!r})",
            file=sys.stderr,
        )
        return 2
    doc = read_run_doc(src)
    stripped = strip_for_commit(doc)
    args.receipts_dir.mkdir(parents=True, exist_ok=True)
    dst = args.receipts_dir / f"{args.run_id}.json"
    dst.write_text(json.dumps(stripped, indent=2) + "\n")
    print(
        f"Promoted {len(stripped['receipts'])} receipt cell(s) to {dst} "
        f"(passage text + answers stripped; IDs + metrics only). "
        f"Review, then `git add` to commit."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
