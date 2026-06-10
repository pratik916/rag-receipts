"""Manual live smoke: ingest the first N documents of a real corpus with REAL vendor keys.

NEVER wired into CI (spec: 5-query live smoke is manual/nightly only). Costs cents.
Run from api/:
    VOYAGE_API_KEY=... uv run python scripts/live_smoke_ingest.py --corpus nq-dev-300
Optionally set COHERE_API_KEY to also smoke the rerank stage.
Artifacts land in a temp dir printed at the end; nothing under data/ is touched.
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

from qdrant_client import QdrantClient

from ragreceipts.config import IngestConfig
from ragreceipts.ingest.chunk_store import read_chunks
from ragreceipts.ingest.loaders import load_passages
from ragreceipts.ingest.pipeline import run_ingest
from ragreceipts.retrieval.dense import VECTOR_CONTEXTUAL, DenseRetriever
from ragreceipts.retrieval.rerank import RerankStage
from ragreceipts.retrieval.sparse import SparseRetriever
from ragreceipts.vendors.cohere_client import CohereClient
from ragreceipts.vendors.voyage_client import VoyageClient


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, help="e.g. nq-dev-300")
    parser.add_argument("--data-dir", type=Path, default=Path("../data"))
    parser.add_argument("--n-docs", type=int, default=5)
    args = parser.parse_args()

    if not os.environ.get("VOYAGE_API_KEY"):
        sys.exit("VOYAGE_API_KEY is required for the live smoke")

    source_corpus_dir = args.data_dir / "corpora" / args.corpus
    passages = load_passages(source_corpus_dir)
    keep_doc_ids: list[str] = []
    for passage in passages:
        if passage.doc_id not in keep_doc_ids:
            keep_doc_ids.append(passage.doc_id)
        if len(keep_doc_ids) >= args.n_docs:
            break
    subset = [p for p in passages if p.doc_id in keep_doc_ids]

    workdir = Path(tempfile.mkdtemp(prefix="ragreceipts-smoke-"))
    smoke_id = f"{args.corpus}-smoke{args.n_docs}"
    smoke_raw = workdir / "corpora" / smoke_id / "raw"
    smoke_raw.mkdir(parents=True)
    with (smoke_raw / "docs.jsonl").open("w", encoding="utf-8") as fh:
        for p in subset:
            fh.write(
                json.dumps(
                    {
                        "doc_id": p.doc_id,
                        "passage_id": p.passage_id,
                        "title": p.title,
                        "text": p.text,
                    }
                )
                + "\n"
            )
    # carry the dataset pins; no queries.jsonl in the doc subset -> n_queries == 0
    shutil.copy(
        source_corpus_dir / "raw" / "download_meta.json",
        smoke_raw / "download_meta.json",
    )

    embed = VoyageClient(api_key=os.environ["VOYAGE_API_KEY"])
    qdrant = QdrantClient(path=str(workdir / "qdrant"))
    manifest = run_ingest(
        corpus_id=smoke_id,
        data_dir=workdir,
        ingest_config=IngestConfig(),
        embed=embed,
        qdrant=qdrant,
    )
    print(json.dumps(manifest, indent=2))

    corpus_dir = workdir / "corpora" / smoke_id
    chunks = read_chunks(corpus_dir / "chunks.jsonl")
    query = subset[0].title or subset[0].text.split(".")[0]
    dense_hits = DenseRetriever(qdrant, smoke_id, VECTOR_CONTEXTUAL, embed).search(query, 3)
    sparse_hits = SparseRetriever.load(corpus_dir / "sparse", chunks).search(query, 3)
    print("query:", query)
    print("dense top-3:", [(s.chunk.chunk_id, round(s.score, 4)) for s in dense_hits])
    print("sparse top-3:", [(s.chunk.chunk_id, round(s.score, 4)) for s in sparse_hits])

    cohere_key = os.environ.get("COHERE_API_KEY")
    if cohere_key and dense_hits:
        stage = RerankStage(CohereClient(api_key=cohere_key))
        reranked = stage.rerank(query, dense_hits + sparse_hits, top_n=3)
        print("rerank top-3:", [(s.chunk.chunk_id, round(s.score, 4)) for s in reranked])
    else:
        print("rerank smoke skipped (COHERE_API_KEY not set)")
    print("smoke artifacts in:", workdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
