"""run_ingest: loaders -> chunker -> contextualizer (BOTH vector sets) -> index writers
-> manifest.json. Full rebuild of every index variant on every ingest.

Reads Spike 0's raw/ layout (R1); n_queries is counted straight off raw/queries.jsonl —
per R2 no eval-queries file is materialized here (Plan B reads raw/queries.jsonl
directly). The manifest's dataset block (incl. "name") comes from download_meta.json."""

from pathlib import Path

from qdrant_client import QdrantClient

from ragreceipts.config import IngestConfig
from ragreceipts.constants import EMBED_MODEL
from ragreceipts.ingest.chunk_store import write_chunks
from ragreceipts.ingest.chunker import chunk_document
from ragreceipts.ingest.contextualizer import embed_corpus
from ragreceipts.ingest.hashing import hash_files, hash_vectors
from ragreceipts.ingest.indexer import write_dense_index
from ragreceipts.ingest.loaders import (
    count_queries,
    group_documents,
    load_dataset_info,
    load_passages,
)
from ragreceipts.ingest.manifest import build_manifest, write_manifest
from ragreceipts.retrieval.sparse import build_sparse_index
from ragreceipts.types import Chunk
from ragreceipts.vendors.base import EmbedTransport


def run_ingest(
    *,
    corpus_id: str,
    data_dir: Path,
    ingest_config: IngestConfig,
    embed: EmbedTransport,
    qdrant: QdrantClient,
    embed_model: str = EMBED_MODEL,
) -> dict:
    corpus_dir = data_dir / "corpora" / corpus_id
    passages = load_passages(corpus_dir)
    dataset = load_dataset_info(corpus_dir)
    documents = group_documents(passages)

    chunks: list[Chunk] = []
    for doc in documents:
        chunks.extend(
            chunk_document(
                corpus_id,
                doc[0].doc_id,
                [(p.passage_id, p.text) for p in doc],
                ingest_config.chunk_size,
                ingest_config.chunk_overlap,
            )
        )
    if not chunks:
        raise ValueError(f"corpus {corpus_id} produced no chunks")
    write_chunks(corpus_dir / "chunks.jsonl", chunks)

    # Regroup chunk texts per document (chunks were generated doc-by-doc, so a doc_id
    # transition marks a new document). Docs that produced zero chunks simply don't appear.
    doc_chunk_texts: list[list[str]] = []
    current_doc: str | None = None
    for chunk in chunks:
        if chunk.doc_id != current_doc:
            doc_chunk_texts.append([])
            current_doc = chunk.doc_id
        doc_chunk_texts[-1].append(chunk.text)

    contextual, isolated = embed_corpus(doc_chunk_texts, embed)
    write_dense_index(qdrant, corpus_id, chunks, contextual, isolated)

    sparse_dir = corpus_dir / "sparse"
    build_sparse_index(chunks, sparse_dir)
    sparse_files = sorted(p for p in sparse_dir.iterdir() if p.is_file())

    manifest = build_manifest(
        corpus_id=corpus_id,
        dataset=dataset,
        chunking={
            "chunk_size": ingest_config.chunk_size,
            "chunk_overlap": ingest_config.chunk_overlap,
        },
        embed_model=embed_model,
        index_hashes={
            "dense_contextual": hash_vectors(contextual),
            "dense_isolated": hash_vectors(isolated),
            "sparse": hash_files(sparse_files),
        },
        tokenizer_artifact="sparse/vocab.tokenizer.json",
        n_docs=len(documents),
        n_chunks=len(chunks),
        n_queries=count_queries(corpus_dir),
    )
    write_manifest(corpus_dir, manifest)
    return manifest
