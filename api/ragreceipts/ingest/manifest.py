"""Corpus manifest (binding JSON shape from contracts). tokenizer_artifact is stored as a
path RELATIVE to the corpus dir (machine-independent); its bytes are hashed into the
sparse hash because hash_files covers every file in sparse/. index_hashes additionally
carries "graph" (G6) when a graph artifact exists; graph_index_hash computes it."""

import json
from datetime import UTC, datetime
from pathlib import Path

from ragreceipts.ingest.hashing import hash_files


def build_manifest(
    *,
    corpus_id: str,
    dataset: dict,
    chunking: dict,
    embed_model: str,
    index_hashes: dict,
    tokenizer_artifact: str,
    n_docs: int,
    n_chunks: int,
    n_queries: int,
) -> dict:
    return {
        "corpus_id": corpus_id,
        "dataset": dataset,
        "chunking": chunking,
        "embed_model": embed_model,
        "index_hashes": index_hashes,
        "tokenizer_artifact": tokenizer_artifact,
        "n_docs": n_docs,
        "n_chunks": n_chunks,
        "n_queries": n_queries,
        "created_at": datetime.now(UTC).isoformat(),
    }


def write_manifest(corpus_dir: Path, manifest: dict) -> Path:
    path = corpus_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def read_manifest(corpus_dir: Path) -> dict:
    return json.loads((corpus_dir / "manifest.json").read_text(encoding="utf-8"))


def graph_index_hash(graph_dir: Path) -> str | None:
    """G6: sha256 over the graph artifact files (sorted by path, via hash_files).

    Returns None when the artifact dir is absent or empty, so callers omit the
    index_hashes["graph"] key entirely (never write a null) — receipts on graph
    presets pin the graph only when it actually exists.
    """
    if not graph_dir.is_dir():
        return None
    files = sorted(p for p in graph_dir.iterdir() if p.is_file())
    if not files:
        return None
    return hash_files(files)
