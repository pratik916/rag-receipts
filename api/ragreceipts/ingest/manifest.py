"""Corpus manifest (binding JSON shape from contracts). tokenizer_artifact is stored as a
path RELATIVE to the corpus dir (machine-independent); its bytes are hashed into the
sparse hash because hash_files covers every file in sparse/."""

import json
from datetime import UTC, datetime
from pathlib import Path


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
