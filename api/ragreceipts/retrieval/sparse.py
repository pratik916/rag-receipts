"""BM25 sparse retrieval on bm25s, fully rebuilt on every ingest (no incremental indexing).

Serialization: bm25s.BM25.save/load for the index matrices, plus the Tokenizer's
vocab + stopwords artifacts (vocab.tokenizer.json / stopwords.tokenizer.json) saved
beside them — the query-time tokenizer MUST use the build-time vocab or scores drift.
Chunk row order comes from chunks.jsonl and is shared with the dense index.
No stemmer: deterministic, zero extra deps (a stemming receipt is possible future work).
"""

from pathlib import Path

import bm25s
from bm25s.tokenization import Tokenizer

from ragreceipts.types import Chunk, ScoredChunk


def _build_tokenizer(stopwords: str | list[str] = "en") -> Tokenizer:
    return Tokenizer(stemmer=None, stopwords=stopwords)


def build_sparse_index(chunks: list[Chunk], index_dir: Path) -> "SparseRetriever":
    """Builds, persists, and returns a live SparseRetriever (full rebuild semantics)."""
    if not chunks:
        raise ValueError("cannot build a sparse index from zero chunks")
    index_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = _build_tokenizer()
    corpus_tokens = tokenizer.tokenize(
        [c.text for c in chunks], return_as="tuple", show_progress=False
    )
    bm25 = bm25s.BM25()
    bm25.index(corpus_tokens, show_progress=False)
    bm25.save(str(index_dir))
    tokenizer.save_vocab(save_dir=str(index_dir))
    tokenizer.save_stopwords(save_dir=str(index_dir))
    return SparseRetriever(bm25, tokenizer, chunks)


class SparseRetriever:
    def __init__(self, bm25: bm25s.BM25, tokenizer: Tokenizer, chunks: list[Chunk]):
        self._bm25 = bm25
        self._tokenizer = tokenizer
        self._chunks = chunks

    @classmethod
    def load(cls, index_dir: Path, chunks: list[Chunk]) -> "SparseRetriever":
        """chunks must be the same list (same order) the index was built from."""
        bm25 = bm25s.BM25.load(str(index_dir))
        tokenizer = _build_tokenizer(stopwords=[])
        tokenizer.load_vocab(str(index_dir))
        tokenizer.load_stopwords(str(index_dir))
        return cls(bm25, tokenizer, chunks)

    def search(self, query: str, k: int) -> list[ScoredChunk]:
        k = min(k, len(self._chunks))  # bm25s raises ValueError when k > corpus size
        if k <= 0:
            return []
        query_tokens = self._tokenizer.tokenize(
            [query], return_as="tuple", update_vocab=False, show_progress=False
        )
        indices, scores = self._bm25.retrieve(query_tokens, k=k, show_progress=False)
        results: list[ScoredChunk] = []
        for idx, score in zip(indices[0].tolist(), scores[0].tolist()):
            if score <= 0.0:  # zero-score padding (e.g. all-stopword queries)
                continue
            results.append(ScoredChunk(chunk=self._chunks[idx], score=float(score), source="bm25"))
        return results
