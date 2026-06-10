"""EmbedTransport over the official voyageai SDK — contextualized chunk embeddings.

Binding verified 2026-06-10 against
https://docs.voyageai.com/docs/contextualized-chunk-embeddings and voyageai 0.4.0:
client.contextualized_embed(inputs=list[list[str]], model=..., input_type=...) returns
.results (each .index/.embeddings); query embedding = inputs=[[query]], input_type="query".
Per-request caps: 1,000 docs / 16,000 chunks / 120,000 total tokens / 32K tokens-per-chunk.
The 120K context window doubles as the per-request budget (spec ingestion plane).
Isolated mode is the CALLER passing single-chunk documents (vendors/base.py contract).
SDK auto-retry is disabled (max_retries=0); retry.call_with_retry owns backoff so the
retry-after header is honored (the SDK does not honor it itself).
"""

import time
from collections.abc import Callable

import voyageai
import voyageai.error

from ragreceipts.constants import EMBED_MODEL
from ragreceipts.vendors.retry import call_with_retry

MAX_TOKENS_PER_REQUEST = 120_000
MAX_CHUNKS_PER_REQUEST = 16_000
MAX_DOCS_PER_REQUEST = 1_000
MAX_TOKENS_PER_CHUNK = 32_000

_RETRYABLE = (
    voyageai.error.RateLimitError,
    voyageai.error.ServerError,
    voyageai.error.ServiceUnavailableError,
    voyageai.error.APIConnectionError,
)


def plan_batches(
    documents: list[list[str]],
    doc_token_counts: list[int],
    max_tokens: int = MAX_TOKENS_PER_REQUEST,
    max_chunks: int = MAX_CHUNKS_PER_REQUEST,
    max_docs: int = MAX_DOCS_PER_REQUEST,
) -> list[tuple[int, int]]:
    """Greedy contiguous batching -> [start, end) doc-index pairs. A document never
    spans requests (Voyage contextualizes within a single request). Documents larger
    than the whole budget must be split upstream (spec: BYO docs >120K tokens are split
    into logical documents at ingest — Plan D concern, disclosed in the manifest)."""
    batches: list[tuple[int, int]] = []
    start = 0
    tokens = 0
    chunks = 0
    for i, doc in enumerate(documents):
        doc_tokens = doc_token_counts[i]
        if doc_tokens > max_tokens:
            raise ValueError(
                f"document {i} has {doc_tokens} tokens > {max_tokens} per-request budget; "
                "split it into logical documents upstream"
            )
        over = i > start and (
            tokens + doc_tokens > max_tokens
            or chunks + len(doc) > max_chunks
            or i - start >= max_docs
        )
        if over:
            batches.append((start, i))
            start, tokens, chunks = i, 0, 0
        tokens += doc_tokens
        chunks += len(doc)
    if start < len(documents):
        batches.append((start, len(documents)))
    return batches


class VoyageClient:
    """count_tokens defaults to the SDK's local tokenizer (downloads its vocab from the
    HF hub on FIRST use — fine for live runs, never reached in CI because tests always
    inject count_tokens)."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = EMBED_MODEL,
        max_attempts: int = 5,
        sleep: Callable[[float], None] = time.sleep,
        sdk: object | None = None,
        count_tokens: Callable[[list[str]], int] | None = None,
    ):
        self._sdk = sdk if sdk is not None else voyageai.Client(api_key=api_key, max_retries=0)
        self._model = model
        self._max_attempts = max_attempts
        self._sleep = sleep
        self._count_tokens = count_tokens or (
            lambda texts: self._sdk.count_tokens(texts, model=self._model)
        )

    def _call(self, fn):
        return call_with_retry(
            fn,
            retryable=_RETRYABLE,
            max_attempts=self._max_attempts,
            sleep=self._sleep,
            label="voyage",
        )

    def embed_documents(self, documents: list[list[str]]) -> list[list[list[float]]]:
        if not documents:
            return []
        doc_token_counts = [self._count_tokens(doc) if doc else 0 for doc in documents]
        for d, (doc, total) in enumerate(zip(documents, doc_token_counts)):
            if total <= MAX_TOKENS_PER_CHUNK:
                continue  # no single chunk can exceed the cap
            for c, chunk in enumerate(doc):
                if self._count_tokens([chunk]) > MAX_TOKENS_PER_CHUNK:
                    raise ValueError(
                        f"doc {d} chunk {c} exceeds the {MAX_TOKENS_PER_CHUNK}-token "
                        "per-chunk cap; re-chunk with a smaller chunk_size"
                    )
        out: list[list[list[float]]] = []
        for start, end in plan_batches(documents, doc_token_counts):
            batch = documents[start:end]
            result = self._call(
                lambda b=batch: self._sdk.contextualized_embed(
                    inputs=b, model=self._model, input_type="document"
                )
            )
            ordered = sorted(result.results, key=lambda r: r.index)  # index is per-request
            out.extend([list(emb) for emb in r.embeddings] for r in ordered)
        return out

    def embed_query(self, query: str) -> list[float]:
        result = self._call(
            lambda: self._sdk.contextualized_embed(
                inputs=[[query]], model=self._model, input_type="query"
            )
        )
        return list(result.results[0].embeddings[0])
