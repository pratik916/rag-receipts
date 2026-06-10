"""VoyageClient: batch planning against the 120K/16K/1K caps, per-chunk 32K cap,
retry honoring retry-after, doc-grouped vs query embedding calls."""

from types import SimpleNamespace

import pytest
import voyageai.error

from ragreceipts.vendors.base import VendorUnavailable
from ragreceipts.vendors.voyage_client import (
    MAX_TOKENS_PER_CHUNK,
    MAX_TOKENS_PER_REQUEST,
    VoyageClient,
    plan_batches,
)


def embed_response(per_doc_chunk_counts: list[int], dim: int = 4) -> SimpleNamespace:
    results = [
        SimpleNamespace(
            index=i,
            embeddings=[[float(i), float(j)] + [0.0] * (dim - 2) for j in range(n)],
        )
        for i, n in enumerate(per_doc_chunk_counts)
    ]
    return SimpleNamespace(results=results, total_tokens=0)


class StubSdk:
    """Scripted voyageai.Client stand-in: pops responses; raises Exception items."""

    def __init__(self, script):
        self.script = list(script)
        self.calls: list[dict] = []

    def contextualized_embed(self, *, inputs, model, input_type):
        self.calls.append({"inputs": inputs, "model": model, "input_type": input_type})
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def words(texts: list[str]) -> int:
    return sum(len(t.split()) for t in texts)


class TestPlanBatches:
    def test_splits_on_token_budget(self):
        docs = [["a"], ["b"], ["c"]]
        assert plan_batches(docs, [50, 60, 30], max_tokens=100) == [(0, 1), (1, 3)]

    def test_splits_on_chunk_cap(self):
        docs = [["x"] * 3, ["y"] * 3]
        assert plan_batches(docs, [1, 1], max_chunks=4) == [(0, 1), (1, 2)]

    def test_splits_on_doc_cap(self):
        docs = [["a"], ["b"], ["c"]]
        assert plan_batches(docs, [1, 1, 1], max_docs=2) == [(0, 2), (2, 3)]

    def test_single_doc_over_budget_raises(self):
        with pytest.raises(ValueError):
            plan_batches([["a"]], [MAX_TOKENS_PER_REQUEST + 1])

    def test_empty(self):
        assert plan_batches([], []) == []


class TestEmbedDocuments:
    def test_batches_and_reassembles_in_order(self):
        # two 10-chunk docs at 7K tokens/chunk = 70K tokens/doc -> two requests
        docs = [[f"d{d}c{c}" for c in range(10)] for d in range(2)]
        sdk = StubSdk([embed_response([10]), embed_response([10])])
        client = VoyageClient(sdk=sdk, count_tokens=lambda texts: 7_000 * len(texts))
        out = client.embed_documents(docs)
        assert len(sdk.calls) == 2
        assert sdk.calls[0]["inputs"] == docs[:1]
        assert sdk.calls[0]["model"] == "voyage-context-3"
        assert sdk.calls[0]["input_type"] == "document"
        assert len(out) == 2 and [len(d) for d in out] == [10, 10]

    def test_per_chunk_token_cap_enforced(self):
        sdk = StubSdk([])
        client = VoyageClient(sdk=sdk, count_tokens=lambda texts: MAX_TOKENS_PER_CHUNK + 1)
        with pytest.raises(ValueError):
            client.embed_documents([["one oversized chunk"]])
        assert sdk.calls == []

    def test_empty_input(self):
        assert VoyageClient(sdk=StubSdk([]), count_tokens=words).embed_documents([]) == []

    def test_retry_honors_retry_after_then_succeeds(self):
        sleeps = []
        sdk = StubSdk(
            [
                voyageai.error.RateLimitError(
                    "rate limited", http_status=429, headers={"retry-after": "2"}
                ),
                embed_response([1]),
            ]
        )
        client = VoyageClient(sdk=sdk, count_tokens=words, sleep=sleeps.append)
        out = client.embed_documents([["tiny doc"]])
        assert sleeps == [2.0]
        assert len(out[0][0]) == 4

    def test_exhaustion_raises_vendor_unavailable(self):
        errors = [voyageai.error.ServerError("boom", http_status=500)] * 2
        client = VoyageClient(
            sdk=StubSdk(errors), count_tokens=words, sleep=lambda s: None, max_attempts=2
        )
        with pytest.raises(VendorUnavailable):
            client.embed_documents([["tiny doc"]])


class TestEmbedQuery:
    def test_query_call_shape_and_result(self):
        sdk = StubSdk([embed_response([1])])
        got = VoyageClient(sdk=sdk, count_tokens=words).embed_query("what is rrf?")
        assert sdk.calls[0]["inputs"] == [["what is rrf?"]]
        assert sdk.calls[0]["input_type"] == "query"
        assert got == [0.0, 0.0, 0.0, 0.0]
