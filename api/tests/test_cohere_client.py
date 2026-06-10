"""CohereClient: rerank call shape, desc-sorted results, status-code-based retry."""

from types import SimpleNamespace

import pytest
from cohere.core.api_error import ApiError

from ragreceipts.vendors.base import VendorUnavailable
from ragreceipts.vendors.cohere_client import CohereClient


def rerank_response(pairs: list[tuple[int, float]]) -> SimpleNamespace:
    return SimpleNamespace(results=[SimpleNamespace(index=i, relevance_score=s) for i, s in pairs])


class StubSdk:
    def __init__(self, script):
        self.script = list(script)
        self.calls: list[dict] = []

    def rerank(self, *, model, query, documents, top_n):
        self.calls.append(
            {
                "model": model,
                "query": query,
                "documents": list(documents),
                "top_n": top_n,
            }
        )
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def test_call_shape_and_mapping():
    sdk = StubSdk([rerank_response([(2, 0.9), (0, 0.5)])])
    got = CohereClient(sdk=sdk).rerank("q", ["a", "b", "c"], top_n=2)
    assert got == [(2, 0.9), (0, 0.5)]
    assert sdk.calls[0] == {
        "model": "rerank-v4.0-pro",
        "query": "q",
        "documents": ["a", "b", "c"],
        "top_n": 2,
    }


def test_results_sorted_desc_even_if_api_misorders():
    sdk = StubSdk([rerank_response([(0, 0.1), (1, 0.8)])])
    assert CohereClient(sdk=sdk).rerank("q", ["a", "b"], top_n=2) == [(1, 0.8), (0, 0.1)]


def test_retries_429_honoring_retry_after():
    sleeps = []
    sdk = StubSdk(
        [
            ApiError(status_code=429, headers={"retry-after": "3"}, body="slow down"),
            rerank_response([(0, 0.7)]),
        ]
    )
    got = CohereClient(sdk=sdk, sleep=sleeps.append).rerank("q", ["a"], top_n=1)
    assert got == [(0, 0.7)]
    assert sleeps == [3.0]


def test_5xx_retries_then_vendor_unavailable():
    errors = [ApiError(status_code=503, body="down")] * 2
    client = CohereClient(sdk=StubSdk(errors), sleep=lambda s: None, max_attempts=2)
    with pytest.raises(VendorUnavailable):
        client.rerank("q", ["a"], top_n=1)


def test_4xx_other_than_429_propagates_unretried():
    sdk = StubSdk([ApiError(status_code=400, body="bad request")])
    with pytest.raises(ApiError):
        CohereClient(sdk=sdk).rerank("q", ["a"], top_n=1)
    assert len(sdk.calls) == 1
