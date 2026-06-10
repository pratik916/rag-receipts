"""RerankTransport over the official cohere SDK v2 (rerank-v4.0-pro, the anchor variant
benchmarked in arXiv 2604.01733; rerank-v4.0-fast available via the model param).

Binding verified 2026-06-10 against https://docs.cohere.com/reference/rerank and
cohere 7.0.3: cohere.ClientV2(api_key=...).rerank(model=, query=, documents=, top_n=)
-> response.results with .index/.relevance_score. Base error:
cohere.core.api_error.ApiError(*, headers=None, status_code=None, body=None).

Env-var contract (R9): callers standardize on COHERE_API_KEY (the SDK's own default is
CO_API_KEY); the CLI/smoke read COHERE_API_KEY and pass it here as api_key.
"""

import time
from collections.abc import Callable

import cohere
from cohere.core.api_error import ApiError

from ragreceipts.constants import RERANK_MODEL
from ragreceipts.vendors.retry import call_with_retry

_RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504}


def _should_retry(err: BaseException) -> bool:
    status = getattr(err, "status_code", None)
    return status is None or status in _RETRYABLE_STATUS


class CohereClient:
    def __init__(
        self,
        api_key: str | None = None,
        model: str = RERANK_MODEL,
        max_attempts: int = 5,
        sleep: Callable[[float], None] = time.sleep,
        sdk: object | None = None,
    ):
        self._sdk = sdk if sdk is not None else cohere.ClientV2(api_key=api_key)
        self._model = model
        self._max_attempts = max_attempts
        self._sleep = sleep

    def rerank(self, query: str, texts: list[str], top_n: int) -> list[tuple[int, float]]:
        response = call_with_retry(
            lambda: self._sdk.rerank(
                model=self._model, query=query, documents=list(texts), top_n=top_n
            ),
            retryable=(ApiError,),
            max_attempts=self._max_attempts,
            sleep=self._sleep,
            label="cohere rerank",
            should_retry=_should_retry,
        )
        pairs = [(r.index, float(r.relevance_score)) for r in response.results]
        return sorted(pairs, key=lambda p: -p[1])  # enforce desc, per protocol contract
