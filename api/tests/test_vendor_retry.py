"""call_with_retry: backoff, retry-after, predicate, exhaustion -> VendorUnavailable."""

import pytest

from ragreceipts.vendors.base import VendorUnavailable
from ragreceipts.vendors.retry import call_with_retry


class Flaky(Exception):
    def __init__(self, message="boom", headers=None, status_code=None):
        super().__init__(message)
        self.headers = headers or {}
        self.status_code = status_code


def test_returns_on_first_success():
    assert (
        call_with_retry(
            lambda: 42, retryable=(Flaky,), max_attempts=3, sleep=lambda s: None, label="x"
        )
        == 42
    )


def test_exponential_backoff_without_retry_after():
    sleeps, attempts = [], []

    def fn():
        attempts.append(1)
        if len(attempts) < 3:
            raise Flaky()
        return "ok"

    assert (
        call_with_retry(fn, retryable=(Flaky,), max_attempts=5, sleep=sleeps.append, label="x")
        == "ok"
    )
    assert sleeps == [1.0, 2.0]


def test_retry_after_header_honored():
    sleeps, script = [], [Flaky(headers={"retry-after": "7"}), "ok"]

    def fn():
        item = script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    assert (
        call_with_retry(fn, retryable=(Flaky,), max_attempts=3, sleep=sleeps.append, label="x")
        == "ok"
    )
    assert sleeps == [7.0]


def test_exhaustion_raises_vendor_unavailable_with_cause():
    def fn():
        raise Flaky("always")

    with pytest.raises(VendorUnavailable) as exc_info:
        call_with_retry(
            fn, retryable=(Flaky,), max_attempts=2, sleep=lambda s: None, label="voyage"
        )
    assert "voyage" in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, Flaky)


def test_should_retry_predicate_reraises_non_retryable():
    def fn():
        raise Flaky("bad request", status_code=400)

    with pytest.raises(Flaky):
        call_with_retry(
            fn,
            retryable=(Flaky,),
            max_attempts=3,
            sleep=lambda s: None,
            label="x",
            should_retry=lambda e: e.status_code == 429,
        )


def test_unlisted_exceptions_propagate():
    def fn():
        raise KeyError("not a vendor error")

    with pytest.raises(KeyError):
        call_with_retry(fn, retryable=(Flaky,), max_attempts=3, sleep=lambda s: None, label="x")
