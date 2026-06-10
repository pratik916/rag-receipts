"""Shared vendor retry loop: exponential backoff (1s doubling, 30s cap) that honors a
retry-after header when the exception carries one. After max_attempts -> VendorUnavailable
(the signal RetrievalCore degrades on). Exceptions outside `retryable`, or rejected by
`should_retry`, propagate untouched — programming errors must stay loud."""

from collections.abc import Callable

from ragreceipts.vendors.base import VendorUnavailable


def retry_after_seconds(err: BaseException) -> float | None:
    headers = getattr(err, "headers", None) or {}
    raw = headers.get("retry-after") or headers.get("Retry-After")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def call_with_retry(
    fn: Callable[[], object],
    *,
    retryable: tuple[type[BaseException], ...],
    max_attempts: int,
    sleep: Callable[[float], None],
    label: str,
    should_retry: Callable[[BaseException], bool] | None = None,
):
    delay = 1.0
    last: BaseException | None = None
    for attempt in range(max_attempts):
        try:
            return fn()
        except retryable as err:
            if should_retry is not None and not should_retry(err):
                raise
            last = err
            if attempt == max_attempts - 1:
                break
            wait = retry_after_seconds(err)
            sleep(wait if wait is not None else delay)
            delay = min(delay * 2.0, 30.0)
    raise VendorUnavailable(f"{label} failed after {max_attempts} attempts: {last!r}") from last
