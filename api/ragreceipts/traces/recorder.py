"""Per-query trace recorder: stamps trace_id + a monotonically increasing seq.

Also serves as the RetrievalCore trace callback (Plan A): __call__ accepts either
a ready TraceEvent (re-stamped onto this trace) or a kwargs dict for emit().
"""

from __future__ import annotations

import dataclasses
import itertools

from ragreceipts.traces.models import TraceEvent
from ragreceipts.traces.store import TraceStore


class TraceRecorder:
    def __init__(self, store: TraceStore, trace_id: str):
        self.store = store
        self.trace_id = trace_id
        self._seq = itertools.count()

    def emit(
        self,
        node: str,
        payload: dict,
        *,
        model: str | None = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        duration_ms: float = 0.0,
    ) -> None:
        self.store.append(
            TraceEvent(
                trace_id=self.trace_id,
                seq=next(self._seq),
                node=node,
                payload=payload,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                duration_ms=duration_ms,
            )
        )

    def __call__(self, event: TraceEvent | dict) -> None:
        if isinstance(event, TraceEvent):
            self.store.append(
                dataclasses.replace(event, trace_id=self.trace_id, seq=next(self._seq))
            )
        else:
            self.emit(**event)
