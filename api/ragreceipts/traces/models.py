"""Trace model. Binding shape from contracts; the SQLite store arrives in Plan C.

TraceCallback is the seam this plan exposes: RetrievalCore emits TraceEvents through it,
and Plan C's TraceStore.append satisfies it without RetrievalCore changing.
"""

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class TraceEvent:
    trace_id: str  # one per query
    seq: int  # ordering within trace
    # node: "route"|"s1_retrieve"|"s1_answer"|"decompose"
    #       |"retrieve_hop"|"grade"|"refine"|"synthesize"
    node: str
    payload: dict  # JSON-serializable inputs/outputs/scores/flags
    model: str | None  # model ID if a Claude call happened
    input_tokens: int
    output_tokens: int
    duration_ms: float


TraceCallback = Callable[[TraceEvent], None]
