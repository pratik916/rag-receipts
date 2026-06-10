"""TraceEvent shape is binding (contracts); TraceCallback is the Plan A → Plan C seam."""

import dataclasses
import json

from ragreceipts.traces.models import TraceCallback, TraceEvent


def _event() -> TraceEvent:
    return TraceEvent(
        trace_id="t1",
        seq=0,
        node="s1_retrieve",
        payload={"query": "q", "results": [], "degraded": []},
        model=None,
        input_tokens=0,
        output_tokens=0,
        duration_ms=1.5,
    )


def test_trace_event_fields_and_frozen():
    event = _event()
    assert (event.trace_id, event.seq, event.node) == ("t1", 0, "s1_retrieve")
    assert dataclasses.fields(TraceEvent)[0].name == "trace_id"
    try:
        event.seq = 9  # type: ignore[misc]
        raised = False
    except dataclasses.FrozenInstanceError:
        raised = True
    assert raised


def test_payload_is_json_serializable():
    json.dumps(dataclasses.asdict(_event()))


def test_trace_callback_is_callable_alias():
    seen: list[TraceEvent] = []
    callback: TraceCallback = seen.append
    callback(_event())
    assert seen[0].trace_id == "t1"
