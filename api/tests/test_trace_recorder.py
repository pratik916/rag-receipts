from ragreceipts.traces.models import TraceEvent
from ragreceipts.traces.recorder import TraceRecorder
from ragreceipts.traces.store import TraceStore


def test_emit_stamps_trace_id_and_increments_seq(tmp_path):
    store = TraceStore(tmp_path / "t.sqlite3")
    rec = TraceRecorder(store, "trace-9")
    rec.emit("route", {"a": 1}, model="m", input_tokens=3, output_tokens=4, duration_ms=1.0)
    rec.emit("s1_retrieve", {"b": 2})
    events = store.get("trace-9")
    assert [(e.seq, e.node) for e in events] == [(0, "route"), (1, "s1_retrieve")]
    assert events[0].model == "m" and events[0].input_tokens == 3
    assert events[1].model is None and events[1].input_tokens == 0


def test_call_accepts_trace_event_and_restamps(tmp_path):
    # Plan A's RetrievalCore on_trace callback delivers a ready TraceEvent (R9).
    store = TraceStore(tmp_path / "t.sqlite3")
    rec = TraceRecorder(store, "trace-9")
    foreign = TraceEvent(
        trace_id="other",
        seq=99,
        node="s1_retrieve",
        payload={"k": 1},
        model=None,
        input_tokens=0,
        output_tokens=0,
        duration_ms=2.0,
    )
    rec(foreign)
    events = store.get("trace-9")
    assert events[0].trace_id == "trace-9" and events[0].seq == 0
    assert events[0].node == "s1_retrieve" and events[0].payload == {"k": 1}


def test_call_accepts_kwargs_dict(tmp_path):
    # The recorder also accepts a kwargs dict for emit() — used by test doubles.
    store = TraceStore(tmp_path / "t.sqlite3")
    rec = TraceRecorder(store, "trace-9")
    rec({"node": "retrieve_hop", "payload": {"hop": 0}, "duration_ms": 3.0})
    events = store.get("trace-9")
    assert events[0].node == "retrieve_hop" and events[0].duration_ms == 3.0
