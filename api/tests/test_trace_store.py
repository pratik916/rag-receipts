import threading

from ragreceipts.traces.models import TraceEvent
from ragreceipts.traces.store import TraceStore


def make_event(seq: int, trace_id: str = "t-1", node: str = "route") -> TraceEvent:
    return TraceEvent(
        trace_id=trace_id,
        seq=seq,
        node=node,
        payload={"query": "q", "n": seq},
        model="claude-haiku-4-5-20251001",
        input_tokens=10,
        output_tokens=5,
        duration_ms=12.5,
    )


def test_append_get_roundtrip(tmp_path):
    store = TraceStore(tmp_path / "traces.sqlite3")
    store.append(make_event(0))
    store.append(make_event(1))
    events = store.get("t-1")
    assert [e.seq for e in events] == [0, 1]
    assert events[0] == make_event(0)  # frozen dataclass equality incl. payload
    assert events[1].payload == {"query": "q", "n": 1}


def test_get_orders_by_seq_and_isolates_traces(tmp_path):
    store = TraceStore(tmp_path / "traces.sqlite3")
    store.append(make_event(2))
    store.append(make_event(0))
    store.append(make_event(1, trace_id="t-2"))
    assert [e.seq for e in store.get("t-1")] == [0, 2]
    assert [e.trace_id for e in store.get("t-2")] == ["t-2"]
    assert store.get("missing") == []


def test_wal_mode_enabled(tmp_path):
    store = TraceStore(tmp_path / "traces.sqlite3")
    mode = store._conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert mode == "wal"


def test_concurrent_appends(tmp_path):
    # The server (Plan D) runs jobs in a worker thread next to request handlers.
    store = TraceStore(tmp_path / "traces.sqlite3")

    def worker(offset: int) -> None:
        for i in range(20):
            store.append(make_event(offset + i))

    threads = [threading.Thread(target=worker, args=(o,)) for o in (0, 100)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(store.get("t-1")) == 40
