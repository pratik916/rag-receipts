from pathlib import Path

from ragreceipts.eval.run_state import RunStore


def test_start_run_is_idempotent(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs.db")
    store.start_run(
        run_id="r1", corpus_id="c", slice_name="smoke", presets=["bm25-only"], spend_cap_usd=5.0
    )
    store.start_run(
        run_id="r1", corpus_id="c", slice_name="smoke", presets=["bm25-only"], spend_cap_usd=5.0
    )  # no error, no dup


def test_record_and_resume_skips_completed(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs.db")
    store.start_run(
        run_id="r1", corpus_id="c", slice_name="smoke", presets=["bm25-only"], spend_cap_usd=5.0
    )
    store.record_result(
        run_id="r1",
        preset="bm25-only",
        query_id="q1",
        status="ok",
        retrieved=[
            {"chunk_id": "d1:0", "passage_id": "p1", "start_token": 0, "end_token": 1, "text": "t"}
        ],
        answer="Paris [1]",
        latency_ms=12.5,
        usd=0.01,
        input_tokens=100,
        output_tokens=20,
        error=None,
    )
    assert store.completed_query_ids("r1", "bm25-only") == {"q1"}
    assert store.completed_query_ids("r1", "rerank") == set()


def test_spent_usd_sums_across_presets(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs.db")
    store.start_run(
        run_id="r1", corpus_id="c", slice_name="smoke", presets=["a", "b"], spend_cap_usd=5.0
    )
    for preset, usd in (("a", 0.01), ("b", 0.02)):
        store.record_result(
            run_id="r1",
            preset=preset,
            query_id="q1",
            status="ok",
            retrieved=[],
            answer="x",
            latency_ms=1.0,
            usd=usd,
            input_tokens=1,
            output_tokens=1,
            error=None,
        )
    assert abs(store.spent_usd("r1") - 0.03) < 1e-9


def test_results_for_round_trips_rows(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "runs.db")
    store.start_run(
        run_id="r1", corpus_id="c", slice_name="smoke", presets=["a"], spend_cap_usd=5.0
    )
    store.record_result(
        run_id="r1",
        preset="a",
        query_id="q9",
        status="failed",
        retrieved=[],
        answer=None,
        latency_ms=3.0,
        usd=0.0,
        input_tokens=0,
        output_tokens=0,
        error="RuntimeError('boom')",
    )
    rows = store.results_for("r1", "a")
    assert len(rows) == 1
    assert rows[0]["status"] == "failed"
    assert rows[0]["error"] == "RuntimeError('boom')"
    assert rows[0]["retrieved"] == []
