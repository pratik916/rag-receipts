"""JobRunner: SQLite-keyed jobs on a single worker thread (contracts §Server)."""

import sqlite3
import time

import pytest

from ragreceipts.server.jobs import JobRunner, JobStatus


def wait_for(predicate, timeout: float = 10.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return
        time.sleep(0.05)
    raise AssertionError("condition not met within timeout")


def test_submit_runs_handler_and_records_events(tmp_path):
    runner = JobRunner(tmp_path / "jobs.sqlite")
    seen = {}

    def handler(ctx):
        seen["params"] = ctx.params
        ctx.emit("halfway", 0.5)
        ctx.emit("done", 1.0)

    runner.register("demo", handler)
    runner.start()
    try:
        job_id = runner.submit("demo", {"corpus_id": "c1"})
        wait_for(lambda: runner.get(job_id).status == JobStatus.SUCCEEDED)
    finally:
        runner.stop()
    assert seen["params"] == {"corpus_id": "c1"}
    events = runner.events(job_id)
    assert [e.message for e in events] == ["halfway", "done"]
    assert events[-1].progress == 1.0
    assert events[0].seq == 1 and events[1].seq == 2


def test_failed_job_records_error(tmp_path):
    runner = JobRunner(tmp_path / "jobs.sqlite")

    def handler(ctx):
        raise RuntimeError("boom")

    runner.register("demo", handler)
    runner.start()
    try:
        job_id = runner.submit("demo", {})
        wait_for(lambda: runner.get(job_id).status == JobStatus.FAILED)
    finally:
        runner.stop()
    assert "boom" in runner.get(job_id).error


def test_submit_unknown_kind_raises(tmp_path):
    runner = JobRunner(tmp_path / "jobs.sqlite")
    with pytest.raises(ValueError, match="no handler registered"):
        runner.submit("nope", {})


def test_crash_recovery_marks_interrupted_and_resume_requeues(tmp_path):
    db = tmp_path / "jobs.sqlite"
    first = JobRunner(db)
    first.register("demo", lambda ctx: None)
    job_id = first.submit("demo", {"n": 1})  # never started -> stays QUEUED
    # Simulate a crash mid-run: force the row to RUNNING, then "restart" the process
    # by constructing a fresh JobRunner over the same DB.
    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE jobs SET status = 'running' WHERE job_id = ?", (job_id,))
    second = JobRunner(db)
    assert second.get(job_id).status == JobStatus.INTERRUPTED

    ran = []
    second.register("demo", lambda ctx: ran.append(ctx.params["n"]))
    second.start()
    try:
        second.resume(job_id)
        wait_for(lambda: second.get(job_id).status == JobStatus.SUCCEEDED)
    finally:
        second.stop()
    assert ran == [1]


def test_resume_rejects_jobs_that_are_not_resumable(tmp_path):
    runner = JobRunner(tmp_path / "jobs.sqlite")
    runner.register("demo", lambda ctx: None)
    job_id = runner.submit("demo", {})  # QUEUED, not started
    with pytest.raises(ValueError, match="not resumable"):
        runner.resume(job_id)


def test_list_filters_by_kind_and_orders_newest_first(tmp_path):
    runner = JobRunner(tmp_path / "jobs.sqlite")
    runner.register("a", lambda ctx: None)
    runner.register("b", lambda ctx: None)
    ja = runner.submit("a", {})
    time.sleep(0.01)
    jb = runner.submit("b", {})
    assert [r.job_id for r in runner.list()] == [jb, ja]
    assert [r.job_id for r in runner.list(kind="a")] == [ja]
