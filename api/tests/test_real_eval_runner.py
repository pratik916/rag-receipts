"""RealEvalRunner: delegates estimate/run to Plan B's R9-pinned entry points."""

from ragreceipts.eval.pricing import PRICING_VERSION
from ragreceipts.eval.runner import (
    EST_QUERY_EMBED_TOKENS,
    EST_SYNTH_INPUT_TOKENS,
    EST_SYNTH_OUTPUT_TOKENS,
)
from ragreceipts.server.evalruns import RealEvalRunner


class RecordingRunner:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def run(self, **kwargs):
        self.calls.append(kwargs)
        return {"receipts": [], "skipped": []}


def test_estimate_delegates_to_plan_b_estimate_run_cost(tmp_path):
    estimate_calls: list[tuple] = []

    def fake_estimate(preset_names, n_queries):
        estimate_calls.append((preset_names, n_queries))
        return 0.2536

    runner = RealEvalRunner(
        data_dir=tmp_path,
        n_queries_fn=lambda corpus_id, slice_name: 15,
        estimate_fn=fake_estimate,
    )
    est = runner.estimate(corpus_id="c1", preset="rerank", slice_name="smoke")
    assert estimate_calls == [(["rerank"], 15)]  # NOT a re-implemented formula
    assert est.n_queries == 15
    assert est.est_usd == 0.2536
    # rerank preset queries dense, so the token estimate includes the query embed
    assert est.est_tokens == 15 * (
        EST_SYNTH_INPUT_TOKENS + EST_SYNTH_OUTPUT_TOKENS + EST_QUERY_EMBED_TOKENS
    )
    assert est.pricing_table_version == PRICING_VERSION


def test_run_invokes_ablation_runner_with_single_preset(tmp_path):
    rec = RecordingRunner()
    messages: list[str] = []
    runner = RealEvalRunner(
        data_dir=tmp_path,
        runner_factory=lambda corpus_id: rec,
        run_id_fn=lambda corpus_id, slice_name: "run-xyz",
    )
    run_id = runner.run(
        corpus_id="c1",
        preset="rerank",
        slice_name="smoke",
        spend_cap_usd=5.0,
        emit=lambda msg, p: messages.append(msg),
    )
    assert run_id == "run-xyz"
    assert rec.calls == [
        {
            "run_id": "run-xyz",
            "corpus_id": "c1",
            "slice_name": "smoke",
            "presets": ["rerank"],
            "spend_cap_usd": 5.0,
        }
    ]
    assert any("run-xyz" in msg for msg in messages)


def test_construction_resolves_pinned_entry_points(tmp_path):
    # Drift guard in test form: the R9 names must import; signature drift reconciles
    # ONLY the adapter, never the EvalRunner protocol or the routes.
    from ragreceipts.cli import _build_core_real, _make_claude  # noqa: F401
    from ragreceipts.eval.run_state import RunStore  # noqa: F401
    from ragreceipts.eval.runner import (  # noqa: F401
        AblationRunner,
        estimate_run_cost,
        new_run_id,
    )

    RealEvalRunner(data_dir=tmp_path)  # constructs without touching Plan B or network
