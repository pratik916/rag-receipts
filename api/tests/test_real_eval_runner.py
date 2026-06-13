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


def test_build_runner_threads_graph_factory_mirroring_serving(tmp_path, monkeypatch):
    """Serving/eval symmetry on the server eval plane: RealEvalRunner._build_runner (the
    production default runner_factory) must pass a graph_factory into AblationRunner that
    mirrors serving (server/pipeline._default_graph_retriever_factory). On a corpus with a
    graph artifact, calling it with router-on yields the SAME `.retrieve`-shaped graph-only
    core the live server builds — so a server-triggered router-on eval reaches graph
    instead of silently falling back to s1 (the unwired eval-plane bug)."""
    import ragreceipts.cli as cli
    import ragreceipts.eval.runner as runner_mod
    from ragreceipts.ingest.chunk_store import write_chunks
    from ragreceipts.retrieval.core import RetrievalCore
    from tests.fakes import FakeEmbed
    from tests.graph_fixtures import fixture_chunks, write_graph_corpus

    captured: dict = {}

    class _Runner:
        def __init__(self, **kw):
            captured.update(kw)

    write_graph_corpus(tmp_path)  # real offline graph artifact + raw/ + manifest
    # serving reads the canonical chunks.jsonl; the graph route-core resolves chunk-by-id
    # against it, so the corpus must carry it (write_graph_corpus only writes raw/+graph/).
    write_chunks(tmp_path / "corpora" / "graph-harness" / "chunks.jsonl", fixture_chunks())
    monkeypatch.setattr(runner_mod, "AblationRunner", _Runner)
    monkeypatch.setattr(cli, "_make_claude", lambda: object())
    monkeypatch.setattr(cli, "build_embed_transport", lambda: FakeEmbed())

    RealEvalRunner(data_dir=tmp_path)._build_runner("graph-harness")
    graph_factory = captured["graph_factory"]
    assert graph_factory is not None
    from ragreceipts.config import PRESETS

    route_core = graph_factory(PRESETS["router-on"])
    assert isinstance(route_core, RetrievalCore)
    assert hasattr(route_core, "retrieve")
    assert route_core._graph is not None
    assert route_core._sparse is None and route_core._dense is None


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
