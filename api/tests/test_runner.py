"""Runner unit tests: skips, cost guard, resume, disclosure, receipt building.

All vendor traffic goes through local Protocol stubs (ClaudeTransport from the
contracts) - zero keys, zero network. Corpus fixtures use Spike 0's raw
layout (R1/R2): raw/queries.jsonl with typed golds + slice-file query-id
lists.
"""

import json
from pathlib import Path

import pytest

from ragreceipts.agents.prompts import PROMPTS_VERSION
from ragreceipts.agents.schemas import FinalAnswer
from ragreceipts.eval.ragas_adapter import RagasScores
from ragreceipts.eval.run_state import RunStore
from ragreceipts.eval.runner import (
    AblationRunner,
    SpendCapExceeded,
    estimate_run_cost,
)
from ragreceipts.traces.store import TraceStore
from ragreceipts.types import Chunk, ScoredChunk
from ragreceipts.vendors.base import ParsedResult
from tests.fakes import FakeRagas

# ---------- local stubs (contracts Protocols) ----------


class StubClaude:
    """ClaudeTransport stub for the graph's S1 path; answers keyed by question.

    force_s1 presets enter the graph at s1_retrieve, so the only Claude call
    per query is s1_answer: parse(output_format=FinalAnswer) with the
    S1_ANSWER_USER prompt ('Question: {query}\n\nContext passages:\n{context}').
    """

    def __init__(self, answers: dict[str, FinalAnswer]) -> None:
        self._answers = answers
        self.parse_calls = 0

    def complete(self, *, model, system, user, max_tokens, temperature=0.0):
        raise AssertionError("the S1 graph path uses parse(), not complete()")

    def parse(self, *, model, system, user, max_tokens, output_format, temperature=0.0):
        self.parse_calls += 1
        question = user.split("Question: ", 1)[1].split("\n", 1)[0]
        return ParsedResult(parsed=self._answers[question], input_tokens=1000, output_tokens=100)


class StubCore:
    """Duck-typed RetrievalCore: fixed results per question text."""

    def __init__(
        self, results: dict[str, list[ScoredChunk]], fail_on: frozenset[str] = frozenset()
    ) -> None:
        self._results = results
        self._fail_on = fail_on

    def retrieve(self, query: str) -> list[ScoredChunk]:
        if query in self._fail_on:
            raise RuntimeError("retrieval exploded")
        return self._results[query]


def sc(passage_id: str, text: str = "some text") -> ScoredChunk:
    chunk = Chunk(
        chunk_id=f"{passage_id}:0",
        corpus_id="c",
        doc_id=passage_id,
        passage_id=passage_id,
        text=text,
        position=0,
        start_token=0,
        end_token=len(text.split()),
    )
    return ScoredChunk(chunk=chunk, score=1.0, source="bm25")


def write_eval_corpus(tmp_path: Path, corpus_id: str, dataset_name: str) -> Path:
    """Spike 0 raw layout (R1) + Plan A's ingest manifest."""
    raw = tmp_path / "corpora" / corpus_id / "raw"
    raw.mkdir(parents=True)
    lines = []
    for i in range(2):
        lines.append(
            json.dumps(
                {
                    "query_id": f"q{i}",
                    "question": f"question {i}?",
                    "answer": f"answer {i}",
                    "answer_aliases": [],
                    "gold": {"type": "passage", "passage_ids": [f"p{i}"]},
                }
            )
        )
    (raw / "queries.jsonl").write_text("\n".join(lines) + "\n")
    (raw / "slice-full.json").write_text(json.dumps(["q0", "q1"]))
    (raw / "slice-smoke.json").write_text(json.dumps(["q0", "q1"]))
    (tmp_path / "corpora" / corpus_id / "manifest.json").write_text(
        json.dumps(
            {
                "corpus_id": corpus_id,
                "dataset": {"name": dataset_name, "hf_id": "x", "split": "dev", "revision": "r"},
                "index_hashes": {
                    "dense_contextual": "sha256:c",
                    "dense_isolated": "sha256:i",
                    "sparse": "sha256:s",
                },
                "n_queries": 2,
            }
        )
    )
    return tmp_path


def make_runner(
    tmp_path: Path,
    dataset: str = "musique",
    *,
    fail_on: frozenset[str] = frozenset(),
    ragas=None,
    abstain_q1: bool = True,
) -> AblationRunner:
    data_dir = write_eval_corpus(tmp_path, "c1", dataset)
    results = {
        "question 0?": [sc("p0", "gold passage zero"), sc("x1"), sc("x2")],
        "question 1?": [sc("y1"), sc("y2"), sc("y3")],  # gold p1 NOT retrieved
    }
    answers = {
        "question 0?": FinalAnswer(text="Answer 0 [1]", citations=[1]),
        "question 1?": FinalAnswer(text="The passages do not contain this.", abstained=abstain_q1),
    }
    return AblationRunner(
        core_factory=lambda cfg: StubCore(results, fail_on=fail_on),
        claude=StubClaude(answers),
        store=RunStore(tmp_path / "runs.db"),
        data_dir=data_dir,
        ragas=ragas,
        trace_store=TraceStore(tmp_path / "traces.sqlite3"),
    )


# ---------- the permanent router-on gate (R10) ----------


def test_router_on_skipped_on_simple_corpus(tmp_path: Path) -> None:
    # PERMANENT gate (MULTI_HOP_DATASETS): a single-hop corpus is refused for
    # THIS reason even after Plan C lands. Plan C keeps and tests this gate (R10).
    runner = make_runner(tmp_path, dataset="nq")
    doc = runner.run(
        run_id="r1", corpus_id="c1", slice_name="smoke", presets=["router-on"], spend_cap_usd=5.0
    )
    assert doc["receipts"] == []
    assert "multi-hop" in doc["skipped"][0]["reason"]
    assert "requires Plan C" not in doc["skipped"][0]["reason"]


# ---------- cost estimate (hand-computed) ----------


def test_estimate_run_cost_hand_computed() -> None:
    # bm25-only/query: sonnet 3300 in x $3/M + 300 out x $15/M = 0.0099+0.0045 = 0.0144
    assert estimate_run_cost(["bm25-only"], 10) == pytest.approx(0.144)
    # rerank/query: 0.0144 + embed 40 x 0.18/1e6 (=0.0000072) + 1 search unit 0.0025
    assert estimate_run_cost(["rerank"], 1) == pytest.approx(0.0169072)
    # router-on/query (R10 S2 upper bound): rerank base 0.0169072
    #   + 5 haiku calls x (1200 x $1/M + 100 x $5/M = 0.0017) = 0.0085
    #   + 2 extra hops x (embed 0.0000072 + rerank 0.0025)    = 0.0050144
    #   = 0.0304216
    assert estimate_run_cost(["router-on"], 100) == pytest.approx(3.04216)


def test_estimate_includes_ragas_judge_heuristic_when_enabled() -> None:
    # Per-ok-query judge heuristic (assumes every query is ok - conservative):
    # sonnet 4000 in x $3/M + 500 out x $15/M = 0.012 + 0.0075 = 0.0195.
    # bm25-only/query 0.0144 + 0.0195 = 0.0339 -> x10 = 0.339
    assert estimate_run_cost(["bm25-only"], 10, ragas=True) == pytest.approx(0.339)
    # router-on with ragas: S2 upper bound 0.0304216 + judge 0.0195
    #   = 0.0499216/query -> x100 = 4.99216
    assert estimate_run_cost(["router-on"], 100, ragas=True) == pytest.approx(4.99216)


# ---------- end-to-end receipt ----------


def test_receipt_metrics_and_fields(tmp_path: Path) -> None:
    runner = make_runner(tmp_path)
    doc = runner.run(
        run_id="r1", corpus_id="c1", slice_name="smoke", presets=["bm25-only"], spend_cap_usd=5.0
    )
    receipt = doc["receipts"][0]["receipt"]
    m = receipt["metrics"]
    # q0: gold p0 at rank 1 -> recall 1.0, mrr 1.0; q1: gold absent -> 0, 0
    assert m["recall_at_5"] == pytest.approx(0.5)
    assert m["mrr_at_3"] == pytest.approx(0.5)
    # q0 answer "Answer 0 [1]" normalizes to "answer 0 1" vs gold "answer 0":
    # EM 0, F1 = 2*(2/3 * 2/2)/(2/3 + 1) = 0.8; q1 abstains -> EM 0, F1 0
    assert m["em"] == pytest.approx(0.0)
    assert m["f1"] == pytest.approx(0.4)
    # usd/query: sonnet 1000 in + 100 out = (3000 + 1500)/1e6 = 0.0045 (no dense/rerank)
    assert m["usd_per_query"] == pytest.approx(0.0045)
    assert m["ragas_faithfulness"] is None  # no judge wired -> disclosed null
    assert receipt["n_total"] == 2
    assert receipt["n_abstained"] == 1
    assert receipt["n_failed"] == 0
    assert receipt["pricing_table_version"] == "2026-06-10"
    assert receipt["prompts_version"] == PROMPTS_VERSION  # R11: populated by Plan C
    assert receipt["index_hashes"] == {"sparse": "sha256:s"}  # bm25-only: sparse only
    assert receipt["models"]["rerank"] == "rerank-v4.0-pro"
    assert receipt["config"]["query"]["route_mode"] == "force_s1"
    # R11: every envelope carries the fixed nondeterminism disclosure
    assert "nondeterministic" in doc["receipts"][0]["nondeterminism_note"]
    # run doc landed in data/receipts-local/
    assert (tmp_path / "receipts-local" / "r1.json").exists()


def test_index_hashes_select_isolated_variant_for_dense_rrf(tmp_path: Path) -> None:
    runner = make_runner(tmp_path)
    doc = runner.run(
        run_id="r1", corpus_id="c1", slice_name="smoke", presets=["dense-rrf"], spend_cap_usd=5.0
    )
    hashes = doc["receipts"][0]["receipt"]["index_hashes"]
    assert hashes == {"sparse": "sha256:s", "dense_isolated": "sha256:i"}


# ---------- failure disclosure ----------


def test_failures_disclosed_and_excluded(tmp_path: Path) -> None:
    runner = make_runner(tmp_path, fail_on=frozenset({"question 1?"}))
    doc = runner.run(
        run_id="r1", corpus_id="c1", slice_name="smoke", presets=["bm25-only"], spend_cap_usd=5.0
    )
    receipt = doc["receipts"][0]["receipt"]
    assert receipt["n_failed"] == 1
    assert receipt["n_total"] == 2
    # metrics computed over the surviving query only
    assert receipt["metrics"]["recall_at_5"] == pytest.approx(1.0)
    failed = [p for p in receipt["per_query"] if p["flags"]["status"] == "failed"]
    assert len(failed) == 1 and "RuntimeError" in failed[0]["flags"]["error"]


# ---------- spend cap + resume ----------


def test_spend_cap_aborts_midrun_then_resumes(tmp_path: Path) -> None:
    runner = make_runner(tmp_path)
    # per-query estimate for bm25-only is 0.0144; actual per query is 0.0045.
    # cap 0.016: q0 admitted (0 + 0.0144 <= 0.016), then before q1:
    # 0.0045 + 0.0144 = 0.0189 > 0.016 -> abort mid-run.
    with pytest.raises(SpendCapExceeded) as exc:
        runner.run(
            run_id="r1",
            corpus_id="c1",
            slice_name="smoke",
            presets=["bm25-only"],
            spend_cap_usd=0.016,
        )
    assert "r1" in str(exc.value)
    # resumable: same run_id, higher cap; q0 must not re-run
    claude_before = runner._claude.parse_calls
    doc = runner.run(
        run_id="r1", corpus_id="c1", slice_name="smoke", presets=["bm25-only"], spend_cap_usd=5.0
    )
    assert runner._claude.parse_calls == claude_before + 1  # only q1 ran
    assert doc["receipts"][0]["receipt"]["n_total"] == 2


# ---------- RAGAS exclusion of abstentions ----------


def test_ragas_runs_on_ok_only_and_is_disclosed(tmp_path: Path) -> None:
    fake = FakeRagas(scores=[RagasScores(faithfulness=0.9, answer_relevancy=0.8)])
    runner = make_runner(tmp_path, ragas=fake)
    doc = runner.run(
        run_id="r1", corpus_id="c1", slice_name="smoke", presets=["bm25-only"], spend_cap_usd=5.0
    )
    receipt = doc["receipts"][0]["receipt"]
    assert len(fake.calls) == 1  # q1 abstained -> excluded from RAGAS
    assert receipt["metrics"]["ragas_faithfulness"] == pytest.approx(0.9)
    assert receipt["metrics"]["ragas_answer_relevancy"] == pytest.approx(0.8)
    ok = [p for p in receipt["per_query"] if p["flags"]["status"] == "ok"][0]
    assert ok["flags"]["ragas_judge_usd_untracked"] is True
