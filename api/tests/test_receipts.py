"""Receipt envelope round-trip + anchor-content golden tests."""

import json

import pytest

from ragreceipts.eval.receipts import (
    ANCHOR_SPECS,
    NONDETERMINISM_NOTE,
    SCHEMA_VERSION,
    PublishedAnchor,
    Receipt,
    build_anchor,
    from_envelope,
    make_run_doc,
    strip_for_commit,
    to_envelope,
)


def sample_receipt() -> Receipt:
    return Receipt(
        run_id="r1",
        corpus_id="musique-dev-300",
        preset="rerank",
        config={"name": "rerank", "ingest": {"contextual": True}, "query": {"rerank": True}},
        index_hashes={"sparse": "sha256:aa", "dense_contextual": "sha256:bb"},
        models={
            "router": "claude-haiku-4-5-20251001",
            "synth": "claude-sonnet-4-6",
            "judge": "claude-sonnet-4-6",
            "rerank": "rerank-v4.0-pro",
            "embed": "voyage-context-3",
        },
        pricing_table_version="2026-06-10",
        prompts_version="n/a",
        n_total=15,
        n_failed=1,
        n_abstained=2,
        metrics={
            "recall_at_5": 0.8,
            "mrr_at_3": 0.6,
            "em": 0.5,
            "f1": 0.55,
            "ragas_faithfulness": None,
            "ragas_answer_relevancy": None,
            "latency_p50_ms": 900.0,
            "latency_p95_ms": 2100.0,
            "usd_per_query": 0.012,
        },
        per_query=[
            {
                "query_id": "q1",
                "retrieved_chunk_ids": ["d1:0", "d2:3"],
                "answer": "Paris [1]",
                "latency_ms": 900.0,
                "usd": 0.011,
                "flags": {"status": "ok", "em": 1.0, "f1": 1.0},
            }
        ],
        anchors=[
            PublishedAnchor(
                source="arXiv 2604.01733 Table I (Cohere Rerank v4.0 Pro on T2-RAGBench)",
                published_value=0.121,
                measured_value=0.09,
                direction_match=True,
                note="financial-domain anchor; direction-match only",
            )
        ],
    )


def test_envelope_round_trip() -> None:
    receipt = sample_receipt()
    env = to_envelope(receipt)
    assert env["schema_version"] == SCHEMA_VERSION == 1
    # must survive a real JSON wire trip, not just dict identity
    restored = from_envelope(json.loads(json.dumps(env)))
    assert restored == receipt


def test_envelope_rejects_unknown_schema_version() -> None:
    env = to_envelope(sample_receipt())
    env["schema_version"] = 999
    with pytest.raises(ValueError):
        from_envelope(env)


def test_envelope_carries_fixed_nondeterminism_note() -> None:
    # R11: every envelope discloses LLM nondeterminism via one fixed string.
    env = to_envelope(sample_receipt())
    assert env["nondeterminism_note"] == NONDETERMINISM_NOTE
    assert "nondeterministic" in env["nondeterminism_note"]


def test_receipt_prompts_version_is_na_in_plan_b() -> None:
    # R11: "n/a" until Plan C populates agents.prompts.PROMPTS_VERSION.
    receipt = sample_receipt()
    assert receipt.prompts_version == "n/a"
    assert to_envelope(receipt)["receipt"]["prompts_version"] == "n/a"


def test_nq_anchor_notes_append_corpus_scale_caveat() -> None:
    # R11: nq-dev-300 anchors carry Spike 0's corpus-scale caveat (D1) -
    # query-derived ~300-page corpus, easier than open-corpus retrieval.
    spec = ANCHOR_SPECS["rerank"][0]
    nq = build_anchor(spec, measured_delta=0.04, corpus_id="nq-dev-300")
    other = build_anchor(spec, measured_delta=0.04, corpus_id="musique-dev-300")
    assert "query-derived" in nq.note and "~300" in nq.note
    assert "easier than" in nq.note
    assert "query-derived" not in other.note
    assert other.note == spec.note


def test_build_anchor_direction_match() -> None:
    spec = ANCHOR_SPECS["rerank"][0]
    assert spec.metric == "recall_at_5"
    assert spec.baseline_preset == "contextual"
    up = build_anchor(spec, measured_delta=0.04)
    down = build_anchor(spec, measured_delta=-0.02)
    assert up.direction_match is True
    assert down.direction_match is False
    assert up.published_value == pytest.approx(0.121)
    assert up.measured_value == pytest.approx(0.04)


def test_rerank_anchor_notes_carry_required_caveats() -> None:
    notes = " ".join(spec.note for spec in ANCHOR_SPECS["rerank"])
    assert "financial" in notes
    assert "direction-match only" in notes
    assert "2604.01733" in ANCHOR_SPECS["rerank"][0].source
    # both Recall@5 (+12.1pp) and MRR@3 (+17.2pp) anchors exist
    assert {s.metric for s in ANCHOR_SPECS["rerank"]} == {"recall_at_5", "mrr_at_3"}
    assert ANCHOR_SPECS["rerank"][1].published_value == pytest.approx(0.172)


def test_contextual_anchor_notes_technique_mismatch() -> None:
    note = ANCHOR_SPECS["contextual"][0].note
    assert "LLM-prefix" in note
    assert "voyage-context-3" in note
    assert "self-benchmark" in note
    assert "cross-index" in note


def test_router_on_anchor_notes_architecture_mismatch() -> None:
    note = ANCHOR_SPECS["router-on"][0].note
    assert "CRAG" in ANCHOR_SPECS["router-on"][0].source
    assert "union_of_hops" in note
    assert "answer-level" in note


def test_bm25_only_has_no_anchor() -> None:
    assert ANCHOR_SPECS["bm25-only"] == []


def test_strip_for_commit_removes_text_keeps_ids_and_metrics() -> None:
    doc = make_run_doc(
        run_id="r1",
        corpus_id="musique-dev-300",
        slice_name="smoke",
        receipts=[sample_receipt()],
        skipped=[],
    )
    stripped = strip_for_commit(doc)
    pq = stripped["receipts"][0]["receipt"]["per_query"][0]
    assert "answer" not in pq  # no model/passage text in committed receipts
    assert pq["retrieved_chunk_ids"] == ["d1:0", "d2:3"]
    assert pq["flags"]["em"] == 1.0
    # original doc untouched (deep copy)
    assert "answer" in doc["receipts"][0]["receipt"]["per_query"][0]
