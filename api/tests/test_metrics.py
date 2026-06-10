"""Golden tests for the binding metric definitions (contracts §Metrics).

Every expected value below is hand-computed in the inline comments.
Span golds are positional token ranges (Spike 0's rule, R3) - never
span_text strings.
"""

import pytest

from ragreceipts.eval.alignment import GoldPassage, GoldSpan
from ragreceipts.eval.metrics import (
    exact_match,
    f1,
    mrr_at_k,
    normalize_answer,
    recall_at_k,
)
from ragreceipts.types import Chunk


def make_chunk(
    passage_id: str, text: str = "", position: int = 0, start_token: int = 0, end_token: int = 1
) -> Chunk:
    return Chunk(
        chunk_id=f"{passage_id}:{position}",
        corpus_id="t",
        doc_id=passage_id,
        passage_id=passage_id,
        text=text,
        position=position,
        start_token=start_token,
        end_token=end_token,
    )


# ---------- recall_at_k ----------


def test_recall_single_gold_hit_in_top5() -> None:
    retrieved = [make_chunk(p) for p in ["a", "b", "gold", "c", "d", "e"]]
    golds = [GoldPassage(query_id="q", passage_id="gold")]
    # gold at rank 3 (within top-5): 1 of 1 golds hit -> 1.0
    assert recall_at_k(retrieved, golds, k=5) == 1.0


def test_recall_single_gold_outside_top5() -> None:
    retrieved = [make_chunk(p) for p in ["a", "b", "c", "d", "e", "gold"]]
    golds = [GoldPassage(query_id="q", passage_id="gold")]
    # gold at rank 6 (outside top-5): 0 of 1 -> 0.0
    assert recall_at_k(retrieved, golds, k=5) == 0.0


def test_recall_multi_gold_partial() -> None:
    retrieved = [make_chunk(p) for p in ["g1", "x", "y", "z", "w"]]
    golds = [GoldPassage(query_id="q", passage_id="g1"), GoldPassage(query_id="q", passage_id="g2")]
    # g1 hit, g2 not retrieved: 1 of 2 golds -> 0.5
    assert recall_at_k(retrieved, golds, k=5) == 0.5


def test_recall_span_gold_50pct_boundary_is_hit() -> None:
    # Gold span [10, 20) = 10 tokens; chunk covers tokens [0, 15) of the
    # same doc -> overlap 5/10 = 50% -> hit (binding rule: chunk covers
    # >=50% of the gold span's tokens; integer form 2*overlap >= gold_len).
    gold = GoldSpan(query_id="q", doc_id="d1", start_token=10, end_token=20)
    chunk = make_chunk("d1", start_token=0, end_token=15)
    assert recall_at_k([chunk], [gold], k=5) == 1.0


def test_recall_span_gold_below_50pct_is_miss() -> None:
    # Chunk covers tokens [0, 14): overlap 4/10 = 40% -> miss
    gold = GoldSpan(query_id="q", doc_id="d1", start_token=10, end_token=20)
    chunk = make_chunk("d1", start_token=0, end_token=14)
    assert recall_at_k([chunk], [gold], k=5) == 0.0


def test_recall_span_gold_requires_same_document() -> None:
    # Full token overlap but a different doc_id -> miss (Spike 0's span_hit)
    gold = GoldSpan(query_id="q", doc_id="d1", start_token=0, end_token=10)
    chunk = make_chunk("d2", start_token=0, end_token=10)
    assert recall_at_k([chunk], [gold], k=5) == 0.0


def test_recall_no_golds_raises() -> None:
    with pytest.raises(ValueError):
        recall_at_k([make_chunk("a")], [], k=5)


# ---------- mrr_at_k ----------


def test_mrr_first_hit_rank1() -> None:
    retrieved = [make_chunk(p) for p in ["gold", "a", "b"]]
    golds = [GoldPassage(query_id="q", passage_id="gold")]
    assert mrr_at_k(retrieved, golds, k=3) == 1.0


def test_mrr_first_hit_rank3() -> None:
    retrieved = [make_chunk(p) for p in ["a", "b", "gold"]]
    golds = [GoldPassage(query_id="q", passage_id="gold")]
    # 1/3
    assert mrr_at_k(retrieved, golds, k=3) == pytest.approx(1 / 3)


def test_mrr_hit_at_rank4_is_zero_within_top3() -> None:
    retrieved = [make_chunk(p) for p in ["a", "b", "c", "gold"]]
    golds = [GoldPassage(query_id="q", passage_id="gold")]
    assert mrr_at_k(retrieved, golds, k=3) == 0.0


def test_mrr_multi_gold_uses_first_hit_of_any_gold() -> None:
    retrieved = [make_chunk(p) for p in ["a", "g2", "g1"]]
    golds = [GoldPassage(query_id="q", passage_id="g1"), GoldPassage(query_id="q", passage_id="g2")]
    # first chunk hitting ANY gold is rank 2 (g2) -> 1/2
    assert mrr_at_k(retrieved, golds, k=3) == 0.5


def test_mrr_span_gold_uses_first_hit_rank() -> None:
    # rank 1 misses (other doc), rank 2 covers the whole span -> 1/2
    gold = GoldSpan(query_id="q", doc_id="d1", start_token=10, end_token=20)
    retrieved = [
        make_chunk("d2", start_token=0, end_token=30),
        make_chunk("d1", start_token=8, end_token=22, position=1),
    ]
    assert mrr_at_k(retrieved, [gold], k=3) == 0.5


# ---------- normalization / EM / F1 (SQuAD-style) ----------


def test_normalize_answer() -> None:
    assert normalize_answer("The  Eiffel Tower!") == "eiffel tower"
    assert normalize_answer("A dog, an apple, the end.") == "dog apple end"


def test_exact_match_normalized() -> None:
    assert exact_match("The Eiffel Tower", ["eiffel tower"]) == 1.0
    assert exact_match("Eiffel", ["eiffel tower"]) == 0.0


def test_exact_match_multi_gold() -> None:
    assert exact_match("Paris", ["London", "paris"]) == 1.0


def test_f1_hand_computed_partial_overlap() -> None:
    # pred tokens: {paris, france} (2); gold tokens: {paris} (1); overlap 1
    # precision = 1/2, recall = 1/1, F1 = 2*(0.5*1)/(0.5+1) = 2/3
    assert f1("Paris France", ["Paris"]) == pytest.approx(2 / 3)


def test_f1_takes_max_over_golds() -> None:
    # vs "Paris": F1 = 2/3 (above). vs "Paris France": F1 = 1.0. max -> 1.0
    assert f1("Paris France", ["Paris", "paris france"]) == 1.0


def test_f1_zero_overlap() -> None:
    assert f1("London", ["Paris"]) == 0.0


def test_f1_empty_prediction_vs_nonempty_gold() -> None:
    assert f1("", ["Paris"]) == 0.0
