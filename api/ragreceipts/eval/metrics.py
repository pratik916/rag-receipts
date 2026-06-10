"""Retrieval + answer metrics.

Binding definitions: docs/superpowers/plans/2026-06-10-contracts.md §Metrics.
Gold-to-chunk alignment is owned by Spike 0's eval/alignment.py (kept
verbatim per R3): GoldPassage(query_id, passage_id) exact-ID match and
GoldSpan(query_id, doc_id, start_token, end_token) positional >=50%
token-overlap. This module never reimplements the hit rule - recall/MRR are
thin wrappers over is_hit / first_hit_rank, which work structurally on
Chunk because Chunk carries start_token/end_token (R3).
EM/F1 use the standard SQuAD normalization (lowercase, strip punctuation,
drop articles a/an/the, collapse whitespace).
"""

from __future__ import annotations

import re
import string
from collections import Counter

from ragreceipts.eval.alignment import Gold, first_hit_rank, is_hit
from ragreceipts.types import Chunk

_ARTICLES = re.compile(r"\b(a|an|the)\b")


def recall_at_k(retrieved: list[Chunk], golds: list[Gold], k: int = 5) -> float:
    """Fraction of golds with >=1 hit in the top-k retrieved chunks (per query)."""
    if not golds:
        raise ValueError("recall_at_k requires at least one gold")
    top = retrieved[:k]
    hits = sum(1 for gold in golds if any(is_hit(chunk, gold) for chunk in top))
    return hits / len(golds)


def mrr_at_k(retrieved: list[Chunk], golds: list[Gold], k: int = 3) -> float:
    """Reciprocal rank (1-based) of the first chunk in the top-k hitting ANY gold; 0 if none."""
    ranks = [
        rank for rank in (first_hit_rank(retrieved, gold, k) for gold in golds) if rank is not None
    ]
    return 1.0 / min(ranks) if ranks else 0.0


def normalize_answer(s: str) -> str:
    """SQuAD-style: lowercase, strip punctuation, drop articles, collapse whitespace."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def exact_match(prediction: str, gold_answers: list[str]) -> float:
    """1.0 iff the normalized prediction equals any normalized gold answer."""
    pred = normalize_answer(prediction)
    return 1.0 if any(pred == normalize_answer(g) for g in gold_answers) else 0.0


def f1(prediction: str, gold_answers: list[str]) -> float:
    """Max token-overlap F1 over gold answers (SQuAD definition)."""
    pred_tokens = normalize_answer(prediction).split()
    best = 0.0
    for gold in gold_answers:
        gold_tokens = normalize_answer(gold).split()
        if not pred_tokens or not gold_tokens:
            best = max(best, float(pred_tokens == gold_tokens))
            continue
        common = Counter(pred_tokens) & Counter(gold_tokens)
        overlap = sum(common.values())
        if overlap == 0:
            continue
        precision = overlap / len(pred_tokens)
        recall = overlap / len(gold_tokens)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best
