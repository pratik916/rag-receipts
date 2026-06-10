import pytest

from ragreceipts.eval.alignment import (
    GoldPassage,
    GoldSpan,
    first_hit_rank,
    is_hit,
    passage_hit,
    span_hit,
)
from ragreceipts.ingest.chunker import ChunkSpan
from ragreceipts.types import Chunk


def _span(doc_id: str, passage_id: str, start: int, end: int, position: int = 0) -> ChunkSpan:
    # Chunk carries token ranges (contracts R3); mirror the ChunkSpan's range onto it so the
    # structural hit rules work whether handed a ChunkSpan or the bare Chunk.
    chunk = Chunk(
        chunk_id=f"{doc_id}:{position}",
        corpus_id="c",
        doc_id=doc_id,
        passage_id=passage_id,
        text="x " * (end - start),
        position=position,
        start_token=start,
        end_token=end,
    )
    return ChunkSpan(chunk=chunk, start_token=start, end_token=end)


def test_passage_hit_is_exact_id_match():
    gold = GoldPassage(query_id="q", passage_id="p1")
    assert passage_hit(_span("d", "p1", 0, 4).chunk, gold)
    assert not passage_hit(_span("d", "p2", 0, 4).chunk, gold)


def test_span_hit_at_exactly_50_percent_boundary():
    gold = GoldSpan(query_id="q", doc_id="d", start_token=10, end_token=20)  # 10 tokens
    assert span_hit(_span("d", "d", 0, 15), gold)  # overlap 5/10 = 50% -> hit
    assert not span_hit(_span("d", "d", 0, 14), gold)  # overlap 4/10 = 40% -> miss
    assert span_hit(_span("d", "d", 12, 30), gold)  # overlap 8/10 = 80% -> hit
    assert span_hit(_span("d", "d", 0, 100), gold)  # full cover -> hit
    assert not span_hit(_span("d", "d", 20, 30), gold)  # adjacent, overlap 0 -> miss


def test_span_hit_requires_same_document():
    gold = GoldSpan(query_id="q", doc_id="d1", start_token=0, end_token=10)
    assert not span_hit(_span("d2", "d2", 0, 10), gold)


def test_span_hit_rejects_empty_gold():
    gold = GoldSpan(query_id="q", doc_id="d", start_token=5, end_token=5)
    with pytest.raises(ValueError):
        span_hit(_span("d", "d", 0, 10), gold)


def test_is_hit_dispatches_on_gold_type():
    ps = _span("d", "p1", 0, 4)
    assert is_hit(ps, GoldPassage(query_id="q", passage_id="p1"))
    assert is_hit(ps, GoldSpan(query_id="q", doc_id="d", start_token=0, end_token=4))


def test_is_hit_works_structurally_on_bare_chunk():
    # Contracts R3: is_hit accepts ANY object with passage_id/doc_id/start_token/end_token,
    # so a bare Chunk (no .chunk attribute) must hit/miss identically to its ChunkSpan.
    chunk = _span("d", "p1", 10, 20).chunk
    assert is_hit(chunk, GoldPassage(query_id="q", passage_id="p1"))
    assert not is_hit(chunk, GoldPassage(query_id="q", passage_id="p2"))
    assert is_hit(chunk, GoldSpan(query_id="q", doc_id="d", start_token=10, end_token=20))
    assert not is_hit(chunk, GoldSpan(query_id="q", doc_id="other", start_token=10, end_token=20))


def test_first_hit_rank_is_one_based_and_k_bounded():
    gold = GoldSpan(query_id="q", doc_id="d", start_token=10, end_token=20)
    ranked = [
        _span("d", "d", 30, 40, position=0),  # miss
        _span("d", "d", 8, 22, position=1),  # hit (full cover)
        _span("d", "d", 10, 20, position=2),  # hit, but rank 2 already found
    ]
    assert first_hit_rank(ranked, gold, k=3) == 2
    assert first_hit_rank(ranked, gold, k=1) is None
    assert first_hit_rank([], gold, k=5) is None


def test_misaligned_gold_provably_never_hits():
    # Harness self-test flavor (spec section Testing): a gold pointing at a passage id
    # absent from every chunk must produce zero hits at any rank.
    gold = GoldPassage(query_id="q", passage_id="not-in-corpus")
    ranked = [_span("d", f"p{i}", 0, 4, position=i) for i in range(10)]
    assert first_hit_rank(ranked, gold, k=10) is None
    assert not any(is_hit(s, gold) for s in ranked)
