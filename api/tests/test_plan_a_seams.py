"""CI-enforced verification of the upstream surfaces Plan B binds to.

These tests pin the cross-plan seams listed in the Plan B Context table:
Spike 0's alignment API (kept verbatim per R3) and Plan A's FakeRerank
final constructor (R5). If one fails, the upstream code drifted from its
binding resolution - reconcile the upstream code or the call sites named
in the Context seam table; do NOT fork a second definition of these names.
"""

from ragreceipts.eval.alignment import GoldPassage, GoldSpan, first_hit_rank, is_hit
from ragreceipts.types import Chunk
from tests.fakes import FakeRerank


def make_chunk(
    passage_id: str, *, start_token: int = 0, end_token: int = 1, text: str = "x", position: int = 0
) -> Chunk:
    return Chunk(
        chunk_id=f"{passage_id}:{position}",
        corpus_id="seam",
        doc_id=passage_id,
        passage_id=passage_id,
        text=text,
        position=position,
        start_token=start_token,
        end_token=end_token,
    )


def test_alignment_passage_seam() -> None:
    gold = GoldPassage(query_id="q", passage_id="p1")
    assert is_hit(make_chunk("p1"), gold) is True
    assert is_hit(make_chunk("p2"), gold) is False


def test_alignment_span_seam_is_positional_50pct() -> None:
    # Spike 0's positional rule (R3): gold [10, 20) = 10 tokens; a chunk
    # covering [0, 15) overlaps 5/10 = 50% -> hit; [0, 14) is 40% -> miss.
    # is_hit works structurally on Chunk because Chunk carries token offsets.
    gold = GoldSpan(query_id="q", doc_id="d1", start_token=10, end_token=20)
    assert is_hit(make_chunk("d1", start_token=0, end_token=15), gold) is True
    assert is_hit(make_chunk("d1", start_token=0, end_token=14), gold) is False
    # same-document requirement: full overlap in the wrong doc is a miss
    assert is_hit(make_chunk("d2", start_token=0, end_token=20), gold) is False


def test_first_hit_rank_seam_is_one_based_and_k_bounded() -> None:
    gold = GoldPassage(query_id="q", passage_id="p1")
    ranked = [make_chunk("x"), make_chunk("p1", position=1), make_chunk("p1", position=2)]
    assert first_hit_rank(ranked, gold, k=3) == 2
    assert first_hit_rank(ranked, gold, k=1) is None


def test_fake_rerank_scores_seam() -> None:
    # R5 final constructor: FakeRerank(script=None, scores=None, fail=False);
    # the text-keyed scores mode exists for Plan B's harness fixture.
    fake = FakeRerank(scores={"high": 0.9, "low": 0.1})
    out = fake.rerank("any query", ["low", "high"], top_n=2)
    assert out == [(1, 0.9), (0, 0.1)]
