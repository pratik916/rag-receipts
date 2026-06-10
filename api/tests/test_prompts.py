from ragreceipts.agents import prompts
from ragreceipts.types import Chunk, ScoredChunk


def sc(i: int, text: str) -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(
            chunk_id=f"d:{i}",
            corpus_id="c",
            doc_id="d",
            passage_id="d",
            text=text,
            position=i,
            start_token=0,
            end_token=len(text.split()),
        ),
        score=1.0,
        source="rrf",
    )


def test_format_numbered_context():
    out = prompts.format_numbered_context([sc(0, "alpha"), sc(1, "beta")])
    assert out == "[1] alpha\n\n[2] beta"
    assert prompts.format_numbered_context([]) == ""


def test_format_hop_context_global_numbering_and_dedupe():
    h1 = {"subquery": "q1", "chunks": [sc(0, "alpha"), sc(1, "beta")]}
    h2 = {"subquery": "q2", "chunks": [sc(1, "beta"), sc(2, "gamma")]}
    text, ordered = prompts.format_hop_context([h1, h2])
    assert [s.chunk.chunk_id for s in ordered] == ["d:0", "d:1", "d:2"]
    assert "[3]" in text and "[4]" not in text  # dedupe: beta numbered once
    assert '(hop: "q2") gamma' in text


def test_prompts_carry_required_instructions():
    # Citation format [n] and structured abstention are load-bearing (spec).
    assert "[1]" in prompts.S1_ANSWER_SYSTEM
    assert "abstained" in prompts.S1_ANSWER_SYSTEM
    assert "abstained" in prompts.SYNTHESIZE_SYSTEM
    assert "contradiction_flag" in prompts.SYNTHESIZE_SYSTEM
    assert "contradictory" in prompts.GRADE_SYSTEM
    assert "{query}" in prompts.ROUTE_USER
    assert "{max_hops}" in prompts.DECOMPOSE_USER
    for name in (
        "ROUTE_SYSTEM",
        "DECOMPOSE_SYSTEM",
        "GRADE_SYSTEM",
        "REFINE_SYSTEM",
        "SYNTHESIZE_SYSTEM",
        "S1_ANSWER_SYSTEM",
    ):
        assert len(getattr(prompts, name)) > 100, name
