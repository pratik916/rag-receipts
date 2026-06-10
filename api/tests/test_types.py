import dataclasses

import pytest

from ragreceipts.types import Chunk, RouteMode, ScoredChunk


def test_chunk_is_frozen_and_carries_alignment_metadata():
    c = Chunk(
        chunk_id="d1:0",
        corpus_id="musique-dev-300",
        doc_id="d1",
        passage_id="p1",
        text="hello world",
        position=0,
        start_token=0,
        end_token=2,
    )
    assert c.passage_id == "p1"
    assert c.chunk_id == f"{c.doc_id}:{c.position}"
    # R3: chunks carry whitespace-token offsets within their parent passage so the
    # eval span-hit rule is computable on retrieved chunks.
    assert (c.start_token, c.end_token) == (0, 2)
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.text = "nope"


def test_scored_chunk_and_route_mode():
    c = Chunk(
        chunk_id="d1:0",
        corpus_id="c",
        doc_id="d1",
        passage_id="d1",
        text="t",
        position=0,
        start_token=0,
        end_token=1,
    )
    s = ScoredChunk(chunk=c, score=1.5, source="bm25")
    assert s.source == "bm25"
    assert RouteMode.FORCE_S1.value == "force_s1"
    assert RouteMode.AUTO.value == "auto"
