"""Guards the binding contract values in constants.py and types.py against drift."""

import dataclasses

import pytest

from ragreceipts import constants
from ragreceipts.types import Chunk, RouteMode, ScoredChunk


def test_model_constants_match_contracts():
    assert constants.ROUTER_MODEL == "claude-haiku-4-5-20251001"
    assert constants.SYNTH_MODEL == "claude-sonnet-4-6"
    assert constants.JUDGE_MODEL == "claude-sonnet-4-6"
    assert constants.EMBED_MODEL == "voyage-context-3"
    assert constants.RERANK_MODEL == "rerank-v4.0-pro"
    assert constants.RAGAS_EMBED_MODEL == "BAAI/bge-small-en-v1.5"
    assert constants.ROUTE_CONFIDENCE_THRESHOLD == 0.7
    assert constants.S2_MAX_HOPS == 3
    assert constants.S2_TOKEN_CEILING == 50_000


def test_chunk_fields_and_id_convention():
    chunk = Chunk(
        chunk_id="doc7:2",
        corpus_id="tiny",
        doc_id="doc7",
        passage_id="doc7-p1",
        text="some text",
        position=2,
        start_token=10,
        end_token=12,
    )
    assert chunk.chunk_id == f"{chunk.doc_id}:{chunk.position}"
    assert chunk.passage_id == "doc7-p1"
    assert (chunk.start_token, chunk.end_token) == (10, 12)  # R3 token range


def test_chunk_is_frozen():
    chunk = Chunk(
        chunk_id="d:0",
        corpus_id="c",
        doc_id="d",
        passage_id="d",
        text="t",
        position=0,
        start_token=0,
        end_token=1,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        chunk.text = "mutated"  # type: ignore[misc]


def test_scored_chunk_carries_source():
    chunk = Chunk(
        chunk_id="d:0",
        corpus_id="c",
        doc_id="d",
        passage_id="d",
        text="t",
        position=0,
        start_token=0,
        end_token=1,
    )
    scored = ScoredChunk(chunk=chunk, score=1.5, source="bm25")
    assert scored.score == 1.5
    assert scored.source == "bm25"


def test_route_mode_values():
    assert RouteMode.AUTO.value == "auto"
    assert RouteMode.FORCE_S1.value == "force_s1"
    assert RouteMode.FORCE_S2.value == "force_s2"
    assert RouteMode("auto") is RouteMode.AUTO
