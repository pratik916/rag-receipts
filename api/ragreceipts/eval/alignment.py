"""Gold-to-chunk alignment rules (Spike 0; lasting code).

Binding metric definitions (contracts section Metrics):
- passage gold: hit iff chunk.passage_id == gold.passage_id
- span gold (NQ long answers): hit iff the chunk covers >= 50% of the gold span's
  tokens (and the chunk belongs to the gold's document)

Token indices are whitespace-token indices into the parent document's text
(indices into text.split()), as produced by ragreceipts.ingest.chunker.chunk_passage.

The hit rules are structural (contracts R3): `is_hit`/`first_hit_rank` accept ANY object
exposing `passage_id`/`doc_id`/`start_token`/`end_token` — both `ChunkSpan` (token range on
the wrapper) and a bare `Chunk` (token range on the chunk itself) — so the same rule serves
the chunker's ChunkSpan and the persisted Chunk identically.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class GoldPassage:
    query_id: str
    passage_id: str


@dataclass(frozen=True)
class GoldSpan:
    query_id: str
    doc_id: str
    start_token: int  # inclusive
    end_token: int  # exclusive; must be > start_token


Gold = GoldPassage | GoldSpan


def _passage_id(candidate: object) -> str:
    """Read the candidate's passage_id, whether it is a ChunkSpan (.chunk.passage_id) or a
    bare Chunk (.passage_id)."""
    inner = getattr(candidate, "chunk", candidate)
    return inner.passage_id


def _span_fields(candidate: object) -> tuple[str, int, int]:
    """Read (doc_id, start_token, end_token) off either a ChunkSpan or a bare Chunk.

    A ChunkSpan exposes only the token range on the wrapper (it has no doc_id field), so doc_id
    always comes from the underlying chunk while the offsets prefer the wrapper: take the
    wrapper's token range when present (ChunkSpan) and otherwise fall back to the bare object's
    range (a bare Chunk carries both doc_id and its own range)."""
    inner = getattr(candidate, "chunk", candidate)
    start = getattr(candidate, "start_token", None)
    end = getattr(candidate, "end_token", None)
    if start is None or end is None:
        start = inner.start_token
        end = inner.end_token
    return inner.doc_id, start, end


def passage_hit(candidate: object, gold: GoldPassage) -> bool:
    return _passage_id(candidate) == gold.passage_id


def span_hit(candidate: object, gold: GoldSpan) -> bool:
    gold_len = gold.end_token - gold.start_token
    if gold_len <= 0:
        raise ValueError(f"gold span must be non-empty: {gold}")
    doc_id, start_token, end_token = _span_fields(candidate)
    if doc_id != gold.doc_id:
        return False
    overlap = min(end_token, gold.end_token) - max(start_token, gold.start_token)
    return 2 * overlap >= gold_len  # integer form of overlap/gold_len >= 0.5


def is_hit(candidate: object, gold: Gold) -> bool:
    if isinstance(gold, GoldPassage):
        return passage_hit(candidate, gold)
    return span_hit(candidate, gold)


def first_hit_rank(ranked: list[object], gold: Gold, k: int) -> int | None:
    """1-based rank of the first hit within the top-k of `ranked`, else None."""
    for rank, candidate in enumerate(ranked[:k], start=1):
        if is_hit(candidate, gold):
            return rank
    return None
