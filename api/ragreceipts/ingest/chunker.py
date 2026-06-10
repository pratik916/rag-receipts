"""Sentence-window chunker (Plan A internals behind Spike 0's binding API).

PUBLIC API — binding per Spike 0 and seam resolution R4:
- chunk_passage(*, corpus_id, doc_id, passage_id, text, chunk_size=512,
  chunk_overlap=64) -> list[ChunkSpan], and the ChunkSpan dataclass, are kept
  forever — eval/alignment.py and scripts/handcheck_alignment.py import them.
- chunk_document is an ingest-internal helper (positions run across passages).

"Token" = whitespace word (indices into text.split()) — the same space Spike 0's
span golds live in. Chunk text is always " ".join(tokens[start:end]), so the
(start_token, end_token) range stored on every ChunkSpan AND Chunk (R3) is exact.
Sentence boundaries are detected per token (a token ending in . ! or ? closes a
sentence — abbreviations like "Dr." split; acceptable for packing, fully offline).
Sentences are packed greedily up to chunk_size tokens, retaining trailing whole
sentences up to chunk_overlap tokens as overlap; a sentence longer than chunk_size
falls back to a sliding token window with stride chunk_size - chunk_overlap
(identical to Spike 0's stub on unpunctuated text). Real Voyage token budgets are
enforced at embed time by VoyageClient's batch planner (120K/16K/32K request caps).
"""

from dataclasses import dataclass, replace

from ragreceipts.types import Chunk

_TERMINALS = (".", "!", "?")


@dataclass(frozen=True)
class ChunkSpan:
    """A chunk plus the whitespace-token range it covers in its parent passage text.

    start_token is inclusive, end_token exclusive; both index into passage_text.split().
    (Kept verbatim from Spike 0; the same range is also stored on the Chunk per R3.)
    """

    chunk: Chunk
    start_token: int
    end_token: int


def _sentence_ranges(tokens: list[str]) -> list[tuple[int, int]]:
    """[start, end) token ranges of sentences; a token ending in .!? closes one."""
    ranges: list[tuple[int, int]] = []
    start = 0
    for i, token in enumerate(tokens):
        if token.endswith(_TERMINALS):
            ranges.append((start, i + 1))
            start = i + 1
    if start < len(tokens):
        ranges.append((start, len(tokens)))
    return ranges


def _units(tokens: list[str], chunk_size: int, stride: int) -> list[tuple[int, int]]:
    """Sentence ranges, with oversized sentences hard-split into sliding windows."""
    units: list[tuple[int, int]] = []
    for start, end in _sentence_ranges(tokens):
        if end - start <= chunk_size:
            units.append((start, end))
            continue
        s = start
        while True:
            e = min(s + chunk_size, end)
            units.append((s, e))
            if e == end:
                break
            s += stride
    return units


def _pack(
    units: list[tuple[int, int]], chunk_size: int, chunk_overlap: int
) -> list[tuple[int, int]]:
    """Greedy sentence packing with trailing-sentence overlap; returns window ranges."""
    windows: list[tuple[int, int]] = []
    window: list[tuple[int, int]] = []
    window_tokens = 0
    for unit in units:
        unit_tokens = unit[1] - unit[0]
        if window and window_tokens + unit_tokens > chunk_size:
            windows.append((window[0][0], window[-1][1]))
            kept: list[tuple[int, int]] = []
            kept_tokens = 0
            for prev in reversed(window):  # retain trailing sentences as overlap
                prev_tokens = prev[1] - prev[0]
                if kept_tokens + prev_tokens > chunk_overlap:
                    break
                kept.insert(0, prev)
                kept_tokens += prev_tokens
            if kept and (kept_tokens + unit_tokens > chunk_size or kept[-1][1] != unit[0]):
                kept, kept_tokens = [], 0  # overlap would overflow / isn't contiguous
            window, window_tokens = kept, kept_tokens
        window.append(unit)
        window_tokens += unit_tokens
    if window:
        windows.append((window[0][0], window[-1][1]))
    return windows


def chunk_passage(
    *,
    corpus_id: str,
    doc_id: str,
    passage_id: str,
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[ChunkSpan]:
    """Split `text` into sentence-packed windows of whitespace tokens.

    Signature and return type are binding (Spike 0 / R4). Empty or whitespace-only
    text yields [].
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError(f"chunk_overlap must be in [0, chunk_size), got {chunk_overlap}")
    tokens = text.split()
    if not tokens:
        return []
    stride = chunk_size - chunk_overlap
    spans: list[ChunkSpan] = []
    for position, (start, end) in enumerate(
        _pack(_units(tokens, chunk_size, stride), chunk_size, chunk_overlap)
    ):
        chunk = Chunk(
            chunk_id=f"{doc_id}:{position}",
            corpus_id=corpus_id,
            doc_id=doc_id,
            passage_id=passage_id,
            text=" ".join(tokens[start:end]),
            position=position,
            start_token=start,
            end_token=end,
        )
        spans.append(ChunkSpan(chunk=chunk, start_token=start, end_token=end))
    return spans


def chunk_document(
    corpus_id: str,
    doc_id: str,
    passages: list[tuple[str, str]],
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    """Ingest-internal helper: chunk each (passage_id, text) of one document via
    chunk_passage, renumbering positions ACROSS the document so chunk_id stays
    unique per doc. start_token/end_token stay passage-relative (R3)."""
    chunks: list[Chunk] = []
    position = 0
    for passage_id, text in passages:
        for span in chunk_passage(
            corpus_id=corpus_id,
            doc_id=doc_id,
            passage_id=passage_id,
            text=text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        ):
            chunks.append(replace(span.chunk, chunk_id=f"{doc_id}:{position}", position=position))
            position += 1
    return chunks
