"""Token-window chunker (Spike 0 stub).

Whitespace-token sliding window. Plan A replaces the internals with the real
sentence-window chunker but MUST keep `chunk_passage`'s signature and `ChunkSpan`
unchanged - eval/alignment.py and the hand-check harness depend on them.
Defaults match contracts IngestConfig (chunk_size=512, chunk_overlap=64).
"""

from dataclasses import dataclass

from ragreceipts.types import Chunk


@dataclass(frozen=True)
class ChunkSpan:
    """A chunk plus the whitespace-token range it covers in its parent passage text.

    start_token is inclusive, end_token exclusive; both index into passage_text.split().
    """

    chunk: Chunk
    start_token: int
    end_token: int


def chunk_passage(
    *,
    corpus_id: str,
    doc_id: str,
    passage_id: str,
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[ChunkSpan]:
    """Split `text` into overlapping windows of whitespace tokens.

    Window stride is chunk_size - chunk_overlap; the final window is truncated at the
    end of the text. Empty/whitespace-only text yields [].
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
    start = 0
    position = 0
    while True:
        end = min(start + chunk_size, len(tokens))
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
        if end == len(tokens):
            break
        start += stride
        position += 1
    return spans
