"""Core shared types (binding: docs/superpowers/plans/2026-06-10-contracts.md)."""

from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class Chunk:
    chunk_id: str  # f"{doc_id}:{position}"
    corpus_id: str
    doc_id: str
    passage_id: str  # parent passage ID for gold alignment (== doc_id when unsegmented)
    text: str
    position: int  # chunk index within document
    start_token: int  # whitespace-token offset within the parent passage (R3)
    end_token: int  # exclusive end offset; persisted to chunks.jsonl + Qdrant payload


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    score: float
    source: str  # "bm25" | "dense" | "rrf" | "rerank"


class RouteMode(str, Enum):  # noqa: UP042 — contracts pin (str, Enum), not StrEnum
    AUTO = "auto"
    FORCE_S1 = "force_s1"
    FORCE_S2 = "force_s2"
