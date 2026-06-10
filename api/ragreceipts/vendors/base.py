"""Vendor transport seam. Binding shapes from docs/superpowers/plans/2026-06-10-contracts.md.

Every network call in the system goes through one of these Protocols; application code
never imports voyageai/cohere/anthropic outside ragreceipts/vendors/. Unit tests inject
fakes from api/tests/fakes.py — zero keys, zero network in CI.
"""

from dataclasses import dataclass
from typing import Protocol


class VendorUnavailable(Exception):
    """A vendor call failed after all retries (429/5xx/connection).

    Real clients raise this after retry exhaustion; fakes raise it when scripted to fail.
    RetrievalCore catches exactly this to degrade visibly (rerank-skipped / dense-skipped).
    """


class EmbedTransport(Protocol):
    def embed_documents(self, documents: list[list[str]]) -> list[list[list[float]]]:
        """documents = list of docs, each a list of chunk texts (doc-grouped).
        Isolated mode is expressed by passing single-chunk documents."""
        ...

    def embed_query(self, query: str) -> list[float]: ...


class RerankTransport(Protocol):
    def rerank(self, query: str, texts: list[str], top_n: int) -> list[tuple[int, float]]:
        """returns (original_index, relevance_score) sorted desc."""
        ...


@dataclass(frozen=True)
class ClaudeResult:
    text: str
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True)
class ParsedResult:
    parsed: object  # the validated Pydantic instance
    input_tokens: int
    output_tokens: int


class ClaudeTransport(Protocol):
    def complete(
        self, *, model: str, system: str, user: str, max_tokens: int, temperature: float = 0.0
    ) -> ClaudeResult: ...

    def parse(
        self,
        *,
        model: str,
        system: str,
        user: str,
        max_tokens: int,
        output_format: type,
        temperature: float = 0.0,
    ) -> ParsedResult: ...


@dataclass(frozen=True)
class Triple:
    subject: str
    relation: str
    object: str


class OpenIETransport(Protocol):
    def extract(self, passages: list[str]) -> list[list[Triple]]:
        """One triple list per input passage (same order, same length)."""
        ...
