"""Retriever protocol. retrieval/ knows nothing about agents or HTTP.

The Phase-2 graph retriever implements this same protocol; no graph flag, enum branch,
or stub ships in v1 code (spec boundary rule)."""

from typing import Protocol

from ragreceipts.types import ScoredChunk


class Retriever(Protocol):
    def search(self, query: str, k: int) -> list[ScoredChunk]: ...
