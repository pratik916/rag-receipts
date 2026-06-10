"""Pydantic response models for Claude structured outputs (messages.parse).

Field names are part of the prompt contract in agents/prompts.py — the prompts
reference `abstained`, `citations`, `unresolved_subqueries`, `contradiction_flag`
by name. Change them in lockstep or not at all.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class RouteDecision(BaseModel):
    route: Literal["simple", "complex"]
    confidence: float = Field(ge=0.0, le=1.0)


class SubQueries(BaseModel):
    items: list[str]


class GradeResult(BaseModel):
    verdict: Literal["sufficient", "insufficient", "contradictory"]


class FinalAnswer(BaseModel):
    text: str
    citations: list[int] = Field(default_factory=list)
    abstained: bool = False
    unresolved_subqueries: list[str] = Field(default_factory=list)
    contradiction_flag: bool = False
