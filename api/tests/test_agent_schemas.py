import pytest
from pydantic import ValidationError

from ragreceipts.agents.schemas import (
    FinalAnswer,
    GradeResult,
    RouteDecision,
    SubQueries,
)


def test_route_decision_literal_and_bounds():
    d = RouteDecision(route="simple", confidence=0.9)
    assert d.route == "simple" and d.confidence == 0.9
    with pytest.raises(ValidationError):
        RouteDecision(route="medium", confidence=0.5)
    with pytest.raises(ValidationError):
        RouteDecision(route="simple", confidence=1.5)


def test_grade_result_verdicts():
    for v in ("sufficient", "insufficient", "contradictory"):
        assert GradeResult(verdict=v).verdict == v
    with pytest.raises(ValidationError):
        GradeResult(verdict="maybe")


def test_subqueries():
    assert SubQueries(items=["a", "b"]).items == ["a", "b"]


def test_final_answer_defaults_and_fields():
    a = FinalAnswer(text="Paris [1]", citations=[1])
    assert a.abstained is False
    assert a.unresolved_subqueries == []
    assert a.contradiction_flag is False
    b = FinalAnswer(
        text="cannot answer",
        abstained=True,
        unresolved_subqueries=["who founded X"],
        contradiction_flag=True,
    )
    assert b.abstained and b.contradiction_flag
