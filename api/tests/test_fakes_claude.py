# api/tests/test_fakes_claude.py
import pytest

from ragreceipts.agents.schemas import RouteDecision
from tests.fakes import FakeClaude, FakeCore, make_chunk


def test_script_pops_in_order_across_parse_and_complete():
    fc = FakeClaude(script=[RouteDecision(route="simple", confidence=0.9), "refined query"])
    r1 = fc.parse(model="m", system="s", user="u", max_tokens=10, output_format=RouteDecision)
    assert r1.parsed.route == "simple"
    r2 = fc.complete(model="m", system="s", user="u", max_tokens=10)
    assert r2.text == "refined query"
    assert [c["method"] for c in fc.calls] == ["parse", "complete"]
    assert fc.calls[0]["output_format"] == "RouteDecision"


def test_token_tuples_and_exhaustion():
    fc = FakeClaude(script=[("answer", 1000, 2000)])
    r = fc.complete(model="m", system="s", user="u", max_tokens=10)
    assert (r.input_tokens, r.output_tokens) == (1000, 2000)
    with pytest.raises(AssertionError):  # script ran dry -> loud failure
        fc.complete(model="m", system="s", user="u", max_tokens=10)


def test_type_mismatch_fails_loud():
    fc = FakeClaude(script=["not a model"])
    with pytest.raises(AssertionError):
        fc.parse(model="m", system="s", user="u", max_tokens=10, output_format=RouteDecision)


def test_fake_core_scripts_and_records_queries():
    hit = [make_chunk(0)]
    core = FakeCore(by_query={"q1": hit})
    assert core.retrieve("q1") == hit
    assert len(core.retrieve("unknown")) == 2  # default corpus
    assert core.queries == ["q1", "unknown"]
