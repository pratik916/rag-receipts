"""The fakes ARE the CI vendor layer — they must be deterministic and scriptable.
Constructor shapes are FINAL per seam resolution R5 and are defined once, here."""

import math

import pytest

from ragreceipts.vendors.base import ClaudeResult, VendorUnavailable
from tests.fakes import FakeClaude, FakeEmbed, FakeRerank


def _norm(vec: list[float]) -> float:
    return math.sqrt(sum(x * x for x in vec))


class TestFakeEmbed:
    def test_deterministic_across_instances(self):
        a = FakeEmbed().embed_query("hello world")
        b = FakeEmbed().embed_query("hello world")
        assert a == b
        assert len(a) == 8
        assert _norm(a) == pytest.approx(1.0)

    def test_isolated_single_chunk_equals_query_embedding_of_same_text(self):
        fake = FakeEmbed()
        [[isolated]] = fake.embed_documents([["alpha beta"]])
        assert isolated == pytest.approx(fake.embed_query("alpha beta"))

    def test_doc_context_changes_chunk_vector(self):
        fake = FakeEmbed()
        [[isolated]] = fake.embed_documents([["alpha beta"]])
        [[contextual, _]] = [fake.embed_documents([["alpha beta", "gamma delta"]])[0]]
        assert contextual != pytest.approx(isolated)

    def test_query_aliases_redirect_query_vector(self):
        fake = FakeEmbed(query_aliases={"q": "target text"})
        assert fake.embed_query("q") == pytest.approx(fake.embed_query("target text"))

    def test_scripted_failures(self):
        with pytest.raises(VendorUnavailable):
            FakeEmbed(fail_query=True).embed_query("q")
        with pytest.raises(VendorUnavailable):
            FakeEmbed(fail_documents=True).embed_documents([["t"]])


class TestFakeRerank:
    def test_default_reverses_order(self):
        got = FakeRerank().rerank("q", ["a", "b", "c"], top_n=3)
        assert [i for i, _ in got] == [2, 1, 0]
        scores = [s for _, s in got]
        assert scores == sorted(scores, reverse=True)

    def test_script_and_top_n(self):
        fake = FakeRerank(script={"q": [1, 0, 2]})
        assert [i for i, _ in fake.rerank("q", ["a", "b", "c"], top_n=2)] == [1, 0]

    def test_scores_mode_orders_by_candidate_text(self):
        # R5 additive mode — Plan B's harness fixture relies on exactly this shape
        fake = FakeRerank(scores={"high": 0.9, "low": 0.1})
        assert fake.rerank("any query", ["low", "high"], top_n=2) == [(1, 0.9), (0, 0.1)]

    def test_scores_mode_unknown_text_gets_zero(self):
        fake = FakeRerank(scores={"a": 0.5})
        assert fake.rerank("q", ["a", "mystery"], top_n=2) == [(0, 0.5), (1, 0.0)]

    def test_scripted_failure(self):
        with pytest.raises(VendorUnavailable):
            FakeRerank(fail=True).rerank("q", ["a"], top_n=1)


class TestFakeClaude:
    def test_str_items_become_claude_results_in_order(self):
        fake = FakeClaude(script=["one", ("two", 30, 7)])
        first = fake.complete(model="m", system="s", user="u", max_tokens=64)
        assert first == ClaudeResult(text="one", input_tokens=10, output_tokens=5)
        second = fake.complete(model="m", system="s", user="u", max_tokens=64)
        assert (second.text, second.input_tokens, second.output_tokens) == ("two", 30, 7)
        assert fake.complete_calls[0]["model"] == "m"
        with pytest.raises(AssertionError):
            fake.complete(model="m", system="s", user="u", max_tokens=64)

    def test_object_items_become_parsed_results(self):
        class Routed:
            complexity = "simple"

        routed = Routed()
        fake = FakeClaude(script=[routed, (Routed(), 99, 3)])
        got = fake.parse(model="m", system="s", user="u", max_tokens=64, output_format=Routed)
        assert got.parsed is routed
        assert (got.input_tokens, got.output_tokens) == (10, 5)
        second = fake.parse(model="m", system="s", user="u", max_tokens=64, output_format=Routed)
        assert (second.input_tokens, second.output_tokens) == (99, 3)
        assert fake.parse_calls[0]["output_format"] is Routed

    def test_one_script_consumed_across_complete_and_parse(self):
        class Graded:
            verdict = "good"

        fake = FakeClaude(script=["text answer", Graded()])
        assert fake.complete(model="m", system="s", user="u", max_tokens=8).text == "text answer"
        got = fake.parse(model="m", system="s", user="u", max_tokens=8, output_format=Graded)
        assert isinstance(got.parsed, Graded)

    def test_item_kind_mismatch_fails_loudly(self):
        class Routed:
            pass

        with pytest.raises(AssertionError):
            FakeClaude(script=[Routed()]).complete(model="m", system="s", user="u", max_tokens=8)
        with pytest.raises(AssertionError):
            FakeClaude(script=["oops"]).parse(
                model="m", system="s", user="u", max_tokens=8, output_format=Routed
            )
