"""OpenIE seam: FakeOpenIE (FINAL R5 constructor), normalize_phrase, and the real
OpenIEExtractor mapping messages.parse output to Triples — all offline."""

import pytest

from ragreceipts.agents.openie import (
    OpenIEExtractor,
    TripleModel,
    TripleSet,
    normalize_phrase,
)
from ragreceipts.constants import OPENIE_MODEL
from ragreceipts.vendors.base import Triple, VendorUnavailable
from tests.fakes import FakeClaude, FakeOpenIE


class TestFakeOpenIE:
    def test_scripted_triples_by_passage_text(self):
        script = {"alpha beta": [Triple("alpha", "is", "beta")]}
        fake = FakeOpenIE(script=script)
        out = fake.extract(["alpha beta", "unknown passage"])
        assert out == [[Triple("alpha", "is", "beta")], []]  # unknown -> []

    def test_deterministic_across_instances(self):
        script = {"p": [Triple("a", "r", "b")]}
        assert FakeOpenIE(script=script).extract(["p"]) == FakeOpenIE(script=script).extract(["p"])

    def test_length_and_order_preserved(self):
        fake = FakeOpenIE(script={"b": [Triple("b", "r", "c")]})
        out = fake.extract(["a", "b", "c"])
        assert len(out) == 3
        assert out == [[], [Triple("b", "r", "c")], []]

    def test_empty_script_default(self):
        assert FakeOpenIE().extract(["anything"]) == [[]]

    def test_fail_raises_vendor_unavailable(self):
        with pytest.raises(VendorUnavailable):
            FakeOpenIE(fail=True).extract(["p"])


class TestNormalizePhrase:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("  Eiffel Tower  ", "eiffel tower"),
            ("EIFFEL   TOWER", "eiffel tower"),
            ("Paris\n", "paris"),
            ("multi   space\tword", "multi space word"),
        ],
    )
    def test_lower_strip_collapse(self, raw, expected):
        assert normalize_phrase(raw) == expected

    def test_same_surface_form_maps_to_one_node(self):
        assert normalize_phrase("The Tower") == normalize_phrase("the   tower ")


class TestOpenIEExtractor:
    def test_one_parse_call_per_passage_maps_to_triples(self):
        claude = FakeClaude(
            script=[
                TripleSet(
                    triples=[TripleModel(subject="Paris", relation="capital of", object="France")]
                ),
                TripleSet(triples=[]),
            ]
        )
        extractor = OpenIEExtractor(claude)
        out = extractor.extract(["Paris is the capital of France.", "Cats hunt mice."])
        assert out == [[Triple("Paris", "capital of", "France")], []]
        # exactly one parse() per passage, on the cheap model, with TripleSet output_format
        assert len(claude.parse_calls) == 2
        assert claude.parse_calls[0]["model"] == OPENIE_MODEL
        assert claude.parse_calls[0]["output_format"] is TripleSet

    def test_empty_passage_list_returns_empty(self):
        claude = FakeClaude(script=[])
        assert OpenIEExtractor(claude).extract([]) == []

    def test_claude_failure_propagates(self):
        # FakeClaude with an empty script raises AssertionError when called; the extractor
        # does not swallow it — a broken OpenIE run must fail loudly, not return [].
        with pytest.raises(AssertionError):
            OpenIEExtractor(FakeClaude(script=[])).extract(["nonempty"])
