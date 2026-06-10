"""OpenIE triple extraction — the real OpenIETransport (LLM orchestration behind
ClaudeTransport). One messages.parse() call per passage on the cheap Haiku model
(OPENIE_MODEL == ROUTER_MODEL), output_format=TripleSet. CI uses FakeOpenIE; this
module is exercised offline against a FakeClaude-backed stub (test_openie.py) and
live only via scripts/build_graph.py (keyed, never CI).

normalize_phrase is the pure surface-form collapser: lower + strip + whitespace-
collapse, so the same phrase maps to one graph node regardless of casing/spacing.
"""

from __future__ import annotations

from pydantic import BaseModel

from ragreceipts.constants import OPENIE_MODEL
from ragreceipts.vendors.base import ClaudeTransport, Triple


class TripleModel(BaseModel):
    subject: str
    relation: str
    object: str


class TripleSet(BaseModel):
    triples: list[TripleModel]


OPENIE_SYSTEM = """\
You extract knowledge-graph triples from a single passage of text.

Return a list of (subject, relation, object) triples. Rules:
- Subjects and objects are noun phrases (named entities, concepts) mentioned in the
  passage; relations are short verb/preposition phrases connecting them.
- Extract only facts STATED in the passage — never world knowledge, never inference.
- Prefer specific entities over pronouns: if the passage says "It was completed in
  1889" and "it" refers to the Eiffel Tower, write subject "Eiffel Tower".
- A passage with no extractable facts yields an empty list. Do not pad."""

OPENIE_USER = "Passage:\n{passage}"


def normalize_phrase(text: str) -> str:
    """Lower + strip + collapse internal whitespace. One node per surface form."""
    return " ".join(text.lower().split())


class OpenIEExtractor:
    """Satisfies OpenIETransport. One parse() call per passage; maps TripleSet -> Triples."""

    def __init__(self, claude: ClaudeTransport, model: str = OPENIE_MODEL):
        self._claude = claude
        self._model = model

    def extract(self, passages: list[str]) -> list[list[Triple]]:
        out: list[list[Triple]] = []
        for passage in passages:
            res = self._claude.parse(
                model=self._model,
                system=OPENIE_SYSTEM,
                user=OPENIE_USER.format(passage=passage),
                max_tokens=1024,
                output_format=TripleSet,
                temperature=0.0,
            )
            parsed: TripleSet = res.parsed
            out.append(
                [
                    Triple(subject=t.subject, relation=t.relation, object=t.object)
                    for t in parsed.triples
                ]
            )
        return out
