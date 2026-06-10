"""The RAGAS seam: Protocol + fake. The real judge is keyed and NEVER runs in CI."""

from ragreceipts.eval.ragas_adapter import RagasJudge, RagasScores, RagasV04Judge
from tests.fakes import FakeRagas  # tests/ is a package (R8)


def test_fake_ragas_conforms_to_protocol_and_scripts_scores() -> None:
    fake = FakeRagas(scores=[RagasScores(faithfulness=0.9, answer_relevancy=0.8)])
    judge: RagasJudge = fake  # structural typing check
    out = judge.score(question="q?", answer="a", contexts=["c1", "c2"])
    assert out == RagasScores(faithfulness=0.9, answer_relevancy=0.8)
    assert fake.calls == [{"question": "q?", "answer": "a", "contexts": ["c1", "c2"]}]


def test_real_judge_class_exists_but_is_not_constructed_offline() -> None:
    # Construction would import ragas + download a sentence-transformers model;
    # CI only asserts the class is importable and documents the keyed path.
    assert RagasV04Judge.__init__ is not object.__init__
