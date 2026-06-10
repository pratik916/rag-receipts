"""RAGAS v0.4 adapter behind a Protocol (transport-seam rule: CI uses FakeRagas).

Verified against docs.ragas.io (stable, fetched 2026-06-10):
- collections metrics: from ragas.metrics.collections import Faithfulness, AnswerRelevancy
    https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/faithfulness/
    https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/answer_relevance/
- direct Anthropic provider (Instructor adapter, no LangChain wrapper):
    llm_factory("<model>", provider="anthropic", client=...)
    https://docs.ragas.io/en/stable/howtos/llm-adapters/
- local sentence-transformers embeddings (no API key, works offline):
    HuggingFaceEmbeddings(model="BAAI/bge-small-en-v1.5")
    https://docs.ragas.io/en/stable/references/embeddings/
WARNING: most online examples show the obsolete v0.2 evaluate()/SingleTurnSample
API - do not use it. The v0.3->v0.4 migration guide confirms .ascore()/.score()
returning MetricResult with .value:
    https://docs.ragas.io/en/stable/howtos/migrations/migrate_from_v03_to_v04/
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from ragreceipts.constants import JUDGE_MODEL, RAGAS_EMBED_MODEL


@dataclass(frozen=True)
class RagasScores:
    faithfulness: float
    answer_relevancy: float


class RagasJudge(Protocol):
    def score(self, *, question: str, answer: str, contexts: list[str]) -> RagasScores: ...


class RagasV04Judge:
    """Real RAGAS v0.4 judge. Requires ANTHROPIC_API_KEY; never constructed in CI.

    All third-party imports are lazy so `import ragreceipts.eval.ragas_adapter`
    stays cheap and offline-safe.
    """

    def __init__(
        self,
        anthropic_client: object,
        judge_model: str = JUDGE_MODEL,
        embed_model: str = RAGAS_EMBED_MODEL,
    ) -> None:
        from ragas.embeddings import HuggingFaceEmbeddings
        from ragas.llms import llm_factory
        from ragas.metrics.collections import AnswerRelevancy, Faithfulness

        llm = llm_factory(judge_model, provider="anthropic", client=anthropic_client)
        embeddings = HuggingFaceEmbeddings(model=embed_model)  # local, zero keys
        self._faithfulness = Faithfulness(llm=llm)
        self._answer_relevancy = AnswerRelevancy(llm=llm, embeddings=embeddings)

    def score(self, *, question: str, answer: str, contexts: list[str]) -> RagasScores:
        faith = self._faithfulness.score(
            user_input=question, response=answer, retrieved_contexts=contexts
        )
        rel = self._answer_relevancy.score(user_input=question, response=answer)
        return RagasScores(faithfulness=float(faith.value), answer_relevancy=float(rel.value))
