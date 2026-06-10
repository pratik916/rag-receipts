"""GraphRetriever — implements the Retriever protocol (search(query, k) -> ScoredChunk).

Faithful HippoRAG-2 query path: embed the query -> score phrase+passage nodes by cosine
-> (recognition='llm') Haiku filters the query-relevant phrases, else keep the top
embedding seeds -> personalized PageRank on the seeds -> blend ppr with the passage's
dense cosine -> top-k chunks (source='graph'). Self-contained: vectors come from the
graph artifact, so no Qdrant at query time. Degrades visibly: missing/empty index or an
embed/claude failure -> VendorUnavailable (RetrievalCore catches it -> 'graph-skipped').
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel

from ragreceipts.constants import GRAPH_BLEND, GRAPH_SEED_TOP_N, OPENIE_MODEL
from ragreceipts.ingest.graph_index import GraphIndex
from ragreceipts.retrieval.graph_ppr import personalized_pagerank
from ragreceipts.types import Chunk, ScoredChunk
from ragreceipts.vendors.base import ClaudeTransport, EmbedTransport, VendorUnavailable


class SeedSelection(BaseModel):
    """Recognition-memory output: the phrases the query is actually about."""

    phrases: list[str]


RECOGNITION_SYSTEM = """\
You are the recognition-memory step of a graph retriever. Given a question and a list of
candidate phrases extracted from a knowledge graph, return ONLY the phrases that the
question is actually asking about — the entities and concepts a correct answer must touch.
Drop phrases that are merely co-located. If unsure, keep the phrase. Return phrases
exactly as given."""

RECOGNITION_USER = """\
Question: {query}

Candidate phrases:
{phrases}"""


class GraphRetriever:
    def __init__(
        self,
        index: GraphIndex,
        *,
        chunks: list[Chunk],
        embed: EmbedTransport,
        claude: ClaudeTransport | None = None,
        recognition: str = "llm",
        blend: float = GRAPH_BLEND,
    ) -> None:
        if recognition not in ("llm", "embedding"):
            raise ValueError(f"recognition must be 'llm' or 'embedding', got {recognition!r}")
        if recognition == "llm" and claude is None:
            raise ValueError("recognition='llm' requires a ClaudeTransport (claude is None)")
        self._index = index
        self._chunk_by_id = {c.chunk_id: c for c in chunks}
        self._embed = embed
        self._claude = claude
        self._recognition = recognition
        self._blend = blend

    def search(self, query: str, k: int) -> list[ScoredChunk]:
        idx = self._index
        if idx.n_phrase == 0 or idx.n_passage == 0:
            raise VendorUnavailable("graph index is empty (no phrase or passage nodes)")
        if k <= 0:
            return []

        try:
            qvec = np.asarray(self._embed.embed_query(query), dtype=np.float32)
        except VendorUnavailable:
            raise
        norm = float(np.linalg.norm(qvec))
        if norm > 0.0:
            qvec = qvec / norm

        # Cosine to every node (passage rows then phrase rows, matching node ids).
        passage_cos = idx.passage_vectors @ qvec  # length n_passage
        phrase_cos = idx.phrase_vectors @ qvec  # length n_phrase
        node_cos = np.concatenate([passage_cos, phrase_cos])  # node_id-aligned

        # Top-N embedding seeds (positive cosine only).
        top = np.argsort(-node_cos)[:GRAPH_SEED_TOP_N]
        seeds: dict[int, float] = {int(n): float(node_cos[n]) for n in top if node_cos[n] > 0.0}
        if not seeds:  # degenerate: nothing positive — seed the single best node
            best = int(np.argmax(node_cos))
            seeds = {best: 1.0}

        if self._recognition == "llm":
            seeds = self._llm_filter(query, seeds)

        ppr = personalized_pagerank(idx.adjacency, seeds)

        scored: list[ScoredChunk] = []
        for node_id, chunk_id in idx.passage_node_to_chunk.items():
            chunk = self._chunk_by_id.get(chunk_id)
            if chunk is None:
                continue
            dense = float(passage_cos[node_id])
            score = self._blend * float(ppr[node_id]) + (1.0 - self._blend) * dense
            scored.append(ScoredChunk(chunk=chunk, score=score, source="graph"))
        scored.sort(key=lambda sc: (-sc.score, sc.chunk.chunk_id))
        return scored[:k]

    def _llm_filter(self, query: str, seeds: dict[int, float]) -> dict[int, float]:
        # Only phrase seeds are candidates for recognition; passages always stay.
        phrase_seed_ids = [nid for nid in seeds if self._index.nodes[nid].kind == "phrase"]
        if not phrase_seed_ids:
            return seeds
        candidates = [self._index.nodes[nid].text for nid in phrase_seed_ids]
        try:
            res = self._claude.parse(
                model=OPENIE_MODEL,
                system=RECOGNITION_SYSTEM,
                user=RECOGNITION_USER.format(query=query, phrases="\n".join(candidates)),
                max_tokens=1024,
                output_format=SeedSelection,
                temperature=0.0,
            )
        except VendorUnavailable:
            raise
        except Exception as exc:  # any recognition failure -> visible degrade
            raise VendorUnavailable(f"graph recognition memory failed: {exc!r}") from exc
        kept = set(res.parsed.phrases)
        filtered = {
            nid: mass
            for nid, mass in seeds.items()
            if self._index.nodes[nid].kind != "phrase" or self._index.nodes[nid].text in kept
        }
        # Never empty the seed set: if recognition dropped every phrase and there are no
        # passage seeds, fall back to the unfiltered embedding seeds.
        return (
            filtered
            if any(self._index.nodes[nid].kind == "phrase" for nid in filtered)
            or not phrase_seed_ids
            else seeds
        )
