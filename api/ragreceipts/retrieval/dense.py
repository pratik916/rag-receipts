"""Dense retrieval over Qdrant named vectors.

Both vector sets live on the same points (names below); IngestConfig.contextual selects
which one to search via vector_name_for(). Point ids are uuid5 of the chunk_id (Qdrant
requires int/UUID ids); the full Chunk — including the R3 start_token/end_token
token-range fields — is stored as payload and reconstructed on read.
"""

import uuid

from qdrant_client import QdrantClient

from ragreceipts.types import Chunk, ScoredChunk
from ragreceipts.vendors.base import EmbedTransport

VECTOR_CONTEXTUAL = "contextual"
VECTOR_ISOLATED = "isolated"


def vector_name_for(contextual: bool) -> str:
    return VECTOR_CONTEXTUAL if contextual else VECTOR_ISOLATED


def point_id_for_chunk(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"ragreceipts:{chunk_id}"))


class DenseRetriever:
    def __init__(
        self, client: QdrantClient, collection: str, vector_name: str, embed: EmbedTransport
    ):
        self._client = client
        self._collection = collection
        self._vector_name = vector_name
        self._embed = embed

    def search(self, query: str, k: int) -> list[ScoredChunk]:
        if k <= 0:
            return []
        vector = self._embed.embed_query(query)  # may raise VendorUnavailable
        response = self._client.query_points(
            collection_name=self._collection,
            query=vector,
            using=self._vector_name,
            limit=k,
            with_payload=True,
        )
        results: list[ScoredChunk] = []
        for point in response.points:
            payload = point.payload or {}
            chunk = Chunk(
                chunk_id=payload["chunk_id"],
                corpus_id=payload["corpus_id"],
                doc_id=payload["doc_id"],
                passage_id=payload["passage_id"],
                text=payload["text"],
                position=int(payload["position"]),
                start_token=int(payload["start_token"]),
                end_token=int(payload["end_token"]),
            )
            results.append(ScoredChunk(chunk=chunk, score=float(point.score), source="dense"))
        return results
