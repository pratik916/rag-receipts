"""Dense index writer: BOTH vector sets as named vectors on the same points, every ingest.

payload = asdict(chunk), so the R3 start_token/end_token fields ride along
automatically and DenseRetriever can reconstruct the full Chunk."""

from dataclasses import asdict

from qdrant_client import QdrantClient, models

from ragreceipts.retrieval.dense import VECTOR_CONTEXTUAL, VECTOR_ISOLATED, point_id_for_chunk
from ragreceipts.types import Chunk


def write_dense_index(
    client: QdrantClient,
    collection: str,
    chunks: list[Chunk],
    contextual_vectors: list[list[float]],
    isolated_vectors: list[list[float]],
) -> None:
    if not chunks:
        raise ValueError("cannot write a dense index from zero chunks")
    if not (len(chunks) == len(contextual_vectors) == len(isolated_vectors)):
        raise ValueError(
            f"chunk/vector count mismatch: {len(chunks)} chunks, "
            f"{len(contextual_vectors)} contextual, {len(isolated_vectors)} isolated"
        )
    dim = len(contextual_vectors[0])
    if client.collection_exists(collection):
        client.delete_collection(collection)  # full rebuild semantics, same as sparse
    client.create_collection(
        collection_name=collection,
        vectors_config={
            VECTOR_CONTEXTUAL: models.VectorParams(size=dim, distance=models.Distance.COSINE),
            VECTOR_ISOLATED: models.VectorParams(size=dim, distance=models.Distance.COSINE),
        },
    )
    points = [
        models.PointStruct(
            id=point_id_for_chunk(chunk.chunk_id),
            vector={VECTOR_CONTEXTUAL: ctx, VECTOR_ISOLATED: iso},
            payload=asdict(chunk),
        )
        for chunk, ctx, iso in zip(chunks, contextual_vectors, isolated_vectors, strict=True)
    ]
    client.upsert(collection_name=collection, points=points)
