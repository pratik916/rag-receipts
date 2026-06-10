"""PipelineConfig: the single source of truth for pipeline behavior.

Query-time flags (bm25/dense/rerank/route_mode) flip code paths on the same index.
Ingest-time flags (contextual, chunking params) select among pre-built index variants —
both dense vector sets are built at every ingest as Qdrant named vectors; `contextual`
selects the named vector at query time and the matching manifest hash for receipts.
Binding shapes from docs/superpowers/plans/2026-06-10-contracts.md.
"""

from dataclasses import dataclass

from ragreceipts.types import RouteMode


@dataclass(frozen=True)
class IngestConfig:  # ingest-time flags → index variants
    contextual: bool = (
        True  # True: doc-grouped voyage-context-3; False: per-chunk isolated, same model
    )
    chunk_size: int = 512  # tokens
    chunk_overlap: int = 64


@dataclass(frozen=True)
class QueryConfig:  # query-time flags → same index, different code path
    bm25: bool = True
    dense: bool = True
    rerank: bool = True
    route_mode: RouteMode = RouteMode.FORCE_S1
    top_k_fuse: int = 50  # candidates into RRF / rerank
    top_k_final: int = 5


@dataclass(frozen=True)
class PipelineConfig:
    name: str  # preset name
    ingest: IngestConfig
    query: QueryConfig


PRESETS: dict[str, PipelineConfig] = {
    "bm25-only": PipelineConfig(
        name="bm25-only",
        ingest=IngestConfig(contextual=False),
        query=QueryConfig(bm25=True, dense=False, rerank=False, route_mode=RouteMode.FORCE_S1),
    ),
    "dense-rrf": PipelineConfig(
        name="dense-rrf",
        ingest=IngestConfig(contextual=False),
        query=QueryConfig(bm25=True, dense=True, rerank=False, route_mode=RouteMode.FORCE_S1),
    ),
    "contextual": PipelineConfig(
        name="contextual",
        ingest=IngestConfig(contextual=True),
        query=QueryConfig(bm25=True, dense=True, rerank=False, route_mode=RouteMode.FORCE_S1),
    ),
    "rerank": PipelineConfig(
        name="rerank",
        ingest=IngestConfig(contextual=True),
        query=QueryConfig(bm25=True, dense=True, rerank=True, route_mode=RouteMode.FORCE_S1),
    ),
    "router-on": PipelineConfig(
        name="router-on",
        ingest=IngestConfig(contextual=True),
        query=QueryConfig(bm25=True, dense=True, rerank=True, route_mode=RouteMode.AUTO),
    ),
}
