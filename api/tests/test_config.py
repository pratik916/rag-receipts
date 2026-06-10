"""PRESETS ladder is binding: keys, order, and every flag value per contracts."""

import dataclasses

import pytest

from ragreceipts.config import PRESETS, IngestConfig, QueryConfig
from ragreceipts.types import RouteMode


def test_preset_keys_in_ladder_order():
    assert list(PRESETS) == [
        "bm25-only",
        "dense-rrf",
        "contextual",
        "rerank",
        "graph",
        "graph-rrf",
        "router-on",
    ]


@pytest.mark.parametrize(
    ("key", "bm25", "dense", "rerank", "contextual", "route_mode"),
    [
        ("bm25-only", True, False, False, False, RouteMode.FORCE_S1),
        ("dense-rrf", True, True, False, False, RouteMode.FORCE_S1),
        ("contextual", True, True, False, True, RouteMode.FORCE_S1),
        ("rerank", True, True, True, True, RouteMode.FORCE_S1),
        ("router-on", True, True, True, True, RouteMode.AUTO),
    ],
)
def test_preset_flags_exact(key, bm25, dense, rerank, contextual, route_mode):
    preset = PRESETS[key]
    assert preset.name == key
    assert preset.query.bm25 is bm25
    assert preset.query.dense is dense
    assert preset.query.rerank is rerank
    assert preset.ingest.contextual is contextual
    assert preset.query.route_mode is route_mode


def test_defaults_match_contracts():
    ingest = IngestConfig()
    assert (ingest.contextual, ingest.chunk_size, ingest.chunk_overlap) == (True, 512, 64)
    query = QueryConfig()
    assert (query.bm25, query.dense, query.rerank) == (True, True, True)
    assert query.route_mode is RouteMode.FORCE_S1
    assert (query.top_k_fuse, query.top_k_final) == (50, 5)


def test_configs_are_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        QueryConfig().bm25 = False  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        PRESETS["rerank"].name = "x"  # type: ignore[misc]


def test_graph_preset_flags():
    g = PRESETS["graph"]
    assert g.name == "graph"
    assert g.ingest.contextual is True
    assert g.query.bm25 is False
    assert g.query.dense is False
    assert g.query.rerank is False
    assert g.query.graph is True
    assert g.query.graph_recognition == "llm"
    assert g.query.route_mode is RouteMode.FORCE_S1


def test_graph_rrf_preset_flags():
    g = PRESETS["graph-rrf"]
    assert g.name == "graph-rrf"
    assert g.query.bm25 is True
    assert g.query.dense is True
    assert g.query.rerank is False
    assert g.query.graph is True
    assert g.query.graph_recognition == "llm"
    assert g.query.route_mode is RouteMode.FORCE_S1


def test_graph_presets_inserted_before_router_on():
    keys = list(PRESETS)
    assert keys.index("graph") < keys.index("router-on")
    assert keys.index("graph-rrf") < keys.index("router-on")
    assert keys.index("rerank") < keys.index("graph")


def test_query_config_graph_defaults_and_frozen():
    q = QueryConfig()
    assert q.graph is False
    assert q.graph_recognition == "llm"
    with pytest.raises(dataclasses.FrozenInstanceError):
        QueryConfig().graph = True  # type: ignore[misc]
