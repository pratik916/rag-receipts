"""Shared fixture graph for graph-retriever / core tests. Deterministic FakeOpenIE
triples over a tiny chunk set + FakeEmbed vectors -> a GraphBuildResult and an
on-disk artifact, all offline."""

from pathlib import Path

from ragreceipts.ingest.graph_index import GraphBuildResult, build_graph_index, write_graph_index
from ragreceipts.types import Chunk
from ragreceipts.vendors.base import Triple
from tests.fakes import FakeEmbed, FakeOpenIE

# Four chunks across three docs; chunk_id == f"{doc}:{position}". Passage gold for the
# graph tests is "the eiffel tower is in paris" living in c0 (doc d1).
FIXTURE_CHUNKS = [
    Chunk("d1:0", "graphfix", "d1", "d1-p0", "The Eiffel Tower is a tower in Paris.", 0, 0, 8),
    Chunk("d1:1", "graphfix", "d1", "d1-p1", "Paris is the capital of France.", 1, 8, 14),
    Chunk("d2:0", "graphfix", "d2", "d2-p0", "Cats are small carnivorous mammals.", 0, 0, 5),
    Chunk(
        "d3:0",
        "graphfix",
        "d3",
        "d3-p0",
        "Solar panels convert sunlight into electricity.",
        0,
        0,
        6,
    ),
]

# FakeOpenIE script keyed by EXACT chunk text. Shared phrases ("paris") link c0 and c1.
FIXTURE_SCRIPT = {
    "The Eiffel Tower is a tower in Paris.": [
        Triple("Eiffel Tower", "located in", "Paris"),
        Triple("Eiffel Tower", "is a", "tower"),
    ],
    "Paris is the capital of France.": [
        Triple("Paris", "capital of", "France"),
    ],
    "Cats are small carnivorous mammals.": [
        Triple("cats", "are", "mammals"),
    ],
    "Solar panels convert sunlight into electricity.": [
        Triple("solar panels", "convert", "sunlight"),
    ],
}


def build_fixture_graph() -> GraphBuildResult:
    return build_graph_index(
        corpus_id="graphfix",
        chunks=FIXTURE_CHUNKS,
        openie=FakeOpenIE(script=FIXTURE_SCRIPT),
        embed=FakeEmbed(),
    )


def write_fixture_graph(graph_dir: Path) -> GraphBuildResult:
    result = build_fixture_graph()
    write_graph_index(result, graph_dir)
    return result
