"""Shared fixture graph for graph-retriever / core tests. Deterministic FakeOpenIE
triples over a tiny chunk set + FakeEmbed vectors -> a GraphBuildResult and an
on-disk artifact, all offline."""

import json
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


# =====================================================================
# Plan F (G4) harness-corpus helpers — APPENDED (RG3). Disjoint from the
# FIXTURE_CHUNKS/build_fixture_graph symbols above: these build a tiny
# *labeled* corpus (queries + slices + manifest + a real graph artifact) the
# AblationRunner can drive end to end. Each query's gold passage shares an
# entity phrase with the query seed text, so query-seeded PPR concentrates on
# the gold passage and the `graph` preset's Recall@5 is a real, deterministic
# signal that can actually fail. Mirrors harness_fixtures.py's raw/ layout (R1).
# =====================================================================

N_QUERIES = 4


def _chunk(passage_id: str, text: str, position: int = 0) -> Chunk:
    return Chunk(
        chunk_id=f"{passage_id}:{position}",
        corpus_id="graph-harness",
        doc_id=passage_id,
        passage_id=passage_id,
        text=text,
        position=position,
        start_token=0,
        end_token=len(text.split()),
    )


def fixture_chunks() -> list[Chunk]:
    """Gold passages g0..g3 each name a distinct entity; fillers name others."""
    chunks: list[Chunk] = []
    for i in range(N_QUERIES):
        chunks.append(_chunk(f"g{i}", f"Entity{i} is located in City{i} near River{i}."))
    for i in range(N_QUERIES):
        chunks.append(_chunk(f"f{i}", f"Filler{i} mentions Topic{i} and Place{i}."))
    return chunks


def fixture_openie_script() -> dict[str, list[Triple]]:
    """One triple list per passage text; entities become phrase nodes/edges."""
    script: dict[str, list[Triple]] = {}
    for c in fixture_chunks():
        i = c.passage_id[1:]
        if c.passage_id.startswith("g"):
            script[c.text] = [
                Triple(subject=f"entity{i}", relation="located in", object=f"city{i}"),
                Triple(subject=f"city{i}", relation="near", object=f"river{i}"),
            ]
        else:
            script[c.text] = [
                Triple(subject=f"filler{i}", relation="mentions", object=f"topic{i}"),
            ]
    return script


def fixture_queries() -> list[dict]:
    """Each query seeds on EntityI -> PPR lands on gold gI (raw/ layout, R1)."""
    out: list[dict] = []
    for i in range(N_QUERIES):
        out.append(
            {
                "query_id": f"q{i}",
                "question": f"Where is Entity{i} located?",
                "answer": f"City{i}",
                "answer_aliases": [],
                "gold": {"type": "passage", "passage_ids": [f"g{i}"]},
            }
        )
    return out


def fixture_query_aliases() -> dict[str, str]:
    """Map each query to its gold chunk text so the retriever's FakeEmbed query
    vector equals the gold passage vector (cosine 1.0). FakeEmbed embeds a
    single-chunk document to _unit_vector(text), the same vector embed_query
    returns for that text — so aliasing the query to the gold text makes the
    graph's dense term peak on the gold passage and PPR concentrates there."""
    gold = {c.passage_id: c.text for c in fixture_chunks() if c.passage_id.startswith("g")}
    return {q["question"]: gold[q["gold"]["passage_ids"][0]] for q in fixture_queries()}


def write_graph_artifact(corpus_dir: Path) -> Path:
    """Build + write the graph/ artifact with Plan E's real builder. Returns the dir."""
    result = build_graph_index(
        corpus_id="graph-harness",
        chunks=fixture_chunks(),
        openie=FakeOpenIE(script=fixture_openie_script()),
        embed=FakeEmbed(),
    )
    graph_dir = corpus_dir / "graph"
    write_graph_index(result, graph_dir)
    return graph_dir


def write_graph_corpus(data_dir: Path, corpus_id: str = "graph-harness") -> Path:
    """raw/ queries + slice files + manifest (dataset name 'musique' to pass R10)
    + the built graph artifact. Returns data_dir."""
    corpus_dir = data_dir / "corpora" / corpus_id
    raw = corpus_dir / "raw"
    raw.mkdir(parents=True)
    queries = fixture_queries()
    (raw / "queries.jsonl").write_text("\n".join(json.dumps(q) for q in queries) + "\n")
    ids = [q["query_id"] for q in queries]
    (raw / "slice-full.json").write_text(json.dumps(ids))
    (raw / "slice-smoke.json").write_text(json.dumps(ids[:15]))
    write_graph_artifact(corpus_dir)
    (corpus_dir / "manifest.json").write_text(
        json.dumps(
            {
                "corpus_id": corpus_id,
                "dataset": {"name": "musique", "hf_id": "fixture", "split": "x", "revision": "0"},
                "index_hashes": {
                    "dense_contextual": "sha256:c",
                    "dense_isolated": "sha256:i",
                    "sparse": "sha256:s",
                    "graph": "sha256:g",
                },
                "n_queries": len(queries),
            }
        )
    )
    return data_dir
