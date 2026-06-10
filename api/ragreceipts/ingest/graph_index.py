"""Knowledge-graph index build (HippoRAG-2 construction) + on-disk artifact.

Nodes: passage nodes (one per chunk, kind="passage", text=chunk_id) FIRST in chunk
order, then phrase nodes (normalized subjects/objects, kind="phrase") sorted by text.
This ordering makes the artifact byte-reproducible so manifest index_hashes["graph"]
pins the corpus exactly. Edges: relation (phrase-phrase from triples), appears_in
(phrase-passage), synonym (phrase-phrase where embedding cosine >= SYNONYM_THRESHOLD).
Vectors: phrase_vectors (per phrase node) + passage_vectors (isolated per chunk),
L2-normalized, row-aligned to their node order -> the GraphRetriever is self-contained
(no Qdrant at query time). The loader is the single place that knows the disk layout.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix

from ragreceipts.agents.openie import normalize_phrase
from ragreceipts.constants import SYNONYM_THRESHOLD
from ragreceipts.types import Chunk
from ragreceipts.vendors.base import EmbedTransport, OpenIETransport


@dataclass(frozen=True)
class GraphNode:
    node_id: int  # dense 0..N-1 (CSR row/col index)
    kind: str  # "phrase" | "passage"
    text: str  # normalized phrase, or chunk_id for passage nodes


@dataclass(frozen=True)
class GraphEdge:
    src: int
    dst: int
    kind: str  # "relation" | "appears_in" | "synonym"
    weight: float


def _unit(vec: list[float]) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    return arr / norm if norm > 0.0 else arr


def _adjacency_from_edges(n: int, edges: list[GraphEdge]) -> csr_matrix:
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for e in edges:  # symmetric: write both directions
        rows += [e.src, e.dst]
        cols += [e.dst, e.src]
        data += [e.weight, e.weight]
    return csr_matrix((data, (rows, cols)), shape=(n, n))


class GraphIndex:
    """In-memory graph loaded from the artifact (or produced by build_graph_index)."""

    def __init__(
        self,
        nodes: list[GraphNode],
        adjacency: csr_matrix,
        phrase_vectors: np.ndarray,
        passage_vectors: np.ndarray,
        passage_node_to_chunk: dict[int, str],
    ) -> None:
        self.nodes = nodes
        self.adjacency = adjacency
        self.phrase_vectors = phrase_vectors
        self.passage_vectors = passage_vectors
        self.passage_node_to_chunk = passage_node_to_chunk

    @property
    def n_passage(self) -> int:
        return len(self.passage_node_to_chunk)

    @property
    def n_phrase(self) -> int:
        return len(self.nodes) - self.n_passage

    @classmethod
    def load(cls, graph_dir: Path) -> GraphIndex:
        nodes: list[GraphNode] = []
        with (graph_dir / "nodes.jsonl").open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    d = json.loads(line)
                    nodes.append(GraphNode(node_id=d["node_id"], kind=d["kind"], text=d["text"]))
        edges: list[GraphEdge] = []
        with (graph_dir / "edges.jsonl").open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    d = json.loads(line)
                    edges.append(
                        GraphEdge(
                            src=d["src"], dst=d["dst"], kind=d["kind"], weight=float(d["weight"])
                        )
                    )
        phrase_vectors = np.load(graph_dir / "phrase_vectors.npy")
        passage_vectors = np.load(graph_dir / "passage_vectors.npy")
        passage_map = {
            int(k): v for k, v in json.loads((graph_dir / "passage_map.json").read_text()).items()
        }
        adjacency = _adjacency_from_edges(len(nodes), edges)
        return cls(nodes, adjacency, phrase_vectors, passage_vectors, passage_map)


@dataclass(frozen=True)
class GraphBuildResult:
    index: GraphIndex
    n_phrase: int
    n_passage: int
    n_edges: int
    n_triples: int
    _edges: list[GraphEdge]

    def edges_view(self) -> list[GraphEdge]:
        """The de-duplicated edge list (one entry per undirected edge)."""
        return self._edges


def build_graph_index(
    *,
    corpus_id: str,
    chunks: list[Chunk],
    openie: OpenIETransport,
    embed: EmbedTransport,
    synonym_threshold: float = SYNONYM_THRESHOLD,
) -> GraphBuildResult:
    # 1. Passage nodes first, in chunk order.
    passage_nodes = [
        GraphNode(node_id=i, kind="passage", text=chunk.chunk_id) for i, chunk in enumerate(chunks)
    ]
    passage_node_to_chunk = {i: chunk.chunk_id for i, chunk in enumerate(chunks)}
    n_passage = len(passage_nodes)

    # 2. OpenIE every chunk; collect normalized phrases + per-chunk phrase sets.
    triples_per_chunk = openie.extract([c.text for c in chunks])
    n_triples = sum(len(ts) for ts in triples_per_chunk)
    phrase_set: set[str] = set()
    for triples in triples_per_chunk:
        for t in triples:
            phrase_set.add(normalize_phrase(t.subject))
            phrase_set.add(normalize_phrase(t.object))

    # 3. Phrase nodes sorted by normalized text, ids after the passage block.
    sorted_phrases = sorted(phrase_set)
    phrase_id: dict[str, int] = {text: n_passage + i for i, text in enumerate(sorted_phrases)}
    phrase_nodes = [
        GraphNode(node_id=phrase_id[text], kind="phrase", text=text) for text in sorted_phrases
    ]
    nodes = passage_nodes + phrase_nodes

    # 4. Edges (de-duplicated by undirected key + kind, keeping max weight).
    edge_weight: dict[tuple[int, int, str], float] = {}

    def add_edge(a: int, b: int, kind: str, weight: float) -> None:
        if a == b:
            return
        key = (min(a, b), max(a, b), kind)
        edge_weight[key] = max(edge_weight.get(key, 0.0), weight)

    for chunk_i, triples in enumerate(triples_per_chunk):
        for t in triples:
            s = phrase_id[normalize_phrase(t.subject)]
            o = phrase_id[normalize_phrase(t.object)]
            add_edge(s, o, "relation", 1.0)
            add_edge(s, chunk_i, "appears_in", 1.0)
            add_edge(o, chunk_i, "appears_in", 1.0)

    # 5. Vectors (L2-normalized rows; phrase rows = embed_query, passage rows = isolated).
    dim = len(embed.embed_query("x")) if chunks or sorted_phrases else 0
    phrase_vectors = np.zeros((len(sorted_phrases), dim), dtype=np.float32)
    for i, text in enumerate(sorted_phrases):
        phrase_vectors[i] = _unit(embed.embed_query(text))
    passage_vectors = np.zeros((n_passage, dim), dtype=np.float32)
    for i, chunk in enumerate(chunks):
        [[vec]] = [embed.embed_documents([[chunk.text]])[0]]
        passage_vectors[i] = _unit(vec)

    # synonym edges: phrase pairs with cosine >= threshold (rows are unit norm).
    if len(sorted_phrases) > 1:
        sims = phrase_vectors @ phrase_vectors.T
        for i in range(len(sorted_phrases)):
            for j in range(i + 1, len(sorted_phrases)):
                cos = float(sims[i, j])
                if cos >= synonym_threshold:
                    add_edge(
                        phrase_id[sorted_phrases[i]], phrase_id[sorted_phrases[j]], "synonym", cos
                    )

    edges = [
        GraphEdge(src=a, dst=b, kind=kind, weight=w) for (a, b, kind), w in edge_weight.items()
    ]
    edges.sort(key=lambda e: (e.src, e.dst, e.kind))  # deterministic on-disk order
    adjacency = _adjacency_from_edges(len(nodes), edges)
    index = GraphIndex(nodes, adjacency, phrase_vectors, passage_vectors, passage_node_to_chunk)
    return GraphBuildResult(
        index=index,
        n_phrase=len(sorted_phrases),
        n_passage=n_passage,
        n_edges=len(edges),
        n_triples=n_triples,
        _edges=edges,
    )


def write_graph_index(result: GraphBuildResult, graph_dir: Path) -> None:
    graph_dir.mkdir(parents=True, exist_ok=True)
    with (graph_dir / "nodes.jsonl").open("w", encoding="utf-8") as fh:
        for n in result.index.nodes:  # already in (passage block, sorted phrases) order
            fh.write(json.dumps({"node_id": n.node_id, "kind": n.kind, "text": n.text}) + "\n")
    with (graph_dir / "edges.jsonl").open("w", encoding="utf-8") as fh:
        for e in result.edges_view():  # already sorted deterministically
            fh.write(
                json.dumps({"src": e.src, "dst": e.dst, "kind": e.kind, "weight": e.weight}) + "\n"
            )
    np.save(graph_dir / "phrase_vectors.npy", result.index.phrase_vectors)
    np.save(graph_dir / "passage_vectors.npy", result.index.passage_vectors)
    (graph_dir / "passage_map.json").write_text(
        json.dumps({str(k): v for k, v in result.index.passage_node_to_chunk.items()}, indent=2)
        + "\n",
        encoding="utf-8",
    )
