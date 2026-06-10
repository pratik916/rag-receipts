"""Tiny fixture corpus in Spike 0's raw/ on-disk format, plus a Chunk factory for tests.

The benchmark slices are unsegmented (doc_id == passage_id); the fixture gives d1 TWO
passages on purpose — the format carries both ids, and the loader seam must handle
segmented documents (contracts: "passage_id == doc_id when unsegmented").
TINY_QUERIES mirrors both real gold shapes: q1 is a MuSiQue-style passage gold
("answer" + "answer_aliases"), q2 an NQ-style span gold ("answer_texts" + "gold_text";
token indices into the doc text's .split()).
"""

import json
from pathlib import Path

from ragreceipts.types import Chunk

TINY_PASSAGES = [
    {
        "doc_id": "d1",
        "passage_id": "d1-p0",
        "title": "Eiffel Tower",
        "text": (
            "The Eiffel Tower is a wrought iron lattice tower in Paris. "
            "It was completed in 1889. Millions of visitors climb the tower every year."
        ),
    },
    {
        "doc_id": "d1",
        "passage_id": "d1-p1",
        "title": "Eiffel Tower",
        "text": ("The tower is the tallest structure in Paris. Its height is about 330 metres."),
    },
    {
        "doc_id": "d2",
        "passage_id": "d2-p0",
        "title": "Cats",
        "text": (
            "Cats are small carnivorous mammals. Domestic cats often hunt mice and birds. "
            "A group of cats is called a clowder."
        ),
    },
    {
        "doc_id": "d3",
        "passage_id": "d3-p0",
        "title": "Solar panels",
        "text": (
            "Solar panels convert sunlight into electricity. "
            "Photovoltaic cells are made of silicon. "
            "Panel efficiency has improved steadily."
        ),
    },
]
TINY_QUERIES = [
    {
        "query_id": "q1",
        "question": "How tall is the Eiffel Tower?",
        "answer": "330 metres",
        "answer_aliases": ["about 330 metres"],
        "gold": {"type": "passage", "passage_ids": ["d1-p1"]},
    },
    {
        "query_id": "q2",
        "question": "What do domestic cats hunt?",
        "answer_texts": ["mice and birds"],
        "gold": {"type": "span", "doc_id": "d2", "start_token": 9, "end_token": 12},
        "gold_text": "mice and birds.",
    },
]
TINY_DOWNLOAD_META = {
    "corpus_id": "tiny",
    "dataset": {
        "hf_id": "local/tiny-fixture",
        "config": "default",
        "split": "test",
        "revision": "fixture-v1",
    },
    "selection_rule": "in-repo fixture",
    "seed": 0,
    "n_queries": 2,
    "n_smoke": 2,
}


def write_tiny_corpus(data_dir: Path, corpus_id: str = "tiny") -> Path:
    """Writes the fixture corpus under data_dir/corpora/{corpus_id}/raw; returns corpus dir."""
    corpus_dir = data_dir / "corpora" / corpus_id
    raw = corpus_dir / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    with (raw / "docs.jsonl").open("w", encoding="utf-8") as fh:
        for row in TINY_PASSAGES:
            fh.write(json.dumps(row) + "\n")
    with (raw / "queries.jsonl").open("w", encoding="utf-8") as fh:
        for row in TINY_QUERIES:
            fh.write(json.dumps(row) + "\n")
    slice_full = [q["query_id"] for q in TINY_QUERIES]
    (raw / "slice-full.json").write_text(json.dumps(slice_full), encoding="utf-8")
    (raw / "slice-smoke.json").write_text(json.dumps(slice_full), encoding="utf-8")
    meta = {**TINY_DOWNLOAD_META, "corpus_id": corpus_id}
    (raw / "download_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return corpus_dir


def make_chunk(
    chunk_id: str,
    text: str = "",
    corpus_id: str = "test",
    passage_id: str | None = None,
    start_token: int = 0,
    end_token: int | None = None,
) -> Chunk:
    doc_id, position = chunk_id.rsplit(":", 1)
    body = text or chunk_id
    return Chunk(
        chunk_id=chunk_id,
        corpus_id=corpus_id,
        doc_id=doc_id,
        passage_id=passage_id or doc_id,
        text=body,
        position=int(position),
        start_token=start_token,
        end_token=end_token if end_token is not None else start_token + len(body.split()),
    )
