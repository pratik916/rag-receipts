"""load_queries reads Spike 0's raw layout directly and normalizes in memory
(R1/R2): no intermediate eval queries file exists anywhere."""

import json
from pathlib import Path

import pytest

from ragreceipts.eval.alignment import GoldPassage, GoldSpan
from ragreceipts.eval.queries import (
    QueryRecord,
    load_queries,
    slice_queries,
    slice_query_ids,
)


def musique_record(i: int) -> dict:
    """Spike 0 raw queries.jsonl record shape (MuSiQue, typed passage gold)."""
    return {
        "query_id": f"mq{i}",
        "question": f"question {i}?",
        "answer": f"answer {i}",
        "answer_aliases": [f"alias {i}"],
        "gold": {"type": "passage", "passage_ids": [f"mu-p{i}a", f"mu-p{i}b"]},
    }


def nq_record(i: int) -> dict:
    """Spike 0 raw queries.jsonl record shape (NQ, typed span gold)."""
    return {
        "query_id": f"nqq-{i}",
        "question": f"who did thing {i}?",
        "answer_texts": [f"short answer {i}"],
        "gold": {"type": "span", "doc_id": f"nq-d{i}", "start_token": 4, "end_token": 24},
        "gold_text": "twenty tokens of gold text",
    }


def write_raw_corpus(tmp_path: Path, corpus_id: str, records: list[dict]) -> Path:
    raw = tmp_path / "corpora" / corpus_id / "raw"
    raw.mkdir(parents=True)
    (raw / "queries.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n")
    full = [r["query_id"] for r in records]
    (raw / "slice-full.json").write_text(json.dumps(full))
    (raw / "slice-smoke.json").write_text(json.dumps(full[:15]))
    return tmp_path


def test_load_queries_normalizes_musique_golds(tmp_path: Path) -> None:
    data_dir = write_raw_corpus(tmp_path, "musique-dev-300", [musique_record(i) for i in range(3)])
    queries = load_queries(data_dir, "musique-dev-300")
    assert len(queries) == 3
    assert isinstance(queries[0], QueryRecord)
    q0 = queries[0]
    assert q0.query_id == "mq0"
    assert q0.question == "question 0?"
    # R2: gold_answers = [answer] + answer_aliases
    assert q0.gold_answers == ["answer 0", "alias 0"]
    assert q0.golds == [
        GoldPassage(query_id="mq0", passage_id="mu-p0a"),
        GoldPassage(query_id="mq0", passage_id="mu-p0b"),
    ]


def test_load_queries_normalizes_nq_span_golds(tmp_path: Path) -> None:
    data_dir = write_raw_corpus(tmp_path, "nq-dev-300", [nq_record(i) for i in range(2)])
    queries = load_queries(data_dir, "nq-dev-300")
    q0 = queries[0]
    # R2: gold_answers = answer_texts for NQ
    assert q0.gold_answers == ["short answer 0"]
    # R3: span golds are positional token ranges, never span_text strings
    assert q0.golds == [GoldSpan(query_id="nqq-0", doc_id="nq-d0", start_token=4, end_token=24)]


def test_slices_come_from_slice_files(tmp_path: Path) -> None:
    data_dir = write_raw_corpus(tmp_path, "c", [musique_record(i) for i in range(20)])
    queries = load_queries(data_dir, "c")
    smoke_ids = slice_query_ids(data_dir, "c", "smoke")
    assert smoke_ids == [f"mq{i}" for i in range(15)]
    smoke = slice_queries(queries, smoke_ids)
    assert [q.query_id for q in smoke] == smoke_ids
    assert len(slice_queries(queries, slice_query_ids(data_dir, "c", "full"))) == 20


def test_slice_order_is_the_files_order_not_line_order(tmp_path: Path) -> None:
    data_dir = write_raw_corpus(tmp_path, "c", [musique_record(i) for i in range(3)])
    raw = data_dir / "corpora" / "c" / "raw"
    (raw / "slice-smoke.json").write_text(json.dumps(["mq2", "mq0"]))
    queries = load_queries(data_dir, "c")
    smoke = slice_queries(queries, slice_query_ids(data_dir, "c", "smoke"))
    assert [q.query_id for q in smoke] == ["mq2", "mq0"]


def test_slice_referencing_unknown_query_id_raises(tmp_path: Path) -> None:
    data_dir = write_raw_corpus(tmp_path, "c", [musique_record(0)])
    with pytest.raises(ValueError) as exc:
        slice_queries(load_queries(data_dir, "c"), ["mq0", "ghost-id"])
    assert "ghost-id" in str(exc.value)


def test_unknown_slice_raises(tmp_path: Path) -> None:
    data_dir = write_raw_corpus(tmp_path, "c", [musique_record(0)])
    with pytest.raises(ValueError):
        slice_query_ids(data_dir, "c", "medium")


def test_missing_queries_file_names_the_download_script(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError) as exc:
        load_queries(tmp_path, "nope")
    assert "queries.jsonl" in str(exc.value)
    assert "scripts/download_data.py" in str(exc.value)  # the real producer (Spike 0)
