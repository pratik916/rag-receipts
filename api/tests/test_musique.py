import pytest

from ragreceipts.ingest.musique import musique_passage_id, musique_records

EXAMPLE = {
    "id": "2hop__460946_294723",
    "question": "Who is the spouse of the Green performer?",
    "answer": "Miquette Giraudy",
    "answer_aliases": [],
    "answerable": True,
    "paragraphs": [
        {"idx": 0, "title": "A", "paragraph_text": "Distractor paragraph.", "is_supporting": False},
        {
            "idx": 1,
            "title": "B",
            "paragraph_text": "Supporting paragraph one.",
            "is_supporting": True,
        },
        {
            "idx": 2,
            "title": "C",
            "paragraph_text": "Supporting paragraph two.",
            "is_supporting": True,
        },
    ],
    "question_decomposition": [
        {"id": 460946, "question": "sub1", "answer": "x", "paragraph_support_idx": 1},
        {"id": 294723, "question": "sub2", "answer": "y", "paragraph_support_idx": 2},
    ],
}


def test_passage_id_is_deterministic_content_address():
    # sha1("T\nhello world")[:16] == "23d5d02c9d6894dc", precomputed
    assert musique_passage_id("T", "hello world") == "mu-23d5d02c9d6894dc"
    assert musique_passage_id("T", "hello world!") != musique_passage_id("T", "hello world")


def test_records_extract_golds_and_dedupable_passages():
    query, passages = musique_records(EXAMPLE)
    assert query["query_id"] == "2hop__460946_294723"
    assert query["question"] == EXAMPLE["question"]
    assert query["answer"] == "Miquette Giraudy"
    assert query["gold"]["type"] == "passage"
    expected_golds = [
        musique_passage_id("B", "Supporting paragraph one."),
        musique_passage_id("C", "Supporting paragraph two."),
    ]
    assert query["gold"]["passage_ids"] == expected_golds
    assert len(passages) == 3
    for p in passages:
        assert p["doc_id"] == p["passage_id"]  # unsegmented: passage == doc
        assert set(p) == {"doc_id", "passage_id", "title", "text"}


def test_passage_ids_dedup_across_examples():
    # Same (title, paragraph_text) shared across two distinct examples must
    # content-address to one corpus doc; a different paragraph must not collide.
    shared = {"idx": 0, "title": "Shared", "paragraph_text": "Shared body.", "is_supporting": True}
    other = {"idx": 0, "title": "Other", "paragraph_text": "Other body.", "is_supporting": True}
    ex1 = {
        **EXAMPLE,
        "id": "e1",
        "question": "q1",
        "paragraphs": [shared],
        "question_decomposition": [
            {"id": 1, "question": "s", "answer": "x", "paragraph_support_idx": 0},
        ],
    }
    ex2 = {
        **EXAMPLE,
        "id": "e2",
        "question": "q2",
        "paragraphs": [shared],
        "question_decomposition": [
            {"id": 2, "question": "s", "answer": "y", "paragraph_support_idx": 0},
        ],
    }
    ex3 = {**ex2, "id": "e3", "paragraphs": [other]}
    _, p1 = musique_records(ex1)
    _, p2 = musique_records(ex2)
    _, p3 = musique_records(ex3)
    assert p1[0]["doc_id"] == p2[0]["doc_id"]  # same content -> one corpus doc
    assert p1[0]["doc_id"] != p3[0]["doc_id"]  # different content -> different doc


def test_support_mismatch_raises():
    broken = dict(EXAMPLE)
    broken["question_decomposition"] = [
        {"id": 1, "question": "sub1", "answer": "x", "paragraph_support_idx": 0},
    ]
    with pytest.raises(
        ValueError,
        match=r"2hop__460946_294723: is_supporting \[1, 2\] != decomposition support \[0\]",
    ):
        musique_records(broken)
