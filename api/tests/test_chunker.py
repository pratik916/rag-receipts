import pytest

from ragreceipts.ingest.chunker import chunk_document, chunk_passage


def _text(n: int) -> str:
    return " ".join(f"t{i}" for i in range(n))


def test_single_window_when_text_fits():
    spans = chunk_passage(
        corpus_id="c", doc_id="d1", passage_id="p1", text=_text(5), chunk_size=8, chunk_overlap=2
    )
    assert len(spans) == 1
    assert (spans[0].start_token, spans[0].end_token) == (0, 5)
    assert spans[0].chunk.chunk_id == "d1:0"
    assert spans[0].chunk.text == "t0 t1 t2 t3 t4"
    assert spans[0].chunk.passage_id == "p1"
    assert spans[0].chunk.corpus_id == "c"


def test_sliding_windows_with_overlap():
    # 10 tokens, size 4, overlap 1 -> stride 3 -> windows (0,4) (3,7) (6,10)
    spans = chunk_passage(
        corpus_id="c", doc_id="d1", passage_id="p1", text=_text(10), chunk_size=4, chunk_overlap=1
    )
    assert [(s.start_token, s.end_token) for s in spans] == [(0, 4), (3, 7), (6, 10)]
    assert [s.chunk.position for s in spans] == [0, 1, 2]
    assert spans[1].chunk.chunk_id == "d1:1"
    assert spans[1].chunk.text == "t3 t4 t5 t6"


def test_empty_text_yields_no_chunks():
    assert (
        chunk_passage(
            corpus_id="c", doc_id="d", passage_id="p", text="   ", chunk_size=4, chunk_overlap=0
        )
        == []
    )


def test_invalid_params_rejected():
    with pytest.raises(ValueError):
        chunk_passage(
            corpus_id="c", doc_id="d", passage_id="p", text="a b", chunk_size=4, chunk_overlap=4
        )
    with pytest.raises(ValueError):
        chunk_passage(
            corpus_id="c", doc_id="d", passage_id="p", text="a b", chunk_size=0, chunk_overlap=0
        )


# --- Plan A appends below: sentence-window packing + internal chunk_document (R4/R3) ---

S1 = "The Eiffel Tower is in Paris."  # 6 tokens
S2 = "It was completed in 1889."  # 5 tokens
S3 = "It is made of wrought iron."  # 6 tokens
S4 = "Millions of people visit it every year."  # 7 tokens
S5 = "The tower is repainted regularly."  # 5 tokens
SENT_TEXT = " ".join([S1, S2, S3, S4, S5])  # 29 tokens


def test_sentence_packing_golden_windows():
    # size 12 / overlap 5: S1+S2=11 fits; S2 (5 <= 5) carried as overlap, +S3=11 fits;
    # S3 (6 > 5) NOT carried; S4+S5=12 fits exactly.
    spans = chunk_passage(
        corpus_id="c", doc_id="d", passage_id="p", text=SENT_TEXT, chunk_size=12, chunk_overlap=5
    )
    assert [s.chunk.text for s in spans] == [f"{S1} {S2}", f"{S2} {S3}", f"{S4} {S5}"]
    assert [(s.start_token, s.end_token) for s in spans] == [(0, 11), (6, 17), (17, 29)]
    assert [s.chunk.position for s in spans] == [0, 1, 2]
    assert [s.chunk.chunk_id for s in spans] == ["d:0", "d:1", "d:2"]
    assert all(s.chunk.passage_id == "p" and s.chunk.corpus_id == "c" for s in spans)


def test_chunk_carries_its_token_range():
    # R3: every Chunk mirrors its ChunkSpan range, and the text IS that token slice
    tokens = SENT_TEXT.split()
    for s in chunk_passage(
        corpus_id="c", doc_id="d", passage_id="p", text=SENT_TEXT, chunk_size=12, chunk_overlap=5
    ):
        assert (s.chunk.start_token, s.chunk.end_token) == (s.start_token, s.end_token)
        assert s.chunk.text == " ".join(tokens[s.start_token : s.end_token])


def test_oversized_sentence_slides_with_stride():
    long_sentence = " ".join(f"w{i}" for i in range(30))  # one 30-token "sentence"
    spans = chunk_passage(
        corpus_id="c",
        doc_id="d",
        passage_id="p",
        text=long_sentence,
        chunk_size=12,
        chunk_overlap=5,
    )
    # stride = 12 - 5 = 7, same sliding rule Spike 0's stub used
    assert [(s.start_token, s.end_token) for s in spans] == [
        (0, 12),
        (7, 19),
        (14, 26),
        (21, 30),
    ]


def test_chunk_document_positions_run_across_passages():
    chunks = chunk_document(
        "tiny",
        "d",
        [("d-p0", f"{S1} {S2}"), ("d-p1", f"{S3} {S4}")],
        chunk_size=100,
        chunk_overlap=10,
    )
    assert [(c.passage_id, c.position, c.chunk_id) for c in chunks] == [
        ("d-p0", 0, "d:0"),
        ("d-p1", 1, "d:1"),
    ]
    # chunks never span passages; token ranges stay PASSAGE-relative (R3)
    assert [(c.start_token, c.end_token) for c in chunks] == [(0, 11), (0, 13)]
