import pytest

from ragreceipts.ingest.chunker import chunk_passage


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
