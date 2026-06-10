from ragreceipts.ingest.nq import (
    MAX_GOLD_SPAN_TOKENS,
    nq_doc_id,
    nq_records,
    remap_span,
    select_long_answer,
    strip_html_tokens,
)


def test_strip_html_tokens_drops_html_and_maps_clean_spans():
    tokens = ["<p>", "Hello", "big world", "<table>", "x", "</p>"]
    is_html = [True, False, False, True, False, True]
    clean, spans = strip_html_tokens(tokens, is_html)
    assert clean == ["Hello", "big", "world", "x"]  # "big world" split into parts
    assert spans == [None, (0, 1), (1, 3), None, (3, 4), None]
    # round-trip invariant that makes whitespace-token indices stable:
    assert " ".join(clean).split() == clean


def test_strip_html_tokens_drops_empty_tokens():
    clean, spans = strip_html_tokens(["a", "   ", "b"], [False, False, False])
    assert clean == ["a", "b"]
    assert spans == [(0, 1), None, (1, 2)]


def test_remap_span_clips_html_edges():
    spans = [None, (0, 1), (1, 3), None, (3, 4), None]
    assert remap_span(spans, 0, 3) == (0, 3)  # <p> Hello big-world -> Hello big world
    assert remap_span(spans, 2, 5) == (1, 4)  # big-world <table> x -> big world x
    assert remap_span(spans, 0, 1) is None  # html-only span
    assert remap_span(spans, 3, 4) is None  # html-only span


def _la(start: int, end: int, cand: int) -> dict:
    return {
        "start_token": start,
        "end_token": end,
        "start_byte": 0,
        "end_byte": 0,
        "candidate_index": cand,
    }


def test_select_long_answer_majority_of_five():
    las = [
        _la(-1, -1, -1),
        _la(10, 20, 5),
        _la(10, 20, 5),
        _la(30, 40, 9),
        _la(-1, -1, -1),
    ]
    assert select_long_answer(las) == (10, 20)


def test_select_long_answer_requires_two_nonnull_annotators():
    las = [_la(10, 20, 5)] + [_la(-1, -1, -1)] * 4
    assert select_long_answer(las) is None


def test_select_long_answer_tie_breaks_deterministically():
    # 1-1 tie between (30,40) and (10,20): smallest start_token wins
    las = [_la(30, 40, 9), _la(10, 20, 5), _la(-1, -1, -1)]
    assert select_long_answer(las) == (10, 20)


def test_nq_doc_id_is_content_addressed():
    # sha1("hello world")[:16] == "2aae6c35c94fcfb4", precomputed
    assert nq_doc_id("hello world") == "nq-2aae6c35c94fcfb4"
    assert nq_doc_id("hello world") == nq_doc_id("hello world")
    assert nq_doc_id("hello world.") != nq_doc_id("hello world")


# --- nq_records: end-to-end normalization into R1 records ----------------------


def _example(
    *, tokens, is_html, long_answers, short_texts=None, question="q?", title="T", example_id=None
):
    ex = {
        "question_text": question,
        "document": {
            "title": title,
            "tokens": {
                "token": tokens,
                "is_html": is_html,
                "start_byte": [0] * len(tokens),
                "end_byte": [0] * len(tokens),
            },
        },
        "annotations": {
            "id": [str(i) for i in range(len(long_answers))],
            "long_answer": long_answers,
            "short_answers": [{"text": short_texts or []}],
            "yes_no_answer": [-1] * len(long_answers),
        },
    }
    if example_id is not None:
        ex["id"] = example_id
    return ex


def test_nq_records_emits_r1_shapes_with_top_level_gold_text():
    # original tokens: <p> Hello big-world <table> x </p>
    # clean tokens:    Hello big world x        (indices 0,1,2,3)
    tokens = ["<p>", "Hello", "big world", "<table>", "x", "</p>"]
    is_html = [True, False, False, True, False, True]
    # majority long answer spans original tokens [1, 3): "Hello" + "big world"
    las = [_la(1, 3, 5), _la(1, 3, 5), _la(-1, -1, -1), _la(-1, -1, -1), _la(-1, -1, -1)]
    ex = _example(
        tokens=tokens, is_html=is_html, long_answers=las, short_texts=["big"], question="Q?"
    )
    query, docs = nq_records(ex)

    # docs.jsonl record shape (R1): {doc_id, passage_id, title, text}
    assert len(docs) == 1
    doc = docs[0]
    assert set(doc) == {"doc_id", "passage_id", "title", "text"}
    assert doc["text"] == "Hello big world x"
    assert doc["title"] == "T"
    assert doc["doc_id"] == doc["passage_id"] == nq_doc_id("Hello big world x")

    # query record shape (R1): query_id, question, answer_texts, gold:{type:span,...}
    # PLUS a TOP-LEVEL gold_text next to gold (NOT nested inside gold).
    assert query["question"] == "Q?"
    assert query["answer_texts"] == ["big"]
    assert query["gold"] == {
        "type": "span",
        "doc_id": doc["doc_id"],
        "start_token": 0,
        "end_token": 3,
    }
    assert "gold_text" not in query["gold"]  # gold_text is top-level, not nested
    assert query["gold_text"] == "Hello big world"  # clean tokens [0:3]


def test_nq_records_query_id_uses_example_id_with_doc_id_fallback():
    # Real NQ rows carry a top-level "id" -> query_id is that id verbatim.
    tokens = ["<p>", "Hello", "world", "</p>"]
    is_html = [True, False, False, True]
    las = [_la(1, 3, 5), _la(1, 3, 5), _la(-1, -1, -1)]
    with_id = _example(tokens=tokens, is_html=is_html, long_answers=las, example_id="nq-dev-123")
    query, _ = nq_records(with_id)
    assert query["query_id"] == "nq-dev-123"

    # No "id" present -> query_id falls back to the content-addressed doc_id.
    without_id = _example(tokens=tokens, is_html=is_html, long_answers=las)
    query, docs = nq_records(without_id)
    assert query["query_id"] == docs[0]["doc_id"]


def test_nq_records_span_offsets_match_text_split_round_trip():
    # The load-bearing coordinate-system contract (R3): slicing the emitted text by
    # the emitted token offsets must select exactly the gold words.
    tokens = ["<h1>", "The", "quick brown", "<b>", "fox", "</b>", "jumps", "</h1>"]
    is_html = [True, False, False, True, False, True, False, True]
    # clean tokens: The quick brown fox jumps  (0,1,2,3,4)
    # gold original [1, 7): The quick-brown <b> fox </b> jumps -> The quick brown fox jumps
    las = [_la(1, 7, 5), _la(1, 7, 5), _la(-1, -1, -1)]
    ex = _example(tokens=tokens, is_html=is_html, long_answers=las)
    query, docs = nq_records(ex)
    text = docs[0]["text"]
    start, end = query["gold"]["start_token"], query["gold"]["end_token"]
    assert text.split()[start:end] == ["The", "quick", "brown", "fox", "jumps"]
    assert " ".join(text.split()[start:end]) == query["gold_text"]


def test_nq_records_excludes_when_fewer_than_two_annotators_agree():
    tokens = ["<p>", "Hello", "world", "</p>"]
    is_html = [True, False, False, True]
    las = [_la(1, 3, 5)] + [_la(-1, -1, -1)] * 4  # only 1 non-null annotator
    ex = _example(tokens=tokens, is_html=is_html, long_answers=las)
    assert nq_records(ex) is None


def test_nq_records_excludes_gold_longer_than_max_span_tokens():
    # A gold clean span longer than MAX_GOLD_SPAN_TOKENS is excluded, not truncated.
    n = MAX_GOLD_SPAN_TOKENS + 50
    tokens = ["w"] * n
    is_html = [False] * n
    las = [_la(0, n, 5), _la(0, n, 5), _la(-1, -1, -1)]  # spans all n clean tokens
    ex = _example(tokens=tokens, is_html=is_html, long_answers=las)
    assert nq_records(ex) is None


def test_nq_records_keeps_gold_at_exactly_max_span_tokens():
    n = MAX_GOLD_SPAN_TOKENS
    tokens = ["w"] * n
    is_html = [False] * n
    las = [_la(0, n, 5), _la(0, n, 5), _la(-1, -1, -1)]
    ex = _example(tokens=tokens, is_html=is_html, long_answers=las)
    result = nq_records(ex)
    assert result is not None
    query, _ = result
    assert query["gold"]["end_token"] - query["gold"]["start_token"] == MAX_GOLD_SPAN_TOKENS


def test_nq_records_excludes_when_gold_remaps_to_empty_span():
    # Majority long answer points only at HTML tokens -> empty clean span -> excluded.
    tokens = ["<p>", "real", "<table>", "</table>", "</p>"]
    is_html = [True, False, True, True, True]
    las = [_la(2, 4, 5), _la(2, 4, 5), _la(-1, -1, -1)]  # spans only html tokens
    ex = _example(tokens=tokens, is_html=is_html, long_answers=las)
    assert nq_records(ex) is None
