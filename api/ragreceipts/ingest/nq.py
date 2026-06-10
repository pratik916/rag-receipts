"""Natural Questions (original NQ) normalization - pure functions, no network.

Verified row shape (datasets-server, google-research-datasets/natural_questions,
config "dev", revision e8103d56, 2026-06-10):
  ex["document"]["tokens"] = {"token": list[str], "is_html": list[bool],
                              "start_byte": list[int], "end_byte": list[int]}
  ex["annotations"] = {"id": list[str] (5 annotators on dev),
                       "long_answer": list[{start_token, end_token, start_byte,
                                            end_byte, candidate_index}],
                       "short_answers": list[{... lists, incl. "text": list[str]}],
                       "yes_no_answer": list[int]}  # null = candidate_index == -1
Long-answer token indices include HTML tokens; these helpers map them into
clean whitespace-token space (the space chunk_passage operates in).
"""

import hashlib
from collections import Counter

TokenSpan = tuple[int, int]

# This is 2x the chunker's default chunk_size (512): a chunk covers at most
# chunk_size tokens, so the >=50% span-hit rule is mathematically unsatisfiable
# for golds longer than 2 * chunk_size tokens. Such golds (typically giant
# tables) are excluded at normalization time and counted in download_meta.json -
# disclosed, never silently truncated (decisions D5.3).
MAX_GOLD_SPAN_TOKENS = 1024


def strip_html_tokens(
    tokens: list[str], is_html: list[bool]
) -> tuple[list[str], list[TokenSpan | None]]:
    """Drop HTML tokens; split tokens containing internal whitespace into parts.

    Returns (clean_tokens, token_spans) where token_spans[i] is the (start, end)
    range the i-th ORIGINAL token occupies in clean_tokens, or None if dropped
    (html or whitespace-only). Guarantees " ".join(clean_tokens).split() == clean_tokens.
    """
    clean: list[str] = []
    spans: list[TokenSpan | None] = []
    for token, html in zip(tokens, is_html, strict=True):
        if html:
            spans.append(None)
            continue
        parts = token.split()
        if not parts:
            spans.append(None)
            continue
        start = len(clean)
        clean.extend(parts)
        spans.append((start, len(clean)))
    return clean, spans


def remap_span(
    token_spans: list[TokenSpan | None], start_token: int, end_token: int
) -> TokenSpan | None:
    """Map an original-token [start_token, end_token) range to clean-token space.

    Returns None when the range contains no visible (non-html) tokens.
    """
    visible = [s for s in token_spans[start_token:end_token] if s is not None]
    if not visible:
        return None
    return visible[0][0], visible[-1][1]


def select_long_answer(long_answers: list[dict]) -> TokenSpan | None:
    """Majority gold over the 5 dev annotators (rule recorded in the decisions doc).

    Require >= 2 annotators with a non-null long answer (candidate_index != -1);
    gold = the most frequent (start_token, end_token) span; ties broken by smallest
    start_token then end_token. Returns None otherwise.
    """
    non_null = [
        (la["start_token"], la["end_token"]) for la in long_answers if la["candidate_index"] != -1
    ]
    if len(non_null) < 2:
        return None
    counts = Counter(non_null)
    return min(counts, key=lambda span: (-counts[span], span[0], span[1]))


def nq_doc_id(text: str) -> str:
    """Content-addressed doc id; dedups the same Wikipedia page across queries."""
    return f"nq-{hashlib.sha1(text.encode()).hexdigest()[:16]}"


def nq_records(example: dict) -> tuple[dict, list[dict]] | None:
    """Normalize one raw NQ example into (query_record, doc_records), or None if excluded.

    Excluded (return None) when, per decisions D3/D5:
      - fewer than 2 annotators give a non-null long answer, or
      - the majority span remaps to an empty clean span (html-only), or
      - the remapped clean span exceeds MAX_GOLD_SPAN_TOKENS.

    Emitted shapes (contracts R1):
      doc record   = {"doc_id", "passage_id", "title", "text"}
      query record = {"query_id", "question", "answer_texts",
                      "gold": {"type": "span", "doc_id", "start_token", "end_token"},
                      "gold_text": <clean words of the gold span>}   # gold_text TOP-LEVEL
    start_token/end_token index the cleaned, whitespace-split token sequence, so
    text.split()[start_token:end_token] selects exactly the gold words (R3).

    query_id is the raw example's "id", falling back to doc_id when absent.
    """
    annotations = example["annotations"]
    majority = select_long_answer(annotations["long_answer"])
    if majority is None:
        return None

    tokens_field = example["document"]["tokens"]
    clean, token_spans = strip_html_tokens(tokens_field["token"], tokens_field["is_html"])

    remapped = remap_span(token_spans, majority[0], majority[1])
    if remapped is None:
        return None
    start_token, end_token = remapped
    if end_token - start_token > MAX_GOLD_SPAN_TOKENS:
        return None

    text = " ".join(clean)
    doc_id = nq_doc_id(text)
    doc_record = {
        "doc_id": doc_id,
        "passage_id": doc_id,
        "title": example["document"]["title"],
        "text": text,
    }

    short_answers = annotations["short_answers"]
    answer_texts: list[str] = []
    for sa in short_answers:
        answer_texts.extend(sa.get("text", []))

    query_record = {
        "query_id": example.get("id", doc_id),
        "question": example["question_text"],
        "answer_texts": answer_texts,
        "gold": {
            "type": "span",
            "doc_id": doc_id,
            "start_token": start_token,
            "end_token": end_token,
        },
        "gold_text": " ".join(clean[start_token:end_token]),
    }
    return query_record, [doc_record]
