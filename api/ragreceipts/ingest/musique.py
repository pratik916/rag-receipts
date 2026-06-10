"""MuSiQue (musique_ans v1.0) normalization - pure functions, no network.

Raw example schema (verified against dgslibisey/MuSiQue revision c8f4f8c9, 2026-06-10):
  id: str, question: str, answer: str, answer_aliases: list[str], answerable: bool,
  paragraphs: list[{idx: int, title: str, paragraph_text: str, is_supporting: bool}],
  question_decomposition: list[{id, question, answer, paragraph_support_idx: int}]
"""

import hashlib


def musique_passage_id(title: str, text: str) -> str:
    """Deterministic content-addressed id; dedups identical paragraphs across examples."""
    digest = hashlib.sha1(f"{title}\n{text}".encode()).hexdigest()
    return f"mu-{digest[:16]}"


def musique_records(example: dict) -> tuple[dict, list[dict]]:
    """Normalize one raw example into (query_record, passage_records).

    Raises ValueError when is_supporting disagrees with question_decomposition's
    paragraph_support_idx - the caller counts and skips such examples (degrade
    visibly, never silently).
    """
    supporting_idx = {p["idx"] for p in example["paragraphs"] if p["is_supporting"]}
    decomp_idx = {d["paragraph_support_idx"] for d in example["question_decomposition"]}
    if supporting_idx != decomp_idx:
        raise ValueError(
            f"{example['id']}: is_supporting {sorted(supporting_idx)} != "
            f"decomposition support {sorted(decomp_idx)}"
        )
    passage_records: list[dict] = []
    gold_passage_ids: list[str] = []
    for p in example["paragraphs"]:
        pid = musique_passage_id(p["title"], p["paragraph_text"])
        passage_records.append(
            {"doc_id": pid, "passage_id": pid, "title": p["title"], "text": p["paragraph_text"]}
        )
        if p["is_supporting"]:
            gold_passage_ids.append(pid)
    query_record = {
        "query_id": example["id"],
        "question": example["question"],
        "answer": example["answer"],
        "answer_aliases": list(example["answer_aliases"]),
        "gold": {"type": "passage", "passage_ids": gold_passage_ids},
    }
    return query_record, passage_records
