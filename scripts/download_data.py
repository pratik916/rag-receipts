#!/usr/bin/env python
"""Download + normalize the Spike 0 benchmark slices (network required, no API keys).

Produces (gitignored):
  data/corpora/musique-dev-300/raw/{queries.jsonl,docs.jsonl,slice-full.json,
                                    slice-smoke.json,download_meta.json}
  data/corpora/nq-dev-300/raw/{...same files...}

Run from repo root:
  uv run --project api python scripts/download_data.py --peek          # schema check
  uv run --project api python scripts/download_data.py --corpus all    # full pull
Selection rules and pins are documented in
docs/superpowers/specs/2026-06-10-spike0-decisions.md (D2, D4, D5).
"""

import argparse
import json
import random
import sys
from datetime import UTC, datetime
from pathlib import Path

import datasets as hf_datasets

# load_dataset(path, name=None, split=..., revision=..., streaming=...) verified against
# https://huggingface.co/docs/datasets/en/package_reference/loading_methods (v4.8.4)
from datasets import load_dataset

from ragreceipts.ingest.musique import musique_records
from ragreceipts.ingest.nq import (
    nq_doc_id,
    remap_span,
    select_long_answer,
    strip_html_tokens,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

MUSIQUE_HF_ID = "dgslibisey/MuSiQue"  # mirror of official musique_ans v1.0 jsonl
MUSIQUE_REVISION = "c8f4f8c9465fb69d31a8eae894c3fd509c4ca321"  # 2023-06-16
NQ_HF_ID = "google-research-datasets/natural_questions"
NQ_CONFIG = "dev"  # parquet-only config: validation split, 7,830 rows
NQ_REVISION = "e8103d566bef4154c2c12b17c6095ec5275840cc"  # 2024-03-11
N_QUERIES = 300
N_SMOKE = 15
SELECTION_SEED = 42
# With chunk_size=512 a chunk covers at most 512 tokens, so the >=50% rule is
# unsatisfiable for golds longer than 2*512 tokens. Exclude and count them.
MAX_GOLD_SPAN_TOKENS = 1024


def _write_outputs(out_dir: Path, queries: list[dict], docs: list[dict], meta: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "queries.jsonl").open("w") as f:
        for q in queries:
            f.write(json.dumps(q) + "\n")
    with (out_dir / "docs.jsonl").open("w") as f:
        for d in docs:
            f.write(json.dumps(d) + "\n")
    slice_full = [q["query_id"] for q in queries]
    (out_dir / "slice-full.json").write_text(json.dumps(slice_full, indent=2))
    (out_dir / "slice-smoke.json").write_text(json.dumps(slice_full[:N_SMOKE], indent=2))
    (out_dir / "download_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"{meta['corpus_id']}: {len(queries)} queries, {len(docs)} docs -> {out_dir}")


def _base_meta(
    corpus_id: str,
    hf_id: str,
    config: str,
    revision: str,
    selection_rule: str,
    extra: dict,
) -> dict:
    return {
        "corpus_id": corpus_id,
        "dataset": {
            "hf_id": hf_id,
            "config": config,
            "split": "validation",
            "revision": revision,
        },
        "selection_rule": selection_rule,
        "seed": SELECTION_SEED,
        "n_queries": N_QUERIES,
        "n_smoke": N_SMOKE,
        "datasets_lib_version": hf_datasets.__version__,
        "created_at": datetime.now(UTC).isoformat(),
        **extra,
    }


def download_musique(peek: bool = False) -> None:
    ds = load_dataset(MUSIQUE_HF_ID, split="validation", revision=MUSIQUE_REVISION)
    if peek:
        ex = ds[0]
        print("musique keys:", sorted(ex.keys()))
        print("paragraph keys:", sorted(ex["paragraphs"][0].keys()))
        print("decomposition keys:", sorted(ex["question_decomposition"][0].keys()))
        return
    examples = sorted(ds, key=lambda ex: ex["id"])
    random.Random(SELECTION_SEED).shuffle(examples)
    queries: list[dict] = []
    docs_by_id: dict[str, dict] = {}
    n_skipped = 0
    for ex in examples:
        if len(queries) == N_QUERIES:
            break
        try:
            query_record, passage_records = musique_records(ex)
        except ValueError as err:
            print(f"skip {ex['id']}: {err}", file=sys.stderr)
            n_skipped += 1
            continue
        queries.append(query_record)
        for p in passage_records:
            docs_by_id.setdefault(p["doc_id"], p)
    if len(queries) < N_QUERIES:
        raise SystemExit(f"musique: only {len(queries)} qualifying queries, need {N_QUERIES}")
    meta = _base_meta(
        "musique-dev-300",
        MUSIQUE_HF_ID,
        "default",
        MUSIQUE_REVISION,
        "sort dev by id, shuffle with seed, take first 300 whose is_supporting set "
        "matches question_decomposition support idx; corpus = union of all 20 "
        "paragraphs per selected example, deduped by content-addressed passage_id",
        {"n_docs": len(docs_by_id), "n_skipped_support_mismatch": n_skipped},
    )
    _write_outputs(
        REPO_ROOT / "data/corpora/musique-dev-300/raw",
        queries,
        list(docs_by_id.values()),
        meta,
    )


def download_nq(peek: bool = False) -> None:
    ds = load_dataset(NQ_HF_ID, NQ_CONFIG, split="validation", streaming=True, revision=NQ_REVISION)
    it = iter(ds)
    if peek:
        ex = next(it)
        print("nq keys:", sorted(ex.keys()))
        print("annotations keys:", sorted(ex["annotations"].keys()))
        print("first long_answer:", ex["annotations"]["long_answer"][0])
        print("document.tokens keys:", sorted(ex["document"]["tokens"].keys()))
        print("question keys:", sorted(ex["question"].keys()))
        return
    queries: list[dict] = []
    docs_by_id: dict[str, dict] = {}
    skip = {
        "no_majority_long_answer": 0,
        "empty_doc": 0,
        "unmappable_span": 0,
        "gold_too_long": 0,
    }
    n_seen = 0
    for ex in it:
        if len(queries) == N_QUERIES:
            break
        n_seen += 1
        gold = select_long_answer(ex["annotations"]["long_answer"])
        if gold is None:
            skip["no_majority_long_answer"] += 1
            continue
        toks = ex["document"]["tokens"]
        clean, token_spans = strip_html_tokens(toks["token"], toks["is_html"])
        if not clean:
            skip["empty_doc"] += 1
            continue
        mapped = remap_span(token_spans, gold[0], gold[1])
        if mapped is None:
            skip["unmappable_span"] += 1
            continue
        start, end = mapped
        if end - start > MAX_GOLD_SPAN_TOKENS:
            skip["gold_too_long"] += 1
            continue
        text = " ".join(clean)
        doc_id = nq_doc_id(text)  # content-addressed: dedups repeated pages
        docs_by_id.setdefault(
            doc_id,
            {
                "doc_id": doc_id,
                "passage_id": doc_id,
                "title": ex["document"]["title"],
                "text": text,
            },
        )
        short_answers = sorted({t for sa in ex["annotations"]["short_answers"] for t in sa["text"]})
        queries.append(
            {
                "query_id": f"nqq-{ex['id']}",
                "question": ex["question"]["text"],
                "answer_texts": short_answers,
                "gold": {
                    "type": "span",
                    "doc_id": doc_id,
                    "start_token": start,
                    "end_token": end,
                },
                "gold_text": " ".join(clean[start:end]),
            }
        )
    if len(queries) < N_QUERIES:
        raise SystemExit(f"nq: only {len(queries)} qualifying queries, need {N_QUERIES}")
    meta = _base_meta(
        "nq-dev-300",
        NQ_HF_ID,
        NQ_CONFIG,
        NQ_REVISION,
        "stream dev/validation in dataset order; accept examples with a >=2/5-annotator "
        "long answer that remaps to a non-empty clean-token span of <= "
        f"{MAX_GOLD_SPAN_TOKENS} tokens; stop at 300",
        {"n_docs": len(docs_by_id), "n_seen": n_seen, "skip_counts": skip},
    )
    _write_outputs(
        REPO_ROOT / "data/corpora/nq-dev-300/raw",
        queries,
        list(docs_by_id.values()),
        meta,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", choices=["musique", "nq", "all"], default="all")
    parser.add_argument(
        "--peek",
        action="store_true",
        help="print the first raw example's structure and exit",
    )
    args = parser.parse_args()
    if args.corpus in ("musique", "all"):
        download_musique(peek=args.peek)
    if args.corpus in ("nq", "all"):
        download_nq(peek=args.peek)


if __name__ == "__main__":
    main()
