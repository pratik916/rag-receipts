#!/usr/bin/env python
"""Render gold-to-chunk alignment samples for human review (Spike 0 acceptance gate).

Offline: reads only data/corpora/ produced by scripts/download_data.py.
Run from repo root:
  uv run --project api python scripts/handcheck_alignment.py
Writes data/handcheck/{corpus}.md and prints a one-line JSON summary per corpus.
"""

import argparse
import json
import random
import statistics
from pathlib import Path

from ragreceipts.eval.alignment import Gold, GoldPassage, GoldSpan, is_hit
from ragreceipts.ingest.chunker import ChunkSpan, chunk_passage

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _golds_for(q: dict) -> list[Gold]:
    g = q["gold"]
    if g["type"] == "passage":
        return [GoldPassage(query_id=q["query_id"], passage_id=pid) for pid in g["passage_ids"]]
    return [
        GoldSpan(
            query_id=q["query_id"],
            doc_id=g["doc_id"],
            start_token=g["start_token"],
            end_token=g["end_token"],
        )
    ]


def render_corpus(
    corpus_dir: Path,
    out_path: Path,
    *,
    n_sample: int,
    seed: int,
    chunk_size: int,
    chunk_overlap: int,
) -> dict:
    raw = corpus_dir / "raw"
    queries = _load_jsonl(raw / "queries.jsonl")
    docs = {d["doc_id"]: d for d in _load_jsonl(raw / "docs.jsonl")}
    by_id = {q["query_id"]: q for q in queries}
    slice_full: list[str] = json.loads((raw / "slice-full.json").read_text())

    # corpus-wide gold-span stats (span golds only) - feeds the decisions doc
    span_lens = [
        q["gold"]["end_token"] - q["gold"]["start_token"]
        for q in queries
        if q["gold"]["type"] == "span"
    ]
    rng = random.Random(seed)
    sample_ids = rng.sample(slice_full, n_sample)

    lines = [
        f"# Hand-check: {corpus_dir.name}",
        f"chunk_size={chunk_size} chunk_overlap={chunk_overlap} seed={seed} n_sample={n_sample}",
        "",
    ]
    if span_lens:
        lines.append(
            f"Gold span tokens over all {len(span_lens)} queries: "
            f"min={min(span_lens)} median={int(statistics.median(span_lens))} "
            f"max={max(span_lens)}"
        )
    n_golds = n_golds_hit = n_queries_ok = 0
    for query_id in sample_ids:
        q = by_id[query_id]
        golds = _golds_for(q)
        answer = q.get("answer") or ", ".join(q.get("answer_texts", [])) or "(none)"
        lines += ["", f"## {query_id}", f"**Q:** {q['question']}", f"**Gold answer:** {answer}"]
        if q["gold"]["type"] == "passage":
            doc_ids = list(q["gold"]["passage_ids"])
            non_gold = sorted(set(docs) - set(doc_ids))
            doc_ids += rng.sample(non_gold, min(2, len(non_gold)))  # distractors
        else:
            g = q["gold"]
            lines += [
                f"**Gold span:** doc={g['doc_id']} tokens "
                f"[{g['start_token']}, {g['end_token']}) "
                f"len={g['end_token'] - g['start_token']}",
                f"**Gold text:** {q['gold_text'][:600]}",
            ]
            doc_ids = [g["doc_id"]]
        spans: list[ChunkSpan] = []
        for doc_id in doc_ids:
            d = docs[doc_id]
            doc_spans = chunk_passage(
                corpus_id=corpus_dir.name,
                doc_id=d["doc_id"],
                passage_id=d["passage_id"],
                text=d["text"],
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            spans.extend(doc_spans)
            lines.append(f"\n**Doc {doc_id}** ({d['title']!r}, {len(doc_spans)} chunks):")
            for s in doc_spans:
                mark = "HIT " if any(is_hit(s, g) for g in golds) else "miss"
                lines.append(
                    f"- [{mark}] {s.chunk.chunk_id} tokens "
                    f"[{s.start_token},{s.end_token}) - {s.chunk.text[:240]}"
                )
        hit_flags = [any(is_hit(s, g) for s in spans) for g in golds]
        n_golds += len(golds)
        n_golds_hit += sum(hit_flags)
        if all(hit_flags):
            n_queries_ok += 1
            lines.append("\nALIGNMENT OK - every gold has at least one hitting chunk.")
        else:
            lines.append("\n**NO HIT for some gold - INVESTIGATE BEFORE SIGN-OFF.**")
    summary = {
        "corpus": corpus_dir.name,
        "queries_sampled": n_sample,
        "golds": n_golds,
        "golds_hit": n_golds_hit,
        "queries_all_golds_hit": n_queries_ok,
        "span_len_max": max(span_lens) if span_lens else None,
    }
    lines += ["", "---", f"Summary: {json.dumps(summary)}"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--chunk-overlap", type=int, default=64)
    args = parser.parse_args()
    for name in ("musique-dev-300", "nq-dev-300"):
        corpus_dir = REPO_ROOT / "data" / "corpora" / name
        out_path = REPO_ROOT / "data" / "handcheck" / f"{name}.md"
        summary = render_corpus(
            corpus_dir,
            out_path,
            n_sample=args.n,
            seed=args.seed,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
        print(json.dumps(summary))
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
