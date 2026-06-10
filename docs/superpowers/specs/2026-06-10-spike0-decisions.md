# Spike 0 Decisions — Gold-to-Chunk Alignment

**Date:** 2026-06-10 · **Status:** draft until the human review gate passes;
the Outcomes section is filled at the end of the spike.
Plan: `docs/superpowers/plans/2026-06-10-spike0-gold-alignment.md`

## D1 — Natural Questions source: original NQ, not KILT-NQ

Options evaluated (both verified against the HF datasets-server on 2026-06-10):

| | KILT-NQ (`facebook/kilt_tasks`, config `nq`) | Original NQ (`google-research-datasets/natural_questions`, config `dev`) |
|---|---|---|
| Validation size | 2,837 | 7,830 |
| Gold format | `provenance` = `wikipedia_id` + `start_paragraph_id`/`end_paragraph_id` + `bleu_score` into the KILT 2019/08 Wikipedia snapshot | 5 annotators' `long_answer` token spans + `short_answers`, natively indexed into the same record's `document.tokens` |
| Corpus dependency | requires `facebook/kilt_wikipedia` (~37 GB, 5.9M pages) to materialize passage text | none — each record carries the full document |
| Gold provenance quality | re-mapped from original NQ via BLEU matching (`bleu_score` exposes mapping confidence < 1.0) — derived, not native | native annotator labels |

**Decision: original NQ** (`google-research-datasets/natural_questions`, config `dev`,
split `validation`, parquet, streamed). Rationale:
1. Self-contained: corpus and gold come from the same record; no 37 GB knowledge-source
   dependency for a laptop-scale project.
2. Native annotator golds — no BLEU-remap noise layered under our alignment rule.
3. It exercises the spec's span-overlap hit rule ("for span-format golds (NQ long
   answers), hit iff the chunk covers ≥50% of the gold span's tokens"). KILT's
   paragraph-id golds would leave that contract-defined rule as dead code.
4. The `dev` config is parquet-only and streams row-group-wise, so we download only
   the prefix we read (~hundreds of MB), not 3.5 GB.

Disclosed consequence: the corpus is the union of the selected queries' own Wikipedia
pages (~300 pages, content-deduplicated). Retrieval difficulty is "find the right
chunk among ~300 pages", not open-Wikipedia retrieval. Every receipt on this corpus
must carry that scale caveat in its `published_anchor.note` (Plan B).

## D2 — Pinned dataset revisions

| Corpus | HF id | Config/split | Revision (commit sha) | Verified |
|---|---|---|---|---|
| musique-dev-300 | `dgslibisey/MuSiQue` | `default` / `validation` (2,417 rows) | `c8f4f8c9465fb69d31a8eae894c3fd509c4ca321` | 2026-06-10 |
| nq-dev-300 | `google-research-datasets/natural_questions` | `dev` / `validation` (7,830 rows) | `e8103d566bef4154c2c12b17c6095ec5275840cc` | 2026-06-10 |

`dgslibisey/MuSiQue` is a community mirror of the official **musique_ans v1.0** JSONL
(official distribution is Zenodo via `github.com/stonybrooknlp/musique`; the mirror's
files are named `musique_ans_v1.0_{train,dev}.jsonl` and its schema matches the
official format field-for-field). Pinning the commit sha makes the mirror tamper-evident.

## D3 — Gold formats and hit rules

- **MuSiQue (passage golds):** each example carries 20 paragraphs
  `{idx, title, paragraph_text, is_supporting}`; gold = the `is_supporting` paragraphs
  (2–4 per question), cross-checked against `question_decomposition[*].paragraph_support_idx`
  (mismatch ⇒ example skipped and counted). Paragraph ids are content-addressed
  (`mu-` + sha1(title\ntext)[:16]) so identical paragraphs shared across examples
  deduplicate to one corpus passage. Hit rule: `chunk.passage_id == gold.passage_id`.
- **NQ (span golds):** gold = the majority long answer over 5 dev annotators
  (rule: require ≥2 non-null; pick the most frequent `(start_token, end_token)` span;
  ties broken by smallest start then end). Original token indices include HTML tokens;
  they are remapped to clean-token space (D5). Hit rule: chunk covers ≥50% of the gold
  span's tokens (integer form: `2*overlap >= gold_len`), same `doc_id` required.

## D4 — Slice definitions (deterministic)

- **musique-dev-300:** sort dev examples by `id`, shuffle with `random.Random(42)`,
  take the first 300 that pass the support-set cross-check. Smoke = first 15 of the
  slice order. Corpus = union of all 20 paragraphs of each selected example, deduped.
- **nq-dev-300:** stream `dev`/`validation` in dataset order; accept examples with a
  ≥2/5-annotator long answer that remaps to a non-empty clean span of ≤1024 tokens;
  stop at 300. Smoke = first 15 accepted. Corpus = content-deduped pages of the 300.
- Slice membership is written to `slice-full.json` / `slice-smoke.json` per corpus;
  the generating logic lives in `scripts/download_data.py` with seed 42.

## D5 — Normalization rules that affect alignment

1. HTML tokens (`is_html == True`) are dropped; each surviving original token maps to
   a `(start, end)` range in clean-token space so gold spans can be remapped exactly.
2. Tokens containing internal whitespace are split into parts so the invariant
   `" ".join(clean_tokens).split() == clean_tokens` holds — whitespace-token indices
   are therefore stable across (text ↔ token list) round trips. This invariant is what
   lets `Chunk.text` and span token indices coexist without storing offsets on `Chunk`.
3. `MAX_GOLD_SPAN_TOKENS = 1024`: with `chunk_size=512`, a chunk covers at most 512
   tokens, so the ≥50% rule is mathematically unsatisfiable for golds longer than
   1024 tokens. Such examples (typically giant tables) are excluded at download time
   and counted in `download_meta.json` — disclosed, never silently dropped.
4. NQ doc ids are content-addressed (`nq-` + sha1(text)[:16]) so the same Wikipedia
   page appearing under multiple queries becomes one corpus document; span golds from
   different queries against the same content share that doc and stay valid.

## Outcomes (recorded 2026-06-10, after the review gate)

### Download results (source: data/corpora/*/raw/download_meta.json)
- musique-dev-300: n_queries=300, n_docs=4679, n_skipped_support_mismatch=0,
  datasets_lib_version=4.8.5 (resolved from the `>=4.8.4,<5` pin).
- nq-dev-300: n_queries=300, n_docs=300, n_seen=571,
  skip_counts: no_majority_long_answer=254, empty_doc=0, unmappable_span=0,
  gold_too_long=17. datasets_lib_version=4.8.5.

### Gold span stats (source: header of data/handcheck/nq-dev-300.md)
- NQ gold span tokens over all 300 queries: min=4, median=107, max=752
  (cap MAX_GOLD_SPAN_TOKENS=1024 enforced at download; 17 examples excluded by it).
- MuSiQue golds are passage-format, so they have no span length (span_len_max=null).

### Hand-check verdicts (source: Task 8 JSON summaries + the review gate)
- musique: queries_sampled=20, golds=51, golds_hit=51, queries_all_golds_hit=20/20.
- nq: queries_sampled=20, golds=20, golds_hit=20, queries_all_golds_hit=20/20.
- Per-miss explanations: none — 0 "NO HIT / INVESTIGATE" markers in either file.
- Reviewer: the project owner (godofcode.pratik) was away during the gate and
  explicitly delegated review authority to the Claude session on 2026-06-10
  ("please take best decisions yourself"). Two independent reviews were performed
  in their stead: (a) the controller read all 20 MuSiQue blocks + a sample of NQ
  blocks against the Task 9 checklist; (b) an independent reviewer subagent
  recomputed every span-overlap verdict in Python (not trusting the rendered
  marks), checked passage-id collisions (zero: no distractor doc-id ever equals a
  gold doc-id), and reconstructed the straddle arithmetic. Verdict: **APPROVED** —
  the alignment coordinate system is trustworthy to build Recall@5/MRR@3 on; no
  off-by-one, no HTML-remap drift, gold-span slices match the printed gold text
  token-for-token. The owner can ratify or reverse this on return.
- Reviewability aid added at the gate (commit 196a6a9, follow-on to the Task 8
  harness at 079416f): for span golds the hand-check now prints, under each chunk
  overlapping the gold token range, the exact chunk-text slice covering the span —
  so a human can visually confirm the gold text physically lands inside the [HIT]
  chunk. The [HIT]/[miss] mark stays positional and never reads that text.

### Surprises & follow-ups for Plans A/B
1. **A single gold span can be hit by TWO chunks** (the chunk overlap is 64 tokens,
   so a span straddling a boundary can clear the ≥50% bar in both neighbors). Live
   example: NQ `nqq-5900977074377897432` (Digital Revolution), gold len 297 —
   chunk0 overlap 174 and chunk1 overlap 187 both satisfy `2·overlap ≥ 297`, two
   HITs. **Plan B's Recall@k/MRR@3 must dedup hits per query (count "was the gold
   found in top-k", not "how many chunks hit")** or recall will be overcounted.
2. **NQ acceptance rate ≈ 53%** (300 accepted / 571 streamed). The dominant skip is
   `no_majority_long_answer=254` (≥2-of-5 annotators must agree on a non-null long
   answer). Plan A's NQ loader should expect to stream ~1.9× the target count.
3. **`gold_too_long=17`**: 17 remapped clean spans exceeded 1024 tokens (giant
   tables/infoboxes) and were excluded — disclosed, never silently dropped.
   `empty_doc=0` and `unmappable_span=0` over 571 rows is a strong signal the HTML
   strip + span remap are robust (no all-HTML gold spans surfaced).
4. **MuSiQue paragraphs ARE shared across examples**: 300×20=6000 raw paragraphs
   deduped to 4679 unique passages (~22% collapse) via content-addressed ids. Corpus
   scale is "find the chunk among ~4.7k passages", not open-Wikipedia. Every receipt
   on this corpus must carry that scale caveat (already noted under D1 for NQ;
   applies to MuSiQue too). `n_skipped_support_mismatch=0`: every one of the first
   300 shuffled examples passed the is_supporting ↔ decomposition cross-check.
5. **NQ corpus = exactly 300 docs for 300 queries**: in this slice no two accepted
   queries shared a Wikipedia page, so content-dedup found no collisions among the
   accepted set (the dedup machinery is still load-bearing — it just had nothing to
   merge here).
6. **MuSiQue gold-label noise inherited**: query `2hop__29191_59955` ("country
   between Lithuania and Poland", gold answer Kaliningrad Oblast) lists "Paris" among
   its gold supporting passages — a MuSiQue multi-hop annotation artifact. Our
   passage-id rule faithfully HITs it; this is dataset noise, not an alignment bug,
   and future receipts inherit MuSiQue's gold-label quality.
7. **CARRY-FORWARD CODE GOTCHA (flagged for Plan A/B wiring):** `nq.py::nq_records`
   reads a FLAT `example["question_text"]`/`short_answers` shape, but LIVE NQ rows
   nest `question["text"]` and carry list-of-dict `short_answers`. The download
   script sidesteps this by composing the low-level helpers (`strip_html_tokens`,
   `remap_span`, `select_long_answer`, `nq_doc_id`) directly rather than calling
   `nq_records`. When Plan A wires `nq_records` against live rows, reconcile its
   accessors to the nested live shape first (and add a test on a nested fixture).
8. Two ugliest examples that still pass cleanly: (a) NQ atmosphere-gases
   (`nqq--8762144002773697012`) — the gold long answer is a composition TABLE
   ("Nitrogen 780,840 78.084 Oxygen O 209,460 20.946 …") that survives HTML strip
   as whitespace tokens and remaps exactly; (b) the Digital Revolution double-hit in
   (1). Both pass because the rules are purely positional and table-tolerant up to
   the 1024-token cap.
