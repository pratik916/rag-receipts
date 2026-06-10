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

## Outcomes

Pending — filled at the end of the spike (Task 10 of the plan) after the human
review gate, with: actual download counts and skip statistics, gold-span length
stats, hand-check verdicts, and surprises for Plans A/B.
