# First keyed eval run (manual - NEVER CI)

CI runs offline with zero keys. This runbook is the one human-driven path
that spends real money. Budget for the full flow below: under $5 of tracked
spend, plus untracked RAGAS judge spend (see steps 4-5: the hard cap
EXCLUDES judge spend).

## 0. Prerequisites
- Spike 0's download script has materialized the raw slices:
  `data/corpora/<corpus_id>/raw/{queries.jsonl,docs.jsonl,slice-full.json,
  slice-smoke.json,download_meta.json}` (from the repo root:
  `uv run --project api python scripts/download_data.py --corpus all`).
- Plan A ingest completed for at least one corpus (e.g. `musique-dev-300`):
  `uv run ragreceipts ingest --corpus musique-dev-300` wrote
  `data/corpora/<corpus_id>/{manifest.json,chunks.jsonl,sparse/}` plus the
  Qdrant collection.
- `ANTHROPIC_API_KEY`, `VOYAGE_API_KEY`, `COHERE_API_KEY` exported (or in
  `api/.env`, loaded into the shell). The CLI names any missing key.
- All commands below run from `api/`; the data dir resolves via
  RAGRECEIPTS_DATA_DIR (default `../data` - R6).

## 1. Verify the composition seams (no spend)
uv run pytest tests/test_plan_a_seams.py tests/test_cli.py \
  tests/test_harness_selftest.py -q

This covers Spike 0's alignment API, Plan A's FakeRerank scores mode, and
the R9-pinned composition root `cli.py::_build_core_real(config, corpus_id,
data_dir)` via its offline construction test - the upstream names are
pinned by the seam resolutions, so there is no signature discovery step.

## 2. Verify rerank pricing (no spend)
Open the Cohere billing dashboard and confirm rerank-v4.0-pro is billed at
$0.0025 per search unit (cohere.com/pricing does not publish it; our value
is corroborated by reseller listings - see eval/pricing.py docstring). If it
differs: update PRICING, bump PRICING_VERSION to today's date, update the
pricing tests, commit, and only then continue.

## 3. Smoke slice first (~$0.90 estimated)
uv run ragreceipts eval --corpus musique-dev-300 --slice smoke \
  --presets bm25-only,dense-rrf,contextual,rerank --spend-cap-usd 2.50

Review the printed estimate (15 queries x 4 presets ~= $0.90), confirm with
`y`. Expect: 4 receipt cells; `router-on` only appears if you listed it, and
is then SKIPPED with the "requires Plan C" reason. Inspect
`data/receipts-local/<run_id>.json`: every envelope carries the fixed
nondeterminism_note; receipts carry prompts_version "n/a"; anchors carry
their caveat notes (on an nq-dev-300 run they additionally end with the
corpus-scale caveat); n_failed / n_abstained are visible; usd_per_query is
plausible (~$0.015).

## 4. RAGAS spot-check
Re-run step 3 with `--ragas` on the smoke slice. First run downloads
BAAI/bge-small-en-v1.5 (~130MB, local; no extra key). The printed estimate
grows by the per-query judge heuristic (~$0.02/query x 15 queries x 4
presets ~= $1.17 -> total ~= $2.07). IMPORTANT: the HARD SPEND CAP EXCLUDES
judge spend - RAGAS judge calls are not token-metered in Plan B, so only
the synthesis/embed/rerank spend counts against the cap; the omission is
disclosed per receipt via the ragas_judge_usd_untracked flag. Confirm
ragas_faithfulness / ragas_answer_relevancy are populated and that flag
appears in per_query flags.

## 5. Headline run (full slice, ~300 queries)
uv run ragreceipts eval --corpus musique-dev-300 --slice full \
  --presets bm25-only,dense-rrf,contextual,rerank --ragas --spend-cap-usd 25

The printed estimate (~$41 with --ragas) includes the judge heuristic, but
the hard cap meters only tracked spend (~$18 estimated) - budget the
untracked judge spend separately. If the cap aborts the run mid-way, re-run
with the printed `--run-id` and a higher cap - completed queries are never
re-billed.

## 6. Promote and commit the headline receipts
uv run ragreceipts receipts promote <run_id>
git -C .. add receipts/<run_id>.json
git -C .. commit -m "docs(receipts): first committed headline receipts (<run_id>)" \
  -m "Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"

Promotion strips passage text AND model answers - committed per-query
records are IDs + metrics only (benchmark redistribution terms). The
default --receipts-dir is ../receipts (R6), matching the data-dir default
so this command works from api/.
