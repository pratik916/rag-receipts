# Demo bootstrap — keyed run (manual, NEVER CI)

One-time human-driven session that produces the **committed** demo-corpus artifacts the
public demo serves from. CI never runs this — the whole pipeline is offline-tested at $0 with
fakes. Budget for the full flow: roughly **$10–15** of tracked spend (corpus is tiny;
`demo/corpus/docs.jsonl` is ~14 KB), plus untracked RAGAS judge spend if you pass `--ragas` in
step 5. The eval ladder in step 5 also produces the committed `receipts/`.

All `ragreceipts` / script steps are resumable: if a keyed eval aborts on the spend cap, re-run
with the printed `--run-id` and a higher `--spend-cap-usd` — completed queries are never re-billed.

## 0. Prerequisites

- Vendor keys exported (or in `api/.env`, loaded into the shell): `VOYAGE_API_KEY`,
  `COHERE_API_KEY`, `ANTHROPIC_API_KEY`. The CLI and `build_deps` name any missing key.
- Qdrant running locally: `docker compose up qdrant -d`, and `QDRANT_URL` exported
  (the server REQUIRES it — never a silent default, R7).
- `demo/corpus/docs.jsonl` committed (Plan H task H1) — this is the demo's source documents.
- `api/` dependencies installed: `cd api && uv sync`.
- The benchmark slices for step 5 materialized (offline, no keys), from the repo root:
  `uv run --project api python scripts/download_data.py --corpus all` — writes
  `data/corpora/{musique-dev-300,nq-dev-300}/raw/`.
- The data dir resolves via `RAGRECEIPTS_DATA_DIR` (default `../data` from `api/` — R6).

```bash
export VOYAGE_API_KEY=...
export COHERE_API_KEY=...
export ANTHROPIC_API_KEY=...
export QDRANT_URL=http://localhost:6333
```

## 1. Verify the offline pipeline first (no spend)

```bash
cd api && uv run pytest -q
```
The full suite runs with fakes and no keys. Green here means every seam this runbook drives
(ingest, graph build, the query runner, demo seeding) is wired before any keyed call.

## 2. Ingest the demo corpus (~$1–2, a few minutes)

The ingest CLI reads `data/corpora/<corpus>/raw/`, so stage the committed docs there first:

```bash
# From the repo root
mkdir -p data/corpora/demo/raw
cp demo/corpus/docs.jsonl data/corpora/demo/raw/

cd api
uv run ragreceipts ingest --corpus demo
```
This writes `data/corpora/demo/{chunks.jsonl,sparse/,manifest.json}` and populates the Qdrant
`demo` collection (named vectors `contextual` / `isolated`). See `first-keyed-run.md` for the
`--chunk-size` / `--chunk-overlap` knobs if you want to deviate from the defaults pinned in the
manifest.

## 3. Build the graph index (~$2–4, a few minutes)

```bash
cd api
uv run python scripts/build_graph.py --corpus demo
```
This runs real OpenIE (Haiku) + Voyage embeddings over `chunks.jsonl`, writes
`data/corpora/demo/graph/`, and updates the manifest's `index_hashes["graph"]` (G6). See
`graph-mode-run.md` for the artifact details and how to confirm it loads.

## 4. Export the committed corpus artifacts

Copy the ingest + graph outputs into the committed `demo/corpus/` location the demo serves from:

```bash
# From the repo root
cp    data/corpora/demo/chunks.jsonl   demo/corpus/
cp -r data/corpora/demo/sparse/        demo/corpus/
cp -r data/corpora/demo/graph/         demo/corpus/
cp    data/corpora/demo/manifest.json  demo/corpus/
```

Dense vectors live only in Qdrant after ingest, so export them to a numpy archive that
`server/demo.py::seed_demo_qdrant` reloads on startup (it reads `data["contextual"]` /
`data["isolated"]`):

```bash
cd api
uv run python scripts/export_demo_vectors.py \
    --corpus-id demo \
    --output ../demo/corpus/dense_vectors.npz
```
Script: `api/scripts/export_demo_vectors.py`. Expect it to print the exported vector count
(matching `manifest.json`'s `n_chunks`).

## 5. Capture the example traces

Run three representative queries and save each full response + trace to a JSON the
`/demo/examples` endpoint loads via `DemoExampleItem(**data)`:

```bash
cd api
uv run python scripts/capture_demo_examples.py \
    --corpus-id demo \
    --output-dir ../demo/examples/
```
Script: `api/scripts/capture_demo_examples.py`. It writes `example_s1.json` (router → System-1,
simple fact), `example_s2.json` (router → System-2, multi-hop), and `example_graph.json`
(`graph-rrf`, graph multi-hop). The script needs all three keys + `QDRANT_URL` (it builds the
real query runner via `build_deps`); it sets `DEMO_MODE=0` so no budget ledger limits the capture.

## 6. Run the eval ladder and promote the receipts (keyed)

Generate the committed receipts on both benchmark corpora. Multi-hop and simple-fact corpora gate
the preset ladder differently — `router-on` / graph presets are **skipped with a disclosed reason**
on a single-hop corpus, never faked.

Multi-hop (musique — carries the graph + router presets and the HippoRAG-2 anchor):
```bash
cd api
uv run ragreceipts eval --corpus musique-dev-300 --slice full \
    --presets bm25-only,dense-rrf,contextual,rerank,graph,graph-rrf,router-on \
    --spend-cap-usd 20 --yes
uv run ragreceipts receipts promote <musique_run_id>
```

Simple-fact (nq — the two-sided "graphs NOT expected to help" side; carries the corpus-scale note):
```bash
cd api
uv run ragreceipts eval --corpus nq-dev-300 --slice full \
    --presets bm25-only,dense-rrf,contextual,rerank,router-on \
    --spend-cap-usd 20 --yes
uv run ragreceipts receipts promote <nq_run_id>
```
The estimate → confirm gate: omit `--yes` to review the printed cost estimate and confirm
interactively; the hard `--spend-cap-usd` aborts (resumably) before any query that would exceed it.
Add `--ragas` for answer-quality metrics (extra, *untracked* judge spend — disclosed per receipt
via `ragas_judge_usd_untracked`, never folded into the hard cap). `receipts promote` strips passage
text and model answers, leaving committed receipts as IDs + metrics only (benchmark redistribution
terms); its `--receipts-dir` defaults to `../receipts` (R6). Review the anchor notes before committing.

## 7. Commit all artifacts

```bash
# From the repo root
git add demo/corpus/ demo/examples/ receipts/
git commit -m "chore(demo): bootstrap corpus artifacts + example traces + receipts"
```
`data/` is gitignored — only `demo/corpus/`, `demo/examples/`, and the stripped `receipts/` are
committed. On startup, `seed_demo_qdrant` reseeds the Qdrant `demo` collection from
`demo/corpus/dense_vectors.npz` + `chunks.jsonl` (idempotent: a no-op if the collection is already
populated, or if `dense_vectors.npz` is absent).
