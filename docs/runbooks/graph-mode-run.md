# Graph mode — keyed run (manual, NEVER CI)

CI builds and tests the entire graph plane at $0 with FakeOpenIE/FakeEmbed/FakeClaude
over the tiny fixture corpus (see `api/tests/test_graph_*.py`). This runbook is the one
human-driven path that spends real money to produce the REAL graph artifact (and, in the
Plan F half, the real "when do graphs help" receipt). Budget for the build: ~$10-30 for
~5k passages (Haiku OpenIE, one structured-output call per chunk) plus Voyage embeddings.

## 0. Prerequisites
- A corpus has been ingested (Plan A): `ragreceipts ingest --corpus musique-dev-300`
  wrote `data/corpora/<id>/{manifest.json,chunks.jsonl,sparse/}` and the Qdrant collection.
- `ANTHROPIC_API_KEY` and `VOYAGE_API_KEY` exported (or in `api/.env`, loaded into the shell).
  `COHERE_API_KEY` is NOT needed for the build (graph mode does not rerank).
- All commands run from `api/`; the data dir resolves via `RAGRECEIPTS_DATA_DIR`
  (default `../data` — R6).

## 1. Verify the offline graph plane first (no spend)
```bash
uv run pytest tests/test_openie.py tests/test_graph_ppr.py tests/test_graph_index.py \
  tests/test_graph_retriever.py tests/test_manifest_graph.py -q
```
This proves OpenIE mapping, deterministic PPR, the reproducible artifact hash, the
retriever (both recognition modes + failure paths), and the manifest hash — all with
fakes, before any keyed call.

## 2. Build the graph artifact (~$10-30 for musique-dev-300)
```bash
ANTHROPIC_API_KEY=... VOYAGE_API_KEY=... \
  uv run python scripts/build_graph.py --corpus musique-dev-300
```
The script:
1. loads `chunks.jsonl`,
2. runs `OpenIEExtractor` (Haiku, one `messages.parse` per chunk) -> triples,
3. embeds phrases + passages with Voyage (`voyage-context-3`),
4. builds phrase + passage nodes (passages first in chunk order, phrases sorted),
   relation / appears-in / synonym (cosine >= 0.85) edges,
5. writes `data/corpora/<id>/graph/{nodes.jsonl,edges.jsonl,phrase_vectors.npy,
   passage_vectors.npy,passage_map.json}`,
6. updates the manifest's `index_hashes["graph"]` (G6).

Expect printed counts (passage/phrase nodes, edges, triples) and the final
`index_hashes['graph'] = sha256:...`. The artifact is deterministic given the same
triples + embeddings, so re-running over an unchanged corpus reproduces the hash
(modulo LLM nondeterminism in OpenIE — the artifact hash pins exactly what was built,
which is what receipts cite).

## 3. Confirm the artifact loads and the retriever runs (no further spend)
```bash
uv run python -c "
from pathlib import Path
from ragreceipts.ingest.graph_index import GraphIndex
idx = GraphIndex.load(Path('../data/corpora/musique-dev-300/graph'))
print('nodes', len(idx.nodes), 'passages', idx.n_passage, 'phrases', idx.n_phrase)
"
```
Expect node counts matching the build output. The `graph` / `graph-rrf` presets can now
be composed by a graph-aware composition root (Plan F wires `GraphRetriever` into
`_build_core_real` and runs the receipt; see the Plan F section appended below once it lands).

## 4. Rebuild after a corpus change
The graph is rebuild-only (no incremental updates, per the Phase-2 non-goals). Re-ingest,
then re-run step 2; the new artifact + new manifest hash supersede the old one. Receipts
always cite the manifest hash, so a stale graph can never be silently attributed.

## 5. Run the two-sided "when do graphs help" receipt (keyed)

The receipt is two-sided **by construction**: the multi-hop side measures the lift; the
simple-fact side measures the (expected) non-help. Both are real, signed deltas vs the
`rerank` baseline — never a magnitude claim, only a direction-match against the HippoRAG-2
anchor (`published_value=0.07`, `baseline_preset="rerank"`).

Multi-hop side (the lift):
```bash
uv run ragreceipts eval --corpus musique-dev-300 --slice full \
    --presets rerank,graph,graph-rrf --spend-cap-usd 30 --yes
```

Simple-fact side (the non-help — the headline insight, not a footnote):
```bash
uv run ragreceipts eval --corpus nq-dev-300 --slice full \
    --presets rerank,graph,graph-rrf --spend-cap-usd 30 --yes
```

The `musique` cells carry the HippoRAG-2 anchor (direction-match only); the `nq` cells
carry the "graphs NOT expected to help simple-fact" note plus the nq corpus-scale caveat,
and show the (likely negative) delta honestly. Latency overhead is measured, not claimed,
in `latency_p50_ms` / `latency_p95_ms`. **Honest USD accounting (RG8):** a graph preset's
`usd_per_query` is **synthesis-only** — the LLM recognition-memory Haiku call has no trace
hook, so its spend is out-of-band and is disclosed as such, never folded into the metered
per-query cost.

The harness self-test (`api/tests/test_harness_selftest_graph.py`, CI-enforced) guarantees
this receipt **can fail**: on the fixture graph the `graph` cell's Recall@5 provably moves
vs a deliberately weak `rerank` cell (1.0 vs 0.0), and a misaligned-gold graph run scores
0.0 — the alignment rule is load-bearing for the graph path too.

## 6. Recognition mini-ablation (keyed, optional)

Does LLM recognition memory add lift over embedding-only seeding?
```bash
uv run ragreceipts eval --corpus musique-dev-300 --slice full \
    --presets graph --graph-recognition embedding --spend-cap-usd 30 --yes --run-id graph-emb
uv run ragreceipts eval --corpus musique-dev-300 --slice full \
    --presets graph --graph-recognition llm --spend-cap-usd 30 --yes --run-id graph-llm
```
Compare the two `graph` cells' Recall@5 and `usd_per_query`: the lift (if any) from LLM
recognition, with cost included. The `llm` mode adds one Haiku recognition call per query
(out-of-band spend, per RG8); `embedding` mode makes no LLM recognition call. The
`--graph-recognition` override flows into the cell's `QueryConfig`, so the receipt's pinned
config records which mode actually ran.

## 7. Router graph-route accuracy (keyed, optional)

```bash
uv run ragreceipts eval --corpus musique-dev-300 --slice full \
    --presets router-on --spend-cap-usd 30 --yes
```
The router-on receipt reports `route_counts` (`n_s1` / `n_s2` / `n_graph`) and the
`graph_route_precision` diagnostic: of the queries the router sent to `graph`, the fraction
whose graph-route top-k contained a gold hit. This is kept **SEPARATE** from the headline
retrieval metrics — "the router chose graph" is never conflated with "graph helps." The
graph route is reachable only on a `MULTI_HOP_DATASETS` corpus (the explicit corpus gate in
the runner, RG6); an off-corpus `graph` decision falls back to `s1`.

## 8. Promote (review first, then commit)

```bash
uv run ragreceipts receipts promote <run_id>
```
Copies the run to `receipts/` stripped to IDs + metrics only — never passage text, never
model answers (benchmark redistribution terms). Review the anchor notes and the two-sided
framing before `git add`. Bump `PRICING_VERSION` first if the live Cohere/Voyage/Anthropic
prices differ from the table (see `first-keyed-run.md`).
