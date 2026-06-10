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

<!-- Plan F appends: producing the two-sided "when do graphs help" receipt (graph/graph-rrf
     presets over musique + nq), the recognition mini-ablation, and the live router-on
     graph-route run. -->
