# rag-receipts — Design Spec

**Date:** 2026-06-10 · **Status:** approved design, adversarially reviewed, pre-implementation
**Revision note:** v2 — incorporates 22 findings from a five-lens adversarial review panel
(placeholders/ambiguity, internal consistency, technical feasibility, scope realism,
research fidelity).

> **Every RAG technique, with receipts.** An adaptive RAG engine that routes each query
> between a fast path and a deliberative agentic loop — and ships with a built-in Ablation
> Lab that measures each component's contribution on labeled data, compared honestly
> against published anchors.

## Why this exists

Thousands of RAG demo repos exist; almost none measure whether their components help.
The differentiator here is honesty-as-a-feature: every pipeline component has a measured
"receipt" — its contribution on labeled data, displayed alongside the published number it
is compared against (never conflated with it; see `published_anchor`). This extends the
portfolio brand established by the sibling project faithfulnessbench: *don't just build
AI systems; validate the measurement.*

## Research grounding

Design decisions are anchored to a 104-agent deep-research report (2026-06-10) with
25 adversarially verified claims — full report committed at
`docs/research/2026-06-10-deep-research-advanced-rag.json`. The load-bearing anchors:

| Finding | Source | Design consequence |
|---|---|---|
| Reranking is the single most impactful component: +12.1pp Recall@5, +17.2pp MRR@3 over hybrid RRF on T2-RAGBench (model: Cohere Rerank v4.0 **Pro**) | arXiv 2604.01733 | Cohere rerank stage is core, not optional |
| BM25 beats SOTA dense embeddings on financial docs (0.644 vs 0.587 R@5); MTEB rank ≠ domain performance | arXiv 2604.01733, 2506.12071 | Hybrid sparse+dense+RRF is non-negotiable |
| Contextualization helps but modestly in the only independent eval (+2–3pp for **LLM-prefix-style** contextual retrieval, vs vendor headline 35–67% failure reduction) | arXiv 2604.01733 vs anthropic.com/news/contextual-retrieval | voyage-context-3 included; its receipt must disclose the technique mismatch (see eval plane) |
| voyage-context-3 embeds whole docs in one pass, one vector per chunk, drop-in downstream | blog.voyageai.com 2025/07/23 | Contextualizer needs no LLM prefix step in v1 |
| Agentic rewriting alone underperforms plain hybrid fusion (CRAG 0.658 < 0.695) | arXiv 2604.01733 | Agency sits ON TOP of strong retrieval, never replaces it |
| Static pipelines fail multi-step reasoning; adaptive routing is the emerging pattern. (The "~10x agentic cost / routing is the production norm" framing is a practitioner caveat carried by the report's verification notes — IBM/practitioner sources — not a survey-verified number.) | arXiv 2501.09136, 2506.10408 + practitioner caveat | System-1/System-2 router is the centerpiece |
| Graphs are conditional: HippoRAG 2 +7% on multi-hop (author-reported, ICML 2025; an independent enterprise replication found only slight gains), while GraphRAG/LightRAG lose badly on simple facts (LightRAG 16.6 vs 61.9 F1 on NQ). Latency overhead is real but the "~2.3x average" is the report's softest number (2-1 vote, method-dependent; LightRAG itself ~10–20x) | arXiv 2502.14802, 2506.05690 | Graph mode deferred to Phase 2, behind the same Retriever protocol, presented honestly |

Caveats inherited from the research: vendor self-benchmarks (Anthropic, Voyage) are
verified-as-stated but not independently replicated at those magnitudes; several
quantitative anchors come from one financial-domain benchmark study (arXiv 2604.01733,
a single **non-peer-reviewed Apr 2026 preprint**) and may not transfer to our corpora;
the graph latency/accuracy magnitude figures are directionally unanimous but soft in
exact values; no claims on eval frameworks survived verification — RAGAS is chosen as
de-facto standard, not as a verified-best option.

## Decisions log (user-approved 2026-06-10; v2 amendments marked)

1. **Corpus:** labeled benchmark slices — Natural Questions (simple) + **MuSiQue
   (primary multi-hop; 2WikiMultihopQA moved to stretch)** — plus
   bring-your-own-documents ingestion (kept in v1 at user's choice; built last,
   designated first cut if v1 drags).
2. **Stack:** Python; LlamaIndex for readers/chunking and retrieval plumbing; LangGraph
   for the agentic control layer. (Framework-free was offered and declined.)
3. **Vendors:** best-in-class multi-vendor — voyage-context-3 embeddings, **Cohere
   Rerank v4.0 Pro** (the variant benchmarked in arXiv 2604.01733; `rerank-v4.0-fast`
   available as config option), Claude for routing/grading/synthesis/judging. Three
   keys in one `.env`. (Anthropic-only default was offered and declined.)
4. **UI:** full web app — FastAPI backend, Next.js frontend; Playground + Ablation Lab
   + Corpora pages.
5. **Deployment:** local-first via docker compose (api + web + qdrant); public hosting
   is Phase 3.
6. **v1 scope:** retrieval-first, receipts-first (Approach 1). Graph mode designed-for
   but deferred to Phase 2.
7. **Name:** `rag-receipts`.
8. **(v2) Non-contextual dense baseline:** `contextual=off` means the *same model*
   (voyage-context-3) embedding each chunk as an isolated single-chunk document. This
   holds the embedder constant so the `+contextual` receipt measures document-context
   value, not model differences.

## Architecture — three planes

### Ingestion plane

loaders (benchmark slices via HF datasets download script → `data/`, gitignored;
BYO PDF/MD/HTML/TXT via LlamaIndex readers) → chunker (sentence-window, configurable
size/overlap; **every chunk carries its parent passage/document ID in metadata** —
this powers metric alignment) → contextualizer → index writers.

- **Contextualizer is a direct Voyage SDK call** (`/v1/contextualizedembeddings`,
  chunks grouped per document) — NOT routed through LlamaIndex's generic per-node
  embedding path, which would silently degrade grouping to single-chunk documents.
  Batching is by token count (the 120K-token window is also the per-request budget;
  caps: 1,000 docs / 16K chunks / 32K tokens-per-chunk per request). BYO documents
  exceeding 120K tokens are split into multiple logical documents at ingest,
  disclosed in the manifest.
- **Ingest builds BOTH dense vector sets upfront** — contextualized and isolated
  (decision #8) — stored as Qdrant named vectors on the same points, plus the bm25s
  sparse index (serialized with its tokenizer config; rebuilt on every ingest —
  bm25s has no incremental indexing; fine at laptop scale).
- Every ingest emits a **corpus manifest** (chunking config, embedder versions,
  content hashes — one hash per index variant) so receipts are traceable to exact
  corpus state. BYO ingest runs as a background job streaming progress to the
  Corpora page.

### Query plane (FastAPI + LangGraph)

- **route** node: Claude (claude-haiku-4-5-20251001, temperature 0) classifies query
  complexity → `simple` | `complex` with confidence. **Confidence is consumed, not
  decorative:** below a configurable threshold (default 0.7), escalate to System-2
  (conservative default). `route_mode ∈ {auto, force_s1, force_s2}`; all
  pre-router ablation presets run `force_s1`.
- **System-1 fast path:** hybrid retrieve (Qdrant + bm25s) → RRF top-50 → Cohere
  rerank → top-5 → Claude (claude-sonnet-4-6) answers with inline `[n]` citations.
  **Abstention is model-decided via prompt and surfaced as a structured
  `abstained: true` field** (not prose); eval reports abstention rate separately and
  excludes abstentions from RAGAS scoring with disclosed counts.
- **System-2 agentic loop** (hard-bounded: 3 hops max + per-query token ceiling,
  default 50K tokens summed input+output across all Claude calls, configurable):
  decompose into ordered sub-queries → per-hop retrieve via the same retrieval core →
  CRAG-style grade per sub-query: `sufficient` → proceed; `insufficient` → refine +
  re-retrieve while budget remains; **`contradictory` → one re-retrieve attempt; if
  still contradictory, synthesize citing both sources with an explicit contradiction
  flag in answer and trace** → synthesize across hops with per-hop citations;
  unresolved sub-queries flagged, never papered over.
- **Shared retrieval core invariant:** both routes and the eval harness execute the
  identical retrieval code, parameterized only by `PipelineConfig`.
- Every node emits TraceEvents (inputs, outputs, scores, model, tokens, ms) → SQLite
  (WAL mode) → Playground trace viewer.

### Eval plane

- **PipelineConfig splits into two flag sets.** Query-time flags (`bm25`, `dense`,
  `rerank`, `route_mode`) flip behavior on the same index. Ingest-time flags
  (`contextual`, chunking params) select among **pre-built index variants**; the
  ablation runner resolves each config to its variant and records the variant's
  manifest hash in the receipt. The invariant is "same code" — the `+contextual`
  cell is labeled in the UI as a cross-index comparison.
- Preset ladder: `bm25-only` → `+dense+rrf` (isolated vectors) → `+contextual`
  (contextualized vectors) → `+rerank` → `router-on`, × a benchmark slice
  (~200–300 queries). A first-class **`smoke` slice (15 queries per corpus)** exists
  so runner changes validate in minutes; full slices are for headline runs. The
  `router-on` cell runs on the multi-hop slice only (System-2 over simple NQ queries
  buys nothing).
- **Retrieval metrics (defined precisely; non-routed cells):** a retrieved chunk is a
  *hit* for a gold passage if its parent passage ID matches; for span-format golds
  (NQ long answers), a chunk is a hit if it covers ≥50% of the gold span's tokens.
  **Recall@5** = fraction of gold passages with ≥1 hit in top-5. **MRR@3** =
  reciprocal rank of the first hit within top-3, 0 if none.
- **`router-on` cell metrics:** retrieval recall is ill-defined across decomposed
  hops, so primary metrics are **answer-level** — EM/F1 vs gold answers + RAGAS —
  with union-of-hops retrieval recall reported as a secondary diagnostic, disclosed
  as such in the receipt.
- **Generation metrics:** RAGAS faithfulness + answer relevancy, judge =
  claude-sonnet-4-6 via RAGAS **v0.4 collections API** (pinned; direct Anthropic
  `llm_factory` support — no LangChain wrapper). Answer-relevancy additionally needs
  an embeddings model: a **local sentence-transformers model** (default
  `BAAI/bge-small-en-v1.5`) — zero extra keys, works offline in CI.
- **Cost metrics:** $ per query = traced token counts × a **versioned pricing table**
  whose as-of date is recorded in every receipt; latency p50/p95.
- **`receipts.json` schema:** per entry — config, index-variant manifest hashes,
  metrics, per-query records, failure + abstention disclosure, and
  `published_anchor {source, published_value, measured_value, direction_match, note}`.
  `note` is **required** and carries the comparability caveats machine-readably —
  e.g. the `+contextual` anchor must state that the independent +2–3pp figure is for
  LLM-prefix contextualization while ours is voyage-context-3 (different technique;
  voyage's own deltas are vendor self-benchmarks); cross-domain anchors (T2-RAGBench
  is financial; our corpora are NQ/MuSiQue) claim **direction-match only**, never
  magnitude reproduction. Domain transfer is itself a finding, not a failure.
- Headline runs are committed to **`receipts/` (top-level, read-only at runtime)**;
  local runs live in SQLite and `data/receipts-local/`. Committed per-query records
  must respect benchmark redistribution terms (IDs + metrics, not passage text).
  The Ablation Lab renders committed and local runs side by side.
- Before any run: cost estimate + confirmation gate + hard spend cap aborting mid-run.

## Repo layout & component boundaries

```
rag-receipts/
├── api/                      # Python backend (uv-managed)
│   ├── ragreceipts/
│   │   ├── ingest/           # loaders, chunker, contextualizer, index writers
│   │   ├── retrieval/        # Retriever protocol, dense/sparse impls, RRF, rerank stage
│   │   ├── agents/           # LangGraph graphs: router, system1, system2
│   │   ├── eval/             # ablation runner, metrics, alignment, RAGAS adapter, receipts schema
│   │   ├── traces/           # TraceEvent capture + SQLite store
│   │   ├── vendors/          # voyage/cohere/anthropic clients behind protocols
│   │   └── server/           # FastAPI app + background job runner
│   └── tests/
├── web/                      # Next.js (pnpm): Playground · Ablation Lab · Corpora
├── receipts/                 # committed headline receipts (read-only at runtime)
├── data/                     # downloaded slices, gold labels, local runs (gitignored)
├── docs/research/            # committed deep-research report
├── docs/superpowers/specs/   # this document
├── docker-compose.yml        # api + web + qdrant
└── README.md
```

Boundary rules:

- **`PipelineConfig` is the single source of truth**, with query-time and ingest-time
  flag sets as defined in the eval plane. Same code everywhere; ingest-time flags
  resolve to pre-built index variants.
- **`retrieval/` knows nothing about agents or HTTP.**
  `Retriever.search(query, k) -> list[ScoredChunk]`; `HybridRRF` composes dense+sparse;
  rerank is a stage. The Phase-2 graph retriever implements the same protocol.
  **"Designed-for" means exactly that: the protocol admits a graph implementation —
  no graph flag, enum branch, or stub ships in v1 code.**
- **`vendors/` is a transport seam** — every network call behind an injectable
  protocol; the whole pipeline unit-tests offline with fakes.
- **`agents/` is pure orchestration** — no retrieval logic inside graph nodes.
- **`web/` computes nothing** — consumes a typed OpenAPI client (openapi-typescript
  v7+, FastAPI's OpenAPI 3.1 output).
- **Server runtime constraints (load-bearing):** api runs single-worker uvicorn
  (in-process job state + progress streams); SQLite in WAL mode; ingest/eval jobs run
  in a dedicated worker thread keyed by SQLite job rows — that's what makes
  "resumable from job state" real. No Celery/Redis; compose stays api+web+qdrant
  (named volumes for qdrant storage, SQLite, bm25s indexes, data/; vendor keys go to
  api only; web gets the API base URL; depends_on gates on a qdrant healthcheck).

Model defaults (re-verify IDs against the claude-api skill at build time):
router/grader = claude-haiku-4-5-20251001; synthesis + RAGAS judge =
claude-sonnet-4-6 (opus-tier configurable); all judging/routing at temperature 0.

## Error handling & cost guards

Principle: **degrade visibly, never silently.**

- Vendor clients retry 429/5xx with exponential backoff honoring `retry-after`.
  Reranker down → serve RRF order with `degraded: rerank-skipped` trace flag + UI
  badge. Query-embedding failure → BM25-only fallback, same treatment. Claude
  failure → surfaced error with retry; no fabricated answers.
- Startup healthcheck reports per-vendor capability; missing keys produce a named
  env-var message, not a stack trace.
- Ingestion: per-document failures collected, never batch-fatal; jobs resumable via
  manifest; voyage calls batched by token count, rate-limit-aware.
- Eval: per-query failures disclosed in the receipt ("n=287/300, 13 failed"),
  excluded from metrics, never hidden; abstentions disclosed separately; runs
  resumable from SQLite job state; pre-run cost estimate + confirmation gate + hard
  spend cap.
- System-2: 3-hop max + 50K-token default ceiling; exhaustion → answer with explicit
  unresolved caveat.
- Reproducibility: temperature 0 for routing/grading/judging; receipts record model
  IDs (including the exact rerank model string), prompt versions, pricing-table
  version, corpus manifest hashes; LLM nondeterminism disclosed.

## Testing

All CI runs offline, zero keys (transport-seam fakes):

- **Golden tests** for RRF fusion, rerank reordering, Recall@5/MRR@3 on hand-computed
  tiny cases — **including the chunk→gold alignment rule** (parent-ID match and
  ≥50%-span-overlap paths). Every `PipelineConfig` flag has a test proving it changes
  behavior.
- **Agent-graph tests** with fake LLM responses asserting state transitions:
  simple→S1, complex→S2, low-confidence→S2 escalation, insufficient→refine loop,
  **contradictory→re-retrieve→flagged synthesis**, budget-exhaustion→caveated
  synthesis, abstention→structured field.
- **Full-path integration** in CI via canned vendor fixtures; a 5-query live smoke
  script for manual/nightly use, never CI.
- **Frontend e2e** (Playwright): query renders trace; Ablation Lab renders committed
  receipts from `receipts/`.
- **Harness self-test (the on-brand one):** a tiny labeled corpus constructed so
  flipping the rerank flag provably changes Recall@5 — and so a deliberately
  misaligned gold mapping provably breaks the alignment tests. If the ablation
  runner stops detecting a change it must detect, CI fails. Receipts that can't fail
  aren't receipts.

## Phase roadmap

- **Spike 0 (time-boxed, before any plan):** gold-to-chunk alignment de-risking —
  pin exact dataset versions/splits (NQ via KILT vs original: decide here), implement
  the alignment function, hand-check it on 20 queries per corpus. This is the
  project's riskiest item; every receipt depends on it.
- **Phase 1 (v1) — four sequential implementation plans, not one:**
  - **Plan A:** ingestion (benchmark slices only) + retrieval core + PipelineConfig +
    golden tests.
  - **Plan B:** eval plane as a CLI producing receipts.json + committed first
    receipts. (The differentiator ships before any UI exists.)
  - **Plan C:** LangGraph router + System-1/System-2 + trace store.
  - **Plan D:** web app (Playground, Ablation Lab, Corpora) + docker compose +
    Playwright e2e + BYO ingest (built last; designated first cut if v1 drags).
- **Phase 2 (v1.1):** HippoRAG-2-style graph retriever in the existing Retriever
  slot — Claude OpenIE triples → KG with passage nodes → query-time triple retrieval
  → recognition-memory filtering → Personalized PageRank (ref arXiv 2502.14802) —
  + router third route + the "when do graphs help" receipt (multi-hop gains
  author-reported +7%, independent replication found slighter gains; loses simple
  facts; latency overhead disclosed per measurement, not per the soft literature
  average).
- **Phase 3:** hosted public demo (rate limiting, server-side key custody, spend
  caps, pre-warmed demo corpus).
- **Phase 4 (stretch):** 2WikiMultihopQA second slice; ColPali-style visual retrieval
  receipt; long-context-vs-RAG receipt; Claude-prefix-vs-voyage-context-3 receipt
  (genuinely open per research — the only independent comparison used a quantized
  Phi-3.5-mini for prefix generation).

## Non-goals (v1)

- No graph retrieval code, flags, or stubs (Phase 2). No web-search fallback in
  System-2. No multi-tenancy, auth, or hosting concerns (Phase 3). No fine-tuned
  models. No corpora beyond laptop + Qdrant-container scale. No framework-free
  rewrite. No incremental sparse indexing (full bm25s rebuild per ingest is
  accepted).

## Open questions deferred to implementation planning

- Chunk size/overlap defaults (start near 512 tokens; sweep is itself a receipt).
- Charting library for Ablation Lab (default recharts).
- Per-run default spend cap value and the pre-run cost-estimation formula.
- Confidence threshold default (0.7) and token-ceiling default (50K) may be tuned
  during Plan C once real traces exist.

(Resolved since v1 of this spec: non-contextual baseline = decision #8; multi-hop
dataset = MuSiQue; rerank variant = v4.0 Pro; RAGAS version/API + embeddings model;
metric definitions; committed-receipts location; dataset-version choice moved to
Spike 0.)
