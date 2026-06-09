# rag-receipts — Design Spec

**Date:** 2026-06-10 · **Status:** approved design, pre-implementation · **Author:** brainstormed with Claude Code

> **Every RAG technique, with receipts.** An adaptive RAG engine that routes each query
> between a fast path and a deliberative agentic loop — and ships with a built-in Ablation
> Lab that proves, with reproduced published deltas, which RAG techniques actually earn
> their latency on a given corpus.

## Why this exists

Thousands of RAG demo repos exist; almost none measure whether their components help.
The differentiator here is honesty-as-a-feature: every pipeline component has a measured
"receipt" — its contribution on labeled data, compared against the published number it
reproduces. This extends the portfolio brand established by the sibling project
faithfulnessbench: *don't just build AI systems; validate the measurement.*

## Research grounding

Design decisions are anchored to a 104-agent deep-research report (2026-06-10) with
25 adversarially verified claims — full report committed at
`docs/research/2026-06-10-deep-research-advanced-rag.json`. The load-bearing anchors:

| Verified finding | Source | Design consequence |
|---|---|---|
| Reranking is the single most impactful component: +12.1pp Recall@5, +17.2pp MRR@3 over hybrid RRF on T2-RAGBench | arXiv 2604.01733 | Cohere rerank stage is core, not optional |
| BM25 beats SOTA dense embeddings on financial docs (0.644 vs 0.587 R@5); MTEB rank ≠ domain performance | arXiv 2604.01733, 2506.12071 | Hybrid sparse+dense+RRF is non-negotiable |
| Contextualization helps but modestly in independent eval (+2-3pp, vs vendor headline 35-67% failure reduction) | arXiv 2604.01733 vs anthropic.com/news/contextual-retrieval | voyage-context-3 included; framed honestly in receipts |
| voyage-context-3 embeds whole docs in one pass, one vector per chunk, drop-in downstream | blog.voyageai.com 2025/07/23 | Contextualizer needs no LLM prefix step in v1 |
| Agentic rewriting alone underperforms plain hybrid fusion (CRAG 0.658 < 0.695) | arXiv 2604.01733 | Agency sits ON TOP of strong retrieval, never replaces it |
| Static pipelines fail multi-step reasoning; adaptive routing (fast path vs agentic loop) is the production norm; agentic loops cost ~10x | arXiv 2501.09136, 2506.10408 | System-1/System-2 router is the centerpiece |
| Graphs are conditional: HippoRAG 2 +7% on multi-hop, but GraphRAG/LightRAG lose badly on simple facts (LightRAG 16.6 vs 61.9 F1 on NQ) at ~2.3x latency | arXiv 2502.14802, 2506.05690 | Graph mode deferred to Phase 2, behind the same Retriever protocol, presented honestly |

Caveats inherited from the research: vendor self-benchmarks (Anthropic, Voyage) are
verified-as-stated but not independently replicated at those magnitudes; several
quantitative anchors come from one financial-domain benchmark and may not transfer;
no claims on eval frameworks (RAGAS vs alternatives) survived verification — RAGAS is
chosen as de-facto standard, not as a verified-best option.

## Decisions log (user-approved 2026-06-10)

1. **Corpus:** labeled benchmark slices — Natural Questions (simple) + MuSiQue and/or
   2WikiMultihopQA (multi-hop) — plus bring-your-own-documents ingestion.
2. **Stack:** Python; LlamaIndex for ingestion/retrieval plumbing; LangGraph for the
   agentic control layer. (Framework-free was offered and declined.)
3. **Vendors:** best-in-class multi-vendor — voyage-context-3 embeddings, Cohere
   Rerank v4.0 (exact model string verified at build time), Claude for routing/
   grading/synthesis/judging. Three keys in one `.env`. (Anthropic-only default was
   offered and declined.)
4. **UI:** full web app — FastAPI backend, Next.js frontend; Playground + Ablation Lab
   + Corpora pages.
5. **Deployment:** local-first via docker compose (api + web + qdrant); public hosting
   is Phase 3.
6. **v1 scope:** retrieval-first, receipts-first (Approach 1). Graph mode designed-for
   but deferred to Phase 2.
7. **Name:** `rag-receipts`.

## Architecture — three planes

### Ingestion plane
loaders (benchmark slices via HF datasets download script → `data/`, gitignored;
BYO PDF/MD/HTML via LlamaIndex readers) → chunker (sentence-window, configurable
size/overlap) → contextualizer (voyage-context-3 contextualized-chunk-embeddings
endpoint, chunks grouped per document, 120K-token window) → index writers:
Qdrant (dense) + bm25s serialized index (sparse), both keyed by `chunk_id`.
Every ingest emits a **corpus manifest** (config + content hashes) so receipts are
traceable to an exact corpus state. BYO ingest runs as a background job streaming
progress to the Corpora page.

### Query plane (FastAPI + LangGraph)
- **route** node: Claude (claude-haiku-4-5-20251001, temperature 0) classifies query
  complexity → `simple` | `complex`, with confidence; recorded in trace.
- **System-1 fast path** (simple): hybrid retrieve (Qdrant + bm25s) → RRF top-50 →
  Cohere rerank → top-5 → Claude (claude-sonnet-4-6) answers with inline `[n]`
  citations; abstains when evidence is thin.
- **System-2 agentic loop** (complex, hard-bounded at 3 hops + per-query token
  ceiling): decompose into ordered sub-queries → per-hop retrieve via the same
  retrieval core (earlier hop answers fill later sub-queries) → CRAG-style grade
  (sufficient / insufficient / contradictory) → refine + re-retrieve while budget
  remains → synthesize across hops with per-hop citations; unresolved sub-queries
  are flagged in the answer, never papered over.
- **Shared retrieval core invariant:** both routes and the eval harness execute the
  identical retrieval code path, parameterized only by `PipelineConfig`. Receipts
  measure exactly what serves queries.
- Every node emits TraceEvents (inputs, outputs, scores, model, tokens, ms) → SQLite
  → Playground trace viewer.

### Eval plane
Ablation runner takes a named preset matrix — `bm25-only` → `+dense+rrf` →
`+contextual` → `+rerank` → `router-on` — × a benchmark slice (~200–300 queries),
flips `PipelineConfig` flags on the same code, and records per-query retrieved chunk
IDs, answer, latency, token cost. Metrics: Recall@5 / MRR@3 vs gold passage IDs;
RAGAS faithfulness + answer relevancy (judge: claude-sonnet-4-6); latency p50/p95;
$ per query. Output: versioned `receipts.json` — per entry: config, metrics,
per-query records, failure disclosure, and a `published_anchor` {source, delta, note}
linking the literature number it reproduces. Headline runs are committed; the
Ablation Lab renders committed and local runs side by side.

## Repo layout & component boundaries

```
rag-receipts/
├── api/                      # Python backend (uv-managed)
│   ├── ragreceipts/
│   │   ├── ingest/           # loaders, chunker, contextualizer, index writers
│   │   ├── retrieval/        # Retriever protocol, dense/sparse impls, RRF, rerank stage
│   │   ├── agents/           # LangGraph graphs: router, system1, system2
│   │   ├── eval/             # ablation runner, metrics, RAGAS adapter, receipts schema
│   │   ├── traces/           # TraceEvent capture + SQLite store
│   │   ├── vendors/          # voyage/cohere/anthropic clients behind protocols
│   │   └── server/           # FastAPI app + background job runner
│   └── tests/
├── web/                      # Next.js (pnpm): Playground · Ablation Lab · Corpora
├── data/                     # downloaded slices + gold labels (gitignored)
├── docs/research/            # committed deep-research report
├── docs/superpowers/specs/   # this document
├── docker-compose.yml        # api + web + qdrant
└── README.md
```

Boundary rules:

- **`PipelineConfig` is the single source of truth** — one dataclass of component
  flags (`bm25`, `dense`, `contextual`, `rerank`, `route_mode`; Phase 2 adds `graph`)
  consumed identically by server and ablation runner. An ablation cell = same code,
  one flag flipped.
- **`retrieval/` knows nothing about agents or HTTP.**
  `Retriever.search(query, k) -> list[ScoredChunk]`; `HybridRRF` composes dense+sparse;
  rerank is a stage. The Phase-2 graph retriever implements the same protocol.
- **`vendors/` is a transport seam** — every network call behind an injectable
  protocol; the whole pipeline unit-tests offline with fakes.
- **`agents/` is pure orchestration** — no retrieval logic inside graph nodes.
- **`web/` computes nothing** — consumes a typed OpenAPI client; backend owns truth.

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
  manifest; voyage calls batched rate-limit-aware.
- Eval: per-query failures disclosed in the receipt ("n=287/300, 13 failed"),
  excluded from metrics, never hidden; runs resumable from SQLite job state; pre-run
  cost estimate + confirmation gate + hard spend cap that aborts mid-run.
- System-2: 3-hop max + token ceiling; exhaustion → answer with explicit unresolved
  caveat.
- Reproducibility: temperature 0 for routing/grading/judging; receipts record model
  IDs, prompt versions, corpus manifest hashes; LLM nondeterminism disclosed.

## Testing

All CI runs offline, zero keys (transport-seam fakes):

- **Golden tests** for RRF fusion, rerank reordering, Recall@5/MRR@3 on hand-computed
  tiny cases. Every `PipelineConfig` flag has a test proving it changes behavior.
- **Agent-graph tests** with fake LLM responses asserting state transitions:
  simple→S1, complex→S2, insufficient→refine loop, budget-exhaustion→caveated
  synthesis.
- **Full-path integration** in CI via canned vendor fixtures; a 5-query live smoke
  script exists for manual/nightly use, never CI.
- **Frontend e2e** (Playwright): query renders trace; Ablation Lab renders committed
  receipts.
- **Harness self-test (the on-brand one):** a tiny labeled corpus constructed so
  flipping the rerank flag provably changes Recall@5; if the ablation runner stops
  detecting a change it must detect, CI fails. Receipts that can't fail aren't
  receipts.

## Phase roadmap

- **Phase 1 (v1, this spec):** full retrieval stack + router + Playground + Ablation
  Lab + BYO ingest + docker compose + README with demo GIF.
- **Phase 2 (v1.1):** HippoRAG-2-style graph retriever (Claude OpenIE triples →
  KG with passage nodes → Personalized PageRank → recognition-memory filtering;
  ref arXiv 2502.14802) in the existing Retriever slot + router third route +
  the headline "when do graphs help" receipt (wins multi-hop, loses simple facts,
  costs latency — presented honestly).
- **Phase 3:** hosted public demo (rate limiting, server-side key custody, spend
  caps, pre-warmed demo corpus).
- **Phase 4 (stretch):** ColPali-style visual retrieval receipt; long-context-vs-RAG
  receipt; Claude-prefix-vs-voyage-context-3 receipt (genuinely open per research).

## Non-goals (v1)

- No graph retrieval (Phase 2). No web-search fallback in System-2. No multi-tenancy,
  auth, or hosting concerns (Phase 3). No fine-tuned models. No support for corpora
  larger than a laptop + Qdrant container can hold. No framework-free rewrite.

## Open questions deferred to implementation planning

- Exact HF dataset versions/splits and slice sizes (target ~200–300 queries each,
  balancing signal vs eval cost; KILT vs original NQ format for gold passages).
- Chunk size/overlap defaults (start near 512 tokens; sweep is itself a receipt).
- Cohere rerank exact model string; charting library for Ablation Lab (default
  recharts); OpenAPI client generator (default openapi-typescript).
- Per-run default spend cap value and cost-estimation formula.
