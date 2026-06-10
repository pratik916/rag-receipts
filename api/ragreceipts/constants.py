"""Model and vendor constants.

Binding values from docs/superpowers/plans/2026-06-10-contracts.md.
"""

ROUTER_MODEL = "claude-haiku-4-5-20251001"  # routing + CRAG grading, temperature=0
SYNTH_MODEL = "claude-sonnet-4-6"  # answer synthesis
JUDGE_MODEL = "claude-sonnet-4-6"  # RAGAS judge
EMBED_MODEL = "voyage-context-3"  # contextualized chunk embeddings
RERANK_MODEL = "rerank-v4.0-pro"  # Cohere Rerank v4.0 Pro (anchor variant)
RAGAS_EMBED_MODEL = (
    "BAAI/bge-small-en-v1.5"  # local sentence-transformers for RAGAS answer-relevancy
)
ROUTE_CONFIDENCE_THRESHOLD = 0.7  # below this, escalate to System-2
S2_MAX_HOPS = 3
S2_TOKEN_CEILING = 50_000  # input+output summed across all Claude calls per query


# --- Phase 2 (graph mode) constants — binding from
# docs/superpowers/plans/2026-06-11-graph-contracts.md ---
OPENIE_MODEL = "claude-haiku-4-5-20251001"  # OpenIE triple extraction (== ROUTER_MODEL, cheap)
SYNONYM_THRESHOLD = 0.85  # cosine >= this => phrase-phrase synonym edge
PPR_DAMPING = 0.5  # HippoRAG-2 personalization/locality balance
PPR_MAX_ITER = 50
PPR_TOL = 1e-6
GRAPH_BLEND = 0.5  # passage score = blend*ppr + (1-blend)*dense
GRAPH_SEED_TOP_N = 30  # query-relevant seed nodes before PPR
