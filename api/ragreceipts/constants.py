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
