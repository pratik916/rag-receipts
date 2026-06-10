"""Contextualizer: builds BOTH dense vector sets on every ingest (decision #8).

Direct EmbedTransport calls — NOT LlamaIndex's per-node embedding path, which would
silently degrade doc-grouping to single-chunk documents (spec ingestion plane).
Contextual = doc-grouped call; isolated = the same chunks as single-chunk documents
(vendors/base.py contract). Outputs are flattened to global chunk order."""

from ragreceipts.vendors.base import EmbedTransport


def embed_corpus(
    doc_chunk_texts: list[list[str]], embed: EmbedTransport
) -> tuple[list[list[float]], list[list[float]]]:
    contextual_nested = embed.embed_documents(doc_chunk_texts)
    isolated_nested = embed.embed_documents([[text] for doc in doc_chunk_texts for text in doc])
    contextual = [vec for doc in contextual_nested for vec in doc]
    isolated = [doc[0] for doc in isolated_nested]
    if len(contextual) != len(isolated):
        raise RuntimeError(
            f"contextual/isolated vector counts diverged: {len(contextual)} vs {len(isolated)}"
        )
    return contextual, isolated
