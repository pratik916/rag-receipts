"""Offline vendor fakes (contracts: FakeEmbed, FakeRerank, FakeClaude).

Constructor shapes are FINAL per seam resolution R5 — Plans B/C consume them as-is
(no constructor migration exists anywhere).

FakeEmbed vectors are sha256-derived unit vectors: deterministic everywhere, no model
downloads. Doc-grouped embeddings mix in a document-level component (0.8*chunk + 0.2*doc,
renormalized) so contextual vectors provably differ from isolated ones — which is what
lets ingest tests assert dense_contextual != dense_isolated.
"""

import hashlib
import math

from ragreceipts.eval.ragas_adapter import RagasScores
from ragreceipts.traces.models import TraceEvent
from ragreceipts.types import Chunk, ScoredChunk
from ragreceipts.vendors.base import ClaudeResult, ParsedResult, Triple, VendorUnavailable


def _unit_vector(text: str, dim: int) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    raw = [digest[i % len(digest)] / 255.0 - 0.5 for i in range(dim)]
    norm = math.sqrt(sum(x * x for x in raw)) or 1.0
    return [x / norm for x in raw]


def _renormalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


class FakeEmbed:
    """Deterministic EmbedTransport.

    query_aliases maps a query string to the text it should embed as — the test's lever
    for making dense retrieval favor a chunk with zero lexical overlap with the query.
    """

    def __init__(
        self,
        dim: int = 8,
        fail_query: bool = False,
        fail_documents: bool = False,
        query_aliases: dict[str, str] | None = None,
    ):
        self.dim = dim
        self.fail_query = fail_query
        self.fail_documents = fail_documents
        self.query_aliases = query_aliases or {}
        self.document_calls: list[list[list[str]]] = []
        self.query_calls: list[str] = []

    def embed_documents(self, documents: list[list[str]]) -> list[list[list[float]]]:
        if self.fail_documents:
            raise VendorUnavailable("FakeEmbed scripted document failure")
        self.document_calls.append(documents)
        out: list[list[list[float]]] = []
        for doc in documents:
            doc_vec = _unit_vector("||".join(doc), self.dim)
            chunk_vecs = []
            for chunk in doc:
                chunk_vec = _unit_vector(chunk, self.dim)
                mixed = [0.8 * c + 0.2 * d for c, d in zip(chunk_vec, doc_vec)]
                chunk_vecs.append(_renormalize(mixed))
            out.append(chunk_vecs)
        return out

    def embed_query(self, query: str) -> list[float]:
        if self.fail_query:
            raise VendorUnavailable("FakeEmbed scripted query failure")
        self.query_calls.append(query)
        return _unit_vector(self.query_aliases.get(query, query), self.dim)


class FakeRerank:
    """Scripted RerankTransport (final R5 shape). Modes, in precedence order:

    - scores: dict keyed by candidate TEXT -> returns (original_index,
      scores.get(text, 0.0)) sorted desc, ties by index (Plan B's harness fixture mode);
    - script: dict keyed by QUERY -> explicit ordering of original indices, best first;
    - default: reversed candidate order (provably different from RRF order, which is
      what the rerank flag-flip test needs).
    """

    def __init__(
        self,
        script: dict[str, list[int]] | None = None,
        scores: dict[str, float] | None = None,
        fail: bool = False,
    ):
        self.script = script or {}
        self.scores = scores
        self.fail = fail
        self.calls: list[tuple[str, list[str], int]] = []

    def rerank(self, query: str, texts: list[str], top_n: int) -> list[tuple[int, float]]:
        if self.fail:
            raise VendorUnavailable("FakeRerank scripted failure")
        self.calls.append((query, list(texts), top_n))
        if self.scores is not None:
            pairs = [(i, float(self.scores.get(text, 0.0))) for i, text in enumerate(texts)]
            pairs.sort(key=lambda p: (-p[1], p[0]))
            return pairs[:top_n]
        order = self.script.get(query, list(reversed(range(len(texts)))))
        return [(idx, 1.0 - 0.01 * pos) for pos, idx in enumerate(order)][:top_n]


class FakeClaude:
    """Scripted ClaudeTransport (final R5 shape): ONE ordered script consumed across
    both complete() and parse(). Script items:

    - str                            -> ClaudeResult(text=item) from complete()
    - any other object (e.g. a Pydantic instance) -> ParsedResult(parsed=item) from parse()
    - (item, input_tokens, output_tokens) tuple   -> same, with explicit token counts

    AssertionError when the script runs dry or the popped item's kind does not match
    the method called (an under- or mis-scripted test must fail loudly, not hang).
    """

    DEFAULT_INPUT_TOKENS = 10
    DEFAULT_OUTPUT_TOKENS = 5

    def __init__(self, script: list | None = None):
        self.script = list(script or [])
        self.complete_calls: list[dict] = []
        self.parse_calls: list[dict] = []
        self.calls: list[dict] = []

    def _pop(self, caller: str) -> tuple[object, int, int]:
        if not self.script:
            raise AssertionError(f"FakeClaude.{caller} called with empty script")
        item = self.script.pop(0)
        if isinstance(item, tuple):
            payload, input_tokens, output_tokens = item
            return payload, input_tokens, output_tokens
        return item, self.DEFAULT_INPUT_TOKENS, self.DEFAULT_OUTPUT_TOKENS

    def complete(
        self, *, model: str, system: str, user: str, max_tokens: int, temperature: float = 0.0
    ) -> ClaudeResult:
        self.calls.append({"method": "complete", "model": model, "system": system, "user": user})
        self.complete_calls.append(
            {
                "model": model,
                "system": system,
                "user": user,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
        )
        payload, input_tokens, output_tokens = self._pop("complete")
        if not isinstance(payload, str):
            raise AssertionError(
                f"FakeClaude.complete expected a str script item, got {type(payload).__name__}"
            )
        return ClaudeResult(text=payload, input_tokens=input_tokens, output_tokens=output_tokens)

    def parse(
        self,
        *,
        model: str,
        system: str,
        user: str,
        max_tokens: int,
        output_format: type,
        temperature: float = 0.0,
    ) -> ParsedResult:
        self.calls.append(
            {
                "method": "parse",
                "output_format": output_format.__name__,
                "model": model,
                "system": system,
                "user": user,
            }
        )
        self.parse_calls.append(
            {
                "model": model,
                "system": system,
                "user": user,
                "max_tokens": max_tokens,
                "output_format": output_format,
                "temperature": temperature,
            }
        )
        payload, input_tokens, output_tokens = self._pop("parse")
        if isinstance(payload, str):
            raise AssertionError("FakeClaude.parse expected a parsed-object script item, got str")
        return ParsedResult(parsed=payload, input_tokens=input_tokens, output_tokens=output_tokens)


class FakeOpenIE:
    """Scripted OpenIETransport (FINAL R5 shape; Plan F consumes it as-is).

    script: dict keyed by passage TEXT -> list[Triple]; an unknown passage yields [].
    fail=True raises VendorUnavailable on extract() (the graph-build degrade path).
    Output length/order always mirror the input passages (OpenIETransport contract).
    """

    def __init__(self, script: dict[str, list[Triple]] | None = None, fail: bool = False):
        self.script = script or {}
        self.fail = fail
        self.calls: list[list[str]] = []

    def extract(self, passages: list[str]) -> list[list[Triple]]:
        if self.fail:
            raise VendorUnavailable("FakeOpenIE scripted failure")
        self.calls.append(list(passages))
        return [list(self.script.get(p, [])) for p in passages]


# --- Plan B: RAGAS judge fake -------------------------------------------------


class FakeRagas:
    """Scripted RagasJudge for CI: returns queued scores in call order."""

    def __init__(self, scores: list[RagasScores]) -> None:
        self._scores = list(scores)
        self.calls: list[dict] = []

    def score(self, *, question: str, answer: str, contexts: list[str]) -> RagasScores:
        self.calls.append({"question": question, "answer": answer, "contexts": contexts})
        return self._scores.pop(0)


# --- Plan C: agent-graph test doubles ----------------------------------------


def make_chunk(i: int, *, doc: str = "d1", text: str | None = None) -> ScoredChunk:
    """Tiny ScoredChunk fixture; chunk_id f'{doc}:{i}', passage_id == doc.

    start_token/end_token (R3) are consecutive positional ranges so span math
    stays valid: chunk i covers tokens [i*n, (i+1)*n) of its parent passage.
    """
    body = text or f"passage text {i}"
    n = len(body.split())
    return ScoredChunk(
        chunk=Chunk(
            chunk_id=f"{doc}:{i}",
            corpus_id="test",
            doc_id=doc,
            passage_id=doc,
            text=body,
            position=i,
            start_token=i * n,
            end_token=(i + 1) * n,
        ),
        score=1.0 / (i + 1),
        source="rrf",
    )


class FakeCore:
    """Duck-types RetrievalCore.retrieve for agent-graph tests (no Qdrant, no keys).

    by_query maps exact query text -> scripted results; anything else returns the
    two-chunk default corpus. Records every query for transition assertions.
    """

    def __init__(
        self,
        by_query: dict[str, list[ScoredChunk]] | None = None,
        default: list[ScoredChunk] | None = None,
    ):
        self.by_query = by_query or {}
        self.default = default if default is not None else [make_chunk(0), make_chunk(1)]
        self.queries: list[str] = []

    def retrieve(self, query: str) -> list[ScoredChunk]:
        self.queries.append(query)
        return self.by_query.get(query, list(self.default))


# --- Plan D: server-layer fakes ----------------------------------------------


class InMemoryTraceStore:
    """Duck-typed TraceStore (append/get per contracts §Traces) for server tests and
    TESTING mode. Single-process only — which is exactly the single-worker constraint."""

    def __init__(self) -> None:
        self._events: dict[str, list[TraceEvent]] = {}

    def append(self, event: TraceEvent) -> None:
        self._events.setdefault(event.trace_id, []).append(event)

    def get(self, trace_id: str) -> list[TraceEvent]:
        return sorted(self._events.get(trace_id, []), key=lambda e: e.seq)


class ScriptedTransport:
    """ClaudeTransport fake with cycling scripts (never exhausts across e2e runs).

    parse() validates the scripted payload into the *requested* output_format, so it
    stays correct even though Plan C owns the route/grade Pydantic models.
    """

    def __init__(self, completions: list[str], parse_payloads: list[dict]) -> None:
        self._completions = completions
        self._parse_payloads = parse_payloads
        self._c = 0
        self._p = 0

    def complete(
        self,
        *,
        model: str,
        system: str,
        user: str,
        max_tokens: int,
        temperature: float = 0.0,
    ) -> ClaudeResult:
        text = self._completions[self._c % len(self._completions)]
        self._c += 1
        return ClaudeResult(text=text, input_tokens=120, output_tokens=40)

    def parse(
        self,
        *,
        model: str,
        system: str,
        user: str,
        max_tokens: int,
        output_format: type,
        temperature: float = 0.0,
    ) -> ParsedResult:
        payload = self._parse_payloads[self._p % len(self._parse_payloads)]
        self._p += 1
        return ParsedResult(
            parsed=output_format.model_validate(payload),
            input_tokens=80,
            output_tokens=20,
        )
