"""ClaudeTransport over the official `anthropic` SDK.

Binding usage per docs/superpowers/plans/2026-06-10-contracts.md (verified against
the claude-api skill 2026-06-10):
- complete() -> client.messages.create(...)
- parse()    -> client.messages.parse(..., output_format=Model) -> resp.parsed_output
- The SDK auto-retries 429/5xx with exponential backoff honoring retry-after;
  `max_retries` is configurable on the client constructor.
- Typed exceptions (anthropic.RateLimitError, anthropic.APIStatusError) propagate
  after retries are exhausted — spec: Claude failure is surfaced, never fabricated.
- Constructing without a key fails fast with the SDK's message naming
  ANTHROPIC_API_KEY (spec: named env-var message, not a stack trace mystery).
- `anthropic` is imported ONLY here (vendors/ boundary rule).
"""

from __future__ import annotations

import anthropic

from ragreceipts.vendors.base import ClaudeResult, ParsedResult


class AnthropicClient:
    def __init__(
        self,
        api_key: str | None = None,
        max_retries: int = 4,
        client: anthropic.Anthropic | None = None,
    ):
        # `client` injection exists for offline tests only.
        self._client = client or anthropic.Anthropic(api_key=api_key, max_retries=max_retries)

    def complete(
        self, *, model: str, system: str, user: str, max_tokens: int, temperature: float = 0.0
    ) -> ClaudeResult:
        resp = self._client.messages.create(
            model=model,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(block.text for block in resp.content if block.type == "text")
        return ClaudeResult(
            text=text,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
        )

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
        resp = self._client.messages.parse(
            model=model,
            system=system,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": user}],
            output_format=output_format,
        )
        return ParsedResult(
            parsed=resp.parsed_output,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
        )
