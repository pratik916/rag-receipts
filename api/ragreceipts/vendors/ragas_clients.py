"""Raw vendor client constructor for the RAGAS judge (transport seam).

Contracts rule: application code never imports `anthropic` outside vendors/.
ragas's llm_factory needs the raw SDK client (not our ClaudeTransport), so the
one place that constructs it lives here. anthropic.Anthropic() reads
ANTHROPIC_API_KEY from the environment.
"""

from __future__ import annotations


def make_anthropic_client() -> object:
    import anthropic

    return anthropic.Anthropic()
