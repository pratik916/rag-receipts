from types import SimpleNamespace

from pydantic import BaseModel

from ragreceipts.vendors.anthropic_client import AnthropicClient


class _Shape(BaseModel):
    answer: str


class _StubMessages:
    def __init__(self):
        self.create_kwargs = None
        self.parse_kwargs = None

    def create(self, **kwargs):
        self.create_kwargs = kwargs
        return SimpleNamespace(
            content=[
                SimpleNamespace(type="text", text="hello "),
                SimpleNamespace(type="thinking", thinking="x"),
                SimpleNamespace(type="text", text="world"),
            ],
            usage=SimpleNamespace(input_tokens=12, output_tokens=7),
        )

    def parse(self, **kwargs):
        self.parse_kwargs = kwargs
        return SimpleNamespace(
            parsed_output=kwargs["output_format"](answer="42"),
            usage=SimpleNamespace(input_tokens=20, output_tokens=9),
        )


def make_client():
    stub = SimpleNamespace(messages=_StubMessages())
    return AnthropicClient(client=stub), stub


def test_complete_maps_text_and_usage():
    client, stub = make_client()
    res = client.complete(model="claude-sonnet-4-6", system="sys", user="hi", max_tokens=4096)
    assert res.text == "hello world"  # text blocks joined, others skipped
    assert (res.input_tokens, res.output_tokens) == (12, 7)
    kw = stub.messages.create_kwargs
    assert kw["model"] == "claude-sonnet-4-6"
    assert kw["system"] == "sys"
    assert kw["messages"] == [{"role": "user", "content": "hi"}]
    assert kw["max_tokens"] == 4096
    assert kw["temperature"] == 0.0  # default per ClaudeTransport contract


def test_parse_returns_parsed_output_and_usage():
    client, stub = make_client()
    res = client.parse(
        model="claude-haiku-4-5-20251001",
        system="sys",
        user="hi",
        max_tokens=1024,
        output_format=_Shape,
    )
    assert res.parsed == _Shape(answer="42")
    assert (res.input_tokens, res.output_tokens) == (20, 9)
    assert stub.messages.parse_kwargs["output_format"] is _Shape
    assert stub.messages.parse_kwargs["messages"] == [{"role": "user", "content": "hi"}]
