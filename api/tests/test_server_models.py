"""API models: validation rules the endpoints rely on."""

import pytest
from pydantic import ValidationError

from ragreceipts.server import models as m


def test_query_request_rejects_unknown_preset():
    with pytest.raises(ValidationError, match="unknown preset"):
        m.QueryRequest(query="q", corpus_id="c1", preset="not-a-preset")


def test_query_request_accepts_every_contract_preset():
    for preset in ["bm25-only", "dense-rrf", "contextual", "rerank", "router-on"]:
        req = m.QueryRequest(query="q", corpus_id="c1", preset=preset)
        assert req.preset == preset


def test_corpus_id_must_be_a_safe_slug():
    with pytest.raises(ValidationError, match="corpus_id"):
        m.QueryRequest(query="q", corpus_id="../etc", preset="rerank")


def test_eval_run_request_defaults():
    req = m.EvalRunRequest(corpus_id="c1", preset="rerank")
    assert req.slice == "smoke"
    assert req.confirm is False
    assert req.spend_cap_usd == 5.0


def test_eval_run_request_rejects_unknown_slice():
    with pytest.raises(ValidationError):
        m.EvalRunRequest(corpus_id="c1", preset="rerank", slice="huge")
