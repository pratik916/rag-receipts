"""G3: the graph/graph-rrf anchors are the honest two-sided receipt's first half.

The anchor's note MUST disclose (a) the soft independent replication and (b) that
graphs are not expected to help simple-fact (nq) queries — these are verbatim
obligations from the contracts addendum. build_anchor still computes the
direction-match and machine-appends the nq corpus-scale caveat (R11)."""

from ragreceipts.eval.receipts import ANCHOR_SPECS, build_anchor


def test_graph_anchor_spec_values_match_contracts():
    specs = ANCHOR_SPECS["graph"]
    assert len(specs) == 1
    spec = specs[0]
    assert spec.source == "HippoRAG 2 (arXiv 2502.14802)"
    assert spec.metric == "recall_at_5"
    assert spec.published_value == 0.07
    assert spec.baseline_preset == "rerank"


def test_graph_anchor_note_is_two_sided_and_discloses_soft_replication():
    note = ANCHOR_SPECS["graph"][0].note
    assert "independent" in note  # the soft replication is disclosed
    assert "slight" in note
    assert "Direction-match only" in note
    assert "never magnitude reproduction" in note
    # the second side of the receipt: graphs not expected to help simple facts
    assert "NOT" in note and "nq" in note


def test_graph_rrf_anchor_present_with_same_caveat():
    specs = ANCHOR_SPECS["graph-rrf"]
    assert len(specs) == 1
    spec = specs[0]
    assert spec.source == "HippoRAG 2 (arXiv 2502.14802)"
    assert spec.metric == "recall_at_5"
    assert spec.baseline_preset == "rerank"
    assert spec.note == ANCHOR_SPECS["graph"][0].note  # same verbatim caveat


def test_build_anchor_direction_match_and_nq_caveat():
    spec = ANCHOR_SPECS["graph"][0]
    # positive measured delta vs positive published -> direction match
    anchor = build_anchor(spec, 0.04, corpus_id="musique-dev-300")
    assert anchor.direction_match is True
    assert anchor.measured_value == 0.04
    assert "corpus-scale caveat" not in anchor.note  # musique: no nq append
    # negative measured delta (graphs hurt) vs positive published -> mismatch,
    # honestly recorded — and the nq run appends the scale caveat (R11)
    nq_anchor = build_anchor(spec, -0.05, corpus_id="nq-dev-300")
    assert nq_anchor.direction_match is False
    assert nq_anchor.measured_value == -0.05
    assert "corpus-scale caveat" in nq_anchor.note
