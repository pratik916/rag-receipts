"""BYO ingest: reader dispatch, oversized-doc split, per-file failure collection,
multipart endpoint + job."""

import json
import time

from fastapi.testclient import TestClient

from ragreceipts.server.app import create_app
from ragreceipts.server.ingest_byo import (
    DOC_TOKEN_LIMIT,
    approx_token_count,
    load_documents,
    split_oversized,
)
from tests.helpers_server import make_test_deps


def test_approx_token_count_is_conservative():
    assert approx_token_count("abcd" * 100) == 100  # 4 chars/token heuristic


def test_split_oversized_discloses_parts():
    text = ("para. " * 200 + "\n\n") * 500  # well above DOC_TOKEN_LIMIT
    docs = split_oversized("big-doc", text, source_file="big.txt")
    assert len(docs) > 1
    assert all(d.n_splits == len(docs) for d in docs)
    assert [d.split_index for d in docs] == list(range(len(docs)))
    assert all(approx_token_count(d.text) <= DOC_TOKEN_LIMIT for d in docs)
    assert "".join(d.text for d in docs).replace("\n\n", "") == text.replace("\n\n", "")


def test_small_doc_is_not_split():
    docs = split_oversized("small", "hello world", source_file="s.txt")
    assert len(docs) == 1 and docs[0].n_splits == 1


def test_load_documents_collects_failures_never_batch_fatal(tmp_path):
    good = tmp_path / "good.txt"
    good.write_text("plain text content")
    md = tmp_path / "notes.md"
    md.write_text("# Title\n\nbody text")
    missing = tmp_path / "missing.pdf"  # never created -> reader raises
    docs, failures = load_documents([good, md, missing])
    assert {d.source_file for d in docs} == {"good.txt", "notes.md"}
    assert len(failures) == 1 and failures[0].file == "missing.pdf"


def test_unsupported_extension_is_a_failure_not_a_crash(tmp_path):
    weird = tmp_path / "data.xyz"
    weird.write_text("???")
    docs, failures = load_documents([weird])
    assert docs == [] and "unsupported" in failures[0].error


class RecordingSink:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def write_corpus(self, *, corpus_id, docs, emit):
        self.calls.append({"corpus_id": corpus_id, "n_docs": len(docs)})
        emit("indexed", 0.9)
        return {
            "corpus_id": corpus_id,
            "dataset": {"name": "byo"},
            "chunking": {"chunk_size": 512, "chunk_overlap": 64},
            "embed_model": "voyage-context-3",
            "index_hashes": {"sparse": "sha256:test"},
            "tokenizer_artifact": "test",
            "n_docs": len(docs),
            "n_chunks": len(docs),
            "n_queries": 0,
            "created_at": "2026-06-10T00:00:00+00:00",
        }


def wait_succeeded(client, job_id, timeout=15.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/jobs/{job_id}").json()
        if body["status"] in ("succeeded", "failed"):
            return body
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_ingest_endpoint_runs_job_and_writes_manifest_with_disclosures(tmp_path):
    deps = make_test_deps(tmp_path, configured=True)
    sink = RecordingSink()
    deps.ingest_sink = sink
    app = create_app(deps_factory=lambda: deps)
    with TestClient(app) as client:
        r = client.post(
            "/corpora/ingest",
            data={"corpus_id": "my-docs"},
            files=[
                ("files", ("a.txt", b"alpha document text", "text/plain")),
                ("files", ("b.md", b"# beta\n\nbody", "text/markdown")),
            ],
        )
        assert r.status_code == 200, r.text
        job = wait_succeeded(client, r.json()["job_id"])
        assert job["status"] == "succeeded"
        listed = client.get("/corpora").json()["corpora"]
        assert any(c["corpus_id"] == "my-docs" for c in listed)
    manifest = json.loads((deps.paths.corpora_dir / "my-docs" / "manifest.json").read_text())
    assert manifest["byo"]["source_files"] == ["a.txt", "b.md"]
    assert manifest["byo"]["failures"] == []
    assert sink.calls[0]["n_docs"] == 2


def test_ingest_endpoint_rejects_when_sink_unavailable(tmp_path):
    deps = make_test_deps(tmp_path, configured=False)
    with TestClient(create_app(deps_factory=lambda: deps)) as client:
        r = client.post(
            "/corpora/ingest",
            data={"corpus_id": "my-docs"},
            files=[("files", ("a.txt", b"x", "text/plain"))],
        )
    assert r.status_code == 503
    assert "VOYAGE_API_KEY" in r.json()["detail"]
