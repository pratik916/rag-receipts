"""BYO document ingestion: PDF/MD/HTML/TXT via LlamaIndex readers (verified imports:
https://developers.llamaindex.ai/python/framework-api-reference/readers/file/).

Documents above the voyage-context-3 contextualization window are split into multiple
logical documents at ingest and DISCLOSED in the manifest (spec §Ingestion plane). Token
counts use a conservative 4-chars-per-token heuristic with a 100K limit, keeping real
token counts safely under the 120K window without a vendor tokenizer dependency.
Per-document failures are collected, never batch-fatal.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

APPROX_CHARS_PER_TOKEN = 4
DOC_TOKEN_LIMIT = 100_000  # conservative vs the 120K-token voyage window
SUPPORTED_EXTS = (".pdf", ".md", ".html", ".txt")


def approx_token_count(text: str) -> int:
    return max(1, len(text) // APPROX_CHARS_PER_TOKEN)


@dataclass(frozen=True)
class LoadedDoc:
    doc_id: str
    text: str
    source_file: str
    split_index: int  # 0-based part number within the source document
    n_splits: int  # total parts the source document became (1 = not split)


@dataclass(frozen=True)
class LoadFailure:
    file: str
    error: str


class IngestSink(Protocol):
    """Chunk, embed (both named-vector sets), and index docs; returns the manifest dict
    (contracts §Corpus manifest). Implemented by the Plan A adapter (production) and
    TestingIngestSink (TESTING mode)."""

    def write_corpus(
        self, *, corpus_id: str, docs: list[LoadedDoc], emit: Callable[[str, float], None]
    ) -> dict: ...


def read_file(path: Path) -> str:
    """Dispatch by extension to the verified LlamaIndex readers; returns full text."""
    from llama_index.readers.file import (
        FlatReader,
        HTMLTagReader,
        MarkdownReader,
        PDFReader,
    )

    ext = path.suffix.lower()
    if ext == ".pdf":
        docs = PDFReader(return_full_document=True).load_data(path)
    elif ext == ".md":
        docs = MarkdownReader().load_data(str(path))  # docs: load_data(file: str)
    elif ext == ".html":
        docs = HTMLTagReader(tag="body").load_data(path)  # default tag is <section>
    elif ext == ".txt":
        docs = FlatReader().load_data(path)
    else:
        raise ValueError(f"unsupported extension: {ext} (supported: {SUPPORTED_EXTS})")
    return "\n\n".join(d.text for d in docs)


def split_oversized(doc_id: str, text: str, *, source_file: str) -> list[LoadedDoc]:
    """Split at paragraph boundaries so no part exceeds DOC_TOKEN_LIMIT approx tokens."""
    limit_chars = DOC_TOKEN_LIMIT * APPROX_CHARS_PER_TOKEN
    if len(text) <= limit_chars:
        return [
            LoadedDoc(doc_id=doc_id, text=text, source_file=source_file, split_index=0, n_splits=1)
        ]
    paragraphs = text.split("\n\n")
    parts: list[str] = []
    current: list[str] = []
    size = 0
    for para in paragraphs:
        # A single paragraph longer than the limit is hard-cut.
        while len(para) > limit_chars:
            if current:
                parts.append("\n\n".join(current))
                current, size = [], 0
            parts.append(para[:limit_chars])
            para = para[limit_chars:]
        if size + len(para) > limit_chars and current:
            parts.append("\n\n".join(current))
            current, size = [], 0
        current.append(para)
        size += len(para) + 2
    if current:
        parts.append("\n\n".join(current))
    n = len(parts)
    return [
        LoadedDoc(
            doc_id=f"{doc_id}#part{i}",
            text=part,
            source_file=source_file,
            split_index=i,
            n_splits=n,
        )
        for i, part in enumerate(parts)
    ]


def load_documents(files: list[Path]) -> tuple[list[LoadedDoc], list[LoadFailure]]:
    """Read every file; per-document failures collected, never batch-fatal (spec)."""
    docs: list[LoadedDoc] = []
    failures: list[LoadFailure] = []
    for path in files:
        try:
            text = read_file(path)
            docs.extend(split_oversized(path.stem, text, source_file=path.name))
        except Exception as exc:
            failures.append(LoadFailure(file=path.name, error=f"{type(exc).__name__}: {exc}"))
    return docs, failures


def make_ingest_handler(sink: IngestSink, corpora_dir: Path):
    """Job handler: idempotent full rebuild from saved uploads (bm25s has no incremental
    indexing — spec accepts full rebuild), which is what makes resume() safe."""
    import json

    def handle(ctx) -> None:
        corpus_id = ctx.params["corpus_id"]
        files = [Path(p) for p in ctx.params["files"]]
        ctx.emit(f"loading {len(files)} files", 0.05)
        docs, failures = load_documents(files)
        ctx.emit(f"loaded {len(docs)} docs ({len(failures)} failed)", 0.3)
        if not docs:
            raise RuntimeError(
                "no readable documents; failures: "
                + "; ".join(f"{f.file}: {f.error}" for f in failures)
            )
        manifest = sink.write_corpus(corpus_id=corpus_id, docs=docs, emit=ctx.emit)
        manifest["byo"] = {
            # loaded files only; failed ones are disclosed under "failures"
            "source_files": sorted({d.source_file for d in docs}),
            "split_documents": [
                {"doc_id": d.doc_id, "source_file": d.source_file, "n_splits": d.n_splits}
                for d in docs
                if d.n_splits > 1
            ],
            "failures": [{"file": f.file, "error": f.error} for f in failures],
        }
        target = corpora_dir / corpus_id
        target.mkdir(parents=True, exist_ok=True)
        (target / "manifest.json").write_text(json.dumps(manifest, indent=2))
        ctx.emit("manifest written", 1.0)

    return handle


class RealIngestSink:
    """IngestSink over Plan A's R9-pinned ingest entry point.

    Pin (contracts §Seam Resolutions R9):
      - ingest/pipeline.py::run_ingest(corpus_id=, data_dir=, ingest_config=,
        embed=, qdrant=) -> manifest dict
    write_corpus materializes the R1 raw/ layout — `raw/docs.jsonl` records
    {"doc_id","passage_id","title","text"} (BYO docs are unsegmented, so
    passage_id == doc_id) plus a BYO `raw/download_meta.json` whose dataset block
    carries {"name": "byo", ...} (the eval runner's multi-hop gate reads
    dataset.name, R10) — then delegates to run_ingest, which chunks, embeds BOTH
    named-vector sets, builds the bm25s index, writes manifest.json, and returns
    the manifest. Constructor seam defaults to the real entry point; tests inject
    a fake.
    """

    def __init__(self, *, data_dir: Path, qdrant, embed, run_ingest_fn=None) -> None:
        if run_ingest_fn is None:
            from ragreceipts.ingest.pipeline import run_ingest  # R9 ingest entry point

            run_ingest_fn = run_ingest
        self._data_dir = data_dir
        self._qdrant = qdrant
        self._embed = embed
        self._run_ingest = run_ingest_fn

    def write_corpus(
        self, *, corpus_id: str, docs: list[LoadedDoc], emit: Callable[[str, float], None]
    ) -> dict:
        import json
        from datetime import UTC, datetime

        from ragreceipts.config import IngestConfig

        raw_dir = self._data_dir / "corpora" / corpus_id / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        with (raw_dir / "docs.jsonl").open("w", encoding="utf-8") as fh:
            for d in docs:
                fh.write(
                    json.dumps(
                        {
                            "doc_id": d.doc_id,
                            "passage_id": d.doc_id,
                            "title": d.source_file,
                            "text": d.text,
                        }
                    )
                    + "\n"
                )
        (raw_dir / "download_meta.json").write_text(
            json.dumps(
                {
                    "corpus_id": corpus_id,
                    "dataset": {
                        "name": "byo",
                        "hf_id": None,
                        "config": None,
                        "split": None,
                        "revision": None,
                    },
                    "created_at": datetime.now(UTC).isoformat(),
                },
                indent=2,
            )
        )
        emit(f"raw layout written ({len(docs)} docs)", 0.45)
        manifest = self._run_ingest(
            corpus_id=corpus_id,
            data_dir=self._data_dir,
            ingest_config=IngestConfig(),
            embed=self._embed,
            qdrant=self._qdrant,
        )
        emit("both dense vector sets + sparse index built", 0.85)
        return manifest


def build_real_ingest_sink(*, paths, qdrant) -> RealIngestSink:
    """Production constructor — wired by deps.build_deps when all three vendor keys
    AND QDRANT_URL (R7) are present."""
    from ragreceipts.vendors.voyage_client import VoyageClient  # Plan A module name

    return RealIngestSink(data_dir=paths.data_dir, qdrant=qdrant, embed=VoyageClient())
