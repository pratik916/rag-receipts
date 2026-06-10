"use client";
import { useCallback, useEffect, useState } from "react";
import { api } from "@/lib/api/client";
import type { components } from "@/lib/api/schema";

type Corpus = components["schemas"]["CorpusModel"];
// Matches the contracts' corpus manifest (open payload by design).
type Manifest = {
  dataset?: { name?: string };
  chunking?: { chunk_size?: number; chunk_overlap?: number };
  embed_model?: string;
  index_hashes?: Record<string, string>;
  n_docs?: number;
  n_chunks?: number;
  n_queries?: number;
  created_at?: string;
  byo?: {
    source_files?: string[];
    split_documents?: { doc_id: string; source_file: string; n_splits: number }[];
    failures?: { file: string; error: string }[];
  };
};

export default function Corpora() {
  const [corpora, setCorpora] = useState<Corpus[]>([]);

  const reload = useCallback(() => {
    api.GET("/corpora").then(({ data }) => setCorpora(data?.corpora ?? []));
  }, []);

  useEffect(() => {
    reload();
  }, [reload]);

  return (
    <>
      <h1 style={{ fontSize: 20 }}>Corpora</h1>
      {corpora.length === 0 && (
        <p className="muted">No corpora ingested yet.</p>
      )}
      {corpora.map((c) => {
        const man = c.manifest as Manifest;
        return (
          <section className="card" key={c.corpus_id} data-testid="corpus-card">
            <div className="row">
              <strong>{c.corpus_id}</strong>
              <span className="badge">{man.dataset?.name ?? "unknown dataset"}</span>
              <span className="muted">
                {man.n_docs ?? "?"} docs · {man.n_chunks ?? "?"} chunks ·{" "}
                {man.n_queries ?? "?"} queries
              </span>
            </div>
            <p className="muted">
              chunk_size {man.chunking?.chunk_size ?? "?"} · overlap{" "}
              {man.chunking?.chunk_overlap ?? "?"} · {man.embed_model ?? "?"} · created{" "}
              {man.created_at ?? "?"}
            </p>
            <table className="chunks">
              <tbody>
                {Object.entries(man.index_hashes ?? {}).map(([name, hash]) => (
                  <tr key={name}>
                    <td>{name}</td>
                    <td>
                      <code>{hash}</code>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            {man.byo?.split_documents && man.byo.split_documents.length > 0 && (
              <p className="muted" data-testid="split-disclosure">
                Oversized documents split at ingest (disclosed per manifest):{" "}
                {man.byo.split_documents
                  .map((s) => `${s.source_file} → ${s.n_splits} parts`)
                  .join(", ")}
              </p>
            )}
            {man.byo?.failures && man.byo.failures.length > 0 && (
              <div data-testid="ingest-failures">
                {man.byo.failures.map((f) => (
                  <p key={f.file} className="error">
                    failed: {f.file} — {f.error}
                  </p>
                ))}
              </div>
            )}
          </section>
        );
      })}
    </>
  );
}
