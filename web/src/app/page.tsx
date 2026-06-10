"use client";
import { useEffect, useState } from "react";
import AnswerView from "@/components/AnswerView";
import TraceViewer from "@/components/TraceViewer";
import { api } from "@/lib/api/client";
import type { components } from "@/lib/api/schema";

type QueryResponse = components["schemas"]["QueryResponse"];
type TraceEvent = components["schemas"]["TraceEventModel"];

// Preset ladder is fixed by contract (api/ragreceipts/config.py PRESETS).
const PRESETS = ["bm25-only", "dense-rrf", "contextual", "rerank", "graph", "graph-rrf", "router-on"];

export default function Playground() {
  const [corpora, setCorpora] = useState<string[]>([]);
  const [corpusId, setCorpusId] = useState("");
  const [preset, setPreset] = useState("rerank");
  const [query, setQuery] = useState("");
  const [result, setResult] = useState<QueryResponse | null>(null);
  const [events, setEvents] = useState<TraceEvent[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    api.GET("/corpora").then(({ data }) => {
      const ids = data?.corpora.map((c) => c.corpus_id) ?? [];
      setCorpora(ids);
      if (ids.length > 0) setCorpusId((cur) => cur || ids[0]);
    });
  }, []);

  async function run() {
    setBusy(true);
    setError(null);
    setResult(null);
    setEvents([]);
    const { data, error: err } = await api.POST("/query", {
      body: { query, corpus_id: corpusId, preset },
    });
    if (err || !data) {
      const detail =
        err && typeof err === "object" && "detail" in err
          ? JSON.stringify((err as { detail: unknown }).detail)
          : "request failed";
      setError(detail);
      setBusy(false);
      return;
    }
    setResult(data);
    const trace = await api.GET("/traces/{trace_id}", {
      params: { path: { trace_id: data.trace_id } },
    });
    setEvents(trace.data?.events ?? []);
    setBusy(false);
  }

  return (
    <>
      <section className="card">
        <textarea
          data-testid="query-input"
          rows={2}
          placeholder="Ask the corpus anything…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
        <div className="row" style={{ marginTop: 10 }}>
          <select
            data-testid="corpus-select"
            value={corpusId}
            onChange={(e) => setCorpusId(e.target.value)}
          >
            {corpora.map((c) => (
              <option key={c}>{c}</option>
            ))}
          </select>
          <select
            data-testid="preset-select"
            value={preset}
            onChange={(e) => setPreset(e.target.value)}
          >
            {PRESETS.map((p) => (
              <option key={p}>{p}</option>
            ))}
          </select>
          <button
            className="primary"
            data-testid="run-query"
            disabled={busy || !query || !corpusId}
            onClick={run}
          >
            {busy ? "Running…" : "Run"}
          </button>
        </div>
      </section>

      {error && <p className="error">{error}</p>}

      {result && (
        <>
          <div className="row">
            <span
              className={
                result.route === "s1"
                  ? "badge badge-s1"
                  : result.route === "graph"
                    ? "badge badge-graph"
                    : "badge badge-s2"
              }
              data-testid="route-badge"
            >
              {result.route === "s1"
                ? "System-1"
                : result.route === "graph"
                  ? "Graph"
                  : "System-2"}
            </span>
            {result.degraded.map((d) => (
              <span key={d} className="badge badge-degraded" data-testid="degraded-flag">
                {d}
              </span>
            ))}
          </div>
          <AnswerView
            answer={result.answer}
            citations={result.citations}
            abstained={result.abstained}
          />
          <section className="card">
            <h2>Trace</h2>
            <TraceViewer events={events} />
          </section>
        </>
      )}
    </>
  );
}
