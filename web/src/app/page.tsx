"use client";
import { useEffect, useState } from "react";
import AnswerView from "@/components/AnswerView";
import TraceViewer from "@/components/TraceViewer";
import { api } from "@/lib/api/client";
import type { components } from "@/lib/api/schema";

type QueryResponse = components["schemas"]["QueryResponse"];
type TraceEvent = components["schemas"]["TraceEventModel"];
type Citation = components["schemas"]["CitationModel"];
type DemoExample = components["schemas"]["DemoExampleItem"];

// Preset ladder is fixed by contract (api/ragreceipts/config.py PRESETS).
const PRESETS = ["bm25-only", "dense-rrf", "contextual", "rerank", "graph", "graph-rrf", "router-on"];

// The shape the result UI needs, shared by a live QueryResponse and a saved
// demo example so both flow through AnswerView + TraceViewer unchanged.
type Display = {
  answer: string;
  route: string;
  degraded: string[];
  abstained: boolean;
  citations: Citation[];
  events: TraceEvent[];
  // When set, the result is a saved showcase rather than the user's own query.
  showcaseLabel?: string;
};

function fromExample(ex: DemoExample): Display {
  return {
    answer: ex.answer,
    route: ex.route,
    degraded: [],
    abstained: false,
    citations: ex.citations,
    events: ex.trace_events,
    showcaseLabel: ex.label,
  };
}

export default function Playground() {
  const [corpora, setCorpora] = useState<string[]>([]);
  const [corpusId, setCorpusId] = useState("");
  const [preset, setPreset] = useState("rerank");
  const [query, setQuery] = useState("");
  const [result, setResult] = useState<Display | null>(null);
  const [examples, setExamples] = useState<DemoExample[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [banner, setBanner] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    api.GET("/corpora").then(({ data }) => {
      const ids = data?.corpora.map((c) => c.corpus_id) ?? [];
      setCorpora(ids);
      if (ids.length > 0) setCorpusId((cur) => cur || ids[0]);
    });
    // Saved showcase examples are optional: pre-bootstrap (and the e2e fixture)
    // return []. Any failure falls back to [] silently — never blocks the page.
    api
      .GET("/demo/examples")
      .then(({ data }) => setExamples(data?.examples ?? []))
      .catch(() => setExamples([]));
  }, []);

  async function run() {
    setBusy(true);
    setError(null);
    setBanner(null);
    setResult(null);
    const { data, error: err, response } = await api.POST("/query", {
      body: { query, corpus_id: corpusId, preset },
    });
    if (err || !data) {
      // Public-demo guards (server/demo.py): budget/rate 429 and corpus 403 get
      // friendly banners; budget additionally falls back to a saved example.
      const detail =
        err && typeof err === "object" && "detail" in err
          ? (err as { detail: unknown }).detail
          : undefined;
      const reason =
        detail && typeof detail === "object" && "reason" in detail
          ? (detail as { reason: string }).reason
          : undefined;
      if (response.status === 429 && reason === "budget") {
        setBanner("Daily live budget reached — showing saved examples.");
        if (examples.length > 0) setResult(fromExample(examples[0]));
      } else if (response.status === 429) {
        setBanner("Too many requests — try again in a moment.");
      } else if (response.status === 403) {
        setBanner("Queries are limited to the demo corpus in the public demo.");
      } else {
        setError(typeof detail === "string" ? detail : JSON.stringify(detail ?? "request failed"));
      }
      setBusy(false);
      return;
    }
    const trace = await api.GET("/traces/{trace_id}", {
      params: { path: { trace_id: data.trace_id } },
    });
    setResult(toDisplay(data, trace.data?.events ?? []));
    setBusy(false);
  }

  // Pre-populate the first saved example so the page is never blank pre-query.
  const shown = result ?? (examples.length > 0 ? fromExample(examples[0]) : null);

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

      {banner && (
        <p className="banner" data-testid="demo-banner">
          {banner}
        </p>
      )}
      {error && <p className="error">{error}</p>}

      {shown && (
        <>
          {shown.showcaseLabel && (
            <p className="muted" data-testid="showcase-note">
              Saved example: {shown.showcaseLabel}. Run your own query above.
            </p>
          )}
          <div className="row">
            <span
              className={
                shown.route === "s1"
                  ? "badge badge-s1"
                  : shown.route === "graph"
                    ? "badge badge-graph"
                    : "badge badge-s2"
              }
              data-testid="route-badge"
            >
              {shown.route === "s1"
                ? "System-1"
                : shown.route === "graph"
                  ? "Graph"
                  : "System-2"}
            </span>
            {shown.degraded.map((d) => (
              <span key={d} className="badge badge-degraded" data-testid="degraded-flag">
                {d}
              </span>
            ))}
          </div>
          <AnswerView
            answer={shown.answer}
            citations={shown.citations}
            abstained={shown.abstained}
          />
          <section className="card">
            <h2>Trace</h2>
            <TraceViewer events={shown.events} />
          </section>
        </>
      )}
    </>
  );
}

function toDisplay(r: QueryResponse, events: TraceEvent[]): Display {
  return {
    answer: r.answer,
    route: r.route,
    degraded: r.degraded,
    abstained: r.abstained,
    citations: r.citations,
    events,
  };
}
