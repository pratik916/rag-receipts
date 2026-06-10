"use client";
import { useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api } from "@/lib/api/client";
import type { components } from "@/lib/api/schema";

type ReceiptEntry = components["schemas"]["ReceiptEntryModel"];
// Matches the contracts' PublishedAnchor dataclass (receipt payload is open by design).
type Anchor = {
  source: string;
  published_value: number;
  measured_value: number;
  direction_match: boolean;
  note: string;
};
type ReceiptBody = {
  run_id?: string;
  preset?: string;
  n_total?: number;
  n_failed?: number;
  n_abstained?: number;
  index_hashes?: Record<string, string>;
  metrics?: Record<string, number>;
  anchors?: Anchor[];
};

const METRICS = [
  "recall_at_5",
  "mrr_at_3",
  "em",
  "f1",
  "ragas_faithfulness",
  "ragas_answer_relevancy",
  "usd_per_query",
];
// Ladder order is fixed by contract (api/ragreceipts/config.py PRESETS).
const PRESET_ORDER = ["bm25-only", "dense-rrf", "contextual", "rerank", "router-on"];

export default function AblationLab() {
  const [receipts, setReceipts] = useState<ReceiptEntry[]>([]);
  const [errors, setErrors] = useState<string[]>([]);
  const [showCommitted, setShowCommitted] = useState(true);
  const [showLocal, setShowLocal] = useState(true);

  useEffect(() => {
    api.GET("/receipts").then(({ data }) => {
      setReceipts(data?.receipts ?? []);
      setErrors(data?.errors ?? []);
    });
  }, []);

  const visible = receipts.filter((r) =>
    r.source === "committed" ? showCommitted : showLocal
  );

  // Cell-level "cross-index" disclosure (contracts R11): a cell is flagged when its
  // dense index hash (index_hashes.dense_contextual ?? index_hashes.dense_isolated)
  // differs from the previous dense-bearing preset in ladder order — i.e. the ladder
  // step it is read against was measured on a DIFFERENT index. With the fixtures,
  // exactly the contextual cell is flagged (dense-rrf:iso -> contextual:ctx).
  const crossIndex = (() => {
    const flagged = new Set<string>();
    let prev: string | null = null;
    for (const preset of PRESET_ORDER) {
      let presetDense: string | null = null;
      for (const r of visible) {
        const body = r.receipt as ReceiptBody;
        if (body.preset !== preset) continue;
        const hashes = body.index_hashes ?? {};
        const dense = hashes["dense_contextual"] ?? hashes["dense_isolated"];
        if (!dense) continue;
        presetDense = dense;
        if (prev !== null && dense !== prev) flagged.add(r.path);
      }
      if (presetDense !== null) prev = presetDense;
    }
    return flagged;
  })();
  const crossIndexPresets = Array.from(
    new Set(
      visible
        .filter((r) => crossIndex.has(r.path))
        .map((r) => (r.receipt as ReceiptBody).preset ?? "?")
    )
  );

  // Grouped bars: one group per preset, one bar per source (recharts: multiple <Bar>
  // elements without stackId render side by side — verified, recharts BarChart API).
  function chartData(metric: string) {
    return PRESET_ORDER.map((preset) => {
      const row: Record<string, string | number> = { preset };
      for (const r of visible) {
        const body = r.receipt as ReceiptBody;
        const value = body.preset === preset ? body.metrics?.[metric] : undefined;
        if (value !== undefined) row[r.source] = value;
      }
      return row;
    }).filter((row) => "committed" in row || "local" in row);
  }

  const anchorRows = visible.flatMap((e) => {
    const body = e.receipt as ReceiptBody;
    return (body.anchors ?? []).map((anchor) => ({
      preset: body.preset ?? "?",
      source: e.source,
      anchor,
    }));
  });

  return (
    <>
      <section className="card">
        <div className="row">
          <h1 style={{ margin: 0, fontSize: 20 }}>Ablation Lab</h1>
          <label>
            <input
              type="checkbox"
              data-testid="toggle-committed"
              checked={showCommitted}
              onChange={(e) => setShowCommitted(e.target.checked)}
            />{" "}
            committed
          </label>
          <label>
            <input
              type="checkbox"
              data-testid="toggle-local"
              checked={showLocal}
              onChange={(e) => setShowLocal(e.target.checked)}
            />{" "}
            local
          </label>
        </div>
        <p className="muted">
          Each preset&apos;s receipt: measured contribution on labeled data. Committed =
          headline runs from <code>receipts/</code>; local = your runs.
        </p>
        {errors.map((err) => (
          <p key={err} className="error">
            unreadable receipt: {err}
          </p>
        ))}
        <table className="chunks">
          <tbody>
            {visible.map((r) => {
              const body = r.receipt as ReceiptBody;
              return (
                <tr key={r.path} data-testid="receipt-row">
                  <td>{body.preset}</td>
                  <td>{r.source}</td>
                  <td>
                    <code>{body.run_id}</code>
                  </td>
                  <td className="muted">
                    n={body.n_total} failed={body.n_failed} abstained={body.n_abstained}
                  </td>
                  <td>
                    {crossIndex.has(r.path) && (
                      <span
                        className="badge badge-degraded"
                        data-testid="cross-index-badge"
                        title="Measured against a different dense index than the preceding ladder cell (see index_hashes)"
                      >
                        cross-index
                      </span>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </section>

      {METRICS.map((metric) => {
        const data = chartData(metric);
        if (data.length === 0) return null;
        return (
          <section className="card" key={metric} data-testid={`metric-chart-${metric}`}>
            <h2 style={{ marginTop: 0 }}>{metric}</h2>
            <ResponsiveContainer width="100%" height={240}>
              <BarChart data={data}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="preset" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="committed" fill="#2563eb" />
                <Bar dataKey="local" fill="#9ca3af" />
              </BarChart>
            </ResponsiveContainer>
            {crossIndexPresets.length > 0 && (
              <p className="muted" data-testid="cross-index-note">
                cross-index: {crossIndexPresets.join(", ")} — measured against a
                different dense index than the preceding ladder cell (the receipt&apos;s
                index_hashes differ), so read that step as a cross-index comparison.
              </p>
            )}
          </section>
        );
      })}

      {anchorRows.length > 0 && (
        <section className="card">
          <h2 style={{ marginTop: 0 }}>Ours vs published anchors</h2>
          <p className="muted">
            Published numbers are anchors, not targets — the note explains why magnitudes
            are not directly comparable.
          </p>
          {anchorRows.map((row, i) => (
            <div key={i} className="anchor" data-testid="anchor-row">
              <div className="anchor-head">
                <strong>{row.preset}</strong>
                <span className="muted">{row.anchor.source}</span>
                <span>
                  published {row.anchor.published_value} · measured{" "}
                  {row.anchor.measured_value}
                </span>
                <span
                  className={
                    row.anchor.direction_match ? "badge badge-ok" : "badge badge-degraded"
                  }
                >
                  {row.anchor.direction_match ? "direction match" : "direction mismatch"}
                </span>
              </div>
              <blockquote data-testid="anchor-note">{row.anchor.note}</blockquote>
            </div>
          ))}
        </section>
      )}
    </>
  );
}
