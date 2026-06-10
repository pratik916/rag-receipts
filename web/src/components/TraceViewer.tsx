import type { components } from "@/lib/api/schema";

type TraceEvent = components["schemas"]["TraceEventModel"];
type ChunkRow = { chunk_id: string; score: number };

const HOP_NODES = new Set(["retrieve_hop", "grade", "refine"]);

export default function TraceViewer({ events }: { events: TraceEvent[] }) {
  return (
    <ol className="trace">
      {events.map((ev) => {
        const payload = ev.payload as Record<string, unknown>;
        const degraded = (payload.degraded as string[] | undefined) ?? [];
        const chunks = (payload.chunks as ChunkRow[] | undefined) ?? [];
        const hop = payload.hop as number | undefined;
        return (
          <li key={ev.seq} className="trace-event" data-testid="trace-event">
            <div className="trace-head">
              <span className="node">{ev.node}</span>
              {HOP_NODES.has(ev.node) && hop !== undefined && (
                <span className="badge">hop {hop}</span>
              )}
              {ev.model && <code className="model">{ev.model}</code>}
              <span className="ms">{ev.duration_ms.toFixed(0)} ms</span>
              {(ev.input_tokens > 0 || ev.output_tokens > 0) && (
                <span className="tokens">
                  {ev.input_tokens}→{ev.output_tokens} tok
                </span>
              )}
              {degraded.map((d) => (
                <span key={d} className="badge badge-degraded" data-testid="degraded-badge">
                  {d}
                </span>
              ))}
            </div>
            {chunks.length > 0 && (
              <table className="chunks">
                <tbody>
                  {chunks.map((c) => (
                    <tr key={c.chunk_id}>
                      <td>
                        <code>{c.chunk_id}</code>
                      </td>
                      <td>{c.score.toFixed(4)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </li>
        );
      })}
    </ol>
  );
}
