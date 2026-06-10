"use client";
import { useState } from "react";
import type { components } from "@/lib/api/schema";

type Citation = components["schemas"]["CitationModel"];

export default function AnswerView({
  answer,
  citations,
  abstained,
}: {
  answer: string;
  citations: Citation[];
  abstained: boolean;
}) {
  const [open, setOpen] = useState<number | null>(null);
  const byN = new Map(citations.map((c) => [c.n, c]));
  const parts = answer.split(/(\[\d+\])/g);
  const current = open !== null ? byN.get(open) : undefined;
  return (
    <div className="card" data-testid="answer">
      {abstained && (
        <span className="badge badge-degraded" data-testid="abstained-badge">
          abstained
        </span>
      )}
      <p>
        {parts.map((part, i) => {
          const match = part.match(/^\[(\d+)\]$/);
          if (!match) return <span key={i}>{part}</span>;
          const n = Number(match[1]);
          return (
            <button
              key={i}
              className="cite"
              data-testid={`cite-${n}`}
              onClick={() => setOpen(open === n ? null : n)}
            >
              [{n}]
            </button>
          );
        })}
      </p>
      {current && (
        <div className="popover" data-testid="citation-popover">
          <div className="popover-head">
            <code>{current.chunk_id}</code>
            <span>score {current.score.toFixed(3)}</span>
          </div>
          <p>{current.text}</p>
        </div>
      )}
    </div>
  );
}
