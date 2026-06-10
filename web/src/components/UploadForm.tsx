"use client";
import { useRef, useState } from "react";
import { API_BASE, api } from "@/lib/api/client";
import type { components } from "@/lib/api/schema";

type JobResponse = components["schemas"]["JobResponse"];

export default function UploadForm({ onDone }: { onDone: () => void }) {
  const [corpusId, setCorpusId] = useState("");
  const [job, setJob] = useState<JobResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  async function poll(jobId: string) {
    const { data } = await api.GET("/jobs/{job_id}", {
      params: { path: { job_id: jobId } },
    });
    if (data) setJob(data);
    if (data && (data.status === "succeeded" || data.status === "failed")) {
      onDone();
      return;
    }
    setTimeout(() => poll(jobId), 1000);
  }

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    const files = fileRef.current?.files;
    if (!files || files.length === 0 || !corpusId) return;
    const form = new FormData();
    form.append("corpus_id", corpusId);
    for (const f of Array.from(files)) form.append("files", f);
    // Multipart goes through raw fetch; the typed client covers the JSON endpoints.
    const res = await fetch(`${API_BASE}/corpora/ingest`, { method: "POST", body: form });
    if (!res.ok) {
      setError(`ingest failed: HTTP ${res.status} ${await res.text()}`);
      return;
    }
    const { job_id } = (await res.json()) as { job_id: string };
    poll(job_id);
  }

  const lastProgress = job?.events.length
    ? job.events[job.events.length - 1].progress
    : 0;

  return (
    <section className="card">
      <h2 style={{ marginTop: 0 }}>Bring your own documents</h2>
      <p className="muted">PDF, Markdown, HTML, or plain text. Runs as a background job.</p>
      <form onSubmit={submit} className="row">
        <input
          type="text"
          data-testid="upload-corpus-id"
          placeholder="corpus-id (lowercase slug)"
          value={corpusId}
          onChange={(e) => setCorpusId(e.target.value)}
        />
        <input
          type="file"
          data-testid="upload-files"
          ref={fileRef}
          multiple
          accept=".pdf,.md,.html,.txt"
        />
        <button className="primary" type="submit" data-testid="upload-submit">
          Ingest
        </button>
      </form>
      {error && <p className="error">{error}</p>}
      {job && (
        <div data-testid="job-progress" style={{ marginTop: 12 }}>
          <div className="row">
            <span className="badge" data-testid="job-status">
              {job.status}
            </span>
            <code>{job.job_id.slice(0, 8)}</code>
          </div>
          <div className="progress" style={{ margin: "8px 0" }}>
            <div style={{ width: `${Math.round(lastProgress * 100)}%` }} />
          </div>
          <ul className="muted" style={{ margin: 0, paddingLeft: 18 }}>
            {job.events.map((ev) => (
              <li key={ev.seq}>{ev.message}</li>
            ))}
          </ul>
          {job.error && <p className="error">{job.error}</p>}
        </div>
      )}
    </section>
  );
}
