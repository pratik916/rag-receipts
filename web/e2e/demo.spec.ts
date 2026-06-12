import { test, expect } from "@playwright/test";

// These tests run against the TESTING=1 server (started by pnpm e2e).
// The fixture's demo_ledger has daily_budget_usd=0.001, so one query exhausts the budget.

test.describe("DEMO_MODE API guards", () => {
  test("POST /query with wrong corpus returns 403", async ({ request }) => {
    const apiBase = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";
    const response = await request.post(`${apiBase}/query`, {
      data: {
        query: "hello",
        corpus_id: "not-a-real-corpus",
        preset: "bm25-only",
      },
    });
    expect(response.status()).toBe(403);
    const body = await response.json();
    expect(body.detail).toMatch(/demo corpus/i);
  });

  test("GET /demo/examples returns a list (empty is fine)", async ({ request }) => {
    const apiBase = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";
    const response = await request.get(`${apiBase}/demo/examples`);
    expect(response.status()).toBe(200);
    const body = await response.json();
    expect(Array.isArray(body.examples)).toBe(true);
  });

  test("POST /corpora/ingest returns 403 in demo mode", async ({ request }) => {
    const apiBase = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";
    const response = await request.post(`${apiBase}/corpora/ingest`, {
      multipart: {
        corpus_id: "test",
        files: {
          name: "a.txt",
          mimeType: "text/plain",
          buffer: Buffer.from("hello"),
        },
      },
    });
    expect(response.status()).toBe(403);
  });
});
