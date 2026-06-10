import { defineConfig } from "@playwright/test";

// Two-server harness per https://playwright.dev/docs/test-webserver (array form
// requires explicit baseURL). The api runs in TESTING=1 mode: vendor transports are
// the contracts' fakes, zero keys, fully offline. RAGRECEIPTS_RECEIPTS_DIR points at
// the committed-format fixture receipts so assertions are hermetic and deterministic.
export default defineConfig({
  testDir: "./e2e",
  timeout: 60_000,
  use: { baseURL: "http://localhost:3000" },
  webServer: [
    {
      command:
        "uv run python -m uvicorn ragreceipts.server.app:app --port 8000 --workers 1",
      cwd: "../api",
      url: "http://localhost:8000/health",
      reuseExistingServer: !process.env.CI,
      timeout: 60_000,
      env: {
        TESTING: "1",
        RAGRECEIPTS_RECEIPTS_DIR: "tests/fixtures/receipts",
      },
    },
    {
      command: "pnpm dev",
      url: "http://localhost:3000",
      reuseExistingServer: !process.env.CI,
      timeout: 120_000,
      env: { NEXT_PUBLIC_API_BASE_URL: "http://localhost:8000" },
    },
  ],
});
