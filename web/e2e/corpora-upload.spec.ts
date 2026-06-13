import { expect, test } from "@playwright/test";

// In TESTING=1 the demo_ledger is active, so /health reports demo_mode:true and
// POST /corpora/ingest returns 403. The Corpora page reads that signal on mount
// and shows a read-only note INSTEAD of the BYO upload form — the block is
// surfaced proactively, not via a failed submit (Task H3).
test("BYO upload hidden in demo mode: shows read-only note, no upload form", async ({ page }) => {
  await page.goto("/corpora");
  await expect(page.getByTestId("ingest-readonly")).toBeVisible();
  await expect(page.getByTestId("ingest-readonly")).toContainText(/disabled in the public demo/i);
  // The upload form (and its controls) must not render in demo mode.
  await expect(page.getByTestId("upload-submit")).toHaveCount(0);
  await expect(page.getByTestId("upload-corpus-id")).toHaveCount(0);
});
