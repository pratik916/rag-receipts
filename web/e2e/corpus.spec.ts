import { test, expect } from "@playwright/test";

// Under TESTING=1 the fixture turns demo mode on: GET /demo/examples returns []
// (no committed examples in the test fixture) and POST /corpora/ingest returns 403.
// These tests assert the graceful public-demo fallback UI (Task H3).
test.describe("Demo graceful states", () => {
  test("Playground renders without crash when demo examples are empty", async ({ page }) => {
    await page.goto("/");
    await expect(page.locator("body")).toBeVisible();
    await expect(page.getByText(/something went wrong|unhandled error/i)).not.toBeVisible();
    // The query form must still be usable even with zero showcase examples.
    await expect(page.getByTestId("query-input")).toBeVisible();
  });

  test("Corpora page shows read-only note when ingest returns 403", async ({ page }) => {
    await page.goto("/corpora");
    const readOnlyNote = page.getByText(/read.only|disabled in the public demo/i);
    await expect(readOnlyNote).toBeVisible({ timeout: 5000 });
  });
});
