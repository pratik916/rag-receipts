import { expect, test } from "@playwright/test";

test("ablation lab renders committed receipts, charts, and verbatim anchor notes", async ({ page }) => {
  await page.goto("/ablation");
  await expect(page.getByTestId("metric-chart-recall_at_5")).toBeVisible();
  // Committed fixture receipts (served from RAGRECEIPTS_RECEIPTS_DIR) render
  await expect(page.getByTestId("receipt-row").filter({ hasText: "bm25-only" })).toBeVisible();
  await expect(page.getByTestId("receipt-row").filter({ hasText: "committed" }).first()).toBeVisible();
  // PublishedAnchor.note rendered VERBATIM
  await expect(page.getByTestId("anchor-note").first()).toContainText(
    "direction-match only, never magnitude reproduction"
  );
  // R11: the contextual cell carries the CELL-level cross-index marker (its dense
  // index hash differs from the preceding dense-bearing ladder cell, dense-rrf),
  // and with the fixture data it is the ONLY flagged cell.
  const contextualRow = page.getByTestId("receipt-row").filter({ hasText: "contextual" });
  await expect(contextualRow).toBeVisible();
  await expect(contextualRow.getByTestId("cross-index-badge")).toHaveText("cross-index");
  await expect(page.getByTestId("cross-index-badge")).toHaveCount(1);
  await expect(page.getByTestId("cross-index-note").first()).toContainText("contextual");
});

test("committed/local toggle filters sources", async ({ page }) => {
  await page.goto("/ablation");
  await expect(page.getByTestId("receipt-row").filter({ hasText: "local" }).first()).toBeVisible();
  await page.getByTestId("toggle-local").uncheck();
  await expect(page.getByTestId("receipt-row").filter({ hasText: "local" })).toHaveCount(0);
  await page.getByTestId("toggle-committed").uncheck();
  await expect(page.getByTestId("receipt-row")).toHaveCount(0);
});
