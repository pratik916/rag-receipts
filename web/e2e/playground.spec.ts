import { expect, test } from "@playwright/test";

test("query renders route badge, cited answer with popover, and trace", async ({ page }) => {
  await page.goto("/");
  await page.getByTestId("query-input").fill("What is the capital of France?");
  await page.getByTestId("preset-select").selectOption("rerank");
  await page.getByTestId("run-query").click();

  await expect(page.getByTestId("route-badge")).toHaveText("System-1");
  await expect(page.getByTestId("answer")).toContainText("Paris");

  await page.getByTestId("cite-1").click();
  await expect(page.getByTestId("citation-popover")).toBeVisible();
  await expect(page.getByTestId("citation-popover")).toContainText("geo-001");

  const nodes = page.getByTestId("trace-event");
  await expect(nodes).toHaveCount(3);
  await expect(nodes.first()).toContainText("route");
  await expect(nodes.nth(1)).toContainText("s1_retrieve");
});

test("degraded retrieval shows a visible badge, never silent", async ({ page }) => {
  await page.goto("/");
  await page.getByTestId("query-input").fill("degrade: capital of France?");
  await page.getByTestId("run-query").click();
  await expect(page.getByTestId("degraded-flag")).toHaveText("rerank-skipped");
  await expect(page.getByTestId("degraded-badge").first()).toBeVisible();
});
