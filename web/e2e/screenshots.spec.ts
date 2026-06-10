import { test } from "@playwright/test";

// Capture-only spec: skipped unless CAPTURE=1 so normal e2e runs don't rewrite images.
test.skip(process.env.CAPTURE !== "1", "screenshot capture only");

test("capture playground", async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 900 });
  await page.goto("/");
  await page.getByTestId("query-input").fill("What is the capital of France?");
  await page.getByTestId("preset-select").selectOption("rerank");
  await page.getByTestId("run-query").click();
  await page.getByTestId("trace-event").first().waitFor();
  await page.getByTestId("cite-1").click();
  await page.screenshot({ path: "../docs/screenshots/playground.png", fullPage: true });
});

test("capture ablation lab", async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 900 });
  await page.goto("/ablation");
  await page.getByTestId("anchor-note").first().waitFor();
  await page.screenshot({ path: "../docs/screenshots/ablation.png", fullPage: true });
});

test("capture corpora", async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 900 });
  await page.goto("/corpora");
  await page.getByTestId("corpus-card").first().waitFor();
  await page.screenshot({ path: "../docs/screenshots/corpora.png", fullPage: true });
});
