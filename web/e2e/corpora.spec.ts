import { expect, test } from "@playwright/test";

test("corpora page lists manifests with chunking and hash disclosure", async ({ page }) => {
  await page.goto("/corpora");
  const card = page.getByTestId("corpus-card").filter({ hasText: "fixture-corpus" });
  await expect(card).toBeVisible();
  await expect(card).toContainText("chunk_size 512");
  await expect(card).toContainText("voyage-context-3");
  await expect(card).toContainText("sha256:fixture");
});
