import { expect, test } from "@playwright/test";

test("layout renders brand and all three page links", async ({ page }) => {
  await page.goto("/");
  await expect(page.locator(".brand")).toContainText("rag-receipts");
  await expect(page.getByRole("link", { name: "Playground" })).toBeVisible();
  await expect(page.getByRole("link", { name: "Ablation Lab" })).toBeVisible();
  await expect(page.getByRole("link", { name: "Corpora" })).toBeVisible();
});
