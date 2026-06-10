import { expect, test } from "@playwright/test";

test("BYO upload streams job progress and the new corpus appears", async ({ page }) => {
  await page.goto("/corpora");
  await page.getByTestId("upload-corpus-id").fill("e2e-byo");
  await page.getByTestId("upload-files").setInputFiles([
    {
      name: "doc-one.txt",
      mimeType: "text/plain",
      buffer: Buffer.from("The first uploaded document, about rivers in France."),
    },
    {
      name: "doc-two.md",
      mimeType: "text/markdown",
      buffer: Buffer.from("# Second doc\n\nAbout capitals of Europe."),
    },
  ]);
  await page.getByTestId("upload-submit").click();
  await expect(page.getByTestId("job-progress")).toBeVisible();
  await expect(page.getByTestId("job-status")).toHaveText("succeeded", { timeout: 30_000 });
  await expect(
    page.getByTestId("corpus-card").filter({ hasText: "e2e-byo" })
  ).toBeVisible();
});
