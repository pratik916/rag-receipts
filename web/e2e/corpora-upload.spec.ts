import { expect, test } from "@playwright/test";

// In TESTING=1 the demo_ledger is active, so POST /corpora/ingest returns 403.
// The UploadForm surfaces the error: "ingest failed: HTTP 403 ...".
test("BYO upload blocked in demo mode: shows 403 error", async ({ page }) => {
  await page.goto("/corpora");
  await page.getByTestId("upload-corpus-id").fill("e2e-byo");
  await page.getByTestId("upload-files").setInputFiles([
    {
      name: "doc-one.txt",
      mimeType: "text/plain",
      buffer: Buffer.from("The first uploaded document, about rivers in France."),
    },
  ]);
  await page.getByTestId("upload-submit").click();
  // Demo mode blocks ingest; the form surfaces the 403 as an error message.
  await expect(page.locator(".error")).toContainText("403");
  // No job progress element should appear (job was never created).
  await expect(page.getByTestId("job-progress")).not.toBeVisible();
});
