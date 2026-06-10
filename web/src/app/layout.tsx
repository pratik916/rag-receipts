import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "rag-receipts",
  description: "Every RAG technique, with receipts.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <header className="nav">
          <span className="brand">
            rag-receipts <span className="tagline">every technique, with receipts</span>
          </span>
          <nav>
            <Link href="/">Playground</Link>
            <Link href="/ablation">Ablation Lab</Link>
            <Link href="/corpora">Corpora</Link>
          </nav>
        </header>
        <main className="main">{children}</main>
      </body>
    </html>
  );
}
