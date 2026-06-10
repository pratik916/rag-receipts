# Full-stack verification checklist (Plan D, Task 17)

The final gate for the whole project. No new code — **evidence before assertions**:
every box below was satisfied by running the command and seeing the stated output.

This run is the **offline gate**: zero vendor keys, fully under `TESTING=1` fakes, only
package/image downloads touch the network. The checklist's **keyed** sections (compose
up with real keys, live-vendor query, billed eval run, post-`down` volume survival) cost
real money and are out of scope here by project policy — they are the human-driven path
documented in [`first-keyed-run.md`](first-keyed-run.md) and are flagged below as
**[keyed — not run]**, never silently skipped.

- Run date: 2026-06-10
- Head commit at verification: `7003f46`
- Toolchain: Python 3.12.8 (uv), Node v22.16.0, pnpm 10.30.3, Docker 29.4.3 /
  Compose v5.1.4

## Offline gates (zero keys) — ALL GREEN

- [x] `cd api && uv run pytest -q` — **308 passed, 1 warning in 11.74s**. No vendor keys
  set (`ANTHROPIC_API_KEY` / `VOYAGE_API_KEY` / `COHERE_API_KEY` all unset), no network.
  The lone warning is the upstream Starlette/httpx `TestClient` deprecation notice, not a
  test failure.
- [x] `cd api && uv run ruff check .` — **All checks passed!** (also `ruff check
  ragreceipts tests` — clean).
- [x] `cd web && pnpm build` — type-safe production build, **Compiled successfully**,
  4 routes prerendered (`/`, `/ablation`, `/corpora`, `/_not-found`), no type errors.
- [x] `cd web && pnpm e2e` (`playwright test`) — **7 passed, 3 skipped** against the
  `TESTING=1` api. The 3 skipped are the `CAPTURE=1`-gated screenshot-capture specs in
  `e2e/screenshots.spec.ts` (skipped by design so a normal e2e run never rewrites the
  committed PNGs). The 7 functional specs: nav layout, playground query +
  cited-answer/trace, degraded-retrieval badge, corpora manifest disclosure, BYO upload
  job progress, ablation receipts/charts/anchor-notes, committed/local toggle.

## Compose stack

- [x] `docker compose config` — **valid (exit 0)** with a placeholder `.env` present;
  services resolve to `qdrant`, `api`, `web`. (Without `.env` the parse errors on the
  missing `env_file`, which is expected — `.env` is gitignored and supplied by the
  operator; the placeholder file used for this check was removed and is never committed.)
- [x] **Offline health-boot smoke** (the offline analogue of the keyed `/health` box):
  booted `uv run python -m uvicorn ragreceipts.server.app:app --workers 1` under
  `TESTING=1`; `GET /health` returned `"status": "ok"`, `"missing_env_vars": []`,
  `"qdrant_ok": true`, **`"testing_mode": true`** (keyed runs report `false`), with all
  three vendors `configured: true` (fake transports). `GET /corpora` served the fixture
  manifest with chunking + index hashes.
- [ ] **[keyed — not run]** `cp .env.example .env`, fill three keys,
  `docker compose up --build -d`; `docker compose ps` healthy; `curl
  localhost:8000/health` → `"testing_mode": false`. → see `first-keyed-run.md`.

## Ingest a smoke corpus (BYO path)

- [ ] **[keyed — not run]** `POST /corpora/ingest` of `/tmp/a.txt` + `/tmp/b.md`; poll
  `GET /jobs/<id>` to `succeeded`; corpus appears at `localhost:3000/corpora`.

  > Covered offline by `e2e/corpora-upload.spec.ts` (BYO upload streams job progress and
  > the new corpus appears) and the api ingest unit/seam tests in the Python suite. The
  > live keyed path adds real LlamaIndex readers + voyage contextualization.

## Query (live vendors — manual 5-query smoke, never CI)

- [ ] **[keyed — not run]** Playground "What is the capital of France?" against
  `smoke-docs`, preset `rerank`: cited answer; trace `route → s1_retrieve → s1_answer`
  with model IDs + token counts; System-1 route badge.

  > Covered offline by `e2e/playground.spec.ts` (route badge, cited answer with popover,
  > trace render; degraded-retrieval badge) against fake transports. The keyed path
  > exercises real Claude/Voyage/Cohere calls.

## Smoke eval + receipts in the UI

- [ ] **[keyed — not run]** `POST /eval/runs` → `needs_confirmation` with estimate +
  `pricing_table_version`; re-POST `confirm: true` → job succeeds; Ablation Lab shows the
  new run under the **local** toggle next to **committed** receipts; anchor notes verbatim;
  charts group committed vs local.
- [ ] **[keyed — not run]** `docker compose down` then `up -d` again — corpora and local
  receipts survive (named volumes `app-data`, `qdrant-storage`).

  > The estimate→confirm gate, the committed/local receipt merge with disclosed errors,
  > and the grouped charts + verbatim anchor notes are all covered offline by the Python
  > eval-runs tests and `e2e/ablation.spec.ts` (receipts/charts/anchor-notes + the
  > committed/local toggle). Named-volume persistence is declared in `docker-compose.yml`
  > (`app-data`, `qdrant-storage`) and exercised only in the keyed runbook.

## Repo hygiene — GREEN

- [x] `git status` — clean tree before this commit; `git ls-files` tracks no `.env` and
  no `data/`; `git check-ignore data/ .env` confirms both are ignored per `.gitignore`.
- [x] README quickstart is the docker-compose flow; its **Development** section documents
  the exact offline commands re-run above (`uv sync && uv run pytest`, `pnpm build`,
  `pnpm e2e`) — a stranger can reproduce every green box here from a clean clone with no
  keys.
- [x] `git log --oneline` — one conventional commit per task, each carrying the
  `Co-Authored-By: Claude Fable 5` trailer.

## Verdict

**Offline gate: GREEN.** Python suite 308/308, ruff clean, web build type-safe, web e2e
7/7, compose config valid, offline health-boot ok, repo hygiene clean. The keyed sections
are deliberately deferred to `first-keyed-run.md` (real spend, human-driven, never CI).
