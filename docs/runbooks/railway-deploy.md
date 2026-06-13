# Railway deploy — public demo (manual, from scratch)

Stand up the public DEMO_MODE demo on [Railway](https://railway.app) as **three services**
(Qdrant, the FastAPI api, the Next.js web) wired over Railway's private network. Rough cost:
**~$5–15/month** for the always-on trio at idle-to-light traffic (one replica each, no
sleep). The demo's own cost-control layer — per-IP rate limits + a hard daily vendor budget
+ read-only ingest/eval — caps vendor spend separately (see the `DEMO_*` env vars in step 6).

This is a human-driven path, never CI. Everything offline-testable about the demo (the
materialize/seed seams, the 429/403 guards, `/health`) is already green in the Python suite at
$0; this runbook is the live deploy on top of that.

## 0. Prerequisites — bootstrap artifacts committed

The demo serves from **committed** artifacts that the keyed bootstrap produces. Before
deploying, confirm these are present in the repo (they get baked into the api image — see
step 6 and `api/Dockerfile`):

- `demo/corpus/dense_vectors.npz` — seeds the Qdrant `demo` collection on startup
- `demo/corpus/chunks.jsonl` — canonical chunk order (Qdrant payloads + the query path)
- `demo/corpus/sparse/` — BM25 index
- `demo/corpus/graph/` — HippoRAG-2 graph artifact (the `graph-rrf` example needs it)
- `demo/corpus/manifest.json` — pinned config + index hashes (also the materialize sentinel)
- `demo/examples/*.json` — the canned Playground examples (`/demo/examples`)
- `receipts/*.json` — committed, stripped receipts (the Ablation Lab)

Produce all of these with the keyed bootstrap: **[`demo-bootstrap.md`](demo-bootstrap.md)**
(~$10–15 of tracked spend, run once). The repo ships only `demo/corpus/docs.jsonl` plus
`.gitkeep` placeholders in `demo/examples/` and `receipts/` until that runbook is run.

> **Graceful pre-bootstrap.** The deploy works even if you skip step 0: on startup the seed
> and materialize steps are no-ops when their source files are absent, so Qdrant stays empty,
> the Playground shows no canned examples, and the Ablation Lab lists no committed receipts —
> all without errors. But the demo is only worth showing once the artifacts are committed.
> Run the bootstrap, commit, and (re)deploy.

## 1. Create a Railway account

Go to [railway.app](https://railway.app) and sign up with **GitHub** (so Railway can deploy
the api/web services straight from this repo). The free trial covers a first look; the
always-on trio needs the usage-based Hobby plan (~$5/mo base + metered usage).

## 2. Install + log in the Railway CLI

```bash
npm i -g @railway/cli      # written against CLI v5.12.x
railway login              # opens a browser to authorize
railway whoami             # confirms the logged-in account
```

## 3. Create the project

From the repo root:

```bash
cd /path/to/rag-receipts
railway init               # name it e.g. "rag-receipts"; creates an empty project
```

The three services below can each be created in the Railway dashboard (Project → **New**) or
via the CLI; the dashboard is clearer for the per-service volume + networking + env settings,
so the steps describe the dashboard. Keep all three in this one project so they share the
private network.

## 4. Qdrant service (Docker image, private only)

Project → **New** → **Empty Service** (or **Docker Image**). Configure:

- **Source:** Docker image `qdrant/qdrant:v1.18.0` (this exact tag — it matches the
  `qdrant-client >=1.18,<2` pin the api builds against).
- **Volume:** mount at **`/qdrant/storage`** (Qdrant's data dir; persists collections across
  restarts/redeploys).
- **Networking:** enable **private networking** and do **not** generate a public domain. Other
  services reach it at the internal host **`qdrant.railway.internal`** on port `6333`. Keeping
  it private is the security stance — nothing outside the project can hit the vector store.

No env vars are required for Qdrant.

## 5. (order) Create the api and web services next

Create both from the GitHub repo so each gets its own build. The api needs the web public URL
for CORS and the web needs the api public URL at build time, so the wiring is: create api
(step 6) → create web (step 7) → set the api's CORS to the web URL (step 8). Generate each
public URL when its service is created; you will paste them across in step 8.

## 6. API service (FastAPI, repo-root Docker build)

Project → **New** → **GitHub Repo** → select this repo. Configure:

- **Root Directory:** **repo root** (`/`, leave blank). This is load-bearing: `api/Dockerfile`
  is built from the **repo-root context** so it can bake `demo/` (the corpus) and `receipts/`
  into the image. A `Root Directory = api/` would put those repo-root dirs outside the build
  context and the demo would have no corpus and no receipts.
- **Dockerfile Path:** **`api/Dockerfile`** (relative to the repo root above).
- **Volume:** mount at **`/data`** — the runtime data dir (SQLite traces/jobs, the bm25s
  indexes, the materialized corpus under `/data/corpora/`, local receipts, the demo ledger).
  This is the only writable persistence the api needs; the corpus + receipts are read-only in
  the image.
- **Networking:** **Generate Domain** to get the api's public URL (e.g.
  `https://rag-receipts-api-production.up.railway.app`). Note it for steps 7–8.

### API environment variables

Set these on the api service (Variables tab). Mark the three vendor keys as **secrets**.

| Variable | Value | Notes |
| --- | --- | --- |
| `ANTHROPIC_API_KEY` | *(secret)* | router/synthesis (Claude) |
| `VOYAGE_API_KEY` | *(secret)* | dense embeddings |
| `COHERE_API_KEY` | *(secret)* | rerank stage |
| `QDRANT_URL` | `http://qdrant.railway.internal:6333` | the step-4 private host; **required** — the server returns 503 (never a silent default) if unset |
| `RAGRECEIPTS_DATA_DIR` | `/data` | the step-6 volume mount |
| `RAGRECEIPTS_RECEIPTS_DIR` | `/app/receipts` | **baked into the image** (not the volume) |
| `RAGRECEIPTS_DEMO_CORPUS_DIR` | `/app/demo/corpus` | baked into the image |
| `RAGRECEIPTS_DEMO_EXAMPLES_DIR` | `/app/demo/examples` | baked into the image |
| `DEMO_MODE` | `1` | turns on the cost-control layer (rate limits, daily budget, read-only ingest/eval, `/query` corpus allow-list) |
| `DEMO_DAILY_BUDGET_USD` | `2.0` | hard global daily vendor budget; over it, `/query` returns 429 `{"reason":"budget"}` |
| `DEMO_RATE_PER_MIN` | `5` | per-IP per-minute cap (429 `{"reason":"rate"}`) |
| `DEMO_RATE_PER_DAY` | `20` | per-IP per-day cap |
| `DEMO_S2_TOKEN_CEILING` | `20000` | System-2 token ceiling for the demo (tighter than the local default) |
| `DEMO_CORPUS_ID` | `demo` | the only corpus `/query` will serve in demo mode; also the Qdrant collection name |
| `RAGRECEIPTS_CORS_ORIGINS` | *(set in step 8)* | the web public URL, no trailing slash |

> **The baked `/app/*` paths are the key correction.** Inside the image the working directory
> is `/app`, so the in-app defaults for these dirs (`../demo/corpus`, `../receipts`, …)
> resolve to `/demo` and `/receipts`, which do **not** exist. The api Dockerfile copies the
> repo dirs to `/app/demo` and `/app/receipts`, so you must point the three `RAGRECEIPTS_*`
> dir vars at those baked paths (mirroring `docker-compose.yml`).

### What happens on first boot (no manual copy needed)

When the api starts with `DEMO_MODE=1`, the lifespan startup automatically:

1. **Materializes the corpus** — `materialize_demo_corpus` copies the query-time artifacts
   (`chunks.jsonl`, `sparse/`, `graph/`, `manifest.json`) from `/app/demo/corpus` into
   `/data/corpora/demo/` on the volume. Idempotent (skips if `/data/corpora/demo/manifest.json`
   already exists), so redeploys don't re-copy.
2. **Seeds Qdrant** — `seed_demo_qdrant` loads `/app/demo/corpus/dense_vectors.npz` and upserts
   the `contextual` / `isolated` named vectors into the `demo` collection. Idempotent (skips if
   the collection already has points).

You do **not** SSH in or copy the corpus by hand — it is wired into startup. (Both steps are
graceful no-ops if the step-0 artifacts were not committed.)

## 7. Web service (Next.js, build-time API URL)

Project → **New** → **GitHub Repo** → the same repo. Configure:

- **Root Directory:** **`web/`** (the web Dockerfile builds from its own `web/` context).
- **Dockerfile Path:** **`web/Dockerfile`** (or `Dockerfile`, relative to the `web/` root).
- **Build Arg:** `NEXT_PUBLIC_API_BASE_URL` = the **api public URL** from step 6 (e.g.
  `https://rag-receipts-api-production.up.railway.app`). This is a Next.js `NEXT_PUBLIC_*`
  value baked at **build time**, so set it as a build arg/variable before the first build; if
  you change the api URL later, the web service must rebuild.
- **Networking:** **Generate Domain** to get the web public URL (this is what you share). Note
  it for step 8.

## 8. Wire CORS (api ↔ web)

Back on the **api** service, set:

```
RAGRECEIPTS_CORS_ORIGINS = https://<your-web-domain>.up.railway.app
```

Use the exact web origin from step 7, **no trailing slash** (it is matched verbatim against
the browser `Origin` header). Setting this var redeploys the api. Multiple origins are
comma-separated if you ever add one.

## 9. Deploy + verify

Trigger a deploy of all three (push to the connected branch, or **Deploy** in the dashboard).
Watch the api deploy logs for `Materialized demo corpus into /data/corpora/demo` and
`Seeded N points into demo collection 'demo'` (or the graceful "skipping seed — run
demo-bootstrap" warnings if step 0 was skipped).

Then verify (substitute your URLs):

```bash
API_URL=https://<your-api-domain>.up.railway.app
WEB_URL=https://<your-web-domain>.up.railway.app

# 1) Health — demo_mode true, no missing env vars, qdrant reachable
curl -s "$API_URL/health" | python -m json.tool
#   expect: "status": "ok", "demo_mode": true, "missing_env_vars": [], "qdrant_ok": true

# 2) Canned examples — the Playground defaults (empty list pre-bootstrap)
curl -s "$API_URL/demo/examples" | python -m json.tool

# 3) A real live query against the demo corpus (counts against the daily budget)
curl -s -X POST "$API_URL/query" \
  -H 'Content-Type: application/json' \
  -d '{"query":"What does the demo corpus say?","corpus_id":"demo"}' | python -m json.tool
#   expect: a cited answer + a trace (route -> retrieve -> synthesize); NOT a 403/503

# 4) Open the web URL in a browser
echo "$WEB_URL"
```

In the browser, confirm: **Playground** loads with the first canned example shown by default
and live queries return cited answers (until the daily budget/rate cap, then a friendly 429);
**Ablation Lab** shows the committed receipts next to their published anchors; **Corpora**
shows the read-only note (BYO upload is disabled in demo mode).

## 10. Troubleshooting

- **`/health` shows a missing env var, or `/query` returns 503 naming one** — that var is
  unset on the api service. Most common: `QDRANT_URL` (must be the private
  `http://qdrant.railway.internal:6333`) or a vendor key. The 503 always names the missing var
  — never a stack trace, never a silent default. Set it; the api redeploys.
- **Playground shows no examples / Ablation Lab is empty** — the step-0 bootstrap artifacts
  were not committed, so the startup seed/materialize were no-ops (check the api logs for the
  "skipping — run demo-bootstrap" warnings). Run [`demo-bootstrap.md`](demo-bootstrap.md),
  commit `demo/corpus/`, `demo/examples/`, `receipts/`, and redeploy.
- **CORS errors in the browser console** (request blocked, no `Access-Control-Allow-Origin`) —
  `RAGRECEIPTS_CORS_ORIGINS` on the api doesn't match the web origin. Set it to the exact web
  URL with **no trailing slash** (step 8) and redeploy the api.
- **429 on `/query`** — expected once an IP hits `DEMO_RATE_PER_MIN` / `DEMO_RATE_PER_DAY`, or
  the global `DEMO_DAILY_BUDGET_USD` is spent for the day. The response body's `reason` is
  `rate` (with `retry_after_s`) or `budget`. Raise the limits via the env vars if you want a
  livelier demo; the budget resets daily (UTC).
- **The graph example fails / `graph-rrf` route errors** — the `demo/corpus/graph/` artifact
  wasn't committed (bootstrap step 3), so it isn't baked into the image. Re-run the graph step
  of the bootstrap, commit, redeploy. (The non-graph routes still work without it — graph
  retrieval degrades with a disclosed `graph-skipped` flag rather than erroring the whole
  query.)
- **Web shows the wrong api URL / calls localhost** — `NEXT_PUBLIC_API_BASE_URL` is baked at
  **build** time. If you changed the api URL, rebuild the web service (a redeploy that
  re-runs the Docker build), not just restart it.
- **Qdrant data vanished after a redeploy** — confirm the `/qdrant/storage` volume is attached
  (step 4). Even so, the api re-seeds the `demo` collection from the baked `dense_vectors.npz`
  on the next boot, so the demo recovers automatically.
