# Built from the REPO ROOT context (see docker-compose.yml / Railway root dir),
# so this Dockerfile can bake the repo-root demo/ corpus + receipts/ into the image.
# The api package layout under /app is identical to a `build: ./api` context:
# /app/ragreceipts/... stays on sys.path and `python -m uvicorn ragreceipts.server.app:app`
# is unchanged.
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# Dependency layer first for caching (paths are repo-root-relative).
COPY api/pyproject.toml api/uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# The api source — contents of api/ land directly in /app, exactly as `build: ./api` did.
COPY api/ ./
RUN uv sync --frozen --no-dev

# Demo corpus + committed receipts. On Railway there are no repo bind-mounts, so these
# MUST be in the image: the startup demo seed/materialize reads /app/demo/corpus and the
# /receipts endpoint reads /app/receipts. (data/ is gitignored and NOT copied — the runtime
# data dir is the mounted volume.)
COPY demo /app/demo
COPY receipts /app/receipts

EXPOSE 8000
# SINGLE worker (load-bearing): JobRunner thread + queue are in-process state.
CMD ["uv", "run", "--no-dev", "python", "-m", "uvicorn", "ragreceipts.server.app:app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
