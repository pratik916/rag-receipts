FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

# Dependency layer first for caching
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY . .
RUN uv sync --frozen --no-dev

EXPOSE 8000
# SINGLE worker (load-bearing): JobRunner thread + queue are in-process state.
CMD ["uv", "run", "--no-dev", "python", "-m", "uvicorn", "ragreceipts.server.app:app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
