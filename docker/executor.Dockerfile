FROM ghcr.io/astral-sh/uv:0.9.26-python3.12-bookworm-slim@sha256:add251a8fbfa14ff5e115adf93cb113c7b58c57bc84f0992164f5b6631b3451f

ENV UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/auto-ds/.venv/bin:${PATH}"

WORKDIR /opt/auto-ds
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --only-group executor

WORKDIR /workspace
