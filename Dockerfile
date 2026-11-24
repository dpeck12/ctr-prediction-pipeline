# Prediction service image.
#
#   uv run python -m ctr.export --models lightgbm        # writes artifacts/lightgbm
#   docker build -t ctr-api --build-arg MODEL_DIR=artifacts/lightgbm .
#   docker run -p 8080:8080 ctr-api
#
# The model is baked into the image, so each image is one immutable model
# version. Cloud Run and similar platforms set $PORT; it defaults to 8080.

FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencies first, so code changes don't reinstall them.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-dev --no-install-project

COPY src ./src
RUN uv sync --locked --no-dev

ARG MODEL_DIR=artifacts/lightgbm
COPY ${MODEL_DIR} /app/model

RUN useradd --create-home --uid 10001 app
USER app

ENV PATH="/app/.venv/bin:$PATH" \
    MODEL_PATH=/app/model \
    PORT=8080
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=3s --start-period=20s \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/health')"

CMD ["sh", "-c", "exec uvicorn ctr.serving.app:app --host 0.0.0.0 --port ${PORT}"]
