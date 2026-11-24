"""FastAPI service that scores impressions with a saved model bundle.

    MODEL_PATH=artifacts/lightgbm uv run python -m uvicorn ctr.serving.app:app --port 8080

Endpoints: GET /health, GET /model (metadata), POST /predict. Interactive docs
are at /docs.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request

from ctr import __version__
from ctr.serving.bundle import ModelBundle
from ctr.serving.schema import PredictRequest, PredictResponse

DEFAULT_MODEL_PATH = "artifacts/model"


def _limit_torch_threads() -> None:
    """One thread per request: for single rows, thread start-up costs more than it saves."""
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a dependency, but serving doesn't need it
        return
    torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "1")))


def create_app(model_path: str | Path | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        path = model_path or os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH)
        _limit_torch_threads()
        app.state.bundle = ModelBundle.load(path)
        yield

    app = FastAPI(
        title="CTR prediction",
        version=__version__,
        description="Click probabilities for ad impressions, from a model trained on Avazu.",
        lifespan=lifespan,
    )

    @app.get("/health")
    def health(request: Request) -> dict[str, str]:
        return {"status": "ok", "model": request.app.state.bundle.name}

    @app.get("/model")
    def model_info(request: Request) -> dict:
        return request.app.state.bundle.metadata

    @app.post("/predict")
    def predict(body: PredictRequest, request: Request) -> PredictResponse:
        bundle: ModelBundle = request.app.state.bundle
        records = [imp.model_dump() for imp in body.impressions]
        probabilities = bundle.model.predict_records(records)
        return PredictResponse(model=bundle.name, probabilities=probabilities.tolist())

    return app


app = create_app()
