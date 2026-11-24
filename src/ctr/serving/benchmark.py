"""Measure single-request latency: in-process and over HTTP.

    uv run python -m ctr.serving.benchmark --model-path artifacts/lightgbm

Three numbers per model:
- pandas: predict_proba on a one-row DataFrame in the same process (the batch path)
- fast: predict_records on one impression dict in the same process (the path
  the API uses)
- http: a POST /predict with one impression to a local uvicorn server, sent
  sequentially over one keep-alive connection (adds JSON, validation and the
  network stack, but no real network)

Impressions are synthetic rows in the Avazu format. Latency depends on the
feature pipeline and model, not on which values are sent.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import httpx
import numpy as np

from ctr.sample import make_synthetic
from ctr.schema import CATEGORICAL_COLUMNS, RAW_TIME
from ctr.serving.app import _limit_torch_threads
from ctr.serving.bundle import ModelBundle
from ctr.serving.schema import PredictRequest, to_frame


def percentiles(samples_ms: Sequence[float]) -> dict[str, float]:
    arr = np.asarray(samples_ms)
    return {
        "n": int(arr.size),
        "p50_ms": round(float(np.percentile(arr, 50)), 3),
        "p95_ms": round(float(np.percentile(arr, 95)), 3),
        "p99_ms": round(float(np.percentile(arr, 99)), 3),
        "mean_ms": round(float(arr.mean()), 3),
    }


def sample_impressions(n: int, seed: int = 0) -> list[dict[str, str]]:
    raw = make_synthetic(n_rows=n, seed=seed)
    return raw[[RAW_TIME, *CATEGORICAL_COLUMNS]].to_dict(orient="records")


def _time_calls(call: Callable[[int], object], n: int, warmup: int) -> list[float]:
    for i in range(warmup):
        call(i)
    samples = []
    for i in range(n):
        t0 = time.perf_counter()
        call(i)
        samples.append((time.perf_counter() - t0) * 1000)
    return samples


def bench_model(
    bundle: ModelBundle, impressions: list[dict], n: int, warmup: int = 50
) -> dict[str, dict]:
    """In-process latency of one prediction: via a one-row DataFrame, and via the fast path."""
    _limit_torch_threads()
    validated = [PredictRequest(impressions=[imp]).impressions for imp in impressions]
    frames = [to_frame(v) for v in validated]
    records = [[v[0].model_dump()] for v in validated]
    model, k = bundle.model, len(impressions)
    return {
        "pandas": percentiles(_time_calls(lambda i: model.predict_proba(frames[i % k]), n, warmup)),
        "fast": percentiles(
            _time_calls(lambda i: model.predict_records(records[i % k]), n, warmup)
        ),
    }


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def bench_http(model_path: Path, impressions: list[dict], n: int, warmup: int = 50) -> dict:
    port = _free_port()
    env = {**os.environ, "MODEL_PATH": str(model_path)}
    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "ctr.serving.app:app", "--port", str(port),
         "--log-level", "warning"],
        env=env,
    )  # fmt: skip
    url = f"http://127.0.0.1:{port}"
    try:
        with httpx.Client(base_url=url, timeout=10) as client:
            deadline = time.time() + 60
            while True:
                try:
                    if client.get("/health").status_code == 200:
                        break
                except httpx.TransportError:
                    pass
                if time.time() > deadline or server.poll() is not None:
                    raise RuntimeError("Server did not start")
                time.sleep(0.2)

            def call(i: int) -> None:
                body = {"impressions": [impressions[i % len(impressions)]]}
                client.post("/predict", json=body).raise_for_status()

            return percentiles(_time_calls(call, n, warmup))
    finally:
        server.terminate()
        server.wait(timeout=10)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--model-path", type=Path, nargs="+", required=True)
    parser.add_argument("--requests", type=int, default=1000)
    parser.add_argument("--out", type=Path, default=None, help="optional JSON file for results")
    args = parser.parse_args(argv)

    impressions = sample_impressions(200)
    results = {}
    for path in args.model_path:
        bundle = ModelBundle.load(path)
        results[bundle.name] = {
            **bench_model(bundle, impressions, args.requests),
            "http": bench_http(path, impressions, args.requests),
        }
        print(bundle.name, json.dumps(results[bundle.name]))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
