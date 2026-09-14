"""Step 3 evidence script for two of the four CAT-1 problems.

Problem 1 -- "batch size is always 1": 100 sequential single-prediction
HTTP calls vs one /predict/batch call carrying the same 100 trips,
against the live deployed API.

Problem 3 -- eager runtime vs ONNX Runtime: times N predictions through
the current pickled XGBoost booster (eager) vs models/model.onnx,
locally, no HTTP involved.

NOTE: models/model.onnx is a static artifact from Module 1 -- there's no
export step in this pipeline that regenerates it when the model retrains
(Step 1's weekly DAG has no ONNX conversion task). So this is comparing
runtime *mechanism* overhead (eager Python/XGBoost calls vs a compiled
ONNX Runtime graph), not two exports of the identical model -- worth
saying explicitly in the report rather than implying they're the same
weights. It's also a live example of problem #4: the ONNX artifact isn't
tracked by anything that knows it can drift from the registry model.

Run:
    python scripts/benchmark_cat1.py
"""

import pickle
import random
import time
from pathlib import Path

import httpx
import numpy as np
import xgboost as xgb

API_URL = "http://localhost:8000"
N_SINGLE = 100
N_RUNTIME = 1000
LOCATION_IDS = list(range(1, 265))


def _random_trip():
    return {
        "PULocationID": random.choice(LOCATION_IDS),
        "DOLocationID": random.choice(LOCATION_IDS),
        "trip_distance": round(random.uniform(0.5, 20.0), 2),
    }


def benchmark_batch_size(client: httpx.Client) -> None:
    trips = [_random_trip() for _ in range(N_SINGLE)]

    start = time.perf_counter()
    for trip in trips:
        client.post(f"{API_URL}/predict", json=trip).raise_for_status()
    single_elapsed = time.perf_counter() - start

    start = time.perf_counter()
    client.post(f"{API_URL}/predict/batch", json={"trips": trips}).raise_for_status()
    batch_elapsed = time.perf_counter() - start

    print("\n=== Problem 1: batch size is always 1 ===")
    print(
        f"{N_SINGLE} sequential single /predict calls: {single_elapsed:.3f}s "
        f"({N_SINGLE / single_elapsed:.1f} req/s)"
    )
    print(
        f"1x /predict/batch call with {N_SINGLE} trips:  {batch_elapsed:.3f}s "
        f"({N_SINGLE / batch_elapsed:.1f} req/s)"
    )
    print(f"Speedup: {single_elapsed / batch_elapsed:.2f}x")
    print(
        "Note: /predict/batch still loops predict_one() row-by-row "
        "server-side (see src/prodml/predict.py) -- most of this win is "
        "saved HTTP round-trips, not real vectorized batching. That gap "
        "is what BentoML's Runner + micro-batching (Step 4) closes."
    )


def benchmark_runtime(models_dir: Path) -> None:
    import onnxruntime as ort

    with open(models_dir / "model.pkl", "rb") as f:
        artifact = pickle.load(f)
    model, dv = artifact["model"], artifact["dv"]

    trip = _random_trip()
    features = {
        "PU_DO": f"{trip['PULocationID']}_{trip['DOLocationID']}",
        "trip_distance": trip["trip_distance"],
    }
    X = dv.transform([features])
    X_dense = X.toarray().astype(np.float32) if hasattr(X, "toarray") else X

    session = ort.InferenceSession(str(models_dir / "model.onnx"))
    expected = session.get_inputs()[0].shape[1]
    if X_dense.shape[1] < expected:
        X_onnx = np.pad(X_dense, ((0, 0), (0, expected - X_dense.shape[1])))
    else:
        X_onnx = X_dense[:, :expected]
    input_name = session.get_inputs()[0].name

    dmat = xgb.DMatrix(X)
    start = time.perf_counter()
    for _ in range(N_RUNTIME):
        model.predict(dmat)
    eager_elapsed = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(N_RUNTIME):
        session.run(None, {input_name: X_onnx})
    onnx_elapsed = time.perf_counter() - start

    print("\n=== Problem 3: eager runtime vs ONNX Runtime ===")
    print(
        f"{N_RUNTIME}x eager XGBoost .predict(): {eager_elapsed:.3f}s "
        f"({eager_elapsed / N_RUNTIME * 1000:.3f} ms/pred)"
    )
    print(
        f"{N_RUNTIME}x ONNX Runtime .run():      {onnx_elapsed:.3f}s "
        f"({onnx_elapsed / N_RUNTIME * 1000:.3f} ms/pred)"
    )
    print(f"Speedup: {eager_elapsed / onnx_elapsed:.2f}x")


if __name__ == "__main__":
    with httpx.Client(timeout=30.0) as client:
        benchmark_batch_size(client)
    benchmark_runtime(Path("models"))
