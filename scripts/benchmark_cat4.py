import argparse
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from prodml.features import engineer_features

MODELS_DIR = Path("models")
ONNX_PATH = MODELS_DIR / "model.onnx"
ONNX_STD_PATH = MODELS_DIR / "model_std.onnx"
OPENVINO_IR_PATH = MODELS_DIR / "openvino_ir" / "model.xml"
DATA_PATH = Path("data/raw/green_tripdata_2024-02.parquet")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="CAT 4 benchmark: ONNX Runtime + OpenVINO")
    p.add_argument("--n-rows", type=int, default=200, help="Rows to score per call")
    p.add_argument(
        "--n-repeat", type=int, default=200, help="Repeats for timing stability"
    )
    p.add_argument("--data-path", type=Path, default=DATA_PATH)
    return p.parse_args(argv)


def load_eager():
    with open(MODELS_DIR / "model.pkl", "rb") as f:
        artifact = pickle.load(f)
    return artifact["model"], artifact["dv"]


def build_batch(dv, n_rows: int, data_path: Path):
    df = pd.read_parquet(data_path)
    df = df.dropna(
        subset=[
            "PULocationID",
            "DOLocationID",
            "trip_distance",
            "lpep_pickup_datetime",
            "lpep_dropoff_datetime",
        ]
    )
    df = df.sample(n=min(n_rows, len(df)), random_state=42).reset_index(drop=True)
    df = engineer_features(df)
    dicts = df[["PU_DO", "trip_distance"]].to_dict(orient="records")
    X = dv.transform(dicts)
    X_dense = (
        X.toarray().astype(np.float32)
        if hasattr(X, "toarray")
        else np.asarray(X, dtype=np.float32)
    )
    return X, X_dense


def align_width(X_dense: np.ndarray, expected: int) -> np.ndarray:
    actual = X_dense.shape[1]
    if actual == expected:
        return X_dense
    kind = "Padding with zeros" if actual < expected else "Truncating"
    print(
        f"  !! width mismatch: current DictVectorizer produces {actual} features, "
        f"model.onnx expects {expected} (stale Module 1 export -- see script docstring). "
        f"{kind} to line up shapes; treat downstream predictions as a mechanics check, "
        f"not a real prediction."
    )
    if actual < expected:
        pad = np.zeros((X_dense.shape[0], expected - actual), dtype=np.float32)
        return np.hstack([X_dense, pad])
    return X_dense[:, :expected]


def time_it(fn, n_repeat: int):
    start = time.perf_counter()
    out = None
    for _ in range(n_repeat):
        out = fn()
    elapsed = time.perf_counter() - start
    return out, elapsed


def report(name: str, elapsed: float, n_repeat: int, n_rows: int) -> float:
    per_call_ms = elapsed / n_repeat * 1000
    per_row_us = per_call_ms * 1000 / n_rows
    print(
        f"{name}: {elapsed:.3f}s for {n_repeat} calls -> {per_call_ms:.4f} ms/call, {per_row_us:.2f} us/row"
    )
    return per_call_ms


def main() -> None:
    args = parse_args()

    booster, dv = load_eager()
    X_sparse, X_dense = build_batch(dv, args.n_rows, args.data_path)
    n_rows = X_dense.shape[0]
    print(f"Scoring {n_rows} real rows x {args.n_repeat} repeats\n")

    # 1. Eager XGBoost
    dmat = xgb.DMatrix(X_sparse)
    eager_preds, eager_elapsed = time_it(lambda: booster.predict(dmat), args.n_repeat)
    eager_ms = report("Eager (XGBoost)", eager_elapsed, args.n_repeat, n_rows)

    # 2. ONNX Runtime, graph optimizations on
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.optimized_model_filepath = str(MODELS_DIR / "model.optimized.onnx")
    ort_session = ort.InferenceSession(
        str(ONNX_PATH), sess_options=so, providers=["CPUExecutionProvider"]
    )
    expected_width = ort_session.get_inputs()[0].shape[1]
    X_aligned = align_width(X_dense, expected_width)
    input_name = ort_session.get_inputs()[0].name
    ort_preds, ort_elapsed = time_it(
        lambda: ort_session.run(None, {input_name: X_aligned})[0], args.n_repeat
    )
    ort_preds = ort_preds.reshape(-1)
    ort_ms = report(
        "ONNX Runtime (graph-optimized, CPUExecutionProvider)",
        ort_elapsed,
        args.n_repeat,
        n_rows,
    )
    print(f"  optimized graph saved to {so.optimized_model_filepath}")

    # 3. OpenVINO, from the rewritten standard-ops graph
    import openvino as ov

    if not ONNX_STD_PATH.exists():
        raise SystemExit(
            f"{ONNX_STD_PATH} not found -- run scripts/convert_onnx_for_openvino.py first."
        )
    core = ov.Core()
    ov_model = core.read_model(str(ONNX_STD_PATH))
    compiled = core.compile_model(ov_model, "CPU")
    output_layer = compiled.output(0)
    ov_preds, ov_elapsed = time_it(
        lambda: compiled([X_aligned])[output_layer], args.n_repeat
    )
    ov_preds = ov_preds.reshape(-1)
    ov_ms = report("OpenVINO runtime (CPU device)", ov_elapsed, args.n_repeat, n_rows)
    OPENVINO_IR_PATH.parent.mkdir(parents=True, exist_ok=True)
    ov.save_model(ov_model, str(OPENVINO_IR_PATH))
    print(f"  IR saved to {OPENVINO_IR_PATH}")
    print(f"  OpenVINO available devices: {core.available_devices}")

    # Accuracy deltas
    print("\n=== Accuracy deltas ===")
    ort_vs_ov = np.abs(ort_preds - ov_preds)
    print(
        f"ONNX Runtime vs OpenVINO (same model_std.onnx weights -- the honest "
        f"same-weights comparison): max abs diff {ort_vs_ov.max():.6f}, "
        f"mean abs diff {ort_vs_ov.mean():.6f}"
    )
    eager_vs_ort = np.abs(eager_preds - ort_preds)
    print(
        f"Eager (XGBoost) vs ONNX Runtime (Ridge weights -- DIFFERENT MODEL, "
        f"not a runtime-conversion accuracy delta): max abs diff {eager_vs_ort.max():.6f}, "
        f"mean abs diff {eager_vs_ort.mean():.6f}"
    )

    print("\n=== Speedup vs eager (mechanism comparison, not same-weights) ===")
    print(f"ONNX Runtime: {eager_ms / ort_ms:.2f}x")
    print(f"OpenVINO:     {eager_ms / ov_ms:.2f}x")


if __name__ == "__main__":
    main()
