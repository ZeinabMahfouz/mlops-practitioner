import argparse
import logging
import resource
import sys
import time
from pathlib import Path

import pandas as pd
import xgboost as xgb

from prodml.features import engineer_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CHUNK_SIZE = 50_000

ASSUMED_HOURLY_RATE_USD = 0.096


def load_model(model_path=None):
    import pickle

    from prodml.config import settings

    path = Path(model_path) if model_path is not None else settings.MODEL_PATH
    with open(path, "rb") as f:
        artifact = pickle.load(f)
    return artifact["model"], artifact["dv"]


def score_chunk(df: pd.DataFrame, booster: xgb.Booster, dv) -> pd.DataFrame:
    df = engineer_features(df)
    if df.empty:
        return df

    dicts = df[["PU_DO", "trip_distance"]].to_dict(orient="records")
    X = dv.transform(dicts)
    dmatrix = xgb.DMatrix(X)
    preds = booster.predict(dmatrix)

    df = df.copy()
    df["predicted_duration"] = preds
    df["pickup_date"] = df["lpep_pickup_datetime"].dt.date.astype(str)
    return df


def run_batch_scoring(input_dir: Path, output_dir: Path, model_path=None) -> dict:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_files = sorted(input_dir.glob("*.parquet"))
    if not input_files:
        raise FileNotFoundError(f"No parquet files found in {input_dir}")

    load_start = time.perf_counter()
    booster, dv = load_model(model_path)
    load_elapsed = time.perf_counter() - load_start
    logger.info("Model loaded in %.3fs", load_elapsed)

    total_rows = 0
    start = time.perf_counter()

    for file_path in input_files:
        logger.info("Scoring %s", file_path.name)
        df = pd.read_parquet(file_path)
        for start_idx in range(0, len(df), CHUNK_SIZE):
            chunk = df.iloc[start_idx : start_idx + CHUNK_SIZE]
            scored = score_chunk(chunk, booster, dv)
            if scored.empty:
                continue
            scored.to_parquet(
                output_dir,
                partition_cols=["pickup_date"],
                index=False,
                engine="pyarrow",
            )
            total_rows += len(scored)

    elapsed = time.perf_counter() - start
    # ru_maxrss: peak resident set size of this process since it started,
    # in KB on Linux (this project runs on Linux; it's bytes on macOS).
    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

    rows_per_sec = total_rows / elapsed if elapsed > 0 else 0.0
    instance_hours_per_million = (
        (1_000_000 / rows_per_sec) / 3600 if rows_per_sec > 0 else float("inf")
    )
    cost_per_million = instance_hours_per_million * ASSUMED_HOURLY_RATE_USD

    summary = {
        "total_rows": total_rows,
        "elapsed_seconds": elapsed,
        "rows_per_sec": rows_per_sec,
        "peak_rss_mb": peak_rss_mb,
        "model_load_seconds": load_elapsed,
        "instance_hours_per_million_rows": instance_hours_per_million,
        "assumed_hourly_rate_usd": ASSUMED_HOURLY_RATE_USD,
        "cost_per_million_predictions_usd": cost_per_million,
    }

    logger.info("=" * 60)
    logger.info("BATCH SCORING SUMMARY")
    logger.info("  Total rows scored:      %s", f"{total_rows:,}")
    logger.info("  Wall-clock time:        %.2fs", elapsed)
    logger.info("  Throughput:             %.1f rows/sec", rows_per_sec)
    logger.info("  Peak RSS:               %.1f MB", peak_rss_mb)
    logger.info("  Model load time:        %.3fs", load_elapsed)
    logger.info(
        "  Cost / million preds:   $%.5f (at $%.3f/hr, %.4f instance-hours/M rows)",
        cost_per_million,
        ASSUMED_HOURLY_RATE_USD,
        instance_hours_per_million,
    )
    logger.info("=" * 60)

    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Batch-score trip data.")
    parser.add_argument("input_dir", type=Path, help="Directory of input Parquet files")
    parser.add_argument(
        "output_dir", type=Path, help="Directory to write predictions to"
    )
    args, _unknown = parser.parse_known_args(argv)
    return args


def main(argv=None) -> None:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    run_batch_scoring(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
