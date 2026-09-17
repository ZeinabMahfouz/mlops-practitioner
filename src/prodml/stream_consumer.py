from __future__ import annotations

import argparse
import logging
import pickle
import time
from pathlib import Path

import pandas as pd
import redis
import xgboost as xgb

from prodml.config import settings
from prodml.features import engineer_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

STREAM_NAME = "trip_events"
GROUP_NAME = "scoring_group"
OUTPUT_STREAM = "trip_predictions"
DLQ_STREAM = "trip_events_dlq"
MAX_DELIVERIES = 3
CLAIM_IDLE_MS = 30_000
BLOCK_MS = 5_000
BATCH_SIZE = 10


def load_model(model_path=None):
    path = Path(model_path) if model_path is not None else settings.MODEL_PATH
    with open(path, "rb") as f:
        artifact = pickle.load(f)
    return artifact["model"], artifact["dv"]


def ensure_group(r: redis.Redis) -> None:
    try:
        r.xgroup_create(name=STREAM_NAME, groupname=GROUP_NAME, id="0", mkstream=True)
        logger.info(
            "Created consumer group '%s' on stream '%s'", GROUP_NAME, STREAM_NAME
        )
    except redis.ResponseError as e:
        if "BUSYGROUP" in str(e):
            logger.info("Consumer group '%s' already exists", GROUP_NAME)
        else:
            raise


def score_event(fields: dict, booster: xgb.Booster, dv) -> float:
    row = {
        "PULocationID": fields["PULocationID"],
        "DOLocationID": fields["DOLocationID"],
        "trip_distance": float(fields["trip_distance"]),
        "lpep_pickup_datetime": pd.to_datetime(fields["lpep_pickup_datetime"]),
        "lpep_dropoff_datetime": pd.to_datetime(fields["lpep_dropoff_datetime"]),
    }
    df = pd.DataFrame([row])
    df = engineer_features(df)
    if df.empty:
        raise ValueError(
            "event failed feature-engineering validation (duration/distance out of range)"
        )

    dicts = df[["PU_DO", "trip_distance"]].to_dict(orient="records")
    X = dv.transform(dicts)
    dmatrix = xgb.DMatrix(X)
    pred = booster.predict(dmatrix)[0]
    return float(pred)


def handle_message(
    r: redis.Redis,
    msg_id: str,
    fields: dict,
    booster: xgb.Booster,
    dv,
    consumer_name: str,
    latencies: list,
    stats: dict,
) -> None:
    try:
        predicted_duration = score_event(fields, booster, dv)
        scored_at = time.time()
        produced_at = float(fields["produced_at"])
        latency_ms = (scored_at - produced_at) * 1000

        r.xadd(
            OUTPUT_STREAM,
            {
                "event_id": fields.get("event_id", "unknown"),
                "predicted_duration": str(predicted_duration),
                "produced_at": fields["produced_at"],
                "scored_at": str(scored_at),
                "latency_ms": str(latency_ms),
                "consumer": consumer_name,
            },
        )
        r.xack(STREAM_NAME, GROUP_NAME, msg_id)
        latencies.append(latency_ms)
        stats["scored"] += 1
        logger.info(
            "[%s] scored %s -> %.2f min (latency %.1f ms)",
            consumer_name,
            fields.get("event_id", msg_id),
            predicted_duration,
            latency_ms,
        )
    except Exception as exc:
        pending = r.xpending_range(
            STREAM_NAME, GROUP_NAME, min=msg_id, max=msg_id, count=1
        )
        times_delivered = pending[0]["times_delivered"] if pending else 1

        permanent = isinstance(exc, (ValueError, KeyError, TypeError))
        if permanent or times_delivered >= MAX_DELIVERIES:
            r.xadd(
                DLQ_STREAM,
                {
                    **{k: str(v) for k, v in fields.items()},
                    "error": str(exc),
                    "times_delivered": str(times_delivered),
                },
            )
            r.xack(STREAM_NAME, GROUP_NAME, msg_id)
            stats["dead_lettered"] += 1
            logger.warning(
                "[%s] DEAD-LETTERED %s after %s attempt(s): %s",
                consumer_name,
                fields.get("event_id", msg_id),
                times_delivered,
                exc,
            )
        else:
            stats["retried"] += 1
            logger.warning(
                "[%s] transient failure on %s (attempt %s), leaving unacked for retry: %s",
                consumer_name,
                fields.get("event_id", msg_id),
                times_delivered,
                exc,
            )


def run_consumer(consumer_name: str, max_events, model_path=None) -> dict:
    r = redis.Redis(host="localhost", port=6379, decode_responses=True)
    ensure_group(r)
    booster, dv = load_model(model_path)

    latencies: list = []
    stats = {"scored": 0, "dead_lettered": 0, "retried": 0}
    start = time.perf_counter()

    while max_events is None or stats["scored"] + stats["dead_lettered"] < max_events:
        try:
            _, claimed, _ = r.xautoclaim(
                STREAM_NAME,
                GROUP_NAME,
                consumer_name,
                min_idle_time=CLAIM_IDLE_MS,
                start_id="0-0",
                count=BATCH_SIZE,
            )
        except redis.RedisError as e:
            logger.debug("xautoclaim transient error (%s), will retry next loop", e)
            claimed = []
        for msg_id, fields in claimed:
            handle_message(
                r, msg_id, fields, booster, dv, consumer_name, latencies, stats
            )

        try:
            resp = r.xreadgroup(
                GROUP_NAME,
                consumer_name,
                {STREAM_NAME: ">"},
                count=BATCH_SIZE,
                block=BLOCK_MS,
            )
        except redis.RedisError as e:
            # redis-py's blocking read can intermittently raise a client-side
            # socket timeout here instead of returning empty -- harmless,
            # just means "no new messages this cycle," so loop and try again.
            logger.debug("xreadgroup transient error (%s), will retry next loop", e)
            continue
        if not resp:
            continue
        for _stream_name, messages in resp:
            for msg_id, fields in messages:
                handle_message(
                    r, msg_id, fields, booster, dv, consumer_name, latencies, stats
                )

    elapsed = time.perf_counter() - start
    latencies.sort()
    n = len(latencies)
    summary = {
        "consumer_name": consumer_name,
        "events_scored": stats["scored"],
        "events_dead_lettered": stats["dead_lettered"],
        "elapsed_seconds": elapsed,
        "p50_latency_ms": latencies[n // 2] if n else None,
        "p95_latency_ms": latencies[int(n * 0.95)] if n else None,
        "max_latency_ms": latencies[-1] if n else None,
        "mean_latency_ms": sum(latencies) / n if n else None,
    }
    logger.info("=" * 60)
    logger.info("STREAMING CONSUMER SUMMARY (%s)", consumer_name)
    logger.info("  Events scored:        %s", summary["events_scored"])
    logger.info("  Events dead-lettered: %s", summary["events_dead_lettered"])
    logger.info("  Elapsed:              %.2fs", elapsed)
    logger.info("  Mean latency:         %.1f ms", summary["mean_latency_ms"] or 0)
    logger.info("  p50 latency:          %.1f ms", summary["p50_latency_ms"] or 0)
    logger.info("  p95 latency:          %.1f ms", summary["p95_latency_ms"] or 0)
    logger.info("  Max latency:          %.1f ms", summary["max_latency_ms"] or 0)
    logger.info("=" * 60)
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Score trip events from a Redis Stream."
    )
    parser.add_argument("--consumer-name", default="consumer-1")
    parser.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Stop after scoring/dead-lettering this many events (default: run forever)",
    )
    parser.add_argument("--model-path", default=None)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    run_consumer(args.consumer_name, args.max_events, args.model_path)


if __name__ == "__main__":
    main()
