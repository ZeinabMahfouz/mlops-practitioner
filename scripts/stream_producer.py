import argparse
import time
import uuid
from pathlib import Path

import pandas as pd
import redis

STREAM_NAME = "trip_events"


def build_event(row: pd.Series) -> dict:
    return {
        "event_id": str(uuid.uuid4()),
        "PULocationID": str(row["PULocationID"]),
        "DOLocationID": str(row["DOLocationID"]),
        "trip_distance": str(float(row["trip_distance"])),
        "lpep_pickup_datetime": str(row["lpep_pickup_datetime"]),
        "lpep_dropoff_datetime": str(row["lpep_dropoff_datetime"]),
        "produced_at": str(time.time()),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Replay trip rows onto a Redis Stream."
    )
    parser.add_argument("parquet_path", type=Path)
    parser.add_argument(
        "--n", type=int, default=200, help="Number of events to produce"
    )
    parser.add_argument("--rate", type=float, default=20.0, help="Events per second")
    parser.add_argument(
        "--poison-every",
        type=int,
        default=0,
        help="Corrupt every Nth event's trip_distance so it can never be scored (0 = disabled)",
    )
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=6379)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)

    df = pd.read_parquet(args.parquet_path)
    df = df.dropna(
        subset=[
            "PULocationID",
            "DOLocationID",
            "trip_distance",
            "lpep_pickup_datetime",
            "lpep_dropoff_datetime",
        ]
    )
    n = min(args.n, len(df))
    df = df.sample(n=n, random_state=42).reset_index(drop=True)

    r = redis.Redis(host=args.host, port=args.port, decode_responses=True)
    delay = 1.0 / args.rate if args.rate > 0 else 0

    poisoned = 0
    for i, row in df.iterrows():
        event = build_event(row)
        if args.poison_every and (i + 1) % args.poison_every == 0:
            event["trip_distance"] = "NOT_A_NUMBER"
            poisoned += 1
            print(f"[{i + 1}/{n}] >>> injecting POISON event {event['event_id']}")
        msg_id = r.xadd(STREAM_NAME, event)
        print(f"[{i + 1}/{n}] produced {msg_id} event_id={event['event_id']}")
        if delay:
            time.sleep(delay)

    print(f"Done. Produced {n} events to stream '{STREAM_NAME}' ({poisoned} poisoned).")


if __name__ == "__main__":
    main()
