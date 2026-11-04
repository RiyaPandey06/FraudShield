from __future__ import annotations

import argparse
import json
import time

from kafka import KafkaProducer

from app.config import get_settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay the deterministic FraudShield demo stream")
    parser.add_argument("--speed", type=float, default=0.05, help="Seconds between events (0 = as fast as possible)")
    parser.add_argument("--limit", type=int, default=0, help="Stop after N events (0 = all)")
    args = parser.parse_args()
    settings = get_settings()
    producer = KafkaProducer(bootstrap_servers=settings.kafka_servers.split(","),
                             value_serializer=lambda v: json.dumps(v).encode("utf-8"), retries=12)
    source = settings.data_dir / "demo_transactions.jsonl"
    sent = 0
    with source.open(encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            producer.send("transactions", key=payload["transaction_id"].encode("utf-8"), value=payload)
            sent += 1
            if args.limit and sent >= args.limit:
                break
            if args.speed:
                time.sleep(args.speed)
    producer.flush()
    print(f"Replay complete: {sent} events from {source}")


if __name__ == "__main__":
    main()
