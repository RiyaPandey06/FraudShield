"""Latency / throughput benchmarks.

Two modes:
  * in-process (default): drives ServiceBundle.process() directly -> isolates features + model + SQLite cost
  * http: drives a running API (POST /api/score) with N concurrent client threads (stdlib only)
Numbers depend on the machine; always report the hardware next to them.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import tempfile
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

from app.config import get_settings
from app.services import ServiceBundle
from app.synthetic import make_transactions, strip_labels


def _percentiles(values_ms: list[float]) -> dict[str, float]:
    arr = np.asarray(values_ms)
    return {"p50_ms": round(float(np.percentile(arr, 50)), 3), "p95_ms": round(float(np.percentile(arr, 95)), 3),
            "p99_ms": round(float(np.percentile(arr, 99)), 3), "mean_ms": round(float(arr.mean()), 3)}


def in_process(requests: int = 3000, train_rows: int = 40_000) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        settings = replace(get_settings(), data_dir=Path(tmp), train_rows=train_rows, auto_retrain=False,
                           drift_check_every=10**9)
        bundle = ServiceBundle.create(settings, label_feed={})
        base_end = max(bundle.state.features.last_ts.values())
        start = datetime.fromtimestamp(base_end, tz=UTC) + timedelta(hours=1)
        stream = [strip_labels(t) for t in make_transactions(requests + 200, seed=99, world_seed=42, start=start,
                                                              id_prefix="bench", drift=None)]
        for tx in stream[:200]:       # warm-up (JIT, caches) excluded from the measurement
            bundle.process(tx)
        totals, parts = [], {"features": [], "model": [], "database": []}
        t0 = time.perf_counter()
        for tx in stream[200:]:
            r = bundle.process(tx)
            totals.append(r["latency_ms"])
            for k in parts:
                parts[k].append(r["timing_ms"][k])
        elapsed = time.perf_counter() - t0
        return {"mode": "in-process", "requests": requests, "throughput_tps": round(requests / elapsed, 1),
                "end_to_end": _percentiles(totals), **{k: _percentiles(v) for k, v in parts.items()},
                "backend": bundle.model.metadata["backend"], "explainer": bundle.model.metadata["explainer"],
                "hardware": {"cpu_count": os.cpu_count(), "python": platform.python_version(),
                             "platform": platform.platform()}}


def http(url: str, requests: int = 3000, concurrency: int = 16) -> dict:
    """Client-side latency against a running API. Transactions are generated, not sent as JSON from disk."""
    stream = [strip_labels(t) for t in make_transactions(requests, seed=123, world_seed=42, id_prefix="http")]
    for i, tx in enumerate(stream):
        tx["transaction_id"] = f"http-{int(time.time())}-{i}"
    latencies, errors, lock = [], [0], threading.Lock()

    def post(tx: dict) -> None:
        body = json.dumps(tx).encode()
        req = urllib.request.Request(f"{url}/api/score", data=body, headers={"Content-Type": "application/json"})
        t = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                resp.read()
            ms = (time.perf_counter() - t) * 1000
            with lock:
                latencies.append(ms)
        except Exception:  # noqa: BLE001
            with lock:
                errors[0] += 1

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        list(pool.map(post, stream))
    elapsed = time.perf_counter() - t0
    return {"mode": "http", "url": url, "requests": requests, "concurrency": concurrency, "errors": errors[0],
            "throughput_tps": round(len(latencies) / elapsed, 1), "end_to_end": _percentiles(latencies or [0.0])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["in-process", "http"], default="in-process")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--requests", type=int, default=3000)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--train-rows", type=int, default=40_000)
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    result = in_process(args.requests, args.train_rows) if args.mode == "in-process" else http(args.url, args.requests, args.concurrency)
    print(json.dumps(result, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
