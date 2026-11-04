"""Deterministic synthetic payment streams.

The generator is deliberately *not* trivially separable. It contains:

* legitimate households that share a device (a graph confounder),
* account-takeover fraud (``solo``) with overlapping amounts and short bursts,
* visible fraud rings that share devices,
* "stealth" rings whose amounts and geography look legitimate, so only graph
  linkage plus *matured* labels can reveal them,
* optional concept/covariate drift (``DriftSpec``) with a ``low_and_slow`` tactic.

Because the structure is injected by us, any gain measured from graph features on
this data demonstrates the *method*, not real-world performance.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

BASE_START = datetime(2026, 1, 1, tzinfo=UTC)
MEAN_GAP_SECONDS = 45.0
HOME_COUNTRIES = ["IN", "US", "GB", "DE", "SG"]
HOME_P = [0.60, 0.15, 0.10, 0.08, 0.07]
FRAUD_KINDS = ["solo", "ring", "stealth"]
FRAUD_P = [0.40, 0.35, 0.25]
ROWS_PER_FRAUD_EVENT = 1.55  # average rows produced by one fraud event (bursts included)


@dataclass(frozen=True)
class DriftSpec:
    """Gradual shift that starts at ``start_index`` and reaches full strength after ``ramp_rows``."""

    start_index: int
    ramp_rows: int = 2000
    amount_inflation: float = 0.6   # covariate: legitimate amounts grow by up to +60%
    new_country_share: float = 0.25  # covariate: legit traffic from unseen countries (travel)
    concept_shift: float = 1.0      # share of fraud that becomes ``low_and_slow`` at full ramp

    def progress(self, index: int) -> float:
        return float(np.clip((index - self.start_index) / max(self.ramp_rows, 1), 0.0, 1.0))


class _World:
    """Static population: accounts, devices, households and fraud rings."""

    def __init__(self, seed: int) -> None:
        r = np.random.default_rng(seed + 10_007)
        self.accounts = [f"acct-{i:05d}" for i in range(3000)]
        self.devices = [f"dev-{i:05d}" for i in range(4200)]
        self.merchants = [f"m-{i:03d}" for i in range(300)]
        self.ring_merchants = self.merchants[:6]
        self.home = {a: HOME_COUNTRIES[int(r.choice(len(HOME_COUNTRIES), p=HOME_P))] for a in self.accounts}
        self.primary = {a: self.devices[i] for i, a in enumerate(self.accounts)}
        self.secondary = {a: self.devices[(i + 1500) % 3000] if r.random() < 0.2 else None
                          for i, a in enumerate(self.accounts)}
        # Legitimate households: 2-4 accounts share one device.
        idx = 300
        while idx < 1300:
            size = int(r.integers(2, 5))
            group = self.accounts[idx: idx + size]
            for member in group:
                self.primary[member] = group[0].replace("acct", "dev")
            idx += size
        # Visible rings (8 x 12 accounts) and stealth rings (4 x 15 accounts), each with 2 devices.
        self.rings: list[tuple[list[str], list[str]]] = []
        for k in range(8):
            members = self.accounts[k * 12:(k + 1) * 12]
            self.rings.append((members, [f"dev-r{k}a", f"dev-r{k}b"]))
        self.stealth: list[tuple[list[str], list[str]]] = []
        for k in range(4):
            members = self.accounts[100 + k * 15:100 + (k + 1) * 15]
            self.stealth.append((members, [f"dev-s{k}a", f"dev-s{k}b"]))
        self.ring_of: dict[str, list[str]] = {}
        for members, devs in self.rings + self.stealth:
            for member in members:
                self.ring_of[member] = devs
        self.fresh_devices = self.devices[3300:]
        weights = r.lognormal(0.0, 0.8, size=len(self.accounts))
        self.activity = weights / weights.sum()


def _foreign(rng: np.random.Generator, home: str) -> str:
    return str(rng.choice([c for c in HOME_COUNTRIES + ["NG", "RU", "BR"] if c != home]))


def _legit(rng: np.random.Generator, w: _World, progress: float, drift: DriftSpec | None) -> dict:
    account = w.accounts[int(rng.choice(len(w.accounts), p=w.activity))]
    device = w.primary[account]
    roll = rng.random()
    if account in w.ring_of and roll < 0.35:
        device = str(rng.choice(w.ring_of[account]))          # mule-farm devices also see legit use
    elif w.secondary[account] and roll < 0.30:
        device = w.secondary[account]
    elif roll > 0.99:
        device = str(rng.choice(w.devices[:3000]))
    amount = float(np.exp(rng.normal(4.4, 0.9)))
    country = w.home[account]
    if drift:
        amount *= 1.0 + drift.amount_inflation * progress
        if rng.random() < drift.new_country_share * progress:
            country = _foreign(rng, country)
    return {"account_id": account, "device_id": device, "merchant_id": str(rng.choice(w.merchants)),
            "amount": amount, "country": country,
            "transaction_type": "transfer" if rng.random() < 0.08 else "card_purchase",
            "is_fraud": 0, "scenario": "legit"}


def _fraud(rng: np.random.Generator, w: _World, kind: str) -> list[tuple[float, dict]]:
    """One fraud event: the primary transaction plus optional burst follow-ups."""
    if kind == "solo":
        account = str(rng.choice([a for a in w.accounts[1300:]][:1500]))
        base = {"account_id": account, "device_id": str(rng.choice(w.fresh_devices)),
                "merchant_id": str(rng.choice(w.merchants)), "amount": float(np.exp(rng.normal(5.4, 0.9))),
                "country": _foreign(rng, w.home[account]) if rng.random() < 0.5 else w.home[account],
                "transaction_type": "transfer" if rng.random() < 0.5 else "card_purchase"}
        followups = int(rng.integers(1, 4)) if rng.random() < 0.5 else 0
        burst_amount = (5.0, 0.7)
    elif kind == "ring":
        members, devs = w.rings[int(rng.integers(len(w.rings)))]
        account = str(rng.choice(members))
        base = {"account_id": account, "device_id": str(rng.choice(devs)) if rng.random() < 0.85 else w.primary[account],
                "merchant_id": str(rng.choice(w.ring_merchants)) if rng.random() < 0.4 else str(rng.choice(w.merchants)),
                "amount": float(np.exp(rng.normal(4.9, 0.9))),
                "country": w.home[account] if rng.random() < 0.7 else _foreign(rng, w.home[account]),
                "transaction_type": "transfer" if rng.random() < 0.6 else "card_purchase"}
        followups = int(rng.integers(1, 3)) if rng.random() < 0.2 else 0
        burst_amount = (4.8, 0.6)
    elif kind == "stealth":
        members, devs = w.stealth[int(rng.integers(len(w.stealth)))]
        account = str(rng.choice(members))
        base = {"account_id": account, "device_id": str(rng.choice(devs)) if rng.random() < 0.9 else w.primary[account],
                "merchant_id": str(rng.choice(w.merchants)), "amount": float(np.exp(rng.normal(4.45, 0.9))),
                "country": w.home[account], "transaction_type": "transfer" if rng.random() < 0.1 else "card_purchase"}
        followups = 0
        burst_amount = (4.4, 0.6)
    else:  # low_and_slow: small amounts, rapid repeats on the account's own device
        account = str(rng.choice(w.accounts[1300:]))
        base = {"account_id": account, "device_id": w.primary[account],
                "merchant_id": str(rng.choice(w.merchants[250:270])), "amount": float(np.exp(rng.normal(3.7, 0.5))),
                "country": w.home[account], "transaction_type": "card_purchase"}
        followups = int(rng.integers(3, 6))
        burst_amount = (3.7, 0.5)
    events = [(0.0, {**base, "is_fraud": 1, "scenario": kind})]
    offset = 0.0
    for _ in range(followups):
        offset += float(rng.uniform(15, 150))
        follow = {**base, "amount": float(np.exp(rng.normal(*burst_amount))), "is_fraud": 1, "scenario": kind}
        events.append((offset, follow))
    return events


def make_transactions(
    rows: int = 40_000,
    seed: int = 42,
    *,
    start: datetime = BASE_START,
    drift: DriftSpec | None = None,
    id_prefix: str = "tx",
    world_seed: int | None = None,
    fraud_rate: float = 0.035,
) -> list[dict]:
    """Generate ``rows`` time-ordered transactions (labels and scenario included)."""
    world = _World(seed if world_seed is None else world_seed)
    rng = np.random.default_rng(seed)
    primary_p = fraud_rate / ROWS_PER_FRAUD_EVENT
    clock = 0.0
    events: list[tuple[float, dict]] = []
    index = 0
    while len(events) < rows + 50:
        clock += float(rng.exponential(MEAN_GAP_SECONDS))
        progress = drift.progress(index) if drift else 0.0
        if rng.random() < primary_p:
            if drift and rng.random() < drift.concept_shift * progress:
                kind = "low_and_slow"
            else:
                kind = str(rng.choice(FRAUD_KINDS, p=FRAUD_P))
            for offset, tx in _fraud(rng, world, kind):
                events.append((clock + offset, tx))
        else:
            events.append((clock, _legit(rng, world, progress, drift)))
        index += 1
    events.sort(key=lambda item: item[0])
    records = []
    for i, (offset, tx) in enumerate(events[:rows]):
        records.append({
            "transaction_id": f"{id_prefix}-{i:07d}",
            "timestamp": (start + timedelta(seconds=offset)).isoformat(),
            **{**tx, "amount": round(tx["amount"], 2)},
        })
    return records


def strip_labels(tx: dict) -> dict:
    return {k: v for k, v in tx.items() if k not in {"is_fraud", "scenario"}}


def write_replay_files(directory: Path, rows: int, base_end: datetime, drift_start: int,
                       seed: int = 73, world_seed: int = 42) -> tuple[Path, Path]:
    """Write the demo stream (no labels) and its delayed label feed (simulated chargebacks)."""
    directory.mkdir(parents=True, exist_ok=True)
    stream = make_transactions(
        rows=rows, seed=seed, world_seed=world_seed, start=base_end + timedelta(hours=1),
        drift=DriftSpec(start_index=drift_start, ramp_rows=max(rows // 5, 50)), id_prefix="live",
    )
    tx_path, label_path = directory / "demo_transactions.jsonl", directory / "demo_labels.jsonl"
    with tx_path.open("w", encoding="utf-8") as tx_handle, label_path.open("w", encoding="utf-8") as lab_handle:
        for tx in stream:
            tx_handle.write(json.dumps(strip_labels(tx)) + "\n")
            lab_handle.write(json.dumps({"transaction_id": tx["transaction_id"], "label": tx["is_fraud"]}) + "\n")
    return tx_path, label_path
