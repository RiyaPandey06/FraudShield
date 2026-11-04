"""Causal behavioural features. Every value is computed from events strictly before the current one."""
from __future__ import annotations

from collections import defaultdict, deque
from datetime import UTC, datetime
from statistics import fmean, pstdev

import numpy as np

TABULAR_FEATURES = [
    "amount_log", "amount_zscore", "amount_vs_max", "velocity_10m", "velocity_1h",
    "seconds_since_last_log", "account_tx_count_log", "merchant_frequency", "merchant_new_for_account",
    "new_device", "device_account_count", "country_new", "hour_sin", "hour_cos", "is_transfer",
]
GRAPH_FEATURES = [
    "graph_degree", "account_risk", "device_risk", "merchant_risk", "neighbor_fraud_accounts",
    "community_size", "community_risk", "ring_flag", "mule_flag",
]
FEATURE_NAMES = TABULAR_FEATURES + GRAPH_FEATURES

REASON_LABELS = {
    "amount_log": "unusually large payment amount",
    "amount_zscore": "amount is far from this account's normal pattern",
    "amount_vs_max": "amount exceeds anything this account has spent before",
    "velocity_10m": "burst of transactions in the last 10 minutes",
    "velocity_1h": "rapid transaction velocity in the last hour",
    "seconds_since_last_log": "very short gap since the previous transaction",
    "account_tx_count_log": "little or no history for this account",
    "merchant_frequency": "unusual merchant activity in the last 24 hours",
    "merchant_new_for_account": "first time this account pays this merchant",
    "new_device": "new device for this account",
    "device_account_count": "device is shared by several accounts",
    "country_new": "country never used before by this account",
    "hour_sin": "unusual time of day", "hour_cos": "unusual time of day",
    "is_transfer": "transfer transaction pattern",
    "graph_degree": "account uses many different devices",
    "account_risk": "account has previously confirmed fraud",
    "device_risk": "device was involved in previously confirmed fraud",
    "merchant_risk": "merchant has an elevated confirmed-fraud rate",
    "neighbor_fraud_accounts": "accounts linked through shared devices have confirmed fraud",
    "community_size": "account sits in a large connected identity cluster",
    "community_risk": "connected cluster contains confirmed-fraud accounts",
    "ring_flag": "account belongs to a suspected fraud ring",
    "mule_flag": "device looks like a mule device (shared and previously abused)",
}


def to_epoch(timestamp: str) -> float:
    value = datetime.fromisoformat(timestamp)
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.timestamp()


class FeatureService:
    """Rolling per-account / per-merchant / per-device state."""

    def __init__(self) -> None:
        self.events: dict[str, deque[float]] = defaultdict(deque)       # account -> recent timestamps (<=1h)
        self.amounts: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=100))
        self.max_amount: dict[str, float] = defaultdict(float)
        self.last_ts: dict[str, float] = {}
        self.tx_count: dict[str, int] = defaultdict(int)
        self.merchant_events: dict[str, deque[float]] = defaultdict(deque)   # rolling 24h (stationary)
        self.account_merchants: dict[str, set[str]] = defaultdict(set)
        self.account_countries: dict[str, set[str]] = defaultdict(set)
        self.account_devices: dict[str, set[str]] = defaultdict(set)
        self.device_accounts: dict[str, set[str]] = defaultdict(set)

    def compute(self, tx: dict, ts: float, ctx: dict) -> dict[str, float]:
        account, device, merchant = tx["account_id"], tx["device_id"], tx["merchant_id"]
        events = self.events[account]
        while events and events[0] < ts - 3600:
            events.popleft()
        previous = self.amounts[account]
        history = self.tx_count[account] > 0
        if len(previous) >= 3:
            mean, std = fmean(previous), pstdev(previous)
            zscore = (tx["amount"] - mean) / max(std, 1.0, mean * 0.1)
        else:
            zscore = 0.0
        m_events = self.merchant_events[merchant]
        while m_events and m_events[0] < ts - 86400:
            m_events.popleft()
        hour = (ts % 86400) / 3600.0
        gap = ts - self.last_ts[account] if account in self.last_ts else 86400.0
        return {
            "amount_log": float(np.log1p(tx["amount"])),
            "amount_zscore": float(zscore),
            "amount_vs_max": float(tx["amount"] / self.max_amount[account]) if self.max_amount[account] else 1.0,
            "velocity_10m": float(sum(1 for t in events if t >= ts - 600)),
            "velocity_1h": float(len(events)),
            "seconds_since_last_log": float(np.log1p(max(gap, 0.0))),
            "account_tx_count_log": float(np.log1p(self.tx_count[account])),
            "merchant_frequency": float(len(m_events)),
            "merchant_new_for_account": float(history and merchant not in self.account_merchants[account]),
            "new_device": float(history and device not in self.account_devices[account]),
            "device_account_count": float(len(self.device_accounts[device] - {account})),
            "country_new": float(history and tx["country"] not in self.account_countries[account]),
            "hour_sin": float(np.sin(2 * np.pi * hour / 24)),
            "hour_cos": float(np.cos(2 * np.pi * hour / 24)),
            "is_transfer": float(tx["transaction_type"] == "transfer"),
            **{name: float(ctx[name]) for name in GRAPH_FEATURES},
        }

    def observe(self, tx: dict, ts: float) -> None:
        account = tx["account_id"]
        self.events[account].append(ts)
        self.amounts[account].append(tx["amount"])
        self.max_amount[account] = max(self.max_amount[account], tx["amount"])
        self.last_ts[account] = ts
        self.tx_count[account] += 1
        self.merchant_events[tx["merchant_id"]].append(ts)
        self.account_merchants[account].add(tx["merchant_id"])
        self.account_countries[account].add(tx["country"])
        self.account_devices[account].add(tx["device_id"])
        self.device_accounts[tx["device_id"]].add(account)
