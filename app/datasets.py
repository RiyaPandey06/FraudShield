"""Dataset adapters. All loaders return the same schema as ``synthetic.make_transactions``:
transaction_id, timestamp (ISO, UTC), account_id, device_id, merchant_id, amount, country,
transaction_type, is_fraud — sorted by time. Nothing is ever split randomly.

IEEE-CIS (Kaggle, licence: competition rules) and PaySim (Kaggle, CC BY-SA 4.0) must be downloaded
by you; set ``DATASET=ieee_cis|paysim`` and ``DATASET_PATH``.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from app.synthetic import BASE_START, make_transactions

IEEE_REFERENCE = datetime(2017, 12, 1, tzinfo=UTC)  # competition's documented TransactionDT origin


def load_paysim(path: str | Path, max_rows: int | None = None) -> list[dict]:
    """PaySim mapping: nameOrig -> account, nameDest -> device *and* merchant link (the beneficiary is the
    shared node that exposes mule accounts), type -> transaction_type. ``step`` is one simulated hour."""
    df = pd.read_csv(path, nrows=max_rows)
    df = df.sort_values("step", kind="stable").reset_index(drop=True)
    within = df.groupby("step").cumcount()
    seconds = df["step"] * 3600 + (within * 3600 / df.groupby("step")["step"].transform("size")).astype(float)
    out = []
    for i, row in enumerate(df.itertuples(index=False)):
        out.append({
            "transaction_id": f"paysim-{i:08d}",
            "timestamp": (BASE_START + timedelta(seconds=float(seconds.iloc[i]))).isoformat(),
            "account_id": str(row.nameOrig), "device_id": f"dest:{row.nameDest}",
            "merchant_id": f"dest:{row.nameDest}", "amount": float(max(row.amount, 0.01)),
            "country": "XX", "transaction_type": str(row.type).lower(), "is_fraud": int(row.isFraud),
        })
    return out


def load_ieee_cis(transaction_csv: str | Path, identity_csv: str | Path | None = None,
                  max_rows: int | None = None) -> list[dict]:
    """IEEE-CIS mapping. The dataset has no account id, so we use the community 'uid' proxy
    (card1 + addr1 + the day the card first appeared). Linkage is therefore limited — which is
    exactly why the README treats it as a tabular benchmark rather than the graph-ablation dataset."""
    df = pd.read_csv(transaction_csv, nrows=max_rows)
    if identity_csv is not None and Path(identity_csv).exists():
        ident = pd.read_csv(identity_csv)[["TransactionID", "DeviceInfo"]]
        df = df.merge(ident, on="TransactionID", how="left")
    else:
        df["DeviceInfo"] = None
    df = df.sort_values("TransactionDT", kind="stable").reset_index(drop=True)
    first_day = (df["TransactionDT"] / 86400 - df.get("D1", 0).fillna(0)).floordiv(1).astype(int)
    out = []
    for i, row in enumerate(df.itertuples(index=False)):
        device = row.DeviceInfo if isinstance(row.DeviceInfo, str) else f"{row.card4}-{row.card6}"
        out.append({
            "transaction_id": f"ieee-{int(row.TransactionID)}",
            "timestamp": (IEEE_REFERENCE + timedelta(seconds=float(row.TransactionDT))).isoformat(),
            "account_id": f"{row.card1}_{row.addr1}_{first_day.iloc[i]}", "device_id": str(device),
            "merchant_id": f"{row.ProductCD}_{row.R_emaildomain}", "amount": float(max(row.TransactionAmt, 0.01)),
            "country": str(row.addr2), "transaction_type": str(row.ProductCD), "is_fraud": int(row.isFraud),
        })
    return out


def load_base_stream(settings) -> list[dict]:
    """The labelled history the model is trained on (and the online state is warmed from)."""
    if settings.dataset == "paysim":
        return load_paysim(settings.dataset_path, settings.train_rows)
    if settings.dataset == "ieee_cis":
        path = Path(settings.dataset_path)
        return load_ieee_cis(path / "train_transaction.csv", path / "train_identity.csv", settings.train_rows)
    return make_transactions(rows=settings.train_rows, seed=42)
