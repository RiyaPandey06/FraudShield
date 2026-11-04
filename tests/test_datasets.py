from __future__ import annotations

import pandas as pd

from app.datasets import load_ieee_cis, load_paysim
from app.features import to_epoch

SCHEMA = {"transaction_id", "timestamp", "account_id", "device_id", "merchant_id", "amount", "country",
          "transaction_type", "is_fraud"}


def test_paysim_loader_maps_schema_and_orders_by_time(tmp_path):
    path = tmp_path / "paysim.csv"
    pd.DataFrame({"step": [2, 1, 1], "type": ["TRANSFER", "CASH_OUT", "PAYMENT"], "amount": [100.0, 5.0, 7.5],
                  "nameOrig": ["C1", "C2", "C3"], "oldbalanceOrg": [1, 1, 1], "newbalanceOrig": [0, 0, 0],
                  "nameDest": ["M9", "M9", "M8"], "oldbalanceDest": [0, 0, 0], "newbalanceDest": [0, 0, 0],
                  "isFraud": [1, 0, 0], "isFlaggedFraud": [0, 0, 0]}).to_csv(path, index=False)
    rows = load_paysim(path)
    assert all(SCHEMA <= set(r) for r in rows) and len(rows) == 3
    times = [to_epoch(r["timestamp"]) for r in rows]
    assert times == sorted(times) and rows[-1]["is_fraud"] == 1 and rows[0]["device_id"] == "dest:M9"


def test_ieee_loader_maps_schema_and_joins_identity(tmp_path):
    tx, ident = tmp_path / "t.csv", tmp_path / "i.csv"
    pd.DataFrame({"TransactionID": [3, 1, 2], "isFraud": [0, 1, 0], "TransactionDT": [300, 100, 200],
                  "TransactionAmt": [10.0, 99.0, 5.0], "ProductCD": ["W", "C", "W"], "card1": [1111, 1111, 2222],
                  "card4": ["visa"] * 3, "card6": ["debit"] * 3, "addr1": [10, 10, 20], "addr2": [87, 87, 87],
                  "D1": [0.0, 1.0, None], "R_emaildomain": ["gmail.com", None, "yahoo.com"]}).to_csv(tx, index=False)
    pd.DataFrame({"TransactionID": [1], "DeviceInfo": ["Windows"]}).to_csv(ident, index=False)
    rows = load_ieee_cis(tx, ident)
    assert [r["transaction_id"] for r in rows] == ["ieee-1", "ieee-2", "ieee-3"]
    assert all(SCHEMA <= set(r) for r in rows) and rows[0]["device_id"] == "Windows" and rows[0]["is_fraud"] == 1
    assert rows[1]["device_id"] == "visa-debit"          # falls back when identity is missing
