from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS transactions (
    transaction_id TEXT PRIMARY KEY, timestamp TEXT NOT NULL, ts REAL NOT NULL, account_id TEXT NOT NULL,
    device_id TEXT NOT NULL, merchant_id TEXT NOT NULL, amount REAL NOT NULL, country TEXT NOT NULL,
    transaction_type TEXT NOT NULL, features_json TEXT NOT NULL, risk_score REAL NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT, transaction_id TEXT NOT NULL UNIQUE, risk_score REAL NOT NULL,
    decision TEXT NOT NULL, threshold REAL NOT NULL, model_version TEXT NOT NULL, reasons_json TEXT NOT NULL,
    graph_json TEXT NOT NULL, explanation TEXT NOT NULL, explanation_source TEXT NOT NULL DEFAULT 'template',
    status TEXT NOT NULL DEFAULT 'open', created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (transaction_id) REFERENCES transactions(transaction_id));
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT, transaction_id TEXT NOT NULL, label TEXT NOT NULL, analyst TEXT NOT NULL,
    note TEXT NOT NULL, review_ms INTEGER, arm TEXT, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS labels (
    transaction_id TEXT PRIMARY KEY, label INTEGER NOT NULL, matured_at_ts REAL NOT NULL);
CREATE TABLE IF NOT EXISTS drift_checks (
    id INTEGER PRIMARY KEY AUTOINCREMENT, stream_ts REAL NOT NULL, n_processed INTEGER NOT NULL,
    results_json TEXT NOT NULL, drifted_json TEXT NOT NULL, triggered INTEGER NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS retrain_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT, trigger TEXT NOT NULL, outcome TEXT NOT NULL, details_json TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP);
CREATE INDEX IF NOT EXISTS idx_alerts_risk ON alerts(risk_score DESC);
CREATE INDEX IF NOT EXISTS idx_tx_ts ON transactions(ts);
"""


class Database:
    """SQLite with one connection per thread (the API and worker are separate processes sharing WAL)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._local = threading.local()

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA busy_timeout=30000")
            self._local.conn = conn
        return conn

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        conn = self._conn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def initialize(self) -> None:
        with self.connection() as conn:
            conn.executescript(SCHEMA)

    # ---- scoring path ----------------------------------------------------
    def save_scored_transaction(self, tx: dict, ts: float, features: dict, score: float, result: dict) -> None:
        with self.connection() as conn:
            conn.execute(
                """INSERT OR IGNORE INTO transactions (transaction_id, timestamp, ts, account_id, device_id,
                merchant_id, amount, country, transaction_type, features_json, risk_score)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (tx["transaction_id"], tx["timestamp"], ts, tx["account_id"], tx["device_id"], tx["merchant_id"],
                 tx["amount"], tx["country"], tx["transaction_type"], json.dumps(features), score))
            if result["decision"] == "alert":
                conn.execute(
                    """INSERT OR IGNORE INTO alerts (transaction_id, risk_score, decision, threshold, model_version,
                    reasons_json, graph_json, explanation) VALUES (?,?,?,?,?,?,?,?)""",
                    (tx["transaction_id"], score, result["decision"], result["threshold"], result["model_version"],
                     json.dumps(result["reasons"]), json.dumps(result["graph_context"]), result["explanation"]))

    def save_labels(self, labels: list[tuple[str, int]], ts: float) -> None:
        if not labels:
            return
        with self.connection() as conn:
            conn.executemany("INSERT OR REPLACE INTO labels(transaction_id, label, matured_at_ts) VALUES (?,?,?)",
                             [(tx_id, label, ts) for tx_id, label in labels])

    # ---- analyst queue -----------------------------------------------------
    def list_alerts(self, limit: int = 50, status: str | None = "open") -> list[dict]:
        query = ("SELECT a.*, t.timestamp, t.account_id, t.device_id, t.merchant_id, t.amount, t.country "
                 "FROM alerts a JOIN transactions t USING(transaction_id)")
        params: list[object] = []
        if status and status != "all":
            query += " WHERE a.status = ?"
            params.append(status)
        query += " ORDER BY a.risk_score DESC, a.created_at DESC LIMIT ?"
        params.append(limit)
        with self.connection() as conn:
            return [self._alert_row(dict(r)) for r in conn.execute(query, params).fetchall()]

    def get_alert(self, transaction_id: str) -> dict | None:
        with self.connection() as conn:
            row = conn.execute(
                """SELECT a.*, t.timestamp, t.account_id, t.device_id, t.merchant_id, t.amount, t.country,
                t.transaction_type, t.features_json FROM alerts a JOIN transactions t USING(transaction_id)
                WHERE a.transaction_id = ?""", (transaction_id,)).fetchone()
        return self._alert_row(dict(row)) if row else None

    def record_feedback(self, transaction_id: str, label: str, analyst: str, note: str,
                        review_ms: int | None = None, arm: str | None = None) -> bool:
        with self.connection() as conn:
            if not conn.execute("SELECT 1 FROM alerts WHERE transaction_id = ?", (transaction_id,)).fetchone():
                return False
            conn.execute("INSERT INTO feedback(transaction_id, label, analyst, note, review_ms, arm) VALUES (?,?,?,?,?,?)",
                         (transaction_id, label, analyst, note, review_ms, arm))
            conn.execute("UPDATE alerts SET status = ? WHERE transaction_id = ?", (label, transaction_id))
        return True

    # ---- LLM explanation upgrade -----------------------------------------------
    def alerts_needing_llm(self, limit: int = 5) -> list[dict]:
        with self.connection() as conn:
            rows = conn.execute(
                """SELECT a.*, t.amount, t.country FROM alerts a JOIN transactions t USING(transaction_id)
                WHERE a.explanation_source = 'template' ORDER BY a.risk_score DESC LIMIT ?""", (limit,)).fetchall()
        return [self._alert_row(dict(r)) for r in rows]

    def update_explanation(self, transaction_id: str, text: str | None, source: str) -> None:
        with self.connection() as conn:
            if text:
                conn.execute("UPDATE alerts SET explanation = ?, explanation_source = ? WHERE transaction_id = ?",
                             (text, source, transaction_id))
            else:  # LLM unavailable/ungrounded: keep the template and stop retrying this alert
                conn.execute("UPDATE alerts SET explanation_source = 'template_final' WHERE transaction_id = ?",
                             (transaction_id,))

    # ---- drift / retrain -----------------------------------------------------
    def record_drift_check(self, stream_ts: float, n_processed: int, results: list[dict], drifted: list[str],
                           triggered: bool) -> None:
        with self.connection() as conn:
            conn.execute("INSERT INTO drift_checks(stream_ts, n_processed, results_json, drifted_json, triggered) "
                         "VALUES (?,?,?,?,?)", (stream_ts, n_processed, json.dumps(results), json.dumps(drifted),
                                                 int(triggered)))

    def latest_drift(self) -> dict | None:
        with self.connection() as conn:
            row = conn.execute("SELECT * FROM drift_checks ORDER BY id DESC LIMIT 1").fetchone()
        if not row:
            return None
        return {"stream_ts": row["stream_ts"], "n_processed": row["n_processed"], "triggered": bool(row["triggered"]),
                "results": json.loads(row["results_json"]), "drifted": json.loads(row["drifted_json"])}

    def drift_history(self, limit: int = 200) -> list[dict]:
        with self.connection() as conn:
            rows = conn.execute("SELECT * FROM drift_checks ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [{"n_processed": r["n_processed"], "triggered": bool(r["triggered"]),
                 "drifted": json.loads(r["drifted_json"]),
                 "max_psi": max((x["psi"] for x in json.loads(r["results_json"])), default=0.0)} for r in rows][::-1]

    def log_retrain(self, trigger: str, outcome: str, details: dict) -> None:
        with self.connection() as conn:
            conn.execute("INSERT INTO retrain_log(trigger, outcome, details_json) VALUES (?,?,?)",
                         (trigger, outcome, json.dumps(details)))

    def retrain_history(self, limit: int = 20) -> list[dict]:
        with self.connection() as conn:
            rows = conn.execute("SELECT * FROM retrain_log ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [{"trigger": r["trigger"], "outcome": r["outcome"], "details": json.loads(r["details_json"]),
                 "created_at": r["created_at"]} for r in rows]

    def labeled_live_rows(self) -> list[dict]:
        """Live transactions with a truth label: matured delayed labels, overridden by analyst feedback."""
        with self.connection() as conn:
            rows = conn.execute(
                """SELECT t.transaction_id, t.ts, t.amount, t.features_json,
                   COALESCE(CASE f.label WHEN 'fraud' THEN 1 WHEN 'legitimate' THEN 0 END, l.label) AS label
                   FROM transactions t
                   LEFT JOIN labels l ON l.transaction_id = t.transaction_id
                   LEFT JOIN (SELECT transaction_id, label FROM feedback WHERE id IN
                              (SELECT MAX(id) FROM feedback GROUP BY transaction_id)) f
                        ON f.transaction_id = t.transaction_id
                   WHERE COALESCE(CASE f.label WHEN 'fraud' THEN 1 WHEN 'legitimate' THEN 0 END, l.label) IS NOT NULL
                   ORDER BY t.ts""").fetchall()
        return [{"transaction_id": r["transaction_id"], "ts": r["ts"], "label": int(r["label"]),
                 "amount": r["amount"], "scenario": "live", **json.loads(r["features_json"])} for r in rows]

    def review_rows(self) -> list[dict]:
        with self.connection() as conn:
            rows = conn.execute("SELECT arm, review_ms FROM feedback WHERE review_ms IS NOT NULL AND arm IS NOT NULL").fetchall()
        return [dict(r) for r in rows]

    def summary(self) -> dict:
        with self.connection() as conn:
            one = lambda q: conn.execute(q).fetchone()[0]  # noqa: E731
            return {
                "transactions_processed": one("SELECT COUNT(*) FROM transactions"),
                "alerts_created": one("SELECT COUNT(*) FROM alerts"),
                "open_alerts": one("SELECT COUNT(*) FROM alerts WHERE status = 'open'"),
                "feedback_count": one("SELECT COUNT(*) FROM feedback"),
                "labels_matured": one("SELECT COUNT(*) FROM labels"),
                "llm_explanations": one("SELECT COUNT(*) FROM alerts WHERE explanation_source = 'llm'"),
            }

    @staticmethod
    def _alert_row(row: dict) -> dict:
        for key in ("reasons_json", "graph_json", "features_json"):
            if key in row:
                row[key.removesuffix("_json")] = json.loads(row.pop(key))
        return row
