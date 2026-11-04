"""FR-13 (optional): a graph-neural comparison without PyTorch Geometric.

Model: two-layer, label-free message passing over the account-device graph (SIGN/SGC style), then a neural classifier.
  layer 1: h1(tx) = mean of the *latest tabular feature vector* of every account that shares a device with this account/device
  layer 2: h2(tx) = mean of those neighbours' own layer-1 aggregates
  classifier: MLP on [x, h1, h2, degree].  Everything is causal (neighbours are read before the event is added).
It is compared against strong tabular baselines on the same strict time split, with paired bootstrap CIs.
This is NOT a PyG implementation; it is a transparent numpy/sklearn message-passing model, named as such everywhere.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from app.evaluation import best_cost_threshold, chronological_split, paired_bootstrap, summarise
from app.features import FEATURE_NAMES, TABULAR_FEATURES
from app.model import GradientBoostedModel
from app.pipeline import compute_features
from app.synthetic import make_transactions

DELAY = 4 * 3600.0
MAX_NEIGHBOURS = 25


def message_passing_features(stream: list[dict], tabular: np.ndarray) -> np.ndarray:
    d = tabular.shape[1]
    last_vec: dict[str, np.ndarray] = {}
    last_h1: dict[str, np.ndarray] = {}
    device_accounts: dict[str, dict[str, None]] = defaultdict(dict)
    account_devices: dict[str, set[str]] = defaultdict(set)
    out = np.zeros((len(stream), 2 * d + 1))
    for i, tx in enumerate(stream):
        account, device = tx["account_id"], tx["device_id"]
        neigh: list[str] = []
        for dev in account_devices[account] | {device}:
            neigh.extend(k for k in device_accounts[dev] if k != account)
        neigh = list(dict.fromkeys(neigh))[-MAX_NEIGHBOURS:]
        h1 = np.mean([last_vec[k] for k in neigh], axis=0) if neigh else np.zeros(d)
        h2 = np.mean([last_h1[k] for k in neigh], axis=0) if neigh else np.zeros(d)
        out[i] = np.concatenate([h1, h2, [len(neigh)]])
        last_vec[account], last_h1[account] = tabular[i], h1
        account_devices[account].add(device)
        device_accounts[device].pop(account, None)
        device_accounts[device][account] = None
    return out


class MessagePassingNet:
    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        self.scaler = StandardScaler()
        self.net = MLPClassifier(hidden_layer_sizes=(64, 32), early_stopping=True, max_iter=80, random_state=seed)

    def fit(self, x: np.ndarray, y: np.ndarray) -> MessagePassingNet:
        rng = np.random.default_rng(self.seed)
        pos = np.flatnonzero(y == 1)
        idx = np.concatenate([np.arange(len(y)), rng.choice(pos, size=len(pos) * 6)])   # oversample the minority
        self.net.fit(self.scaler.fit_transform(x[idx]), y[idx])
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self.net.predict_proba(self.scaler.transform(x))[:, 1]


def run(rows: int = 60_000, seeds: int = 3, n_boot: int = 500) -> dict:
    started = time.time()
    stream = make_transactions(rows=rows, seed=42)
    frame, _ = compute_features(stream, DELAY)
    tab = frame[TABULAR_FEATURES].to_numpy(float)
    mp = message_passing_features(stream, tab)
    for j in range(mp.shape[1]):
        frame[f"mp_{j}"] = mp[:, j]
    mp_cols = [f"mp_{j}" for j in range(mp.shape[1])]
    train, val, test = chronological_split(frame, embargo_seconds=DELAY)
    y_tr, y_val, y_te = (a["label"].to_numpy() for a in (train, val, test))
    cols_gnn = TABULAR_FEATURES + mp_cols

    def gbm(cols, seed):
        model = GradientBoostedModel(cols, seed=seed).fit(train, y_tr)
        return model.predict_proba(val), model.predict_proba(test)

    def gnn(seed):
        model = MessagePassingNet(seed).fit(train[cols_gnn].to_numpy(float), y_tr)
        return model.predict_proba(val[cols_gnn].to_numpy(float)), model.predict_proba(test[cols_gnn].to_numpy(float))

    runners = {"gbm_tabular": lambda s: gbm(TABULAR_FEATURES, s), "gbm_tabular_graph": lambda s: gbm(FEATURE_NAMES, s),
               "gbm_tabular_messagepassing": lambda s: gbm(cols_gnn, s), "gnn_messagepassing_mlp": gnn}
    per_model: dict[str, list[dict]] = {k: [] for k in runners}
    first_scores: dict[str, np.ndarray] = {}
    for seed in range(seeds):
        for name, fn in runners.items():
            s_val, s_te = fn(seed)
            thr, _ = best_cost_threshold(y_val, s_val)
            per_model[name].append(summarise(y_te, s_te, thr))
            first_scores.setdefault(name, s_te)
    summary = {name: {k: round(float(np.mean([r[k] for r in rs])), 4) for k in rs[0]} | {
        "pr_auc_std": round(float(np.std([r["pr_auc"] for r in rs])), 4)} for name, rs in per_model.items()}
    boots = {}
    for ref in ("gbm_tabular", "gbm_tabular_graph"):
        boots[f"gnn_vs_{ref}"] = {m: paired_bootstrap(y_te, first_scores[ref], first_scores["gnn_messagepassing_mlp"], m, n_boot)
                                  for m in ("pr_auc", "recall_at_90_precision")}
    return {"rows": rows, "test_rows": len(test), "test_frauds": int(y_te.sum()), "seeds": seeds, "n_boot": n_boot,
            "backend": GradientBoostedModel(TABULAR_FEATURES).backend, "models": summary, "bootstrap": boots,
            "message_passing": "2 layers, mean aggregation over accounts sharing a device, <=25 neighbours, label-free",
            "seconds": round(time.time() - started, 1)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=60_000)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--boot", type=int, default=500)
    parser.add_argument("--out", default="results/gnn.json")
    args = parser.parse_args()
    result = run(args.rows, args.seeds, args.boot)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ("models", "bootstrap", "seconds")}, indent=2))


if __name__ == "__main__":
    main()
