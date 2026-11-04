"""Model training, versioned storage, MLflow registry (best effort) and TreeSHAP reasons."""
from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.config import Settings
from app.evaluation import (
    FN_COST, FP_COST, best_cost_threshold, chronological_split, pr_auc, summarise,
)
from app.features import FEATURE_NAMES, REASON_LABELS, TABULAR_FEATURES

try:  # LightGBM is the production backend; sklearn HistGradientBoosting is a documented fallback.
    import lightgbm as lgb
    HAVE_LIGHTGBM = True
except Exception:  # noqa: BLE001
    lgb = None
    HAVE_LIGHTGBM = False

log = logging.getLogger("fraudshield.model")
REGISTRY_NAME = "fraudshield"


def _sigmoid(x: float | np.ndarray) -> float | np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class GradientBoostedModel:
    """Gradient boosting with a uniform interface: ``fit``, ``predict_proba`` and per-prediction contributions.

    * LightGBM: exact TreeSHAP via ``pred_contrib`` (score is the sigmoid of the contribution sum).
    * Fallback (no LightGBM installed): HistGradientBoosting + *occlusion* attributions, which are
      clearly labelled ``explainer="occlusion"`` and are NOT SHAP values.
    """

    def __init__(self, feature_names: list[str], seed: int = 42, backend: str | None = None) -> None:
        self.feature_names, self.seed = list(feature_names), seed
        self.backend = backend or ("lightgbm" if HAVE_LIGHTGBM else "sklearn_hgb")
        self.estimator: Any = None
        self.ref_vector: np.ndarray | None = None

    @property
    def explainer(self) -> str:
        return "treeshap" if self.backend == "lightgbm" else "occlusion"

    def fit(self, x: pd.DataFrame, y: np.ndarray) -> GradientBoostedModel:
        data = x[self.feature_names].to_numpy(dtype=float)
        self.ref_vector = np.median(data, axis=0)
        if self.backend == "lightgbm":
            imbalance = max((y == 0).sum() / max((y == 1).sum(), 1), 1.0)
            self.estimator = lgb.LGBMClassifier(
                n_estimators=250, num_leaves=31, learning_rate=0.05, subsample=0.85, subsample_freq=1,
                colsample_bytree=0.9, reg_lambda=1.5, scale_pos_weight=imbalance, objective="binary",
                n_jobs=2, random_state=self.seed, verbosity=-1)
        else:
            self.estimator = HistGradientBoostingClassifier(
                max_iter=250, learning_rate=0.06, max_leaf_nodes=31, l2_regularization=1.0,
                class_weight="balanced", random_state=self.seed)
        self.estimator.fit(data, y)
        return self

    def predict_proba(self, x: pd.DataFrame | np.ndarray) -> np.ndarray:
        data = x[self.feature_names].to_numpy(dtype=float) if isinstance(x, pd.DataFrame) else x
        return self.estimator.predict_proba(data)[:, 1]

    def score_with_contributions(self, row: np.ndarray) -> tuple[float, np.ndarray]:
        """Single-row score plus a contribution per feature (log-odds scale)."""
        if self.backend == "lightgbm":
            contrib = self.estimator.booster_.predict(row.reshape(1, -1), pred_contrib=True)[0]
            return float(_sigmoid(contrib.sum())), contrib[:-1]
        batch = np.tile(row, (len(row) + 1, 1))
        for j in range(len(row)):
            batch[j + 1, j] = self.ref_vector[j]
        probs = np.clip(self.estimator.predict_proba(batch)[:, 1], 1e-6, 1 - 1e-6)
        logits = np.log(probs / (1 - probs))
        return float(probs[0]), logits[0] - logits[1:]

    def to_state(self) -> dict:
        return {"backend": self.backend, "seed": self.seed, "feature_names": self.feature_names,
                "estimator": self.estimator, "ref_vector": self.ref_vector}

    @classmethod
    def from_state(cls, state: dict) -> GradientBoostedModel:
        model = cls(state["feature_names"], state["seed"], state["backend"])
        model.estimator, model.ref_vector = state["estimator"], state["ref_vector"]
        return model


def fit_baseline(x: pd.DataFrame, y: np.ndarray, seed: int = 42) -> Pipeline:
    pipe = Pipeline([("scale", StandardScaler()),
                     ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed))])
    pipe.fit(x[TABULAR_FEATURES].to_numpy(dtype=float), y)
    return pipe


def baseline_scores(baseline: Pipeline, x: pd.DataFrame) -> np.ndarray:
    return baseline.predict_proba(x[TABULAR_FEATURES].to_numpy(dtype=float))[:, 1]


class ModelService:
    """Owns the champion model: train -> evaluate -> (gate) -> version on disk -> alias in MLflow."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model: GradientBoostedModel | None = None
        self.metadata: dict[str, Any] = {}
        self._pointer_mtime = 0.0

    # ---- paths ---------------------------------------------------------
    def _dir(self, version: str):
        return self.settings.models_dir / version

    # ---- loading -------------------------------------------------------
    def ensure_model(self, base_frame_provider) -> None:
        if self.load_champion():
            return
        self.train(base_frame_provider(), extra=None, force_promote=True)

    def load_champion(self) -> bool:
        pointer = self.settings.champion_path
        if not pointer.exists():
            return False
        version = json.loads(pointer.read_text())["version"]
        meta_path = self._dir(version) / "metadata.json"
        if not meta_path.exists():
            return False
        metadata = json.loads(meta_path.read_text())
        model, source = None, "local-cache"
        if self.settings.mlflow_uri:
            model = self._load_from_registry(metadata)
            source = "mlflow-registry" if model else "local-cache"
        if model is None:
            model = GradientBoostedModel.from_state(joblib.load(self._dir(version) / "model.joblib"))
        metadata["model_source"] = source
        self.model, self.metadata = model, metadata
        self._pointer_mtime = pointer.stat().st_mtime
        return True

    def reload_if_changed(self) -> bool:
        pointer = self.settings.champion_path
        if pointer.exists() and pointer.stat().st_mtime > self._pointer_mtime:
            return self.load_champion()
        return False

    def _load_from_registry(self, metadata: dict) -> GradientBoostedModel | None:
        try:
            import mlflow
            from mlflow import MlflowClient
            mlflow.set_tracking_uri(self.settings.mlflow_uri)
            version = MlflowClient().get_model_version_by_alias(REGISTRY_NAME, "champion")
            if version.tags.get("fs_version") != metadata["model_version"]:
                return None  # registry and local metadata disagree: use the verified local artefact
            estimator = mlflow.sklearn.load_model(f"models:/{REGISTRY_NAME}@champion")
            return GradientBoostedModel.from_state({
                "backend": metadata["backend"], "seed": metadata["seed"], "feature_names": metadata["feature_names"],
                "estimator": estimator, "ref_vector": np.asarray(metadata["ref_vector"])})
        except Exception as exc:  # noqa: BLE001 - serving must survive a registry outage
            log.warning("MLflow registry unavailable, using local cache: %s", exc)
            return None

    # ---- training ------------------------------------------------------
    def train(self, base: pd.DataFrame, extra: pd.DataFrame | None, force_promote: bool = False,
              seed: int = 42) -> dict[str, Any]:
        """Train a candidate on ``base`` (+ ``extra`` labelled live rows), evaluate on the untouched
        chronological test window, and promote it only if it is not worse than the current champion."""
        frame = base if extra is None or extra.empty else pd.concat([base, extra], ignore_index=True)
        train, val, test = chronological_split(frame, embargo_seconds=self.settings.label_delay_seconds)
        candidate = GradientBoostedModel(FEATURE_NAMES, seed=seed).fit(train, train["label"].to_numpy())
        baseline = fit_baseline(train, train["label"].to_numpy(), seed)
        y_val, y_test = val["label"].to_numpy(), test["label"].to_numpy()
        s_val, s_test = candidate.predict_proba(val), candidate.predict_proba(test)
        threshold, val_cost = best_cost_threshold(y_val, s_val)
        base_threshold, _ = best_cost_threshold(y_val, baseline_scores(baseline, val))
        b_test = baseline_scores(baseline, test)
        champion_pr = None
        if self.model is not None and self.metadata.get("feature_names") == FEATURE_NAMES:
            champion_pr = pr_auc(y_test, self.model.predict_proba(test))
        candidate_pr = pr_auc(y_test, s_test)
        promote = force_promote or champion_pr is None or candidate_pr >= champion_pr - 1e-9
        version = "v" + datetime.now(UTC).strftime("%Y%m%d%H%M%S")
        # Reference = the most recent 40% of training rows: skips the cold-start period where per-account
        # history is still empty, which would otherwise look like drift once state is warm.
        recent = train.iloc[int(len(train) * 0.6):]
        sample = recent.sample(min(2000, len(recent)), random_state=seed)
        metadata = {
            "model_version": version, "backend": candidate.backend, "explainer": candidate.explainer, "seed": seed,
            "feature_names": FEATURE_NAMES, "ref_vector": candidate.ref_vector.tolist(),
            "trained_at": datetime.now(UTC).isoformat(), "label_delay_seconds": self.settings.label_delay_seconds,
            "split": {"scheme": "strict chronological 70/15/15 with label-maturity embargo",
                      "train_rows": len(train), "val_rows": len(val), "test_rows": len(test),
                      "test_frauds": int(y_test.sum()), "live_rows_added": 0 if extra is None else len(extra)},
            "metrics": {
                **summarise(y_test, s_test, threshold),
                "cost_threshold": round(threshold, 6), "validation_cost": val_cost,
                "baseline": {**summarise(y_test, b_test, base_threshold), "name": "logistic_regression_tabular"},
                "fn_cost": FN_COST, "fp_cost": FP_COST,
            },
            "reference": {"features": {name: sample[name].round(6).tolist() for name in FEATURE_NAMES},
                          "scores": candidate.predict_proba(sample).round(6).tolist()},
            "promotion": {"champion_pr_auc_on_new_test": champion_pr, "candidate_pr_auc": candidate_pr,
                          "promoted": bool(promote)},
        }
        out = self._dir(version)
        out.mkdir(parents=True, exist_ok=True)
        joblib.dump(candidate.to_state(), out / "model.joblib")
        (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        registry_version = self._log_mlflow(candidate, metadata)
        if promote:
            self.settings.champion_path.write_text(json.dumps({"version": version}), encoding="utf-8")
            self._set_alias(registry_version)
            metadata["model_source"] = "local-cache"
            self.model, self.metadata = candidate, metadata
            self._pointer_mtime = self.settings.champion_path.stat().st_mtime
        else:
            (out / "REJECTED").write_text("Candidate did not beat the champion on the new test window.\n")
        return metadata

    # ---- MLflow (best effort; never blocks scoring) ----------------------
    def _log_mlflow(self, model: GradientBoostedModel, metadata: dict) -> str | None:
        if not self.settings.mlflow_uri:
            return None
        try:
            import mlflow
            mlflow.set_tracking_uri(self.settings.mlflow_uri)
            mlflow.set_experiment("FraudShield")
            with mlflow.start_run(run_name=metadata["model_version"]):
                mlflow.log_params({"backend": model.backend, "explainer": model.explainer,
                                   "label_delay_s": metadata["label_delay_seconds"], **metadata["split"]})
                flat = {k: v for k, v in metadata["metrics"].items() if isinstance(v, (int, float))}
                mlflow.log_metrics(flat)
                mlflow.log_metrics({f"baseline_{k}": v for k, v in metadata["metrics"]["baseline"].items()
                                    if isinstance(v, (int, float))})
                info = mlflow.sklearn.log_model(model.estimator, "model", registered_model_name=REGISTRY_NAME)
                from mlflow import MlflowClient
                client = MlflowClient()
                latest = max(client.search_model_versions(f"name='{REGISTRY_NAME}'"), key=lambda v: int(v.version))
                client.set_model_version_tag(REGISTRY_NAME, latest.version, "fs_version", metadata["model_version"])
                del info
                return latest.version
        except Exception as exc:  # noqa: BLE001
            log.warning("MLflow logging skipped: %s", exc)
            return None

    def _set_alias(self, registry_version: str | None) -> None:
        if not (self.settings.mlflow_uri and registry_version):
            return
        try:
            from mlflow import MlflowClient
            MlflowClient(self.settings.mlflow_uri).set_registered_model_alias(REGISTRY_NAME, "champion", registry_version)
        except Exception as exc:  # noqa: BLE001
            log.warning("MLflow alias not set: %s", exc)

    # ---- inference -----------------------------------------------------
    def predict(self, features: dict[str, float]) -> tuple[float, list[dict]]:
        assert self.model is not None, "model not loaded"
        row = np.array([features[name] for name in self.model.feature_names], dtype=float)
        score, contributions = self.model.score_with_contributions(row)
        reasons = []
        for index in np.argsort(-np.abs(contributions))[:3]:
            name = self.model.feature_names[int(index)]
            value = float(contributions[index])
            reasons.append({"feature": name, "value": round(float(features[name]), 4),
                            "contribution": round(value, 4),
                            "direction": "increases_risk" if value >= 0 else "reduces_risk",
                            "label": REASON_LABELS.get(name, name.replace("_", " "))})
        return score, reasons
