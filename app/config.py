from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _flag(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    threshold: float | None          # None -> use the cost-tuned threshold stored with the model
    kafka_servers: str
    mlflow_uri: str | None
    ollama_url: str | None
    ollama_model: str
    label_delay_seconds: float
    train_rows: int
    dataset: str                     # synthetic | paysim | ieee_cis
    dataset_path: str | None
    drift_check_every: int
    drift_window: int
    drift_min_rows: int
    drift_psi_threshold: float
    drift_min_features: int
    drift_cooldown_rows: int
    auto_retrain: bool
    retrain_min_labels: int
    demo_rows: int
    demo_drift_start: int

    @property
    def database_path(self) -> Path:
        return self.data_dir / "fraudshield.db"

    @property
    def models_dir(self) -> Path:
        path = self.data_dir / "models"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def champion_path(self) -> Path:
        return self.data_dir / "champion.json"

    @property
    def base_frame_path(self) -> Path:
        return self.data_dir / "base_features.joblib"


def get_settings() -> Settings:
    data_dir = Path(os.getenv("DATA_DIR", "data")).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    threshold = os.getenv("FRAUD_THRESHOLD")
    return Settings(
        data_dir=data_dir,
        threshold=float(threshold) if threshold else None,
        kafka_servers=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:19092"),
        mlflow_uri=os.getenv("MLFLOW_TRACKING_URI") or None,
        ollama_url=os.getenv("OLLAMA_URL") or None,
        ollama_model=os.getenv("OLLAMA_MODEL", "llama3.2:3b"),
        label_delay_seconds=float(os.getenv("LABEL_DELAY_HOURS", "4")) * 3600.0,
        train_rows=int(os.getenv("TRAIN_ROWS", "40000")),
        dataset=os.getenv("DATASET", "synthetic"),
        dataset_path=os.getenv("DATASET_PATH") or None,
        drift_check_every=int(os.getenv("DRIFT_CHECK_EVERY", "50")),
        drift_window=int(os.getenv("DRIFT_WINDOW", "300")),
        drift_min_rows=int(os.getenv("DRIFT_MIN_ROWS", "200")),
        drift_psi_threshold=float(os.getenv("DRIFT_PSI_THRESHOLD", "0.2")),
        drift_min_features=int(os.getenv("DRIFT_MIN_FEATURES", "2")),
        drift_cooldown_rows=int(os.getenv("DRIFT_COOLDOWN_ROWS", "600")),
        auto_retrain=_flag("AUTO_RETRAIN", "true"),
        retrain_min_labels=int(os.getenv("RETRAIN_MIN_LABELS", "150")),
        demo_rows=int(os.getenv("DEMO_ROWS", "1500")),
        demo_drift_start=int(os.getenv("DEMO_DRIFT_START", "500")),
    )
