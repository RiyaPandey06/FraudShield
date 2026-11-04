"""Initialise the shared volume: cache the feature history, train + register the champion, write the demo stream."""
from __future__ import annotations

from datetime import UTC, datetime

from app.config import get_settings
from app.database import Database
from app.model import ModelService
from app.services import load_base_frame
from app.synthetic import write_replay_files


def main() -> None:
    settings = get_settings()
    Database(settings.database_path).initialize()
    frame = load_base_frame(settings)
    model = ModelService(settings)
    model.ensure_model(lambda: frame)
    base_end = datetime.fromtimestamp(float(frame["ts"].max()), tz=UTC)
    write_replay_files(settings.data_dir, settings.demo_rows, base_end, settings.demo_drift_start)
    m = model.metadata["metrics"]
    print(f"FraudShield initialised: model={model.metadata['model_version']} backend={model.metadata['backend']} "
          f"PR-AUC={m['pr_auc']} (baseline {m['baseline']['pr_auc']}) data={settings.data_dir}")


if __name__ == "__main__":
    main()
