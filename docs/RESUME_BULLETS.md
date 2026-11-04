# Resume bullets: measured values and what still needs your machine

Use a bullet only after re-running the command next to it on your own hardware (with LightGBM installed) and replacing
the numbers. The values below were measured in a 1-vCPU sandbox on **synthetic data** with the scikit-learn fallback model.

* Built a streaming fraud-detection system (Redpanda, FastAPI, LightGBM) with graph features, TreeSHAP reasons and drift
  monitoring. In-process scoring measured **p50 0.85 ms / p95 1.4 ms, ~600 tx/s** on one vCPU
  (`python -m experiments.loadtest`). *HTTP/Kafka end-to-end numbers: run `--mode http` on your stack.*
* Added account-device graph features (Louvain communities + delayed-label risk propagation): **PR-AUC 0.911 -> 0.940
  (paired-bootstrap 95% CI +0.016 to +0.043) and recall at 90% precision 0.73 -> 0.82** on a strict time split with a label
  embargo, synthetic data (`python -m experiments.ablation`).
* Implemented PSI/KS drift monitoring with a champion-gated retraining path: trigger fired **400 transactions (~5 h) before a
  delayed-label metric showed the drop**, with 0 false alarms in a 236-check control stream (`python -m experiments.drift_experiment`).
* LLM analyst explanations (local, grounded): **review-time reduction not yet measured** - run the dashboard's study mode with
  real reviewers, then `python -m experiments.review_time_report`. Do not publish a percentage until then.
