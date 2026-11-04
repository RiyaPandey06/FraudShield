# Architecture

```mermaid
flowchart LR
    subgraph Ingest
      P[Producer<br/>replays time-ordered stream] -->|transactions topic| K[(Redpanda / Kafka)]
      LF[(Delayed label feed<br/>simulated chargebacks)]
    end
    K --> W
    subgraph Worker["Scoring worker (same code path as API)"]
      W[StreamState.featurize] --> F[Feature service<br/>velocity, amount deviation,<br/>merchant frequency, device links]
      W --> G[Graph service<br/>account-device-merchant graph<br/>Louvain communities<br/>matured-label risk]
      F --> M[Model<br/>LightGBM + TreeSHAP]
      G --> M
      M --> D{score >= cost-tuned<br/>threshold?}
    end
    LF -. released only after label delay .-> G
    D --> DB[(SQLite: transactions, alerts,<br/>feedback, labels, drift, retrains)]
    DB --> X[Explanation upgrader<br/>local LLM via Ollama<br/>grounding check, template fallback]
    X --> DB
    DB --> API[FastAPI<br/>/api/score /alerts /feedback /drift /retrain /metrics]
    API --> UI[React analyst console<br/>queue, case detail, feedback,<br/>review-time study]
    UI -->|fraud / legitimate + review time| DB
    W --> MON[Drift monitor<br/>PSI + KS on inputs and score]
    MON -->|trigger| R[Retrain<br/>base history + matured labels + feedback]
    R --> GATE{candidate >= champion<br/>on new test window?}
    GATE -->|yes| REG[(MLflow registry<br/>alias: champion)]
    GATE -->|no| REJ[keep champion, log rejection]
    REG --> M
    API & W --> PR[Prometheus] --> GR[Grafana]
```

## Design decisions that matter

* **One feature code path.** `app/pipeline.py::StreamState` is used by training, the experiments, the worker and the API.
  `tests/test_core.py::test_train_serve_parity` proves the live service reproduces the training features exactly.
* **Labels are delayed everywhere.** `GraphService.schedule_label` reveals a label only after `LABEL_DELAY_HOURS`. Training
  purges rows whose labels would not yet be mature at the split boundary (embargo). The earlier version of this project
  fed true labels into the graph instantly in training but used its own alerts when serving, which is leakage plus
  train/serve skew; both are fixed.
* **Strict time splits.** 70/15/15 chronological, never random. Retraining sorts base + live rows by timestamp before splitting.
* **Safe retraining.** A candidate is promoted only if its PR-AUC on the new chronological test window is not below the
  current champion's. Retraining is deferred until enough matured labels exist.
* **Explanations never block scoring.** Every alert gets raw top-3 contributions and a template sentence immediately;
  the LLM rewrite is asynchronous and accepted only if every number it states appears in the supplied facts.
* **Backends.** LightGBM with exact TreeSHAP is the production path. If LightGBM cannot be imported the code falls back to
  scikit-learn HistGradientBoosting with *occlusion* attributions, recorded as `explainer: occlusion` (these are not SHAP values).
