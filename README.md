# FraudShield

Real-time fraud detection with graph analysis, grounded LLM explanations and drift monitoring. A portfolio project built to
show production-minded ML: time-correct evaluation, a working streaming system, and honest limitations.

**In plain words:** payments stream in; each gets a risk score from a gradient-boosted model; a graph of accounts, devices and
merchants catches fraud rings; analysts see a ranked queue with reasons they can check; and the system watches itself for
drift and can retrain safely.

## Run it

```bash
docker compose up --build                      # full stack
docker compose --profile llm up --build        # + local Ollama for LLM explanations
docker compose exec ollama ollama pull llama3.2:3b   # once, if using the llm profile; then set OLLAMA_URL=http://ollama:11434
```

| URL | What |
|---|---|
| http://localhost:8000 | Analyst console (React) |
| http://localhost:8000/docs | API docs |
| http://localhost:3000 | Grafana: latency p50/p95/p99, throughput, alert rate, drift PSI, model version |
| http://localhost:5000 | MLflow tracking + model registry (`fraudshield@champion`) |
| http://localhost:9090 | Prometheus |

The init container trains the model (about a minute), registers it, and writes a 1,500-transaction demo stream that begins
with normal traffic and then drifts. Watch the console's drift banner turn amber and the retrain log appear.

Without Docker: `pip install -e .[dev]`, then `python -m app.bootstrap && uvicorn app.main:app` (direct scoring via `POST /api/score`).

## Results (synthetic data; read the caveat)

Full tables, CIs and drift timeline: [`docs/RESULTS.md`](docs/RESULTS.md). Reproduce everything with `python -m experiments.run_all`.

| Model (strict time split, 4h label embargo) | PR-AUC | Recall @ 90% precision | Cost @ tuned threshold |
|---|---|---|---|
| Logistic regression (tabular) | 0.870 | 0.633 | 17,280 |
| GBM tabular only | 0.911 | 0.730 | 12,272 |
| GBM tabular + graph | 0.940 | 0.818 | 7,552 |

Graph ablation (paired bootstrap): PR-AUC **+0.029 [+0.016, +0.043]**, recall@90% precision **+0.079 [+0.033, +0.128]**.
In-process scoring: p50 0.85 ms, p95 1.4 ms, ~600 tx/s (1 vCPU, excludes HTTP/Kafka). The optional message-passing neural net lost to the tabular baseline (reported). Drift: label-free trigger fired
400 transactions (~5 h) before a matured-label metric showed the drop, 0 false alarms on a control stream. **Retraining did not
recover performance in the drift experiment** (reported as a limitation).

> These numbers come from the author-built synthetic generator and from the scikit-learn fallback model (LightGBM was not
> installable where they were produced). They validate the method and measurement discipline, not real-world fraud
> performance. Re-run on your machine to regenerate with LightGBM.

## PRD coverage and verification status

"Verified" means executed in the build sandbox. "Written" means implemented and syntax-checked but needing your machine,
because the sandbox had no network, Docker, Kafka, MLflow, FastAPI or LightGBM.

| ID | Requirement | Status |
|---|---|---|
| FR-1 | Time-ordered replay to Redpanda | Written (`app/producer.py`); stream content verified |
| FR-2 | Rolling features (velocity, deviation, merchant frequency, device links) | **Verified** (causality + parity tests) |
| FR-3 | FastAPI scoring with SHAP reasons, model from MLflow | Scoring path **verified** in-process; HTTP layer + registry load written |
| FR-4 | LightGBM + logistic baseline | Baseline and pipeline **verified** with the sklearn fallback; LightGBM path written |
| FR-5 | Cost-tuned configurable threshold | **Verified**; served threshold is the tuned one unless `FRAUD_THRESHOLD` is set |
| FR-6 | Graph + community detection for rings/mules | **Verified** (Louvain, ring/mule flags, delayed-label risk) |
| FR-7 | LLM explanations | Async upgrade, grounding check and fallback **verified against a mock Ollama**; real Ollama written |
| FR-8 | Drift monitoring + retrain trigger | **Verified** (PSI/KS, 2-signal trigger, experiment). Evidently is an optional HTML helper only (untested) |
| FR-9 | Prometheus + Grafana | Written (metrics, scrape config, provisioned dashboard JSON validated); not run |
| FR-10 | React dashboard with feedback | Written (no-build React via CDN, syntax-checked); not run in a browser |
| FR-11 | Feedback labels used in retraining | **Verified** (`labeled_live_rows`, feedback overrides matured labels) |
| FR-12 | MLflow tracking + registry | Written (graceful fallback to local cache); not run |
| FR-13 | Optional GNN, compared against tabular baselines | **Verified** as a numpy/scikit-learn two-layer message-passing network (not PyG). **It lost** to the tabular baseline (PR-AUC 0.868 vs 0.910) and to the graph-feature GBM (0.939); reported in `docs/RESULTS.md` |
| FR-14 | One-command Docker Compose | Written; not run |

Evaluation items from the PRD: strict time splits, label-delay simulation, PR-AUC/recall@precision/precision@k/cost,
p50/p95/throughput, ablation with CIs and drift lead time are all measured. **Not measured:** analyst review-time change
(needs real reviewers; the console has a study mode and `experiments/review_time_report.py` analyses it), and anything on
real datasets.

### What is still left for you (cannot be done without your machine or people)
1. `docker compose up --build` and fix anything that only shows up when the full stack runs (API, Kafka, MLflow, Grafana, React console were written but not executed in the build sandbox).
2. `python -m experiments.run_all` with LightGBM installed, to replace the fallback-model numbers.
3. A review-time study with a few real reviewers (`docs/RESUME_BULLETS.md` explains it).
4. Record the demo video from `docs/DEMO_SCRIPT.md` and add screenshots.
5. Optionally try IEEE-CIS / PaySim (loaders are ready) and a tuned PyG GNN.

### How the PRD's open questions were resolved
Redpanda (lighter on a laptop). NetworkX only (Neo4j cut). Graph ablation runs on the synthetic generator because IEEE-CIS
lacks account linkage and PaySim is synthetic anyway; both loaders exist for you to try. Label delay is simulated by a maturity
queue plus a training embargo. The GNN was cut.

## Using real datasets

```bash
DATASET=ieee_cis DATASET_PATH=/path/to/ieee-fraud-detection python -m app.bootstrap   # train_transaction.csv (+ train_identity.csv)
DATASET=paysim   DATASET_PATH=/path/to/PS_20174392719_1491204439457_log.csv python -m app.bootstrap
```
Respect each dataset's licence. Mapping assumptions are documented in `app/datasets.py`. The demo stream is still synthetic.

## Experiments

```bash
python -m experiments.ablation                 # graph vs tabular, 5 seeds, paired bootstrap
python -m experiments.drift_experiment         # lead time, false alarms, retraining recovery
python -m experiments.gnn                      # message-passing network vs tabular baselines
python -m experiments.loadtest                 # in-process latency/throughput
python -m experiments.loadtest --mode http --url http://localhost:8000 --concurrency 16   # against the running API
python -m experiments.review_time_report --db data/fraudshield.db   # after real reviewers used study mode
python -m experiments.run_all                  # everything + docs/RESULTS.md (~6 min)
pytest                                         # unit tests (API tests need fastapi + httpx)
```

## Repository layout

```
app/            synthetic.py datasets.py features.py graph.py pipeline.py evaluation.py model.py drift.py
                explain.py services.py database.py metrics.py main.py worker.py producer.py bootstrap.py
experiments/    ablation.py drift_experiment.py gnn.py loadtest.py review_time_report.py run_all.py
monitoring/     prometheus.yml + provisioned Grafana datasource and dashboard
static/         React analyst console (no build step)
docs/           architecture.md architecture.svg RESULTS.md DEMO_SCRIPT.md INTERVIEW_PREP.md RESUME_BULLETS.md
tests/          causality, label maturity, train/serve parity, metrics, drift, explain, datasets, GNN, API
```

## Honest limitations
Synthetic data; very small `TRAIN_ROWS` (below ~20k) gives noisy models that may not beat the baseline (default is 40k); fixed simulated label delay; analyst-feedback selection bias; retraining did not fix the synthetic drift;
the fallback explainer is occlusion, not SHAP; latency numbers exclude network and Kafka; nothing here is PCI-grade or has
seen real customer data. See [`MODEL_CARD.md`](MODEL_CARD.md).
