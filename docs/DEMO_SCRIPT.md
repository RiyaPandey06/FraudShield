# 2-minute demo script (record this yourself)

I cannot record video, so this is the exact storyboard plus a capture checklist. Run `docker compose up --build` first.

| Time | Show | Say |
|---|---|---|
| 0:00 | README title + `docs/architecture.svg` | "FraudShield scores payments in real time, finds fraud rings with a graph, explains alerts, and watches itself for drift." |
| 0:15 | Terminal: `docker compose up --build`, then the init log line with PR-AUC vs baseline | "One command. The model is trained with a strict time split and beats a logistic baseline on PR-AUC, not accuracy." |
| 0:30 | http://localhost:8000: metrics row and alert queue filling | "The producer replays a time-ordered stream; the worker scores each event with features, graph context and top reasons." |
| 0:50 | Click the top alert: reasons, linked accounts, ring flag, explanation text | "Reasons are raw model contributions; the sentence is a grounded explanation. Linked accounts expose the ring." |
| 1:10 | Click Confirm fraud, then enable study mode and label another case | "Analyst labels feed retraining. Study mode times reviews with and without explanations." |
| 1:25 | Grafana (localhost:3000): latency p95, throughput, alert rate, PSI panel | "Latency, throughput and drift are observable. The demo stream starts drifting partway through." |
| 1:45 | Console drift banner turning amber; retrain log / MLflow registry champion alias | "PSI flags the shift before labels can; retraining is gated so a worse model is never promoted." |
| 1:55 | `docs/RESULTS.md` table | "Everything here is measured on synthetic data, with confidence intervals and honest losses, including a GNN that lost." |

**Capture checklist:** console with populated queue; case detail; Grafana dashboard after ~60 s of replay; amber drift banner; MLflow registry page; `docs/RESULTS.md` tables.
Save screenshots to `docs/screenshots/` and link them from the README.
