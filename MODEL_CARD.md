# Model card: FraudShield risk model

## Intended use
A portfolio/demonstration model that ranks payment transactions by fraud risk for a simulated analyst queue. It is
**not** validated for any real payment decision and must not be used to block real customers.

## Data
* **Default: synthetic** (`app/synthetic.py`), 40,000 training transactions, about 3.5% fraud, over about 3 weeks of
  simulated time. Fraud scenarios: account takeover with bursts, visible fraud rings, stealth rings with legitimate-looking
  amounts, and (in drift tests) a low-and-slow tactic. Legitimate households that share a device act as confounders.
* **Optional real benchmarks:** IEEE-CIS (Kaggle) and PaySim (Kaggle, synthetic) loaders exist in `app/datasets.py`
  (`DATASET=ieee_cis|paysim`). They are fixture-tested only; **no result in this repository comes from them.**
  IEEE-CIS has no account id (we use the card1+addr1+first-day proxy), so graph linkage there is weak.

## Model
LightGBM gradient boosting (scikit-learn HistGradientBoosting fallback), class-weighted, on 15 tabular and 9 graph features.
Baseline: class-weighted logistic regression on the 15 tabular features. Threshold: exact minimiser of
`500*missed_fraud + 20*false_positive` on the validation window; served unless `FRAUD_THRESHOLD` is set.

## Evaluation protocol
Strict chronological train/validation/test with a label-maturity embargo; PR-AUC primary; recall at 90% precision,
precision@1%, and cost reported; accuracy never reported. Confidence intervals by paired bootstrap; seed variance from 5 trainings.
Figures: [`docs/RESULTS.md`](docs/RESULTS.md).

## Graph neural comparison
A two-layer message-passing network (numpy/scikit-learn, not PyG) was evaluated and **lost** to the tabular GBM; see `docs/RESULTS.md` section 5.

## Known limitations and failure modes
* **Synthetic data.** The graph gain (+0.029 PR-AUC, CI [+0.016, +0.043] in `docs/RESULTS.md`) is a property of how the
  generator makes rings depend on shared devices and delayed labels. It shows the method works end to end, not that graph
  features will help on real portfolios.
* **Drift response.** The label-free drift monitor warned before label-based monitoring could, with no false alarms on a
  control stream, but retraining after the drift did not recover performance in the experiment.
* **Label delay is simulated** as a fixed 4 hours. Real chargeback delays vary and can be weeks.
* **Selection bias in analyst feedback.** Only alerted cases receive analyst labels, so feedback-only retraining would
  be biased. The matured-label feed is what makes retraining data less biased in the demo.
* **Explanations.** The local LLM only paraphrases supplied facts; a numeric grounding check rejects invented numbers, but
  it cannot catch every wording error, which is why the raw contributions are always shown beside the text.
* **Cold start.** Accounts and devices with no history have weak behavioural features.
* **No fairness analysis** (no demographic data exists in the synthetic set).
* The occlusion fallback explainer is not SHAP and is slower; production should use the LightGBM path.

## Ethical and operational notes
No real customer data. Scores should drive review prioritisation, with a human making the final call.
