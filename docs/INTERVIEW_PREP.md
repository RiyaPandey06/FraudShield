# Interview preparation (answers grounded in this repo)

**Why PR-AUC instead of ROC-AUC or accuracy?** Fraud is ~3.5% of rows. A model that flags nothing is 96.5% "accurate", and
ROC-AUC is dominated by the huge negative class, so a model can look great while most alerts are wrong. PR-AUC and
recall at fixed precision measure what an analyst experiences: how many alerts are real, and how much fraud is caught.

**How did you prevent leakage and handle delayed labels?** Features use only prior events (`test_features_are_causal`).
Splits are chronological. Graph risk uses labels only after a maturity delay, and training purges rows whose labels would
not yet be known at the split boundary. I also found and fixed an earlier bug where training used instant labels while
serving used its own alerts. Parity between training and serving features is tested.

**How do you choose the threshold in production?** From a cost matrix on the validation window (missed fraud 500, false
positive 20 in this demo), as an exact minimiser, and it is stored with the model. In a real system the costs come from
finance and analyst capacity, and I would also cap alert volume.

**What happens when fraud patterns drift?** PSI/KS on inputs and scores flag drift without waiting for labels; a trigger
needs two signals to avoid single-feature noise. Retraining runs on matured labels and is promoted only through a champion
gate. Honest result: the monitor gave modest lead time, and retraining did not recover performance on my synthetic
low-and-slow drift, so new features or faster labels would be needed.

**Why a graph, and what did it add?** Rings share devices, and confirmed fraud on one account is evidence about its
neighbours. On the synthetic data it added +0.029 PR-AUC (CI +0.016 to +0.043) and +0.079 recall at 90% precision, mostly on
ring scenarios. I would not claim that transfers to real data without testing on it.

**Did you try a GNN?** Yes, a two-layer message-passing network over the account-device graph (numpy/scikit-learn, not PyG). It lost to the tabular GBM (PR-AUC 0.868 vs 0.910) and to the graph-feature GBM (0.939), CI excluding zero. I reported the loss; on this data the engineered graph features carried the signal, and I did not test a tuned PyG model.
