# Cross-stage findings

## 2026-09-27 — MIPRO prune and full-evaluation interaction

Pinned DSPy utils.eval_candidate_program catches TrialPruned into score0/results[].
Strict propagation reaches actual Optuna PRUNED with null values and complete
actual survivors, exercised by one authored installed full/minibatch compile.
However, pruning at the original scheduled full boundary skips the evaluation
and study.add_trial; a selected-source/fake-Study counterexample changes the
winner from good to seed. This defect remains unresolved, not merely a logging
issue. Keep native-equivalence/scientific deployment blocked until a shared
cadence-preserving adaptation is verified. Diagnostic is not method benefit.

Source043/044 are same-family/provisional. Fresh doc unavailable due thread
limit; author execution is not independent environment acceptance. Real history
unchanged1170calls; authored fixture ledger never initializes scientific costs.
Details: research/ARIS_MIPRO_SEARCH.md. Earlier access/failed observations retained.
