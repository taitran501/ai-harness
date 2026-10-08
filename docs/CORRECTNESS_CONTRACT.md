# Run evidence and correctness contract (schema v2)

The canonical local and remote runner is `harness.runner`. Legacy `train.py` delegates to it. Supported built-in estimators are `baseline`, `logistic_regression`, `ridge`, `tfidf_logistic_regression`, and `tfidf_ridge`; `baseline` resolves from the target type. Other architectures require a task plugin or explicit factory. Invalid parameters and unsupported generic metrics fail instead of silently selecting another estimator/scorer.

## Predictions and scoring

`PredictionBatch` carries IDs, labels, probabilities and class order. Probability metrics reject labels alone. Generic metrics: accuracy, macro/micro F1, ROC-AUC, logloss, RMSE, MAE and R2. Plugin scorers retain their official aggregate. Generic weighted aggregation requires matching subtask metric and direction. Every declared target must be complete.

`primary_score` is the original metric; `utility` negates it for minimize tasks. Ranking, UCB and improvement use utility. Fold metrics are diagnostics; the primary score is recomputed on complete OOF predictions for the selected input variant. Multiple variants require `data.selection_variant`. Task seed is authoritative; otherwise use runtime seed or 42. Declared split/dev paths never fall back when missing.

## Immutable artifacts

Full validation claims `<artifact-directory>/<exp_id>` exclusively. Use a new experiment ID for another full run. `--force` does not overwrite evidence. Debug, selected-fold and sample probes use `probes/<exp_id>/<run_id>` and never enter the main journal/candidate pool.

Manifest v2 records run identity/kind/status, task/scorer identities, resolved-config/data/split hashes, actual estimator parameters and artifact hashes. Required artifacts include metrics, OOF predictions, evaluation truth and resolved config. Consumers reject incomplete or changed hashes. Historical artifacts remain visible as `LEGACY_UNVERIFIED` and cannot establish a new baseline or candidate. No automatic migration or re-certification occurs.

## Candidate and submission gates

Promotion requires the exact `exp_id`. Candidate IDs are unique and retain source IDs and manifest hashes. Sources must share task/scorer/data/split identities and identical ID sets. Joins are one-to-one. Classification voting uses declared class order for ties; probability blending requires matching probability class order; regression blending operates on continuous predictions. All positive-weight targets pass diversity checks and a combined candidate must outperform its best constituent before publication.

Submission requires a task plugin implementing both builder and validator, and a hashed `test_predictions.csv` explicitly marked `test` in the full-run manifest. OOF is not test evidence. Generic tasks without submission hooks fail closed. Standalone audit supports an explicit `--id-column`; it compares prediction columns only and rejects null/duplicate IDs and invalid domains.

## Governance, state and budgets

Dashboard candidate/freeze/unfreeze/cloud and score endpoints delegate to Agent Bridge. Approval remains mandatory for sensitive actions. Freeze validates eligible candidates, unique IDs and slot count; unfreeze requires a reason. Audit history is retained.

Resource reservations are durable JSON transactions, separate for training, probes and candidate operations. Terminal accounting is idempotent. Failed runs count consumed resources; unknown GPU allocation remains null. A local wall-time budget terminates the process tree. GPU hard budgets and remote hard runtime budgets are rejected because these backends currently lack verified allocation/termination accounting. Kaggle/Modal remain usable without a hard runtime budget. Stopping a monitor never certifies remote cancellation.

JSON state uses reentrant thread locks plus Windows `msvcrt` / Unix `fcntl` locks, fresh reads and atomic writes. Locks cover state updates only. Unrelated snapshot changes and append-only events merge; conflicting scalar updates fail. Task caching hashes file content. TLS always verifies certificates; `HARNESS_CA_BUNDLE` selects an explicit trusted CA file.

## Capability boundaries

Text metadata features work in `UniversalFeaturePipeline`. Calibration and threshold optimization remain utilities requiring explicit integration; they are not built-in runner models. Optional diagnostic failure is recorded visibly; scorer/evidence persistence failure propagates. Provider calls in regression tests are mocked. Live cloud validation requires a separately budgeted run.
