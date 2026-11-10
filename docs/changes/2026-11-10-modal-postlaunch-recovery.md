# Terminal-gated failed-run recovery

Owned attached Modal jobs now record a unique job/config/task identity and canonical generated app name before attempting launch. Failure recovery queries the recorded environment for exactly one matching app, requires explicit stopped state and zero running tasks, downloads provider log tail and full child failure snapshots, verifies snapshot identities/SHA-256 and retained local inputs, rechecks terminal app identity, then deletes only the receipt's owned volume. A local recovery receipt pins the evidence manifests. No absent-app or log-only terminal inference is allowed.

The normal cloud-runner failure handler attempts this path and retains receipts/storage on any uncertainty. Missing snapshots, container death before snapshot creation, provider unavailability and integrity failures remain incomplete recovery rather than automatic cleanup. Provider logs are a bounded tail; the snapshot contains the full child-runner stream. Success/failure is never converted to a quality score.

Seven recovery cases plus related staging/preview/export checks give21 passing local tests;11 existing Modal framework tests and Ruff also pass. Cases cover explicit stopped success, running/missing/ambiguous apps, nonzero tasks, corrupt snapshot and wrong job identity. These are simulated provider responses; no live resource was created, downloaded or deleted.
