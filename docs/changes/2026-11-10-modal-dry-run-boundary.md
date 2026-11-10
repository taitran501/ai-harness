# Make runner Modal dry-run return before dispatch

The main runner accepted --dry-run but did not forward it to run_via_modal. Modal therefore followed the normal upload/dispatch path.

The runner now forwards the flag, and run_via_modal returns a static model/resource plan before remote preparation, account checks, consent, worker updates, uploads or importing Modal. It respects effective/base configuration for model admission. The result explicitly states that provider readiness, model volume contents and actual inference remain unverified. Default execution is unchanged.

Regression checks call the function with actual Track1 metadata while remote preparation, upload, worker updates and subprocess execution are replaced by failures, and inspect CLI forwarding. No provider operation is needed for this verification. This patch does not add model-volume staging or prove GPU readiness.
