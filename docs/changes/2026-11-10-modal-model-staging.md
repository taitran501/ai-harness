# Verified model staging and cleanup primitives

New model_assets.modal_staging primitives verify declared source bytes/SHA-256 before touching Modal, persist an intent receipt, create a UUID-named model volume with allow_existing=False, upload exact files without overwrite and retain state after failures. They use public APIs verified against the installed Modal SDK signatures and CLI source. Upload completion does not claim remote hash verification; the existing runtime verifier remains mandatory.

Cleanup requires a receipt with a matching owned-volume name and confirmed creation, then validates completed local full-run artifacts against the expected experiment before deleting that exact volume in the recorded environment. Shared names and incomplete artifacts are rejected. Failed/aborted-run recovery and dispatch integration are not implemented by this change; no live resources were created during development.

Four mocked-provider tests cover staging, corrupt local inputs, create collisions, and cleanup refusal. This evidence verifies control flow only, not provider interoperability or live model inference.
