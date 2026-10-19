# Data bundle: skip (not abort on) cloned-dataset .git metadata

- **Date**: 2026-10-19
- **PR**: [#109](https://github.com/taitran501/ai-harness/pull/109)
- **Branch**: `fix/arc-prize-audit-remediation`
- **Status**: open
- **Commits**: `09ccfe6`

## Why

Staging the arc-prize-2026 competition data failed because the dataset
was cloned from GitHub and therefore contained a `.git/` directory.
`data_bundle._files_under` aborted the entire bundle on *any* file with
a blocked path part, so ordinary version-control metadata inside a
legitimate data directory made the dataset un-uploadable. Two genuinely
different situations were being handled by one rule:

1. Protected content (credentials, harness evidence, provider state)
   that must **never** be uploaded — aborting is correct.
2. Directory metadata (`.git`) that legitimately ships inside a
   cloned dataset — skipping is correct; the data itself is fine.

## What changed

- **`harness/data_bundle.py`** — split `BLOCKED_PATH_PARTS` into:
  - `PROTECTED_PATH_PARTS` (`.env`, `.kaggle`, `.modal`, `.ai-harness`,
    `artifacts`, `submissions`) — still abort staging wherever found.
  - `METADATA_SKIP_PARTS` (`.git`) — files under declared data
    directories are **skipped**, not fatal.
  - `BLOCKED_PATH_PARTS` kept as the union, still applied to
    *declared* paths (explicitly declaring `.git` as task data is
    still refused).
  - New guard: a declared directory whose files are *all* skipped
    metadata now fails with a clear "no uploadable files" error instead
    of silently contributing nothing to the bundle.
- **`tests/test_data_bundle_protected_paths.py` (new)** — 4 tests:
  `.git` inside declared dataset is skipped (bundle contains only the
  real data files); explicit `.git` declaration refused; protected
  file (`.env`) inside dataset still aborts; metadata-only directory
  fails with the clear error.

## Approach

Keep the strict protection semantics (a `.env` in a data dir must still
fail loudly) while classifying `.git` as skippable metadata. The
declared-path check deliberately keeps the union: a user who lists
`dataset/.git` as data has a broken config, not a cloned dataset. The
metadata-only guard closes the only silent hole the skip introduces.
Alternative considered — a per-config `allow_metadata` flag — rejected
as unnecessary surface: skipping is safe by construction (metadata
bytes add nothing to training and bloat the bundle).

## Evidence

- `tests/test_data_bundle_protected_paths.py`: 4/4 pass.
- Existing `test_workspace_workflow.py::test_data_bundle_consent_fingerprint_and_eda_privacy`
  (bundle staging + fingerprint + privacy): still passes.

## Impact & follow-ups

- Cloned datasets (GitHub/GitLab with `.git/`) can now be staged;
  bundle manifests simply omit the metadata files.
- Consent fingerprints are computed on the staged file set, so a
  metadata-only change to the dataset (e.g. a fresh clone of the same
  content) does not invalidate granted consent.
- Follow-up (none blocking): nothing.
