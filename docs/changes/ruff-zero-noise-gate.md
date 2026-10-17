# Ruff zero-noise gate + B10–B16 cloud/ops & architecture fixes

- **Date**: 2026-10-17
- **PR**: [#56](https://github.com/taitran501/ai-harness/pull/56)
- **Status**: merged (squash)
- **Area**: lint gate / cloud runner / models registry

## 1. Why (motivation + thought)

Clean every remaining ruff finding in the pre-push gate scope, then re-verify
each B10–B16 module-review item against live code and fix the real ones. The
method (verify first) mattered: most B-items turned out to be false positives;
only a few were real, so a blind "fix all" would have added churn.

## 2. What (concrete changes)

- **Ruff** (gate scope `--select F,E7` → 0 errors): 63× F541 (f-string without
  placeholder, auto-fixed); 10× E701 + 5× E702 (split in `baseline.py`,
  `task_spec.py`, `tree_journal.py`, `dashboard_server.py`); 4× E741
  (ambiguous `l`, renamed in `cloud_job_runner.py`, `diagnostics.py`,
  `tools/remove_dead_functions.py`).
- **B12 — real latent bug**: `CloudJobManager.stream_logs` opened the log in
  TEXT mode then `f.seek(start_byte)` (undefined behaviour on CPython for
  non-zero offsets). Now binary mode + byte seek + per-line decode. The
  dashboard `/api/cloud/logs/stream` endpoint now parses/validates a
  `start_byte` query param (backward compatible: omit → resume from 0).
- **B15 — real per-dispatch cost**: `run_full_preflight_check` loaded the
  ENTIRE training frame locally before every cloud dispatch. Added
  `full_data_check` (default True); the dispatch path
  (`cloud_runner._prepare_remote_run`) passes `full_data_check=False` and only
  checks dataset/split file presence — the remote job re-validates its data as
  its first step.
- **B16 — real architecture pollution**: `DecomposedMappingEquityModel`
  (~195 lines, Zindi-specific literals incl. `maricopa-az` offsets) lived in
  the generic `harness/models.py` registry. Moved to the zindi workspace as its
  own task-plugin package (`zindi/plugin/`, entry point `zindi_bias_bounty`,
  installed editable); the runner's `build_model` plugin hook serves all 8
  model names there, and `create_model` rejects them everywhere else.
- **False positives (no change, documented)**: B10 (full `lock_payload` already
  persisted; enhancement kept — `EMERGENCY_UNFREEZE` now records
  `released_slots` / `lock_receipt_path` / `frozen_at_git_commit`); B11
  (measured ~1 ms/node, the 1.5 s journal load is discovery not hashing); B13
  (empirically verified `os.kill(pid, 0)` is safe on this platform); B14
  (already resolved in Wave 1).

## 3. Approach (decisions + rejected alternatives)

Verify each B-item against current code *before* changing anything (4 of 7 were
false positives). Text-mode `seek()` is UB on CPython, so byte-offset resume
must go through a binary stream. Competition-specific models with hardcoded
literals must live in the competition workspace via the `build_model` plugin
hook, not the generic factory.

## 4. Evidence

- `tests/test_ruff_b10_b16.py` — 6 regression tests (B10 audit
  self-containment; B12 byte-offset resume + past-EOF safety; B15 frame-load
  flag; B16 factory agnosticism + zindi plugin serving).
- B15/B16 bite-checked by re-introducing the bug and confirming failure.
  Full suite 653 OK; ruff gate clean.

## 5. Impact & follow-ups

- **Downstream**: establishes the "verify before fixing" discipline for
  module-review items and the zero-noise ruff gate policy (F/E7 clean).
- One-liner ruff noise (E7xx) in exception paths hides real structure —
  splitting them costs nothing and makes review easier.
