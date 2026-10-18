# Repair integrity gate + recursive artifact sealing

- **Date**: 2026-10-18
- **PR**: — (branch `fix/repair-integrity-and-recursive-sealing`)
- **Branch**: `fix/repair-integrity-and-recursive-sealing`
- **Status**: open
- **Commits**: —

## Why

An external integrity review (verified line-by-line against this repo) found
two evidence-integrity gaps in the repair/sealing surface:

1. **C1 — repair re-seals tampered scores.** `ai-harness repair` re-sealed
   `files` hashes straight from disk bytes. A doctored `metrics.json` (or
   OOF/truth CSV) was laundered back into the sealed bundle: the tampered
   score became the sealed truth, `validate_run` passed again, and nothing
   recorded that the run had been re-sealed. `validate_run` never recomputes
   the score from predictions — so the only way a repair can be trusted on
   evidence files is an explicit re-derivation.
2. **M2 — nested run files were never sealed.** The runner sealed only
   top-level files (`exp_dir.iterdir()`), so pipeline runs' per-stage
   evidence (`stages/*.log`, nested logs) lived outside the hash surface and
   could be rewritten freely while the run still validated.

Design constraints honored: additive/backward-compatible (old top-level-only
manifests validate exactly as before), single scoring path (re-derivation
reuses the runner's exact extraction + evaluation calls, no second metric
engine), and out-of-scope of the `scorer_sha256` hash seal (only
`harness.evaluation` + `harness.task_spec` sources are sealed — new code
calls them, never modifies them).

## What changed

**New: `harness/oof_revalidation.py`** — score re-derivation module:
loads a run's `oof_predictions.csv` (fallback `predictions.csv`) and
`evaluation_truth.csv`, rebuilds the per-variant prediction mapping via the
same `build_variant_predictions` the runner now imports, scores through
`task_plugin.evaluate_predictions` (+ optional `run_config
paths.metrics_evaluator` hook, mirroring `UniversalRunner._evaluate`) and
`resolve_evaluation`, then compares recorded vs recomputed
(`SCORE_REL_TOL = 1e-6`). Exports `is_evidence_file` (the evidence-file set)
shared by both repair entry points.

**`harness/repair.py`** — the integrity gate:
- Evidence drift (drifted/missing `metrics.json`, OOF, `predictions.csv`,
  `test_predictions.csv`, `evaluation_truth.csv`) is **never re-sealed by
  default** — the run stays visibly broken, CLI exits **3**
  (`blocked_evidence_drift`).
- `--verify` re-derives the score (lazy project/plugin load, paid only when
  exercised) and re-seals only on an exact match; mismatches report both
  values (`metric_mismatch`); missing evidence is `not_verifiable`.
- `--force` is an explicit operator override, always recorded.
- Every repair action (including blocked attempts) appends to
  `<artifacts_dir>/repair_log.jsonl` — beside the run folders, outside every
  sealed bundle, with operator, old/new hashes, and the metric-check report.
- Re-sealed manifests gain `repaired: true` + `repaired_at` (re-seal
  provenance); `--check` now surfaces runs carrying the marker.
- The public API `repair_run_dir` enforces the same gate (raises
  `EvidenceDriftError` without `force=True`), so no caller bypasses it.
- File maps are now built recursively via `artifacts.seal_file_map`.

**`harness/artifacts.py`** — `seal_file_map()`: recursive sha256 map with
POSIX relative keys (top-level keeps bare names; nested → `stages/train.log`;
manifest excluded). `validate_run` now accepts safe relative keys while
rejecting traversal (`..`), absolute, and empty names — old bare-name
manifests are unaffected.

**`harness/runner.py`** — `_save_and_finalize` seals with `seal_file_map`
(new runs include nested files), and the per-variant `pred_/proba_/classes_`
extraction now calls the shared `build_variant_predictions` so runner and
re-derivation cannot drift apart.

**Consumers** updated for relative keys: `harness/import_local.py`
(`_content_verified` safe-relpath check + nested copy), `harness/kaggle_runtime.py`
(`export_run` materializes nested outputs), `harness/templates/kaggle_template.py`
(notebook validation accepts safe relpaths; import copy creates parent dirs),
`harness/templates/modal_template.py` (bundle save creates parent dirs).

**Tests** — `tests/test_repair_integrity.py` (new, 8): tampered metrics
blocked by default (exit 3, audit log, not re-sealed), re-derivation matches
the recorded score, `--verify` passes on unchanged score / blocks on mismatch
(reporting both values), `--force` audited, missing-evidence blocked and
not-verifiable, log append-only outside the bundle. `tests/test_recursive_sealing.py`
(new, 9): recursive sealing with relpath keys, nested tamper detected,
traversal/absolute keys rejected, old top-level manifests still validate,
import/export consumers handle nested files. `tests/test_repair.py` +
`tests/test_pr_b_guardrails.py` updated to the new contract (mechanism tests
use explicit `force=True`; CLI tests assert block-then-restore).

## Approach

- **Gate, not recompute-by-default.** Re-derivation needs the task spec +
  plugin + evidence files; doing it on every repair would couple a
  lightweight maintenance path to full project loading. Instead: default
  posture *refuses* evidence re-seals (fail-safe), and `--verify`/`--force`
  are explicit, audited opt-ins. Benign drift (non-evidence files) keeps the
  classic instant re-seal.
- **Single scoring path.** Rather than writing a second scorer in
  `repair.py`, the runner's extraction block was extracted into
  `oof_revalidation.build_variant_predictions` and imported by the runner —
  so recorded and recomputed scores come from one implementation, and
  `evaluate`/`resolve_evaluation` are called exactly as `_evaluate` does
  (including the custom-hook branch).
- **Self-contained truth.** Re-derivation reads `evaluation_truth.csv` from
  the run folder (the runner mirrors it content-addressed per run), so
  verification never needs the original dataset.
- **Additive sealing namespace.** Recursive keys use POSIX relpaths; bare
  names never contain `/`, so old manifests and new manifests share one
  namespace without collision. Traversal is rejected at validation,
  import, and notebook-import time.
- **Audit trail placement.** `repair_log.jsonl` lives beside the run folders
  (under the artifacts dir), never inside a run folder, so it is outside
  every sealed bundle and cannot perturb `validate_run`.

## Evidence

- Red proof of the C1 bug: with the gate disabled (monkeypatched
  `is_evidence_file -> False`), a tampered `metrics.json` (0.5 → 9.99) is
  re-sealed, `validate_run` passes again — the reviewer scenario. With the
  gate: exit 3, run stays broken, `repair_log.jsonl` records the blocked
  attempt.
- New tests: `tests/test_repair_integrity.py` (8) + `tests/test_recursive_sealing.py`
  (9) — all pass; nested `stages/train.log` tamper now fails `validate_run`
  (previously invisible).
- Updated contract tests in `tests/test_repair.py` / `tests/test_pr_b_guardrails.py`
  pass with explicit `force=True`.
- Full suite: `python tools/run_tests.py` → **892 tests OK** (skipped=1,
  pre-existing). `ruff check harness/ tests/ tools/` clean. `--smoke` tier
  OK.
- Backward compatibility: old top-level-only manifests validate unchanged
  (`test_old_top_level_manifest_still_validates` + full suite covering
  runner/journal/import paths).

## Impact & follow-ups

- **Behavior change (intended):** evidence-drifted runs no longer repair
  silently. Operators use `--verify` (recommended) or `--force` (audited).
  Exit code 3 is new: "blocked by the evidence integrity gate".
- **New runs seal nested files**; existing top-level-only runs are
  unaffected, and repair still never adopts undeclared files (bundle
  exclusivity preserved).
- Modal bundle upload is still flat-keyed: importing a nested-sealed run
  through the Modal template fails loudly (missing-key hash mismatch)
  rather than silently — acceptable for v1; a nested-aware bundle is a
  follow-up if Modal is used for pipeline runs.
- `SCORE_REL_TOL = 1e-6` (relative) is the mismatch threshold; if a real
  task's CSV round-trip exceeds it, revisit with live-run evidence.
- Deferred (out of scope, tracked separately): M3 scorer-hash scope, M1
  grounding source verification, M5 benchmarking, M4 module splitting.
