# Truthful observations, Python 3.10 hashing, and declarative plugin contracts

- **Date**: 2026-11-14
- **PR**: #137
- **Branch**: `fix/harness-invariants-and-audit-hardening`
- **Status**: open (uncommitted working tree)
- **Commits**: None; host assistant will review and commit.

## Why

Experiment inspection used truthiness to select scores, replacing measured
`0.0` with another score or the task baseline. It fabricated fold dispersion
(`0.015`) and missing metric deltas (`0.0`), and reported success for an explicit
unknown experiment by inspecting the overall best candidate instead.

Artifact publication called the Python 3.11-only `hashlib.file_digest` without
a Python 3.10 fallback. Declarative submission verification swallowed test-data
loading failures, and submission ID padding was tied to the `GEOID` column name.

## What changed

- `harness/bridge_actions.py`: resolve scores with explicit `None` checks;
  preserve measured zeroes; leave unknown scores, fold dispersion, and deltas
  as `None`; render unavailable scores safely; fail explicit unknown experiment
  requests with the requested ID instead of falling back to another candidate.
- `harness/artifacts.py`: retain native hashing when available and otherwise
  stream SHA-256 in 65,536-byte chunks through the same open file handle.
- `harness/plugins.py`: propagate test split loading failures and share ID
  formatting across explicit-column and inferred-column submission builders.
  `data_config.id_zfill` overrides the legacy 11-character `GEOID` default.
- `docs/TASK_SPEC_TEMPLATE.md`: document the configurable submission ID width.
- `tests/unit/test_truthful_observations.py`: seven tests for score precedence,
  missing/measured dispersion and deltas, explicit missing IDs, default candidate
  selection, declared/absent baselines, and missing diagnostics.
- `tests/unit/test_python_compatibility.py`: two tests comparing native and
  fallback hashes to independently computed digests for empty, one-byte,
  exact-chunk, and multiple-chunk files.
- `tests/unit/test_plugin_contracts.py`: three tests for loader failures
  (`RuntimeError`, `OSError`, and `ValueError`), valid/invalid submission IDs and
  row counts, and configured/default/overridden ID padding in both builder paths.
- `tests/integration/test_agent_bridge.py`: correct the old assertion that an
  explicit baseline without candidate evidence should succeed.

## Approach

Keep the observation schema and default candidate selection intact, replacing
truthiness and fabricated numeric defaults at the inspection boundary. Review
of adjacent inspection handlers found no score fallback in diagnostics; its
missing-evidence warning and unset metric fields are covered by a regression.
This checkout has no `_handle_inspect_candidate` handler.

Remove the submission validator's blanket exception handler so callers receive
the original loading failure. Preserve its existing row/ID checks and the
optional-test-path behavior. Compute formatted IDs once so both submission
construction paths honor the same declarative width and compatibility default.

Add regressions before production changes: the original implementation failed
the new suite with 14 subtest failures and eight subtest errors. After the fixes,
all 12 new tests pass.

## Evidence

- `python tools/run_tests.py tests.unit.test_truthful_observations tests.unit.test_python_compatibility tests.unit.test_plugin_contracts`:
  **12 tests passed** (3.1 seconds including runner overhead).
- `ruff check harness/ tools/ tests/`: **passed**.
- `python tools/run_tests.py --smoke`: **84 tests passed**, no skips, failures,
  or errors (24.8 seconds including runner overhead).
- `python tools/run_tests.py --fast`: **1,134 tests run; 1,132 passed and two
  skipped**, no failures or errors across 47 modules (478.5 seconds including
  runner overhead). The run emitted a restricted-network warning when fetching
  the Hugging Face model catalog and used the existing fallback catalog.
- `git diff --check`: **passed**.

## Impact & follow-ups

Consumers must handle `None` for unavailable inspection metrics and `FAILED`
for explicit unknown experiments. Test loader failures now stop submission
verification rather than silently accepting unverified submissions.

Hash compatibility was exercised on Python 3.11.9 with the native API removed
to simulate Python 3.10 availability; no actual Python 3.10 interpreter run is
claimed. No commits, pushes, or PR updates are part of this change.
