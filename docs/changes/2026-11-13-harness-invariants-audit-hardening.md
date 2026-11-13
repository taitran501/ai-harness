# Enforce harness invariants after the adversarial audit

- **Date**: 2026-11-13
- **PR**: Pending host review
- **Branch**: `fix/harness-invariants-and-audit-hardening`
- **Status**: Working tree; not committed
- **Commits**: None

## Why

The audit identified twelve correctness and invariant gaps spanning task edits,
scorer identity, prediction alignment, immutable recovery, research caches,
backend enforcement, workspace imports, remote configuration, bridge input,
test discovery, and hygiene reporting. These gaps could discard configuration,
admit incompatible evidence, score misaligned rows, overwrite completed runs,
or hide unimported outputs.

## What changed

- Preserve wrapped and flat task documents when changing execution settings in
  `harness/cli.py` and `harness/bridge_actions.py`.
- Require both differing generic scorer hashes to be explicitly recognized in
  `harness/artifacts.py`; register the audited previous and updated built-in
  scorer hashes. Custom and plugin scorers retain exact identity checks.
- Validate explicit prediction IDs for flat as well as structured mappings in
  `harness/evaluation.py`, excluding the ID field from target labels.
- Validate downloaded Modal manifests, required files, metrics, and identity in
  staging before publishing a complete directory. Identical recovery is a
  read-only success; conflicting existing content produces an explicit error.
- Include recursive artifact file paths, timestamps, and sizes in research
  cache keys so nested evidence changes trigger validation.
- Reject unknown requested and contract backends. Enforce the backend lock
  before applying the remote kernel recursion guard.
- Isolate custom scorer imports, local dependency modules, and temporary search
  paths across scoring, runner initialization, and artifact validation.
- Exclude only `execution.backend` and `execution.locked` from scientific task
  hashes. Recognize legacy hashes for the finite set of valid operational
  variants without rewriting sealed evidence; scientific fields remain bound.
- Pass resolved cloud preflight configuration to remote research preparation,
  keeping inherited research rules and citation inputs consistent.
- Normalize bridge boolean and list inputs before guards and handlers, reject
  non-finite numeric inputs, and return exit code 1 for failed CLI actions.
- Move research scope tests into one canonical module, remove class aliases,
  and align slow-tier membership with consolidated test module paths.
- Classify run-output directories before cosmetic allow-lists and continue
  tracking them even after the ordinary finding limit is reached. Include
  `run_*` directories as well as `kernel_output_*` outputs.
- Add adversarial regression coverage and upgrade the Modal recovery fixture
  to a complete sealed run. Update dispatch context fixtures to include the
  effective configuration, assert restored scorer import state, and check the
  CLI failure code for an unsupported keep claim without verified evidence.

## Approach

Use existing validation and locking primitives at the boundary where each
invariant applies. Avoid broad identity exceptions: scorer compatibility is a
two-sided allow-list, and legacy task hash compatibility varies only known
execution metadata. Recovery compares complete trees before touching an
existing destination. Scorer imports restore previous workspace modules and
paths even when evaluation raises.

## Evidence

Final local verification completed on 2026-11-14:

- `ruff check harness/ tools/ tests/`: **All checks passed** (exit 0).
- `python tools/run_tests.py --smoke`: **49 tests, OK**, 9.8 s including
  discovery/imports (3.708 s in the test runner; exit 0).
- `python tools/run_tests.py --fast`: **1,051 tests, OK (skipped=2)**,
  481.2 s including discovery/imports (473.262 s in the test runner; exit 0).
  There were no failures or errors; the two existing environment-dependent
  skips also occurred in the initial fast run.
- Final targeted verification: **64 tests, OK**, 110.7 s including
  discovery/imports (106.349 s in the test runner; exit 0), covering the
  custom-hook validation class, all 14 audit-hardening regression methods,
  cloud-runner tests, and Kaggle runtime tests.
- `git diff --check`: exit 0.

The new regression methods exercise sibling preservation in both document
shapes and both update entrypoints, flat/structured ID mismatch, two-sided
scorer compatibility, scientific identity and legacy operational hashes,
nested cache tampering, unknown backends, remote-flag lock enforcement,
cross-workspace scorer/dependency imports and exception cleanup, bridge input
normalization, CLI failure codes, allow-listed run-output tracking, immutable
Modal recovery, inherited remote research configuration, and discovery counts.

Local verification logs are retained under
`.ai-harness/qa/audit-hardening/{ruff,smoke,fast,targeted-final}.log`
(ignored workspace evidence, not committed artifacts).

## Impact & follow-ups

No commits, pushes, cloud dispatches, or provider submissions were performed.
Provider recovery is exercised with mocked downloads and sealed local fixtures;
live Modal/Kaggle execution is outside this change's verification scope.
Filesystem cache detection retains the existing timestamp-and-size trust model.
