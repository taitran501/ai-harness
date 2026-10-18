# Dead-code trim: remove unreferenced vendored AIDE journal package

- **Date**: 2026-10-18
- **PR**: [#90](https://github.com/taitran501/ai-harness/pull/90)
- **Branch**: `perf/journal-sync-dedupe-and-dead-code-trim`
- **Status**: open
- **Commits**: `00ec8eb`

## Why

The "heaviness" audit that produced the journal-sync-dedupe change ran a
dead-code/duplication scanner over the 8 largest modules plus a repo-wide
reference sweep (`.py/.md/.yaml/.json/.toml/.ipynb`, including `tests/`,
`docs/`, `configs/`). One repo-tracked candidate survived de-biasing
(reference verification against the live tree); several other scanner
candidates were confirmed **live** and deliberately left untouched.

## What changed

- Removed `harness/vendor/aide_journal.py` (~159 lines) and
  `harness/vendor/__init__.py`. `aide_journal.py` is a direct port of
  `WecoAI/aideml/aide/journal.py` (MIT) holding `AIDEOriginalNode` /
  `AIDEOriginalJournal` — zero references anywhere in code, tests, docs,
  configs, or the packaging entry points. The harness has its own
  `TreeJournal` (`harness/tree_journal.py`) which is what every consumer
  actually uses.
- Removed the `vendor/*.py` package-data glob from `pyproject.toml`
  (`[tool.setuptools.package-data] harness = [...]`) since the vendored
  module no longer ships.

## Approach

De-biasing (the guard against deleting live code) was done by
reference-sweep, not by the scanner's verdict alone:

- `harness/vendor/aide_journal.py` — grep for `aide_journal`,
  `AIDEOriginalNode`, `AIDEOriginalJournal` across all source/tests/docs
  → only the file's own definition lines matched. Not imported, not
  referenced in `pyproject.toml` entry points, not used by any test.
  Safe to delete.

**Confirmed live and NOT deleted** (scanner false-positives, caught by
the sweep): `ApprovalTier`, `SessionBudget`, `TypedActionCall`
(dataclasses in `agent_bridge.py`) are all referenced inside the module
(`DEFAULT_ACTION_APPROVAL_TIERS`, `_get_spec_min_improvement`,
`execute_action`/`_dispatch_typed_action`). The `ai_harness`/`ml_harness`
shim packages are live (shipped + asserted by
`tests/test_agent_usability_audit.py`, probed by `status_report.py`).
`tools/remove_dead_functions.py` is the repo's own AST-safe deletion
meta-tool (referenced in the ruff-gate change doc) — kept.

**Scope note — git-ignored generated configs are not repo changes.** The
`configs/exp00X_feat_feature_pipeline.yaml` experiment configs flagged by
the scan are git-**ignored** by design (`.gitignore`:
`configs/exp[0-9][0-9][0-9]_*.yaml`; they are generated per workspace by
`create_experiment`/`write_experiment_config`). Fifteen of them
(`exp002`…`exp016`, all with zero references) were therefore removed as
a **local working-tree cleanup only** — they never existed in the
repository and are not part of this PR.

## Evidence

- Reference sweep: `aide_journal` / `AIDEOriginal*` → 0 external
  matches.
- `python tools/run_tests.py --fast` → 799 tests OK (skipped=1) after
  removal (same count as before; nothing imported the removed package).
- `ruff check harness/ tools/ tests/ --select F,E7` → clean.

## Impact & follow-ups

- ~160 repo lines and one vendored dependency removed; no behavior
  change, no public API removed (the `vendor` package was internal and
  unreferenced).
- The `harness/vendor/` directory is gone; if a future change re-vendors
  an algorithmic core, re-add it under `harness/vendor/` and restore the
  `vendor/*.py` package-data glob.
- None.
