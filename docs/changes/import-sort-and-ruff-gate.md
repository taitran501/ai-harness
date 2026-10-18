# Import sort (I001) + canonical ruff gate (`ruff.toml`)

**PR:** #94 · **Branch:** `fix/style-import-sort-and-ruff-config` · **Date:** 2026-10-18
**Status:** Implemented

## Why

Audit item C (post-#93): the repo had **no ruff config** — the lint gate lived
only as ad-hoc CLI flags in AGENTS.md (`--select F,E7`), so gate policy was not
machine-enforced and invalid-escape regressions (W605; 3 instances fixed in
#92) could return silently. Separately, **139 imports across 97 files were
unsorted** (I001) — a cosmetic debt that made `ruff format`-style tooling
noisy.

Before fixing, the I001 risk surface was verified:

- No `__init__.py` affected (97 files, none are package inits).
- `sys.path` bootstrap hacks in `tools/*.py` sit *between* import blocks;
  ruff preserves block boundaries (non-import statements never move), so the
  "reorder imports above the path hack" failure mode does not occur
  (verified in sample diffs).
- Template modules (`kaggle_template.py`, `modal_template.py`) only have
  module-level imports reordered; embedded template strings are untouched.
- Residual risk (import-order side effects) is covered by the full test
  suite.

## What

- **`ruff check ... --select I001 --fix`** — 139/139 import sorts applied
  (66 harness, 8 tools, 65 tests), pure reordering + multi-line wrapping of
  long import statements.
- **`ruff.toml`** (new): canonical gate — `select = ["F", "E7", "W605"]` with
  `line-length = 120` (informational; **E501 intentionally not gated** — most
  long lines are docstrings/code-samples/URLs where re-wrapping corrupts
  markdown/JSON). `line-length` sits at top level (this ruff version rejects
  it under `[lint]`); the `[lint]` table carries the select policy.
- **`AGENTS.md`** pre-push gate line updated: `ruff check harness/ tools/
  tests/` (no ad-hoc flags; policy now in `ruff.toml`).

## Approach

Policy in config rather than in muscle memory: the gate command in AGENTS.md
drops its flags, so a plain `ruff check` anywhere in the repo reproduces the
exact same gate. W605 joins the gate at zero cost (0 violations at add-time)
and prevents the #92-class escape regressions. The I001 pass is mechanical
(`--fix`) and verified by the suite — the two gates (old ad-hoc `--select
F,E7` and the new config-driven gate) both run green on the same tree.

## Evidence

- `ruff check --select I001 --fix` → "Found 139 errors (139 fixed, 0
  remaining)".
- Post-fix: `ruff check harness/ tools/ tests/` (config-driven) → "All checks
  passed!" (exit 0); `--select F,E7` (old ad-hoc gate) → "All checks passed!"
  (exit 0) — gate equivalence proven.
- Import smoke: `pkgutil.walk_packages` over the whole `harness` package →
  every module imports cleanly after the reorders (no order side effects).
- Gates: fast + full suite OK (see commit / CI on this branch).

## Impact

- The lint gate is now **config-enforced** (W605 included) and flag-free;
  AGENTS.md and the tooling can no longer drift.
- Import blocks across the repo are canonically ordered; future tooling
  (ruff format, IDE isort) starts from a stable baseline.
- E501 stays out of the gate by documented decision, keeping the gate
  zero-noise.
