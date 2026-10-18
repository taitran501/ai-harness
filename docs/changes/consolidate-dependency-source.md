# Change: consolidate dependency source of truth in pyproject.toml

- **Date**: 2026-10-18
- **PR**: #105
- **Branch**: `chore/dep-source`
- **Status**: open

## Why

`requirements.txt` (tracked, 9 entries) duplicated the dependency list that
`pyproject.toml` already owns, and it had **drifted**: it pre-dated the
2026-10-18 dependency fix (missing `pyarrow`/`lightgbm`/`xgboost`,
`modal>=0.62.0` vs the `>=1.5.0` the package declares) and was **referenced
by nothing** — no code, docs, or workflow install from it. Two live lists
guarantee the next drift. The file's only unique content was the dev-path
pair `pytest>=8.0` / `pytest-xdist>=3.6` (parallel `pytest tests -n auto`
path bootstrapped by `conftest.py`; the canonical gate is unittest via
`tools/run_tests.py`).

## What changed

- **`pyproject.toml`** — new `test` extra: `["pytest>=8.0", "pytest-xdist>=3.6"]`,
  with a comment documenting the dev path (`pip install -e ".[test]" &&
  python -m pytest tests -n auto`) and that the canonical gate stays
  unittest-based.
- **`requirements.txt`** — deleted.

## Approach

Single source of truth: `pyproject.toml` (installed by `pip install -e .` /
`.[cloud]`, the path CI and docs use). The pytest pair moved into an optional
extra rather than core dependencies because no gate or runtime code requires
pytest — only the parallel dev path does. A pointer-stub
requirements.txt was considered and rejected: it would still be a second
machine-editable list, just a smaller one.

## Evidence

- `tomllib` parse: 8 core deps; extras = `cloud`, `test`.
- Repo-wide grep: zero remaining references to `requirements.txt`.
- `python tools/run_tests.py --smoke`: **47 tests in 6.7 s — OK**
  (dependency surface unchanged for the canonical runner).

## Impact & follow-ups

- Fresh installs are unaffected (they never used the file). The parallel
  pytest dev path is preserved via `.[test]`.
- `pip install -r requirements.txt` is no longer a supported install path;
  `pip install -e .` / `.[cloud]` / `.[test]` are.
- None.
