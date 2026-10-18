# Change: minimal CI gate (ruff + full test suite) for main

- **Date**: 2026-10-18
- **PR**: [#104](https://github.com/taitran501/ai-harness/pull/104) (open)
- **Status**: open

## Why

Audit finding (path E): the repository had **no `.github/` at all** — no CI. Every
quality requirement in AGENTS.md ("Pre-Push Quality Gates": `run_tests.py` full
suite + `ruff check`) was **manual and unenforced**. Any push — including direct
pushes to `main` and force-pushes — landed without a gate (this tech-debt session
itself performed two force-pushes + two squash merges without one).

The gate it enforces is already documented and proven: ruff (F, E7, W605 via
`ruff.toml`) + the full 925-test suite (`python tools/run_tests.py`, ~5 min) —
the exact commands the pre-push policy requires.

## What changed

New `.github/workflows/ci.yml` — two parallel jobs on `push` (main) and
`pull_request`:

| Job | Command | Purpose |
|---|---|---|
| `lint` | `ruff check harness/ tools/ tests/` | Static gate, fails fast in ~1s |
| `test` | `python tools/run_tests.py` | Full 925-test suite (the documented pre-push gate) |

**Plus two packaging fixes the simulation surfaced** (see Evidence): `pyarrow`
was used by runtime code — `harness/runner.py:908` writes
`oof_predictions.parquet` and `harness/candidates.py` reads it back — but was
never declared; and `lightgbm`/`xgboost` are the documented tabular model
families (AGENTS.md) that the suite trains, also undeclared. Local dev installs
carried all three incidentally; a clean install failed 9 tests total across two
simulation rounds. Added to core `dependencies`: `pyarrow>=14.0.0`,
`lightgbm>=4.0.0`, `xgboost>=2.0.0` (CatBoost stays optional — graceful
`ImportError` by design, no test trains it).

Design choices:

- **Python 3.11** — matches `requires-python = ">=3.10"` and the local environment
  (suite verified on CPython 3.11).
- **`pip install -e ".[cloud]"`** — core deps plus the `cloud` extra (modal, kaggle)
  for environment parity with the local checkout where the full suite is verified.
  No test imports kaggle/modal at module level (availability is a lazy check in
  `harness/cloud_runner.py`), but parity removes any doubt for the slow cloud tests.
- **ruff pinned to `0.16.2`** — the exact version behind the local gate, so CI and
  local agree on rule behavior.
- **Full suite, not `--fast`** — CI enforces the actual pre-push policy; 925 tests
  in ~5 min is comfortably within CI budgets.
- **No pip cache** — no lockfile exists for `cache: pip` to key on; install is
  wheel-based and fast enough without it.

## Evidence

- YAML validated (`yaml.safe_load`: jobs `lint`, `test`; triggers `push`,
  `pull_request`).
- **Clean-venv CI simulation** (fresh venv → `pip install -e ".[cloud]"` →
  `python tools/run_tests.py`):
  - run 1: **FAILED — 8 errors, one root cause**: `ImportError: Unable to find
    a usable engine; tried using: 'pyarrow', 'fastparquet'` in
    `test_bootstrap_task`, `test_task_validate` (×6), `test_upgrades_roadmap` —
    `runner.py:908` writes OOF via `to_parquet` and tests write parquet fixtures,
    but the dependency was undeclared.
  - run 2 (after `pyarrow>=14.0.0` added): **FAILED — 1 error**: `LightGBM is
    not installed` in `test_upgrades_roadmap` (full multi-variant training run
    with the LightGBM family). AGENTS.md documents LightGBM + XGBoost as the
    implemented tabular model paths, so both were added to core
    `dependencies` (`lightgbm>=4.0.0`, `xgboost>=2.0.0`); CatBoost remains
    intentionally optional (no test trains it).
  - run 3 (fresh venv, final dependency set): **925 tests OK (288.7s, 0 errors**,
    3 skips). Bonus: the 6 `test_e2e_pipeline` LightGBM/XGBoost tests guarded
    by `skipUnless(HAS_LIGHTGBM/HAS_XGBOOST)` were silently skipped in run 2
    (9 skips) and now actually execute (3 skips) — declaring the dependencies
    increased clean-environment coverage, not just gate stability.

**The real CI run then exposed Linux-runner-only test bugs** (masked by
the local Windows dev env), fixed across follow-up commits:

- `test_dashboard.test_static_html_served` asserted an exact `<title>` that
  only matches the **gitignored local `dashboard/dist` build**; clean
  checkouts serve the committed `tools/web/index.html` (different title).
  Fix: assert stable SPA markers, not a build-specific title.
- `test_sync_command._make_workspace` ran git with `check=True`, swallowing
  stderr so CI failures were undiagnosable. Fix: `_git()` helper includes
  stderr+stdout in the raised error.
- `_make_workspace` cloned the live repo, **inheriting its state**. GitHub
  Actions checkouts are shallow (depth 1) AND detached, so the clone was
  shallow + detached: push rejected (`shallow update not allowed`),
  `pull --ff-only` failed (no upstream). Fix: build the fixture from
  `git archive HEAD` (tracked tree, immune to shallow/detached state) into
  a fresh `git init -b main` single-commit repo with a bare origin at the
  same commit. Verified by simulating the exact CI scenario (shallow +
  detached source): push rc=0, ff pull rc=0, no-op pull rc=0.
- Both gate commands verified on the commit's tree: `ruff check` clean;
  `python tools/run_tests.py` → **925 tests OK** (262.3s).
- Workflow takes effect on the PR that introduces it (GitHub Actions evaluates
  the PR head's workflow file).

**Final CI evidence** (run `39144652521`, commit `9adc007`, 2026-10-18): `test`
job — **925 tests OK in ~40 s**, `lint` OK (earlier runs on this branch:
`d41e4d5`/`fafca0c`/`df5fb67`/`3737f22` failed on the issues above;
`776d0c5` was the first green run). Note: `actions/checkout@v4` / 
`actions/setup-python@v5` emit Node.js 20 deprecation annotations; they
migrate to Node 24 automatically — cosmetic, no action required.

## Impact

- `main` now has an enforced gate: direct pushes and PRs must pass ruff + the full
  925-test suite before merge.
- The documented AGENTS.md pre-push policy is now mechanically verified, closing
  the gap where a broken merge could reach `main` unnoticed.
- Fresh installs (pip/clone/CI) now include `pyarrow`, `lightgbm`, and
  `xgboost`, so the runner's OOF parquet path, parquet dataset loading, and the
  documented tabular model families work out of the box instead of failing on
  any machine without a coincidental dev-environment install.
