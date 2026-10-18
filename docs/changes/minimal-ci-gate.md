# Change: minimal CI gate for main — built, verified, removed before merge

- **Date**: 2026-10-18
- **PR**: [#104](https://github.com/taitran501/ai-harness/pull/104) (open)
- **Branch**: `ci/minimal-gate`
- **Status**: open — **CI workflow removed before merge at the operator's
  decision**; the dependency declarations and test-portability fixes below
  are the net changes that land.

## Why (original)

Audit finding (path E): the repository had **no `.github/` at all** — no CI.
Every quality requirement in AGENTS.md ("Pre-Push Quality Gates":
`run_tests.py` full suite + `ruff check`) was **manual and unenforced**. Any
push — including direct pushes to `main` and force-pushes — landed without a
gate. The gate enforced was already documented and proven: ruff (F, E7, W605
via `ruff.toml`) + the full 925-test suite — the exact commands the pre-push
policy requires.

## Decision: remove the CI workflow before merge

Same day, after the gate was green on every run, the operator decided the
GitHub-runner overhead was not worth it right now (~2 min per run; the test
job's ~40 s of `pip install` is the main cost, plus runner queue time on top
of the job clocks). `.github/workflows/ci.yml` was deleted, so PR #104 lands
**without** CI. The verified workflow recipe survives in this file and in the
branch's run history, so re-adding CI later is a two-file restore (workflow +
this changelog entry).

## Net changes that land

- **`pyproject.toml`** — core `dependencies` gains `pyarrow>=14.0.0`,
  `lightgbm>=4.0.0`, `xgboost>=2.0.0`. All three were used by documented code
  paths but never declared (clean installs failed the OOF-parquet tests and
  the LightGBM/XGBoost e2e tests); CatBoost stays intentionally optional
  (graceful `ImportError`, no test trains it).
- **`tests/test_dashboard.py`** — `test_static_html_served` now asserts stable
  SPA markers instead of a build-specific `<title>` (the exact title only
  matches the gitignored local `dashboard/dist` build; clean checkouts serve
  the committed `tools/web/index.html`).
- **`tests/test_sync_command.py`** — two portability fixes:
  - `_git()` helper surfaces git stderr+stdout on failure (plain
    `check=True` swallowed it, making non-Windows failures undiagnosable).
  - `_make_workspace` builds the throwaway framework checkout from
    `git archive HEAD` (tracked tree) into a fresh `git init -b main`
    single-commit repo with a bare origin at the same commit, instead of
    `git clone` of the live repo. A clone inherits the source's state: on a
    CI-style checkout (shallow + detached) the clone was shallow + detached
    with no upstream, where `git push` is rejected (`shallow update not
    allowed`) and `git pull --ff-only` fails. The archive-built fixture is
    immune to source shallow/detached state on any platform.

## Evidence

- **Clean-venv simulation** (fresh venv → `pip install -e ".[cloud]"` →
  `python tools/run_tests.py`) drove the dependency fixes:
  - run 1 (before): **FAILED — 8 errors**, one root cause:
    `ImportError: Unable to find a usable engine; tried using: 'pyarrow',
    'fastparquet'` in the OOF-parquet path (`harness/runner.py:908` writes
    `oof_predictions.parquet`) and parquet fixture tests.
  - run 2 (after `pyarrow`): **FAILED — 1 error**: `LightGBM is not
    installed` in `test_upgrades_roadmap` → added `lightgbm`/`xgboost`.
  - run 3 (final set): **925 tests OK (288.7 s, 0 errors, 3 skips)** — and
    the 6 `skipUnless(HAS_LIGHTGBM/HAS_XGBOOST)` e2e tests now actually
    execute instead of silently skipping.
- **Real GitHub Actions runs** (while the workflow existed) exposed and
  confirmed the test-portability fixes:
  - failed runs `d41e4d5`/`fafca0c`/`df5fb67`/`3737f22` (dashboard title
    assertion; sync fixture push/pull failures).
  - green runs `776d0c5`, `9adc007` (run `39144652521`: test job **925 tests
    OK in ~40 s**, lint OK) and final `4267c8f` (run `39144875541`: success).
- **Shallow+detached scenario simulation** of the exact CI checkout state:
  archive-built fixture has no `.git/shallow`, push rc=0, fast-forward pull
  rc=0, no-op pull rc=0.
- Local (Windows, CPython 3.11): `ruff check` clean; `python
  tools/run_tests.py` → **925 tests OK** (227.7 s, receipt `a3260687`).

## Impact & follow-ups

- `main` has **no enforced CI gate** (status quo before this PR); the
  AGENTS.md pre-push policy remains a manual convention.
- Fresh installs (pip/clone/any clean environment) now include `pyarrow`,
  `lightgbm`, and `xgboost`, so the runner's OOF parquet path and the
  documented tabular model families work out of the box.
- `test_sync_command` and `test_dashboard` are now environment-independent
  (verified on both a Windows dev machine and the Linux runner).
- Follow-up (optional, deferred): if CI is wanted later, restore
  `.github/workflows/ci.yml` from the branch history (`git log
  ci/minimal-gate -- .github/workflows/ci.yml`) — the last version was
  verified green. Consider a `setup-python` `cache: pip` if re-adding, to cut
  the install step.
