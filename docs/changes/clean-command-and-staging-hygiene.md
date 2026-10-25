# Safe clean command and cloud staging hygiene

- **Date**: 2026-10-25
- **PR**: Not opened
- **Branch**: `main` (local changes)
- **Status**: local implementation
- **Commits**: Not committed

## Why

The workspace hygiene allow-list hid `.ml-harness` left after the framework
rename. Cloud dispatch allocated persistent staging directories without
finally-based cleanup. Single-fold probes also accumulated without an explicit
cleanup command. Deleting arbitrary root clutter would endanger user data;
cleanup needs a narrow inventory and a separately authorized apply operation.

## What changed

- `harness/cleaner.py`: typed cleanup inventory, strict location eligibility,
  ancestor/descendant link and Windows junction rejection, protected task inputs,
  age and changed-inventory guards, live staging PID checks and verified legacy
  ZIP backups. System temp uses exactly four prefixes and a two-hour floor.
- `harness/cli.py`: `clean` with explicit categories, preview by default,
  confirmation for `--apply`, `--yes`, age selection and legacy `--archive`.
- `harness/cloud_runner.py`: Kaggle and Modal staging is cleaned in `finally`
  across success, failures and Kaggle dry-run. API `keep_staging=True` and
  `python -m harness.cloud_runner --keep-staging` retain debug bundles.
- `harness/hygiene.py`: legacy state now produces an advisory pointing to the
  safe preview command, even though it is a hidden directory.
- `tests/test_cleaner.py`: isolated adversarial safety and staging lifecycle
  tests. Existing Kaggle dispatch verification reads metadata while staging
  exists and checks that the directory disappears after dispatch returns.

## Approach

See [architecture and test plan](../plans/clean_feature_plan.md). No generic
filesystem deletion option is provided. Only individual probe runs and inactive
legacy state are eligible in workspaces. Legacy state still used for attachment
or artifacts is retained. Source/data directories inside a legacy candidate
cause the whole candidate to be retained. Declared task data in temp is also
protected using the selected workspace, including revalidation at apply time.

Examples (first command writes nothing):

```powershell
ai-harness clean --project-root . --legacy --probes --temp
ai-harness clean --project-root . --legacy --apply --archive
ai-harness clean --project-root . --probes --older-than-hours 24 --apply -y
```

`--archive` backs up legacy only; probe/temp deletion remains destructive with
`--apply`. ZIP backups live in `.ai-harness/archives/` and are never pruned.
Reported deleted source bytes are not net reclaimed disk space when archiving.

## Evidence

- Codex CLI code review (`gpt-6-luna`, `reasoning_effort=xhigh`): passed with
  2 addressed P2 findings:
  1. Active probe runs (`artifact_manifest.json` status `RUNNING` or `.state.lock`)
     are explicitly refused and excluded from cleanup scans.
  2. `dispatch_kernel()` now marks staging with `STAGING_MARKER` containing the
     producer PID to protect active in-flight pushes from premature temp cleanup.
- `python tools/run_tests.py test_cleaner`: 29 tests pass, including Windows
  link/junction guards, archive failures, forged plans, active staging, active
  probe locks, and both cloud runner lifecycles.
- `python tools/run_tests.py --smoke`: 47 tests pass.
- Affected Kaggle dispatch regression plus `tests.agent_bridge.test_cloud_hygiene_bypass`:
  9 tests pass.
- `ruff check harness/ tests/` and `git diff --check`: clean. Both CLI help
  entry points expose the new flags.
- The broader `test_kaggle_runtime` module has an existing failure in
  `test_mount_resolution_matches_selected_owner_and_rejects_unrelated_data`:
  it expects `missing`, but the runtime raises `Ambiguous selected Kaggle
  dataset mount`. Reproduced on unmodified `HEAD` extracted via `git archive`
  into a disposable directory. This unrelated failure was not changed. The
  full repository suite was not run; no commit or push was requested.

All cleanup tests use disposable directories; real workspace and system temp
are not pruned. Cloud calls are mocked; these results do not establish live
Kaggle or Modal execution success.

## Impact & follow-ups

No artifact/config migration or real cleanup is performed by this change.
Two-hour temp retention cannot be disabled; recent descendants reset eligibility.
PID markers protect new active dispatches. Historical unmarked staging is
eligible by location and age, so users should review the preview. Portable
path-based operations cannot guarantee immunity to hostile concurrent renames;
links and changed inventories are checked before apply and entries are checked
again during deletion. Filesystem errors can leave a partially cleaned candidate
and are reported; cleanup does not silently broaden its scope or retry elsewhere.
