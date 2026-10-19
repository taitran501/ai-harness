# Advisory CLI auth probe in `ensure_cli_ready`

- **Date**: 2026-10-19
- **PR**: TBD (this change)
- **Branch**: `feat/cli-auth-probe`
- **Status**: open
- **Commits**: —

## Why

arc-prize-2026 audit gap **G4**: the dispatch preflight (PR #109) only
probes `kaggle --version`, so a CLI that is **present but unauthenticated**
sailed through and failed only mid-push with `Authentication required to
call the Kaggle API` — four times in the arc-prize conversation (exp005
push stage, conversation step indices 6846, 6962, 7184, 7238). The agent
then spent turns hunting for `kaggle.json` (`Get-ChildItem -Recurse`)
after compute had already been queued. Crucially, the failure is
CLI-specific: the SDK-based dispatch path was authenticated (OAuth) while
the CLI was not, so a blanket "fail if CLI unauthenticated" would have
broken a working path.

## What changed

- **`harness/templates/kaggle_template.py`** —
  `KaggleKernelManager.ensure_cli_ready(*, check_python=True,
  check_auth=True)`:
  - New **advisory** auth probe: `kaggle kernels list --mine -v 1`
    (timeout 60s, runs once per process on the same non-cached path as the
    version probe).
  - A non-zero exit whose combined stdout/stderr contains
    `auth` / `login` / `credential` / `kaggle.json` (case-insensitive)
    prints a WARNING to stderr pointing at the fix (`kaggle auth login` or
    `~/.kaggle/kaggle.json`) and noting that SDK-based dispatch may still
    work. **Never fatal** — dispatch continues.
  - Probe timeout or any unexpected exception is skipped silently; a
    non-auth probe failure (e.g. quota) does not warn. Preflight stays a
    pure early-signal layer and cannot become a new failure mode.
  - `check_auth=False` restores the previous probe set (used by tests and
    by callers that intentionally skip the advisory).
  - Both existing call sites (`cloud_runner.run_via_kaggle` preflight and
    `push_code_bundle`) use the default and pick the probe up with no
    change.
- **Tests** — `tests/test_kaggle_preflight.py`: new `CliAuthProbeTests`
  (auth-fail output → WARNING on stderr + version still returned + no
  raise; authenticated → no warning; probe timeout → skipped safely;
  non-auth failure → no warning; `check_auth=False` → single probe; second
  call served from cache → no second auth probe). Existing
  `test_version_is_reported_and_cached` now expects 2 subprocess calls
  (version + auth probe) on the first invocation.

## Approach

The probe is an actual authenticated API call rather than a
`kaggle.json` existence check because the live failure case was exactly
the opposite of "file present": SDK OAuth was valid, the CLI was not, and
no credentials file existed. An existence check would have said "all
good". It is deliberately **advisory** for the same reason — blocking on
it would have failed dispatches whose SDK path was perfectly
authenticated, converting a CLI-only gap into a hard gate. Cost: one extra
CLI API call (≤60s worst case) per process, only on the first
`ensure_cli_ready` call; timeouts degrade to "no warning", never to
failure. Alternative considered: probing `kaggle competitions list` —
rejected, `kernels list --mine` is the cheapest authenticated endpoint
that matches what dispatch actually does (kernel push).

## Evidence

- `tests/test_kaggle_preflight.py` + `tests/test_kaggle_runtime.py`: 44
  passed (6 new auth-probe tests + 1 existing test updated).
- `ruff check harness/templates/kaggle_template.py
  tests/test_kaggle_preflight.py` — clean.
- Full suite: recorded in the PR (green at merge).

## Impact & follow-ups

- Behavior is strictly additive: same success path, one extra advisory
  WARNING line in the worst case. No contract change.
- Follow-up: none for this gap; E2E re-dispatch of the arc-prize run
  validates the probe on the real (now fixed) environment.
