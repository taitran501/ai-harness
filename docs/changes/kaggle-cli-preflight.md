# Kaggle dispatch preflight: verify the CLI before uploading anything

- **Date**: 2026-10-19
- **PR**: [#109](https://github.com/taitran501/ai-harness/pull/109)
- **Branch**: `fix/arc-prize-audit-remediation`
- **Status**: merged
- **Commits**: `abb485b` (squash)

## Why

Remote dispatch assumed the Kaggle CLI was present and healthy. When it
was not (missing, broken, or on a Python/runtime mismatch), failures
surfaced mid-dispatch — after dataset upload attempts — with poor
diagnostics (see the empty-error failure of job_20261019_095919). The
documented environment contract (`docs/kaggle_execution.md`: kaggle
CLI, Python 3.11+) was never verified at the single most useful moment:
before any network call.

## What changed

- **`harness/templates/kaggle_template.py`** — new
  `KaggleKernelManager.ensure_cli_ready()`:
  - Probes `kaggle --version` (30s timeout, cached per process in
    `KaggleKernelManager._cli_version_cache`).
  - Missing CLI (`FileNotFoundError`) → `RuntimeError` with install
    guidance pointing at `docs/kaggle_execution.md` + `kaggle auth
    login`.
  - Probe timeout or non-zero exit → `RuntimeError` carrying the CLI
    output.
  - Python < 3.11 → visible `WARNING` on stderr (the documented kernel
    image contract is 3.11+; a mismatched local runtime produces errors
    that do not reproduce remotely) — warning, not fatal, so legacy
    machines keep working.
  - Prints the resolved CLI + Python versions on first probe.
- **`harness/cloud_runner.py`** — `run_via_kaggle` calls
  `ensure_cli_ready()` before dataset upload, bundling, or kernel push.
- **`harness/templates/kaggle_template.py`** — `push_code_bundle`
  calls `ensure_cli_ready()` as well (covers the staged-package CLI
  entry point).
- **`tests/test_kaggle_preflight.py` (new, part 2)** —
  `EnsureCliReadyTests`: missing CLI guidance; broken CLI output
  surfaced; version reported once and cached (single subprocess call).
- **`tests/test_kaggle_runtime.py`** — the dispatch flow test now
  patches `ensure_cli_ready` (it previously patched only
  `subprocess.run`, which the new preflight consumes with Mock-shaped
  output).

## Approach

Fail-fast preflight at the two entry points that touch the network
(`run_via_kaggle` orchestrator and `push_code_bundle`), not sprinkled
around each CLI call: one cached probe per process keeps the overhead
negligible while guaranteeing the first failure message is an
environment diagnosis, not a mid-upload mystery. Version equality
against the kernel image is intentionally **not** enforced (Kaggle
controls the kernel runtime; only the documented minimum — Python 3.11
— is checked, as a warning). Alternative considered — asserting the
exact kaggle CLI major version — rejected as brittle across CLI
releases; the probe reports the version for diagnostics and the
workspace can pin it in its own config if needed.

## Evidence

- `tests/test_kaggle_preflight.py::EnsureCliReadyTests`: 3/3 pass.
- `test_dispatch_uses_submitted_id_and_cli_can_be_in_a_separate_environment`
  (full dispatch flow with mocked CLI): passes.
- Live probe on the dev machine: `Kaggle CLI 2.2.4`, Python 3.11.9 —
  preflight prints and caches.

## Impact & follow-ups

- Any Kaggle dispatch now starts with a visible, attributable
  environment check (`[kaggle preflight] kaggle CLI ready: ...`).
- No behaviour change for healthy environments beyond the one-line
  preflight print.
- Follow-up (none blocking): nothing.
