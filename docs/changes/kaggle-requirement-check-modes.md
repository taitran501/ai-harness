# `requirement_check` modes (strict | warn | off) + detected versions in `ensure_dependencies`

- **Date**: 2026-10-19
- **PR**: TBD (this change)
- **Branch**: `feat/requirement-check-modes`
- **Status**: open
- **Commits**: —

## Why

The arc-prize-2026 audit (gap G3) found multiple remote kernels dying at
**startup** on the dependency gate, not on the task code. The bundle's
`runtime.requirements` carry exact pins derived from the local dev
environment (`numpy==1.26.4, pandas==3.0.5, scikit-learn==1.8.0, scipy==1.17.1,
xgboost==3.2.0`), while the Kaggle script-kernel image (Python 3.13, pandas
2.x, as observed in the kernel error logs) cannot satisfy them offline.
`ensure_dependencies` then raised a bare "image does not satisfy: …" — the
agent could not tell whether the gap was one minor version or a major
incompatible one (e.g. `pandas>=3.0.0` vs image pandas 2.x means the task
code itself would break, not just the pin), and the only escape hatch was
flipping `enable_internet`, changing the pin by guesswork, or re-dispatching.
At least three dispatches (exp002, exp005 ×2, exp006 in the arc-prize
conversation) were lost to this.

## What changed

- **`harness/kaggle_runtime.py`** — `ensure_dependencies(runtime, *,
  enable_internet=False, requirement_check=None)`:
  - New mode resolution: explicit arg > `runtime["requirement_check"]`
    manifest value > `strict`. Invalid modes raise a `ValueError` naming the
    allowed values.
  - Every unsatisfied requirement is now reported with its **detected image
    version** via `importlib.metadata.version()` (no module import — the
    function is loaded by path before dependencies exist):
    `pandas>=3.0.0 (detected: pandas=2.2.2)`, or `… (detected: xgboost=not
    installed)` for missing distributions.
  - Mode semantics:
    - `strict` (default, unchanged contract): offline + unsatisfied →
      `RuntimeError` (now with detected versions); online → install, then
      re-verify and raise if still unsatisfied.
    - `warn`: offline → log `[requirement_check=warn] WARNING: … (detected:
      …)` to the kernel log and continue with image versions; online →
      install as before, but a still-unsatisfied post-install check logs a
      WARNING instead of raising.
    - `off`: skip the version gate **and** any install; detected versions are
      still logged (`[requirement_check=off] skipping dependency version
      gate; detected: …`).
  - The packaging `Requirement` import is lazy (`_requirement_cls()`), so the
    module still loads before harness dependencies exist.
- **`harness/templates/kaggle_template.py`** — `setup_kernel_package(...,
  requirement_check="strict")`: validated at dispatch time (typo →
  `ValueError` before any push), pinned into the bundle's
  `ai-harness-runtime.json` (so the kernel reads the same mode the dispatch
  side saw) and into the generated `run_kaggle.py`
  (`ensure_dependencies(runtime, enable_internet=…, requirement_check=…)`).
  `requirement_check` is part of the bundle cache key, so changing the mode
  rebuilds the bundle.
- **`harness/cloud_runner.py`** — `run_via_kaggle` reads
  `task.runtime.requirement_check` from the task contract (same source as
  `enable_internet` / `kaggle_bundle_mode`) and passes it to
  `setup_kernel_package`; the dry-run dispatch plan now reports it too.
- **Docs** — `docs/TASK_SPEC_TEMPLATE.md` new section 2.7 (Kaggle runtime
  options incl. `requirement_check`); `docs/kaggle_execution.md` new
  "Dependency version policy in the kernel" subsection (modes table, image
  facts, when to use `warn`/`off`).
- **Tests** — `tests/test_kaggle_runtime.py`: strict error carries detected
  versions (incl. `not installed`), `warn` continues offline without install,
  `off` skips gate + install but logs detected versions, mode resolution
  (explicit > manifest > default) + invalid-value `ValueError`, online
  `warn` downgrades the post-install gate to a warning, and the codegen pin
  (generated script + `ai-harness-runtime.json` carry the mode; invalid value
  rejected at dispatch).

## Approach

The mode travels with the **code bundle**, not as a new CLI flag: the kernel
script is generated once per dispatch and cannot ask the operator questions,
so the gate decision must be deterministic at codegen time. Storing it in
`ai-harness-runtime.json` (which the script already loads) plus passing it
explicitly keeps the two sources consistent and survives the bundled-harness
cache. `importlib.metadata` was chosen over `importlib` for version lookup
because the gate exists precisely to run *before* the (potentially broken)
native modules are importable; reading distribution metadata never imports
the package. Alternatives considered: (a) auto-relaxing pins to detected
versions — rejected, silently downgrading `pandas==3.0.5` to 2.x would run
task code against an incompatible major and produce wrong scores instead of a
loud failure; (b) a per-requirement toggle — rejected, one knob covers the
observed failure pattern and keeps the contract surface small.

## Evidence

- `tests/test_kaggle_runtime.py`: 28 passed (7 new for G3: strict detected
  versions, `not installed` rendering, `warn` offline, `off` skip+log, mode
  resolution/validation, online `warn` downgrade, codegen pin + dispatch
  validation).
- `ruff check harness/ tests/` — clean.
- `python tools/run_tests.py --smoke` — 47 tests OK.
- Full suite: recorded in the PR (green at merge).

## Impact & follow-ups

- Default behavior is **unchanged** for existing task configs (`strict`);
  only the error message gained the detected versions.
- Follow-up (G4, separate PR): CLI auth probe in `ensure_cli_ready` — the
  same conversation shows unauthenticated CLIs failing only at push time.
- Note: the arc-prize local env gap (competition wheels requiring Python
  ≥3.12 vs local 3.11) is an agent-side environment issue, not a framework
  gate; tracked as a future advisory, out of scope here.
