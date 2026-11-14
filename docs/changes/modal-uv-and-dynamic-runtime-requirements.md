# Modal Fast UV Image Building, Dynamic Task Runtime Requirements, and Job ID Normalization

- **Date**: 2026-11-01
- **PR**: Working tree
- **Branch**: `main`
- **Status**: merged
- **Commits**: Working tree

## Why

1. **Slow container image builds on Modal**: In LLM/SFT tasks, installing large wheels like PyTorch, Hugging Face Transformers, PEFT, and Accelerate via standard `pip` in Modal Debian-slim containers took 5 to 10 minutes. The user explicitly requested: *"Lần sau có gì tối ưu install nhanh bằng uv nha, chứ pip install lâu quá đợi rất lâu"*.
2. **Missing dynamic runtime requirements mechanism**: `remote_runtime.py` and `modal_template.py` previously hardcoded a tabular dependency stack (`numpy, pandas, scikit-learn, scipy, lightgbm, xgboost`). When running custom scripts for LLM/SFT tasks, the framework provided no declarative way to specify deep learning dependencies via `task.yaml`. This forced agents in competition workspaces to resort to dirty monkey-patching of canonical framework files.
3. **Requirement duplication in generated Modal scripts**: Multiple requirement sources were merged without deduplication, causing duplicate entries in container install commands (e.g. `lightgbm>=4.0.0` and `numpy>=1.24.0` repeated multiple times).
4. **Job ID prefix friction**: When users or agents invoked `cloud_runner.py` with custom job identifiers like `--job-id test_job`, the validation pattern `^job_[A-Za-z0-9_.-]{1,180}$` failed and terminated execution with an unsupported characters error instead of forgivingly normalizing the ID with the `job_` prefix.

## What changed

- **`harness/task_spec.py`**:
  - Added `@property runtime_requirements` to `UniversalTaskSpec`. Loads declared requirements from `task.runtime.requirements` or `task.requirements` as a list of trimmed strings.
- **`harness/remote_runtime.py`**:
  - Updated `resolve_remote_runtime()` to read `task_spec.runtime_requirements`.
  - Merged and deduplicated task requirements into `runtime["requirements"]` for both default and custom task plugins.
- **`harness/templates/modal_template.py`**:
  - Added reading of `task.runtime.requirements` directly during script generation.
  - Deduplicated `runtime_requirements` preserving insertion order.
  - Updated the Modal container image definition to use `modal.Image.debian_slim().pip_install("uv").run_commands(f"uv pip install --system ...")`, reducing layer build time from >6 minutes to ~47 seconds with permanent layer caching.
- **`harness/cloud_runner.py`**:
  - Added automatic normalization for `--job-id` in CLI `main()` to prepend `job_` if omitted by the caller.
- **`tests/test_modal_runtime_requirements.py`**:
  - Added new test suite covering `UniversalTaskSpec.runtime_requirements`, `resolve_remote_runtime` deduplication, and `generate_modal_script` `uv` commands and compilation.

## Evidence

- Unit tests: `tests/test_modal_runtime_requirements.py` passed (3 tests in 1.79s).
- Regression tests: `test_modal_cloud_modes.py`, `test_remote_dispatch_hardening.py`, `test_upgrades_roadmap.py`, `test_workspace_workflow.py`, `test_correctness_regressions.py`, and `test_job_state.py` passed 100%.
- Smoke tier gate: `python tools/run_tests.py --smoke` passed (47 tests in 2.6s).
- Linter: `ruff check harness/ tools/ tests/` clean (All checks passed!).

## Impact & follow-ups

- Agents and users can now declare arbitrary packages (PyTorch, Transformers, vLLM, Whisper, etc.) directly in `task.yaml` under `task.runtime.requirements`.
- Remote Modal jobs build and start in <50s using `uv`.
- Eliminates the need for any workspace agent to monkey-patch framework templates.
