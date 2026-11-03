# Modal Attached Live-Stream and Cloud Detached Modes

- **Date**: 2026-11-01
- **PR**: Working tree
- **Branch**: `main`
- **Status**: merged
- **Commits**: working tree

## Why

Modal serverless GPU execution supports two natural developer workflows:
1. **Attached / Live-stream Mode** (`modal run app.py`): Connected directly in the terminal (CMD/PowerShell), streaming container build and training logs live line-by-line, and importing output artifacts upon completion.
2. **Cloud / Detached Mode** (`modal run --detach app.py`): The local terminal dispatches the training job to Modal Cloud and disconnects immediately; containers run independently on cloud GPUs without risk of disruption if the local machine sleeps, restarts, or disconnects; logs are queried via `modal app logs <app_id>` or the web dashboard, and artifacts persist on a Modal Volume.

Prior to this change, `ai-harness` failed under both modes:
- **Attached Mode Failures**:
  - `push_dataset_to_modal` invoked `modal volume put ...` with `capture_output=True`, silently uploading large datasets (e.g. 6 GB in competitive computer vision/NLP) with 0 bytes of stdout progress, giving users and agents the false impression of an indefinite freeze. Furthermore, it re-uploaded datasets on every dispatch even if the content fingerprint was already present in the Modal Volume.
  - On Windows, `CloudJobManager.start_job` spawned background processes with `creationflags = CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS`. Detaching standard console handles broke Python Click/Rich terminal handshakes inside Modal CLI, causing worker processes to crash immediately at startup (`pid died before completion`).
  - In `generate_modal_script`, `excluded_data_roots` stored absolute Windows paths instead of workspace-relative paths, which prevented `_ignore_workspace_path` from filtering declared data directories when `image.add_local_dir(".")` packaged the image. This bundled multi-GB dataset trees into the container image and triggered a fatal runtime exception: `RuntimeError: Uploaded task data conflicts with bundled task code`.
- **Cloud Detached Mode Inability**:
  - `cloud_runner.py` had no `--detach` flag and required the local process to stay alive until the entire multi-hour training run completed.
  - `CloudJobManager.get_job()` checked liveness strictly via local operating system PID (`_pid_running(job.pid)`). When the local process exited, the harness immediately marked the job as `FAILED` (`pid died before completion`), even though the training container was actively executing on Modal Cloud.
  - Output artifacts were only returned via in-memory values from `@app.local_entrypoint()`, which are unavailable in detached mode.

## What changed

- **`harness/templates/modal_template.py`**:
  - Fixed `excluded_data_roots` to use relative workspace paths so declared data directories are correctly ignored from `image.add_local_dir(".")`.
  - Added smart volume check in `push_dataset_to_modal`: checks if `aih-data-bundle.json` exists in `/task-data` via `modal volume ls` before uploading, skipping multi-GB re-uploads and printing explicit progress milestones.
  - Added volume artifact persistence in `run_training_remote`: container now copies essential output artifacts (`metrics.json`, `oof_predictions.csv`, `artifact_manifest.json`, etc.) directly to `/mnt/cloud_data/artifacts/<exp_id>` and commits the volume with `data_volume.commit()`.
  - Added `ModalAppManager` class with:
    - `get_app_status(app_id)`: checks `modal app list --json` and inspects recent logs to distinguish `RUNNING`, `COMPLETED`, and `FAILED` states.
    - `fetch_app_logs(app_id, tail)`: fetches remote container logs.
    - `wait_for_app(app_id, ...)`: polls until terminal state.
    - `pull_app_output(exp_id, volume_name, artifact_dir, ...)`: downloads and verifies artifacts from Modal Volume into `artifacts/<exp_id>/`.
- **`harness/cloud_runner.py`**:
  - Added `--detach` CLI flag and `detach: bool = False` argument to `run_via_modal`.
  - In detached mode, `run_via_modal` executes `modal run --detach`, queries the deployed `app_id` via `modal app list --json`, registers `remote_id = app_id`, outputs live log inspection instructions, and returns cleanly with status `RUNNING`.
- **`harness/cloud_job_runner.py`**:
  - Added `_recover_modal_job(job)` to `CloudJobManager`: when a local PID is no longer running, queries Modal provider state via `ModalAppManager`. If still running, retains status `RUNNING`; if completed, pulls artifacts from Modal Volume and runs post-run verification; if failed, imports failure logs.
  - In Windows process launch, removed `DETACHED_PROCESS` from `popen_kwargs["creationflags"]` (retaining `CREATE_NEW_PROCESS_GROUP`), preventing Click/Rich CLI crashes.
  - Added `detach: bool = False` to `start_job()`.
- **`harness/runner.py`**:
  - Added `--detach` and `--gpu` arguments to `build_parser()` and forwarded them to `run_via_modal()`.
- **`harness/bridge_actions.py`**:
  - Updated `_handle_inspect_cloud_job_or_wait_cloud_job` to handle `ap-` Modal App IDs directly as a fallback.
  - Forwarded `detach` flag in `_handle_run_full_validation`.
- **`tests/test_modal_cloud_modes.py`**:
  - Added unit test suite covering `ModalAppManager`, `_recover_modal_job`, dataset bundle caching/skipping, and relative excluded data path generation.

## Approach

1. **Dual-Mode Parity**: Mirror the robust recovery pattern proven by Kaggle (`KaggleKernelManager` and `_recover_kaggle_job`). For attached runs, standard `modal run` executes with live output streaming. For detached runs, `modal run --detach` delegates execution to Modal Cloud and tracks status via App ID (`ap-xxxxxx`).
2. **Persistent Storage for Detached Artifacts**: Decoupled artifact retrieval from the local entrypoint return bundle by writing artifacts to the persistent Modal Volume mounted at `/mnt/cloud_data`. This guarantees artifacts survive disconnects and can be downloaded anytime via `ModalAppManager.pull_app_output`.
3. **Smart Data Volume Reuse**: Instead of copying and re-uploading gigabytes of data on every experiment run, the volume is fingerprinted. If `aih-data-bundle.json` is already detected in `/task-data` on the volume, the upload step completes in seconds.

## Evidence

- **Unit Tests**:
  - `python -m unittest tests/test_modal_cloud_modes.py`: 8 tests passed in 2.1s (App status states, manifest verification, relative excluded roots, volume skip logic, job recovery).
  - `python -m unittest tests/test_cleaner.py`: 29 tests passed in 3.6s.
  - `python -m unittest tests/test_cloud_jobs_and_env.py`: 8 tests passed in 9.5s.
  - `python -m unittest tests/test_remote_dispatch_hardening.py`: 15 tests passed in 7.4s.
- **Invariants Gate**:
  - `python tools/run_tests.py --smoke`: 47 tests passed in 2.9s.
- **Static Gate**:
  - `ruff check harness/ tools/ tests/`: All checks passed.

## Impact & follow-ups

Users and agents can now seamlessly dispatch long-running GPU experiments to Modal either attached (`ai-harness-run --backend modal` / `ai-harness-agent act --action run_full_validation --backend modal`) or detached (`--detach`). Disconnecting the terminal or restarting the machine no longer marks running cloud jobs as failed.
