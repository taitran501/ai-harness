# StereoQueerEval task adapter

This adapter keeps StereoQueerEval data loading, model construction, output normalization, and task metrics outside the ML-Harness core. It loads the competition workspace's `src.data`, `src.models`, and `src.metrics` modules only when this adapter is selected.

From the root of the ML-Harness checkout, after placing or cloning the competition repository beside it:

```powershell
python -m pip install -e .
python -m pip install -e .\task_plugins\stereoqueereval
ml-harness plugins
ml-harness attach --project-root C:\path\to\competition --task-config configs/task.yaml --plugin stereoqueereval
ml-harness-agent --project-root C:\path\to\competition context
```

Run an experiment through the installed harness CLI:

```powershell
ml-harness-run --project-root C:\path\to\competition --config configs/exp001_baseline.yaml --debug --fold 0
```

The plugin currently supports training-data loading, local grouped validation metrics, the task model factory, and a versioned adapter wheel for Kaggle/Modal remote execution. Extra remote dependencies can be pinned in `task.runtime.requirements` in the workspace task config. If no split file is declared, it creates deterministic local GroupKFold assignments from the configured group column. Its submission builder and official submission validator remain unavailable until the organizers publish their final formats and scorer contract. The current official task page identifies the shared task but does not publish those technical details.

Local dataset upload requires a separate confirmation for each provider and data fingerprint. The harness creates a new private Kaggle dataset or Modal volume for that fingerprint; it does not overwrite a caller-selected resource name. Kaggle quota and Modal monthly billing details are read from the official CLI when available. The Modal summary reports spend, not remaining quota.
