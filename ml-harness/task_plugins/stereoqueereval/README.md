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

The plugin currently supports training-data loading, local grouped validation metrics, and the task model factory. If no split file is declared, it creates deterministic local GroupKFold assignments from the configured group column. Its submission builder and official submission validator remain unavailable until the organizers publish their final formats and scorer contract. The current official task page identifies the shared task but does not publish those technical details.
