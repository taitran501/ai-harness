# Task plugins and local agent integration

ML-Harness is installed as a Python distribution. Competition adapters are separate distributions discovered through the `ml_harness.task_plugins` entry-point group. The harness does not inspect a workspace's `src` package or guess a competition from its directory name.

## Install and attach

Install the core runtime, then the adapter selected for a competition:

```powershell
python -m pip install -e .
python -m pip install -e .\task_plugins\stereoqueereval
ml-harness plugins
ml-harness attach --project-root C:\path\to\competition --task-config configs/task.yaml --plugin stereoqueereval
```

Attach writes a small `.ml-harness/project.yaml` manifest and `.agents/skills/ml-harness/SKILL.md`. It does not rewrite the competition's task config. All experiment state, including the generated experiment leaderboard, belongs to `.ml-harness/artifacts/<task-id>/`.

The agent and runner accept explicit workspace paths, so a console script installed in a virtual environment can work from any current directory:

```powershell
ml-harness-agent --project-root C:\path\to\competition context
ml-harness-agent --project-root . act --action inspect_data_readiness
ml-harness-run --project-root . --config configs\exp001.yaml --debug --fold 0
```

## Adapter contract

Implement `harness.plugins.TaskPlugin` and register it in your distribution's `pyproject.toml`:

```toml
[project.entry-points."ml_harness.task_plugins"]
my_competition = "my_adapter:MyCompetitionPlugin"
```

The public hooks cover task data loading and validation, model construction, prediction normalization, metric evaluation, submission validation, and submission building. The generic implementation delegates metrics to `UniversalTaskSpec` and raises `PluginCapabilityError` for submission operations until the adapter supplies those hooks.

`ml-harness-agent context` reports the selected plugin ID and which capabilities it implements. Missing adapters fail with the installed entry-point list. A plugin that lacks official submission hooks cannot be treated as submission-ready.

## StereoQueerEval adapter boundary

`task_plugins/stereoqueereval` is an optional local adapter. It explicitly loads the selected task workspace's `src.data`, `src.models`, and `src.metrics`; those imports are absent from the core runner. The task-specific config example also lives inside that adapter.

The adapter provides local training-data loading, deterministic grouped folds, and local evaluation. Remote execution also fails closed because it does not install the adapter in the remote runtime yet. Its official submission builder and validator deliberately remain unavailable because the current [official task page](https://stereoqueereval.github.io/) does not publish their formats or scorer contract. These limits are reported through plugin capabilities and submission operations fail closed.

## Distribution layout

```text
ml-harness/                         # installable generic runtime
task_plugins/stereoqueereval/       # optional competition adapter distribution
competition workspace/
  configs/task.yaml                 # task-owned declarative contract
  .ml-harness/project.yaml          # local attachment manifest
  .ml-harness/artifacts/<task-id>/  # isolated state and run outputs
```
