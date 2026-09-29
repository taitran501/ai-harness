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

The public hooks cover task data loading and validation, model construction, prediction normalization, metric evaluation, validation split construction and checking, EDA extensions, submission validation, and submission building. The generic implementation delegates metrics to `UniversalTaskSpec` and raises `PluginCapabilityError` for submission operations until the adapter supplies those hooks.

`ml-harness-agent context` reports the selected plugin ID and which capabilities it implements. Missing adapters fail with the installed entry-point list. A plugin that lacks official submission hooks cannot be treated as submission-ready.

## Remote runtime, phases, and workspace ownership

Remote execution requires a plugin to declare a runtime manifest with its exact installed distribution and version. The manifest can identify a wheel or a pinned package requirement, plus adapter-specific dependencies and workspace-contained bundle files. Kaggle and Modal receive the selected code and declared dependencies; credentials and task data are handled separately. Uploading local task data requires explicit consent for that provider and the current data fingerprint. Generated Kaggle datasets and Modal volumes remain private by default.

The attached workspace owns `prompts/ml-harness/phase-pack.yaml`, its ordered phase Markdown files, and `.ml-harness/artifacts/<task-id>/phase_history.json`. The IDE agent performs reasoning and web search. ML-Harness renders a phase prompt with task configuration, EDA, run state, and prior responses, then records the response, cited URLs, and workspace artifact paths. The first pass is sequential; experiment, selection, and external-evaluation phases can repeat after evidence changes. A baseline phase result cannot be recorded until measured run metrics are locked as the baseline.

Each dashboard process is bound to loopback and one task workspace. It rejects non-loopback Host headers and cross-origin requests; start a separate dashboard from a separate workspace to work on another task.

## StereoQueerEval adapter boundary

`task_plugins/stereoqueereval` is an optional local adapter. It explicitly loads the selected task workspace's `src.data`, `src.models`, and `src.metrics`; those imports are absent from the core runner. The task-specific config example also lives inside that adapter.

The adapter provides local training-data loading, deterministic grouped folds, and local evaluation. For cloud dispatch it builds a versioned wheel from the installed adapter module and includes that wheel in the remote code bundle. Extra runtime dependencies belong in the task's `runtime.requirements` list. Its official submission builder and validator deliberately remain unavailable because the current [official task page](https://stereoqueereval.github.io/) does not publish their formats or scorer contract. Submission operations therefore fail closed.

## Distribution layout

```text
ml-harness/                         # installable generic runtime
task_plugins/stereoqueereval/       # optional competition adapter distribution
competition workspace/
  configs/task.yaml                 # task-owned declarative contract
  .ml-harness/project.yaml          # local attachment manifest
  .ml-harness/artifacts/<task-id>/  # isolated state and run outputs
```
