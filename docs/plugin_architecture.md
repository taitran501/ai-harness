# Task plugins and local agent integration

> [!IMPORTANT]
> **Verification Scope: Strictly NLP / Text Only**
> AI-Harness is currently **strictly verified and working well ONLY on Natural Language Processing (NLP / Text)** tasks. Non-NLP modalities are experimental WIP on the roadmap.

AI-Harness is installed as a Python distribution. Competition adapters are separate distributions discovered through the `ai_harness.task_plugins` entry-point group. The harness does not inspect a workspace's `src` package or guess a competition from its directory name.

## Install and attach

Install the core runtime, then the adapter selected for a competition:

```powershell
python -m pip install -e .
ai-harness plugins
ai-harness attach --project-root C:\path\to\competition --task-config configs/task.yaml --plugin <plugin-name>
```

Attach writes a small `.ai-harness/project.yaml` manifest and `.agents/skills/ai-harness/SKILL.md`. It does not rewrite the competition's task config. All experiment state, including the generated experiment leaderboard, belongs to `.ai-harness/artifacts/<task-id>/`.

The agent and runner accept explicit workspace paths, so a console script installed in a virtual environment can work from any current directory:

```powershell
ai-harness-agent --project-root C:\path\to\competition context
ai-harness-agent --project-root . act --action inspect_data_readiness
ai-harness-run --project-root . --config configs\exp001.yaml --debug --fold 0
```

## Adapter contract

Implement `harness.plugins.TaskPlugin` and register it in your distribution's `pyproject.toml`:

```toml
[project.entry-points."ai_harness.task_plugins"]
my_competition = "my_adapter:MyCompetitionPlugin"
```

The public hooks cover task data loading and validation, model construction, prediction normalization, metric evaluation, validation split construction and checking, EDA extensions, submission validation, and submission building. The generic implementation delegates metrics to `UniversalTaskSpec` and raises `PluginCapabilityError` for submission operations until the adapter supplies those hooks.

`ai-harness-agent context` reports the selected plugin ID and which capabilities it implements. Missing adapters fail with the installed entry-point list. A plugin that lacks official submission hooks cannot be treated as submission-ready.

## Remote runtime, phases, and workspace ownership

Remote execution requires a plugin to declare a runtime manifest with its exact installed distribution and version. The manifest can identify a wheel or a pinned package requirement, plus adapter-specific dependencies and workspace-contained bundle files. Kaggle and Modal receive the selected code and declared dependencies; credentials and task data are handled separately. Uploading local task data requires explicit consent for that provider and the current data fingerprint. Generated Kaggle datasets and Modal volumes remain private by default.

The attached workspace owns `prompts/ai-harness/phase-pack.yaml`, its ordered phase Markdown files, and `.ai-harness/artifacts/<task-id>/phase_history.json`. The IDE agent performs reasoning and web search. AI-Harness renders a phase prompt with task configuration, EDA, run state, and prior responses, then records the response, cited URLs, and workspace artifact paths. The first pass is sequential; experiment, selection, and external-evaluation phases can repeat after evidence changes. A baseline phase result cannot be recorded until measured run metrics are locked as the baseline.

Each dashboard process is bound to loopback and one task workspace. It rejects non-loopback Host headers and cross-origin requests; start a separate dashboard from a separate workspace to work on another task.

## Distribution layout

```text
ai-harness/                         # installable generic runtime
competition workspace/
  configs/task.yaml                 # task-owned declarative contract
  .ai-harness/project.yaml          # local attachment manifest
  .ai-harness/artifacts/<task-id>/  # isolated state and run outputs
```
