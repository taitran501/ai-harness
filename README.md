# ML-Harness

ML-Harness connects an IDE agent to a competition workspace. The agent handles research, web search, data analysis, and solution development; ML-Harness provides task context, phase prompts, experiment tools, provider integrations, and artifact tracking.

## Install

Requires Python 3.10+ and Git. The base install supports workspace attachment, phase prompts, plugin discovery, and basic task context. Add the `ml` extra when you need EDA, experiment execution, full Agent Bridge diagnostics, or the dashboard.

Using [`uv`](https://docs.astral.sh/uv/getting-started/installation/):

~~~powershell
$HarnessEnvPath = Join-Path $env:USERPROFILE ".venvs/ml-harness"
uv venv $HarnessEnvPath --python 3.11
uv pip install --python "$HarnessEnvPath\Scripts\python.exe" "ml-harness @ git+https://github.com/taitran501/ml-harness.git@feature/ml-harness-installable-plugin"
& "$HarnessEnvPath\Scripts\Activate.ps1"
ml-harness plugins
~~~

Enable the local ML tools when you need them:

~~~powershell
uv pip install "ml-harness[ml] @ git+https://github.com/taitran501/ml-harness.git@feature/ml-harness-installable-plugin"
~~~

Install the `kaggle` or `modal` extra only for the provider you use. To use a task adapter, install it into the same environment; see the [StereoQueerEval adapter guide](task_plugins/stereoqueereval/README.md) for an example.

With pip, create and activate a virtual environment, then run the same package commands using `python -m pip install`.

## Attach and work with a task

Run these commands from a competition workspace containing `configs/task.yaml`:

~~~powershell
ml-harness attach --project-root . --task-config configs/task.yaml
ml-harness phase validate --project-root .
ml-harness phase render --project-root . --phase setup
ml-harness-agent --project-root . context
~~~

For EDA, model runs, full diagnostics, and the dashboard, install the `ml` extra first. For example:

~~~powershell
ml-harness eda run --project-root .
ml-harness-run --project-root . --config configs/exp000_baseline.yaml --debug --fold 0
ml-harness-dashboard --project-root .
~~~

## Documentation

- [PowerShell end-to-end cookbook](docs/cookbook.md)
- [Installation profiles and task attachment](docs/installation.md)
- [Agent and phase workflow](docs/getting-started.md)
- [Cloud execution](docs/cloud-execution.md)
- [Task adapter API](docs/task-adapters.md)
- [Task specification template](docs/TASK_SPEC_TEMPLATE.md)
- [Architecture](docs/architecture.md)
- [StereoQueerEval adapter](task_plugins/stereoqueereval/README.md)
