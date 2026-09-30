# ML-Harness

ML-Harness is an installable Python package for working through machine-learning competitions with an IDE agent. The IDE agent handles research, native web search, analysis, and coding. ML-Harness provides task context, ordered prompts, EDA, experiment tracking, local/cloud execution, and artifact history.

## Start here

**End users do not need to clone this repository, build a wheel, install Node.js, or build dashboard assets.** Install the packages with pip, then run the provided commands from your competition workspace.

- [Copy/paste PowerShell cookbook](docs/cookbook.md) — install, attach, bootstrap, EDA, phases, experiments, dashboard, Kaggle, and Modal.
- [Installation and task attachment](docs/installation.md)
- [Agent and phase workflow](docs/getting-started.md)
- [Cloud execution](docs/cloud-execution.md)
- [Task adapter API](docs/task-adapters.md)
- [Task specification template](docs/TASK_SPEC_TEMPLATE.md)
- [Optional StereoQueerEval adapter](task_plugins/stereoqueereval/README.md)
- [Architecture](docs/architecture.md)

## Install directly from GitHub

Python 3.11 and Git are required. Create one reusable environment and install the core package:

~~~powershell
$HarnessEnvPath = Join-Path $env:USERPROFILE ".venvs/ml-harness"
New-Item -ItemType Directory -Force (Split-Path $HarnessEnvPath) | Out-Null
if (-not (Test-Path "$HarnessEnvPath\Scripts\python.exe")) { py -3.11 -m venv $HarnessEnvPath }
& "$HarnessEnvPath\Scripts\Activate.ps1"
python -m pip install "ml-harness @ git+https://github.com/taitran501/ml-harness.git@feature/ml-harness-installable-plugin"
ml-harness plugins
~~~

Install a task adapter only when the competition requires one. The cookbook includes the StereoQueerEval adapter command and end-to-end task workflows.

The feature branch is the current install target. Replace it with a release tag or the default branch after a release is published.

## What gets installed

The core distribution installs the ml-harness, ml-harness-agent, ml-harness-run, ml-harness-dashboard, and ml-harness-bootstrap commands, including the dashboard assets. Each competition remains in its own workspace with its own dataset, task config, prompts, experiment code, and artifacts.

The IDE agent is external to the dashboard. Attaching ML-Harness writes a task-local skill and editable eight-phase prompt pack so the agent can work with that competition's context.
