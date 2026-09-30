# ML-Harness

ML-Harness is a Python toolkit for agent-assisted machine-learning competition work. It connects an IDE agent to a competition workspace containing the task specification, dataset, experiment code, and run artifacts.

The IDE agent performs task research, web search, analysis, and code development. ML-Harness provides workspace context, phase prompts, data profiling, experiment execution, provider integration, and artifact tracking. It does not host an LLM agent.

## Features

- Workspace-scoped task configuration, prompt packs, phase history, and artifacts.
- Agent Bridge commands for task context and typed harness actions.
- General-purpose EDA and experiment execution.
- Local runs and optional Kaggle and Modal backends.
- Task adapters for competition-specific data, validation, metrics, and submission rules.
- Local dashboard for inspecting workspace and job state.

## Requirements

- Python 3.10 or newer; Python 3.11 is recommended.
- Git, for installing the current package directly from the repository.

## Installation

Create and activate a virtual environment:

~~~powershell
py -3.11 -m venv .venv
& .\.venv\Scripts\Activate.ps1
~~~

On macOS or Linux:

~~~bash
python3.11 -m venv .venv
source .venv/bin/activate
~~~

Install the core package:

~~~bash
python -m pip install --upgrade pip
python -m pip install "ml-harness @ git+https://github.com/taitran501/ml-harness.git@feature/ml-harness-installable-plugin"
ml-harness plugins
~~~

For declarative tasks supported by the core, no task adapter is needed. Install a task adapter into the same environment when the competition requires specialized behavior. For example, the optional StereoQueerEval adapter is installed with:

~~~bash
python -m pip install "ml-harness-stereoqueereval @ git+https://github.com/taitran501/ml-harness.git@feature/ml-harness-installable-plugin#subdirectory=task_plugins/stereoqueereval"
~~~

## Quick start

Run these commands from the competition workspace. The workspace must contain a task specification at configs/task.yaml.

~~~bash
ml-harness attach --project-root . --task-config configs/task.yaml
ml-harness phase validate --project-root .
ml-harness phase render --project-root . --phase setup
ml-harness-agent --project-root . context
ml-harness eda run --project-root .
ml-harness-dashboard --project-root .
~~~

To select an installed task adapter, pass its plugin ID during attachment:

~~~bash
ml-harness attach --project-root . --task-config configs/task.yaml --plugin stereoqueereval
~~~

When starting with only a dataset and competition context, bootstrap a task specification and starter baseline configuration first:

~~~bash
ml-harness-bootstrap --project-root . --dataset dataset --context docs/TASK.md
ml-harness attach --project-root . --task-config configs/task.yaml --force
~~~

Review generated task and experiment configurations against the official competition contract before running experiments.

## Documentation

- [PowerShell end-to-end cookbook](docs/cookbook.md)
- [Installation and task attachment](docs/installation.md)
- [Agent and phase workflow](docs/getting-started.md)
- [Cloud execution](docs/cloud-execution.md)
- [Task adapter API](docs/task-adapters.md)
- [Task specification template](docs/TASK_SPEC_TEMPLATE.md)
- [Architecture](docs/architecture.md)
- [StereoQueerEval adapter](task_plugins/stereoqueereval/README.md)
