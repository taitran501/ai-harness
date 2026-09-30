# ML-Harness

ML-Harness is an installable Python workspace for running competition projects with an IDE agent. Install the core package once in a Python environment, add a task adapter when the competition needs task-specific data, split, metric, or submission logic, then attach the installation to each task workspace.

The IDE agent performs research, web search, analysis, and scientific reasoning. ML-Harness supplies the task context and ordered phase prompts, runs EDA and experiments, tracks local or cloud jobs, and stores responses and artifacts. It does not run an LLM agent inside the dashboard.

## Install

These commands install directly from the GitHub feature branch that contains the installable package.

~~~powershell
py -3.11 -m venv .venv
& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# Core harness
python -m pip install "ml-harness @ git+https://github.com/taitran501/ml-harness.git@feature/ml-harness-installable-plugin"

# Optional adapter for a StereoQueerEval workspace
python -m pip install "ml-harness-stereoqueereval @ git+https://github.com/taitran501/ml-harness.git@feature/ml-harness-installable-plugin#subdirectory=task_plugins/stereoqueereval"

ml-harness plugins
~~~

For a declarative task supported by the core, install only the core package. Add the competition's adapter only when its task contract requires custom behavior.

## Attach it to a competition workspace

The task workspace owns its dataset, task config, code, and Git history. Use one workspace per competition task.

~~~powershell
cd C:/work/my-competition
ml-harness attach --project-root . --task-config configs/task.yaml

# Select an installed adapter only when the task uses one:
# ml-harness attach --project-root . --task-config configs/task.yaml --plugin stereoqueereval

ml-harness-agent --project-root . context
ml-harness phase validate --project-root .
ml-harness phase render --project-root . --phase setup
~~~

Attach creates a task-local manifest, an IDE-agent skill, and an editable eight-phase prompt pack. The dashboard and CLIs read and write state for the selected workspace only.

## Work through the task

~~~powershell
ml-harness eda run --project-root .
ml-harness phase render --project-root .
ml-harness-agent --project-root . context
ml-harness-run --project-root . --config configs/exp001.yaml --debug --fold 0
ml-harness-dashboard --project-root .
~~~

The first pass follows the eight phases in order. Experiment, candidate selection, and external evaluation phases can be revisited after new run or score evidence. The IDE agent uses its own web search and records inspected sources, responses, and workspace artifact paths through the phase workflow.

For optional Kaggle and Modal setup, privacy-scoped dataset consent, provider preflight, and remote job details, see [Cloud execution](docs/cloud-execution.md).

## Documentation

- [Installation and task attachment](docs/installation.md)
- [Agent and phase workflow](docs/getting-started.md)
- [Cloud execution](docs/cloud-execution.md)
- [Task adapter API](docs/task-adapters.md)
- [Task specification template](docs/TASK_SPEC_TEMPLATE.md)
- [Architecture](docs/architecture.md)
- [StereoQueerEval adapter](task_plugins/stereoqueereval/README.md)

## Repository layout

~~~text
ml-harness/
├── harness/                         # Task-neutral runtime and plugin API
├── tools/                           # Agent, runner, dashboard, and audit CLIs
├── dashboard/                       # Dashboard source
├── configs/tasks/                   # Generic example task configs
├── task_plugins/stereoqueereval/   # Optional task adapter distribution
├── tests/
└── docs/
~~~

The Python wheel contains the runtime and dashboard assets. Competition datasets, prompts, run history, and submissions stay in the task workspace.
