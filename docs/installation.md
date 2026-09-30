# Installation and task attachment

ML-Harness is installed into a Python environment as a core distribution. A competition-specific adapter is a second, optional distribution discovered by Python entry points. Use the same Python environment for installing packages and running the CLIs.

These commands pin the GitHub feature branch that contains the installable package. Replace the branch with a release tag or the default branch after a release is published.

## Install the core

Python 3.10 or newer is required. Python 3.11 is recommended when using the current Kaggle CLI.

~~~powershell
py -3.11 -m venv .venv
& .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install "ml-harness @ git+https://github.com/taitran501/ml-harness.git@feature/ml-harness-installable-plugin"
~~~

To install directly from a local clone instead:

~~~powershell
git clone --branch feature/ml-harness-installable-plugin https://github.com/taitran501/ml-harness.git
cd ml-harness
python -m pip install -e .
~~~

## Install an optional task adapter

For a declarative task supported by the core, no adapter installation is needed. For StereoQueerEval, install its adapter into the same environment:

~~~powershell
python -m pip install "ml-harness-stereoqueereval @ git+https://github.com/taitran501/ml-harness.git@feature/ml-harness-installable-plugin#subdirectory=task_plugins/stereoqueereval"
ml-harness plugins
~~~

When installing from a local clone, use:

~~~powershell
python -m pip install -e task_plugins/stereoqueereval
~~~

The plugin list should include the built-in declarative adapter and every installed task adapter. An adapter must be installed before selecting its plugin ID during attachment.

## Attach to a task workspace

Keep each competition in its own workspace. For an existing project, use the task config already owned by that workspace:

~~~powershell
cd C:/work/my-competition
ml-harness attach --project-root . --task-config configs/task.yaml
ml-harness-agent --project-root . context
ml-harness phase validate --project-root .
~~~

For a task that uses the optional adapter, add its installed entry-point ID:

~~~powershell
ml-harness attach --project-root . --task-config configs/task.yaml --plugin stereoqueereval
~~~

Attachment creates these task-local files:

- .ml-harness/project.yaml: task config, selected adapter, and artifact directory.
- .agents/skills/ml-harness/SKILL.md: instructions for the IDE agent working in this task.
- prompts/ml-harness/: editable phase prompts and their output contracts.
- .ml-harness/artifacts/<task-id>/: phase history, responses, experiment runs, and imported remote results.

Attach does not copy the harness source into the task or replace its task config. Use --force only when intentionally replacing an existing attachment.

### Starting from only a dataset and task context

If the workspace has no task config, create one from the competition rules and the [task specification template](TASK_SPEC_TEMPLATE.md), or bootstrap a starter from local data and a context document:

~~~powershell
ml-harness-bootstrap --project-root . --dataset dataset --context docs/TASK.md
~~~

Review the generated metric, target, split, and submission fields against the official task instructions before running experiments. Bootstrap also writes a starter agent skill. When attaching after bootstrap, pass --force so ML-Harness can write its task manifest and task-specific agent skill:

~~~powershell
# Declarative task
ml-harness attach --project-root . --task-config configs/task.yaml --force

# Task with the StereoQueerEval adapter
ml-harness attach --project-root . --task-config configs/task.yaml --plugin stereoqueereval --force
~~~

## Install cloud support

Kaggle and Modal are optional dependencies. Add them only in an environment that will submit cloud jobs:

~~~powershell
python -m pip install "ml-harness[cloud] @ git+https://github.com/taitran501/ml-harness.git@feature/ml-harness-installable-plugin"
~~~

Then use the providers' official CLI authentication flows and run a task-scoped preflight. See [Cloud execution](cloud-execution.md).

## Check or remove the installation

~~~powershell
python -m pip show ml-harness
ml-harness plugins
python -m pip uninstall ml-harness ml-harness-stereoqueereval
~~~

Uninstalling a package does not remove task workspace files or artifacts.
