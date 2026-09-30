# Installation and task attachment

ML-Harness is installed as a Python package. For normal use, you do not need to clone the repository, build a wheel, install Node.js, or build dashboard assets: pip installs the runtime, CLI commands, and dashboard assets directly from GitHub. Use the same Python environment for the core, optional adapter, and CLI commands.

Start with the [copy/paste PowerShell cookbook](cookbook.md) for the complete install-to-experiment workflow.

## Prerequisites

Python 3.10 or newer is required; Python 3.11 is recommended. Installing from the GitHub URL also requires Git to be available on PATH.

## Install the core

Create one reusable environment outside competition workspaces, activate it, and install the package:

~~~powershell
$HarnessEnvPath = Join-Path $env:USERPROFILE ".venvs/ml-harness"
New-Item -ItemType Directory -Force (Split-Path $HarnessEnvPath) | Out-Null
if (-not (Test-Path "$HarnessEnvPath\Scripts\python.exe")) {
    py -3.11 -m venv $HarnessEnvPath
}
& "$HarnessEnvPath\Scripts\Activate.ps1"
python -m pip install --upgrade pip
python -m pip install "ml-harness @ git+https://github.com/taitran501/ml-harness.git@feature/ml-harness-installable-plugin"
ml-harness plugins
~~~

These commands install the current feature branch. Replace the branch with a release tag or the default branch after a release is published.

## Install an optional task adapter

For a declarative task supported by the core, no adapter installation is needed. For StereoQueerEval, install its adapter into the same environment:

~~~powershell
python -m pip install "ml-harness-stereoqueereval @ git+https://github.com/taitran501/ml-harness.git@feature/ml-harness-installable-plugin#subdirectory=task_plugins/stereoqueereval"
ml-harness plugins
~~~

An adapter must be installed before selecting its plugin ID during attachment. For details about writing adapters, see [Task adapters](task-adapters.md).

## Attach to a task workspace

Keep each competition task in its own workspace. Activate the environment, change to the task folder, and attach the workspace's own task config:

~~~powershell
& "$HarnessEnvPath\Scripts\Activate.ps1"
Set-Location "C:\work\my-competition"
ml-harness attach --project-root . --task-config configs/task.yaml
ml-harness-agent --project-root . context
ml-harness phase validate --project-root .
~~~

For an installed task adapter, select its entry-point ID:

~~~powershell
ml-harness attach --project-root . --task-config configs/task.yaml --plugin stereoqueereval
~~~

Attachment creates these task-local files:

- .ml-harness/project.yaml: task config, selected adapter, and artifact directory.
- .agents/skills/ml-harness/SKILL.md: instructions for the IDE agent working in this task.
- prompts/ml-harness/: editable phase prompts and output contracts.
- .ml-harness/artifacts/<task-id>/: phase history, responses, experiment runs, and imported remote results.

Attach does not copy the harness source into the task or replace its task config. Use --force only when intentionally replacing an existing attachment or when attaching after bootstrap generated a starter skill.

## Start from only a dataset and competition context

Create docs/TASK.md with the official competition contract and put local competition files under dataset/. Then run:

~~~powershell
ml-harness-bootstrap --project-root . --dataset dataset --context docs/TASK.md
Get-Content configs/task.yaml
ml-harness attach --project-root . --task-config configs/task.yaml --force
~~~

Review the generated target, metric, data paths, and split settings against organizer instructions before running experiments. Bootstrap creates a starter experiment config at configs/exp000_baseline.yaml when it can infer the required fields. The [cookbook](cookbook.md) includes the complete setup commands.

## Install optional cloud support

Add Kaggle and Modal support only in the environment that will launch cloud jobs:

~~~powershell
$HarnessEnvPath = Join-Path $env:USERPROFILE ".venvs/ml-harness"
& "$HarnessEnvPath\Scripts\Activate.ps1"
python -m pip install "ml-harness[cloud] @ git+https://github.com/taitran501/ml-harness.git@feature/ml-harness-installable-plugin"
~~~

Then authenticate with the provider CLI and run task-scoped preflight. See [Cloud execution](cloud-execution.md).

## Check or remove the installation

~~~powershell
python -m pip show ml-harness
ml-harness plugins
python -m pip uninstall ml-harness ml-harness-stereoqueereval
~~~

Uninstalling a package does not remove task workspace files or artifacts.
