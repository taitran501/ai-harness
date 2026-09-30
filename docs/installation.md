# Installation and task attachment

ML-Harness is distributed as a Python package with optional dependency profiles. The base profile supports workspace attachment, phase prompts, plugin discovery, and basic task context. Install the `ml` extra for EDA, experiments, full Agent Bridge diagnostics, bootstrap, and the local dashboard.

## Requirements

Python 3.10 or newer and Git are required. [`uv`](https://docs.astral.sh/uv/getting-started/installation/) is recommended for creating the environment and installing the package. Its cache lets separate environments reuse downloaded wheels.

## Create an environment and install the base profile

~~~powershell
$HarnessEnvPath = Join-Path $env:USERPROFILE ".venvs/ml-harness"
uv venv $HarnessEnvPath --python 3.11
uv pip install --python "$HarnessEnvPath\Scripts\python.exe" "ml-harness @ git+https://github.com/taitran501/ml-harness.git@feature/ml-harness-installable-plugin"
& "$HarnessEnvPath\Scripts\Activate.ps1"
ml-harness plugins
~~~

To use pip instead, create and activate the environment, then install the same package requirement:

~~~powershell
py -3.11 -m venv $HarnessEnvPath
& "$HarnessEnvPath\Scripts\Activate.ps1"
python -m pip install "ml-harness @ git+https://github.com/taitran501/ml-harness.git@feature/ml-harness-installable-plugin"
~~~

## Enable local ML tools

Install this extra in the same environment when you need EDA, model runs, full Agent Bridge context and actions, task bootstrap, or the dashboard:

~~~powershell
uv pip install "ml-harness[ml] @ git+https://github.com/taitran501/ml-harness.git@feature/ml-harness-installable-plugin"
~~~

This adds NumPy, pandas, scikit-learn, and SciPy. With pip, run `python -m pip install "ml-harness[ml] @ git+https://github.com/taitran501/ml-harness.git@feature/ml-harness-installable-plugin"` in the activated environment. The base workflow remains available without these packages. If an ML-only command is used before installing the extra, it prints the missing profile and install command.

## Install a task adapter

For a declarative task supported by the core, no adapter is needed. Install a specialized adapter into the same environment before selecting its plugin ID. For StereoQueerEval:

~~~powershell
uv pip install "ml-harness-stereoqueereval @ git+https://github.com/taitran501/ml-harness.git@feature/ml-harness-installable-plugin#subdirectory=task_plugins/stereoqueereval"
ml-harness plugins
~~~

The adapter requires the `ml` profile. For adapter development from a checkout, see [Task adapters](task-adapters.md).

## Attach to a task workspace

Use a separate workspace for each competition task. Activate the environment, change to the task folder, and attach its task config:

~~~powershell
& "$HarnessEnvPath\Scripts\Activate.ps1"
Set-Location "C:\work\my-competition"
ml-harness attach --project-root . --task-config configs/task.yaml
ml-harness-agent --project-root . context
ml-harness phase validate --project-root .
~~~

For an installed adapter, pass its plugin ID:

~~~powershell
ml-harness attach --project-root . --task-config configs/task.yaml --plugin stereoqueereval
~~~

Attachment creates these task-local files:

- `.ml-harness/project.yaml`: task config, selected adapter, and artifact directory.
- `.agents/skills/ml-harness/SKILL.md`: instructions for the IDE agent working in this task.
- `prompts/ml-harness/`: editable phase prompts and output contracts.
- `.ml-harness/artifacts/<task-id>/`: phase history, responses, experiment runs, and imported remote results.

## Start from a dataset and competition context

Create `docs/TASK.md` with the official competition contract and put local competition files under `dataset/`. Bootstrap requires the `ml` profile:

~~~powershell
ml-harness-bootstrap --project-root . --dataset dataset --context docs/TASK.md
Get-Content configs/task.yaml
ml-harness attach --project-root . --task-config configs/task.yaml --force
~~~

Review the generated target, metric, data paths, and split settings against organizer instructions before running experiments. The [cookbook](cookbook.md) contains a complete setup sequence.

## Install provider support

Install only the provider package you plan to use, along with `ml` for experiment execution. For Kaggle:

~~~powershell
uv pip install "ml-harness[ml,kaggle] @ git+https://github.com/taitran501/ml-harness.git@feature/ml-harness-installable-plugin"
kaggle auth login --force
~~~

Or use **Account Settings → Connect Kaggle** in the local dashboard. Approve access on the Kaggle page it opens; the dashboard then refreshes the account status.

For Modal:

~~~powershell
uv pip install "ml-harness[ml,modal] @ git+https://github.com/taitran501/ml-harness.git@feature/ml-harness-installable-plugin"
modal setup
~~~

The `[cloud]` extra installs the ML profile and both provider packages together. Provider quotas, availability, and charges depend on the account and service; see [Cloud execution](cloud-execution.md).

## Check or remove the installation

~~~powershell
python -m pip show ml-harness
ml-harness plugins
python -m pip uninstall ml-harness ml-harness-stereoqueereval
~~~

Uninstalling a package does not remove task workspace files or artifacts.
