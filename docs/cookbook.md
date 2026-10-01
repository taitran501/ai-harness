# ML-Harness PowerShell cookbook

Use this Windows workflow to install ML-Harness, attach a competition workspace, and work with an IDE agent. It uses `uv` for environment and package management.

You provide the competition dataset and organizer context. The IDE agent uses those inputs to research the task, run EDA and experiments, and create task outputs; ML-Harness tracks the workflow and runtime.

## 0. Prerequisites

Install Python 3.11, Git, and [uv](https://docs.astral.sh/uv/getting-started/installation/), then open PowerShell. If `uv` is not installed, Windows users can install it with WinGet:

~~~powershell
winget install --id astral-sh.uv -e
~~~

Open a new PowerShell window after installing, then check the commands:

~~~powershell
py -3.11 --version
git --version
uv --version
~~~

## 1. Install ML-Harness once

The reusable virtual environment stays outside competition folders. This end-to-end cookbook uses the `ml` profile for EDA, model runs, full Agent Bridge actions, and the dashboard. If you only need attachment, phase prompts, plugin discovery, and basic task context, install the base package without `[ml]`; add the profile later when you need ML tools.

~~~powershell
$HarnessEnvPath = Join-Path $env:USERPROFILE ".venvs/ml-harness"
uv venv $HarnessEnvPath --python 3.11
& "$HarnessEnvPath\Scripts\Activate.ps1"
# Full local workflow: EDA, experiments, diagnostics, and dashboard
uv pip install --python "$HarnessEnvPath\Scripts\python.exe" "ml-harness[ml] @ git+https://github.com/taitran501/ml-harness.git@feature/ml-harness-installable-plugin"

# For attachment and phase workflow only, install the base profile instead:
# uv pip install --python "$HarnessEnvPath\Scripts\python.exe" "ml-harness @ git+https://github.com/taitran501/ml-harness.git@feature/ml-harness-installable-plugin"

# Optional: only for a StereoQueerEval task workspace
uv pip install --python "$HarnessEnvPath\Scripts\python.exe" "ml-harness-stereoqueereval @ git+https://github.com/taitran501/ml-harness.git@feature/ml-harness-installable-plugin#subdirectory=task_plugins/stereoqueereval"

ml-harness plugins
~~~

For a general declarative task, omit the adapter install. The installed adapter list should show the built-in declarative plugin; StereoQueerEval appears only if you installed it.

## 2. Point commands at a competition workspace

Set this to the folder that contains one competition's dataset, task config, and code. Use a separate folder for each task.

~~~powershell
$TaskRoot = "C:\work\my-competition"
& "$HarnessEnvPath\Scripts\Activate.ps1"
Set-Location $TaskRoot
~~~

Open this folder in your IDE. Run the commands below in its integrated PowerShell terminal with the ML-Harness environment activated. Attachment places agent instructions inside the task workspace.

## 3A. Attach an existing task workspace

If the workspace already has configs/task.yaml, attach the core plugin:

~~~powershell
ml-harness attach --project-root . --task-config configs/task.yaml
~~~

For a StereoQueerEval workspace, install that adapter in step 1 and select it during attachment:

~~~powershell
ml-harness attach --project-root . --task-config configs/task.yaml --plugin stereoqueereval
~~~

Attachment creates .ml-harness/project.yaml, .agents/skills/ml-harness/SKILL.md, and the editable eight-phase prompt pack under prompts/ml-harness/. It does not copy the harness source into your project. Run attachment once per task workspace.

## 3B. Start from a dataset and competition context

Use this when the workspace does not yet have a task config. Put the competition files under dataset/ and create docs/TASK.md with the official goal, data schema, metric, split rules, submission format, and restrictions.

~~~powershell
New-Item -ItemType Directory -Force dataset, docs | Out-Null
@"
# Competition task context

- Official competition or task page:
- Goal and prediction target:
- Training and test files:
- Input and label columns:
- Official metric and direction:
- Required validation or grouping rules:
- Submission file format:
- Restrictions, deadlines, or external data rules:
"@ | Set-Content -LiteralPath docs/TASK.md -Encoding utf8
notepad docs/TASK.md
~~~

Save the official task information in docs/TASK.md, copy the competition files into dataset/, and run bootstrap:

~~~powershell
ml-harness-bootstrap --project-root . --dataset dataset --context docs/TASK.md
Get-Content configs/task.yaml
~~~

Bootstrap prints `NEEDS_REVIEW` and lists every local data file as an unclassified candidate. It does not choose train/test roles, infer a metric or platform, generate splits, or create a baseline. Keep the task config as a draft until the official contract has been checked. Then attach the task. Bootstrap writes a starter skill, so use `--force` for this first attachment:

~~~powershell
# General declarative task
ml-harness attach --project-root . --task-config configs/task.yaml --force

# Or, for a StereoQueerEval task with its adapter installed
ml-harness attach --project-root . --task-config configs/task.yaml --plugin stereoqueereval --force
~~~

For specialized task contracts, install and select the relevant task adapter. Do not create experiment configs or splits until the task contract is confirmed.

## 4. Give the IDE agent the phase prompt

Check the generated prompt pack and current phase:

~~~powershell
ml-harness phase validate --project-root .
ml-harness phase status --project-root .
ml-harness phase render --project-root . --phase setup
ml-harness-agent --project-root . context
~~~

Ask the IDE agent to follow .agents/skills/ml-harness/SKILL.md and the rendered phase prompt. Reuse this request, changing the phase and output paths as needed:

> Work on the current ML-Harness phase for this competition. Read the supplied task brief, local file inventory, and phase prompt. Use your native web search when useful; cite the official pages you actually inspected. Save the completed response to notes/setup.md and inspected sources as a JSON array in notes/sources.json. Save generated code and other outputs in this task workspace. Do not claim an experiment or score that you did not run and measure.

For a new onboarding draft, the first response should identify the competition and cite official task, scoring, and submission sources. Have the agent propose updates only for fields those sources support in `configs/task.yaml`, and leave unknowns unset. Review the source links and proposed values yourself before approving the contract. Run the approval-gated Agent Bridge action only after you approve:

~~~powershell
ml-harness-agent --project-root . act --action confirm_task_spec --user-approved --params '{"official_sources":[{"url":"https://example.org/official-rules","title":"Official rules"}],"review_summary":"Metric, validation rules, and submission format checked against official sources."}'
~~~

After the agent writes its response and sources, record them:

~~~powershell
New-Item -ItemType Directory -Force notes | Out-Null
ml-harness phase record --project-root . --phase setup --response-file notes/setup.md --sources-file notes/sources.json
ml-harness phase status --project-root .
~~~

The sources file is optional; append --sources-file notes/sources.json when you want the harness to store inspected web sources. For later phases, change the phase and response path, for example --phase understand --response-file notes/understand.md. Attach relevant outputs with repeated --artifact path/to/file options.

## 5. Run EDA and a measured baseline

Run EDA, then ask the agent to interpret the reports with the task context:

~~~powershell
ml-harness eda run --project-root .
ml-harness-agent --project-root . context
~~~

Bootstrap does not create an experiment config. After the contract is confirmed, ask the IDE agent to create a task-appropriate config from the adapter and official contract. Existing workspaces should use their own experiment config; set $ExperimentConfig to that file.

~~~powershell
$ExperimentConfig = "configs/exp001_baseline.yaml"
if (-not (Test-Path $ExperimentConfig)) { throw 'Set $ExperimentConfig to an experiment YAML that exists in this task workspace.' }
ml-harness-run --project-root . --config $ExperimentConfig --debug --fold 0
ml-harness-run --project-root . --config $ExperimentConfig
~~~

The first command is a small smoke run. Only a completed full run and its measured primary metric count as a baseline. Do not record a baseline score before the run produces it.

Start a bounded experiment session and ask the IDE agent to propose and run one controlled change at a time:

~~~powershell
ml-harness-agent --project-root . start-session --goal "Improve the declared validation metric" --max-experiments 3
ml-harness-agent --project-root . context
ml-harness phase render --project-root . --phase experiment
~~~

The IDE agent does the reasoning and code changes. The harness records runs, metrics, phase responses, and artifacts so the agent can continue from the latest evidence.

## 6. Open the local dashboard

~~~powershell
ml-harness-dashboard --project-root .
~~~

Keep the terminal running and open the printed local URL in a browser. The dashboard is scoped to this task workspace and binds to loopback.

## 7. Optional: run on Kaggle or Modal

Install the ML profile and only the provider you plan to use in the same environment. Then sign in using the official provider CLI:

~~~powershell
$HarnessEnvPath = Join-Path $env:USERPROFILE ".venvs/ml-harness"
& "$HarnessEnvPath\Scripts\Activate.ps1"

# Kaggle
uv pip install "ml-harness[ml,kaggle] @ git+https://github.com/taitran501/ml-harness.git@feature/ml-harness-installable-plugin"
kaggle auth login --force

# For Modal instead, run these commands:
# uv pip install "ml-harness[ml,modal] @ git+https://github.com/taitran501/ml-harness.git@feature/ml-harness-installable-plugin"
modal setup
~~~

Back in the task workspace, run the provider check. Replace the example config if this workspace uses a different experiment file:

~~~powershell
Set-Location $TaskRoot
ml-harness cloud status --project-root .
ml-harness cloud preflight --project-root . --provider kaggle --config $ExperimentConfig
ml-harness cloud preflight --project-root . --provider modal --config $ExperimentConfig
~~~

Uploading a local dataset requires explicit, task-scoped consent for each provider and current dataset fingerprint. Review the files before running the relevant consent command:

~~~powershell
# Run only the provider you intend to use, after reviewing the dataset.
ml-harness cloud consent --project-root . --provider kaggle --confirm-upload
# Or:
ml-harness cloud consent --project-root . --provider modal --confirm-upload
~~~

Dispatch a tracked full-validation job through the Agent Bridge:

~~~powershell
ml-harness-agent --project-root . act --action run_full_validation --params '{"config_path":"configs/exp001_baseline.yaml","backend":"kaggle"}'
# Or change backend to modal:
ml-harness-agent --project-root . act --action run_full_validation --params '{"config_path":"configs/exp001_baseline.yaml","backend":"modal"}'
~~~

The returned job ID and status are shared with the dashboard. When the provider finishes, the harness imports verified outputs and metrics into the task artifact tree for the IDE agent to inspect. Provider capacity, quotas, and charges can change; Kaggle or Modal is not guaranteed to be free.

See [Cloud execution](cloud-execution.md) for official authentication links and runtime details.

## 8. Finish and prepare a submission

Ask the IDE agent to audit the task's submission contract and candidate outputs. Freeze and submission actions remain behind human approval. If the selected adapter does not provide an official submission builder or validator, ML-Harness reports that gap instead of guessing the required format.

## Common commands

~~~powershell
ml-harness plugins
ml-harness-agent --project-root . context
ml-harness phase status --project-root .
ml-harness phase render --project-root . --phase experiment
ml-harness eda run --project-root .
ml-harness cloud status --project-root .
ml-harness-dashboard --project-root .
~~~

For command options, append --help, for example ml-harness phase --help or ml-harness cloud preflight --help.
