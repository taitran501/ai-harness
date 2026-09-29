# ML-Harness

A modular experimentation harness and dashboard for competitive machine learning and data science projects (Kaggle, Codabench, DrivenData, and academic benchmarks).

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dashboard](https://img.shields.io/badge/Dashboard-React%20%2B%20Vite-purple.svg)](dashboard/)

---

## Overview

**ML-Harness** is a Python framework designed to streamline running, tracking, and evaluating machine learning experiments. It works for standard data science workflows whether you run models locally, dispatch heavy training jobs to cloud GPUs (Modal or Kaggle), or collaborate with AI coding assistants (such as Antigravity or Codex).

The harness is **task-agnostic**. Its core runtime provides experiment tracking, validation orchestration, agent sessions, and generic guardrails. Competition-specific data loading, model factories, metrics, and official submission contracts live in separately installed task adapters.

## Install into a competition workspace

Install the core runtime and only the adapter needed by the task, then attach it to that workspace:

```powershell
python -m pip install .
python -m pip install -e .\task_plugins\stereoqueereval
ml-harness plugins
ml-harness attach --project-root C:\path\to\competition --task-config configs/task.yaml --plugin stereoqueereval
ml-harness-agent --project-root C:\path\to\competition context
```

For a fresh environment, install directly from this repository's package subdirectories:

```powershell
python -m pip install "ml-harness @ git+https://github.com/taitran501/ml-harness.git#subdirectory=ml-harness"
python -m pip install "ml-harness-stereoqueereval @ git+https://github.com/taitran501/ml-harness.git#subdirectory=ml-harness/task_plugins/stereoqueereval"
```

`ml-harness attach` writes `.ml-harness/project.yaml` and an agent skill under `.agents/skills/ml-harness/`. The task config stays owned by the competition workspace. Session state and experiment artifacts are namespaced under `.ml-harness/artifacts/<task-id>/`.

See [Plugin architecture and lifecycle](docs/plugin_architecture.md) for the entry-point API, attach flow, and capability rules. The StereoQueerEval adapter is maintained in [`task_plugins/stereoqueereval`](task_plugins/stereoqueereval/README.md); its local data and metric code is not imported unless that adapter is installed and selected.

---

## Features

### 1. Broad Modeling Support
- **Classical ML & Gradient Boosting**: Native support for `scikit-learn`, `LightGBM`, `XGBoost`, and `CatBoost` with tabular preprocessing pipelines.
- **Deep Learning Transformers**: Fine-tuning pretrained text encoders (`DeBERTa`, `RoBERTa`, `ModernBERT`, etc.) via Hugging Face with customizable heads, learning rate schedules, and mixed precision.
- **PEFT / QLoRA**: Script generation and execution for 4-bit quantized LoRA fine-tuning using `Unsloth` and Hugging Face PEFT for consumer-grade GPU memory footprints.
- **LLM Prompting & Reasoning**: Support for zero-shot, few-shot, and Chain-of-Thought (CoT) prompting with structured JSON output parsing across OpenAI, Gemini, and DeepSeek backends.

### 2. Flexible Validation Protocols
- **Holdout Validation**: Fast validation against a designated evaluation dataset (`dev_split_name`).
- **Cross-Validation**: $k$-fold cross-validation with anti-leakage grouping (`GroupKFold`, `StratifiedKFold`, `TimeSeriesSplit`).
- **Debug Probes**: Low-cost, sampled-row smoke tests (`--debug --fold 0`) to verify syntax, pipeline wiring, and convergence before launching long runs.

### 3. Experiment Tracking & Solution Tree
- **Lineage DAG**: Experiments are tracked with parent-child relationships, hypotheses, and evaluation metrics. Attached projects keep this state in `.ml-harness/artifacts/<task-id>/`.
- **Upper Confidence Bound (UCB) Search**: Recommends next steps by balancing exploitation of the current best model with exploration of under-tested model families.
- **Human-Readable Logs**: Summaries and notes are maintained in `EXPERIMENTS.md` (tracking best models and an idea graveyard of approaches that did not pan out).

### 4. Verification & Guardrails
- **Epistemic Grounding**: Optional metadata verification ensuring configurations specify their rationale, reference sources, and assumptions before running.
- **Ensemble Diversity Guard**: Checks candidate diversity against thresholds declared by the active task or plugin.
- **Pre-Submission Audit**: Uses the selected task plugin's submission builder and validator; the generic audit applies only to declarative tasks.

### 5. Multi-Backend Execution
- **Local**: Run directly on your development machine (`ml-harness-run`).
- **Modal.com**: Serverless GPU execution with persistent volume caching and GPU selection (A10G, A100, H100, T4, L4).
- **Kaggle API / CLI**: Automated script packaging, kernel push, remote execution monitoring, and artifact download.

### 6. Interactive Web Dashboard
A local web UI built with React, Vite, and Tailwind CSS provides:
- **Agent workspace**: Ordered, workspace-owned phase prompts; copied prompts go to the user's IDE agent, while responses, inspected source URLs, and artifacts are saved into task history. The dashboard does not run an LLM agent.
- **EDA and provider readiness**: Local data profiling plus Kaggle/Modal CLI and account status, quota/cost caveats, and data-fingerprint scoped upload consent.
- **Solution Tree View**: Interactive graph of all experiment iterations and their performance deltas.
- **Leaderboard**: Compare validation scores, parameters, and training logs across models.
- **Execution Runner**: Configure and trigger runs directly from the browser.
- **LLM Studio**: Test prompts against sample data, search Hugging Face Hub models, and generate Unsloth training scripts.
- **Task Configurator**: Edit a task definition while keeping task-specific adapters independently installable.
- **Submissions Hub**: Track generated submission files and audit receipts.

---

## Package layout

```text
ml-harness/
├── harness/                    # Task-neutral runtime, runner, and plugin API
├── tools/                      # Agent Bridge, dashboard, audit, and bootstrap CLIs
├── dashboard/                  # Dashboard source; packaged static assets are in tools/web/
├── configs/tasks/              # Generic example task presets
└── task_plugins/               # Separately installable competition adapters
    └── stereoqueereval/        # Optional adapter for the SemEval task
```

Task-owned configs, datasets, split assignments, generated skills, run history, and submissions stay in the attached competition workspace. The harness writes its attachment manifest and managed state under `.ml-harness/` there. The Python wheel contains the runtime and dashboard assets, not competition data or run artifacts.

## Install and attach to a task

From this checkout, install the core package and only the task adapter you need:

```powershell
cd C:\path\to\semeval_2027\ml-harness
python -m pip install .
python -m pip install .\task_plugins\stereoqueereval
ml-harness plugins
ml-harness attach --project-root C:\path\to\competition --task-config configs/task.yaml --plugin stereoqueereval
```

Or install the packages directly from this repository's feature branch:

```powershell
python -m pip install "ml-harness @ git+https://github.com/taitran501/ml-harness.git@feature/ml-harness-installable-plugin#subdirectory=ml-harness"
python -m pip install "ml-harness-stereoqueereval @ git+https://github.com/taitran501/ml-harness.git@feature/ml-harness-installable-plugin#subdirectory=ml-harness/task_plugins/stereoqueereval"
```

The task workspace must own the task specification named by `--task-config`. Attach records the chosen plugin and config in `.ml-harness/project.yaml`, then creates `.agents/skills/ml-harness/SKILL.md` so a local coding agent can follow the task contract. Use `--force` only when intentionally replacing an existing attachment.

## Work with an agent

```powershell
ml-harness-agent --project-root C:\path\to\competition context
ml-harness-agent --project-root C:\path\to\competition start-session --goal "Improve the declared validation metric" --max-experiments 3
ml-harness-agent --project-root C:\path\to\competition act --action inspect_data_readiness
```

The bridge returns task/plugin capabilities and enforces typed action budgets and approval gates. For a local experiment, use a config inside the task workspace:

```powershell
ml-harness-run --project-root C:\path\to\competition --config configs/exp001.yaml --debug --fold 0
ml-harness-dashboard --project-root C:\path\to\competition
```

All experiment output from an attached task is routed under `.ml-harness/artifacts/<task-id>/`, even if an experiment config contains an older `artifacts/` path. The task's dataset and declared split/config paths remain in that workspace.

## Prompt phases, EDA, and cloud jobs

Each attached task gets its own editable prompt pack under `prompts/ml-harness/`. The first pass follows the eight phases in order. The experiment, candidate selection, and external evaluation phases can be revisited after run or score evidence changes:

```powershell
ml-harness phase validate --project-root C:\path\to\competition
ml-harness phase render --project-root C:\path\to\competition
ml-harness eda run --project-root C:\path\to\competition
ml-harness phase record --project-root C:\path\to\competition --phase setup --response-file notes\setup.md --sources-file notes\sources.json
```

Install the optional cloud tools and authenticate through their official CLIs. Review provider readiness before granting upload consent; consent is specific to this workspace, provider, and current data fingerprint:

```powershell
python -m pip install ".[cloud]"
kaggle auth login
modal setup
ml-harness cloud preflight --project-root C:\path\to\competition --provider kaggle
ml-harness cloud consent --project-root C:\path\to\competition --provider kaggle --confirm-upload
ml-harness cloud preflight --project-root C:\path\to\competition --provider kaggle --config configs\exp001.yaml
ml-harness-dashboard --project-root C:\path\to\competition
```

Dataset uploads are private by default and require a separate confirmation for each provider and data fingerprint. A changed dataset or split file requires new consent. Provider quotas, GPU availability, and charges depend on the account; the harness does not promise that a provider is always free. Kaggle jobs are polled to a terminal state, and results are imported only after task, job, experiment, config, and checksum validation. Modal cloud runs require Modal CLI/SDK 1.5.0 or newer.

The browser only displays state and captures the IDE agent's output. Use the agent's native web search, experiments, and reasoning; ML-Harness records the results and supplies the next prompt with current task context.

## Task plugin contract

A task adapter is a separate Python distribution registered under the `ml_harness.task_plugins` entry-point group. It can provide data loading and validation, a model factory, prediction normalization, task metrics, and official submission build/validation hooks. See [Plugin architecture and lifecycle](docs/plugin_architecture.md) and the [task specification template](docs/TASK_SPEC_TEMPLATE.md).

The optional StereoQueerEval adapter loads the selected workspace's `src.data`, `src.models`, and `src.metrics` only when installed and selected. It supports local training-data loading, grouped validation, model creation, and task metrics. Its official submission hooks are unavailable until the organizers publish the required format and scorer contract, so the harness blocks package creation for that adapter. It packages the installed adapter module into a versioned remote wheel; declare extra remote dependencies in `task.runtime.requirements`.

## Develop the dashboard

```powershell
cd dashboard
npm install
npm run build:package
```

`build:package` type-checks and builds the UI, then copies the static files into `tools/web/` so the installed wheel serves the same dashboard.

## License

This project is licensed under the [MIT License](LICENSE).
