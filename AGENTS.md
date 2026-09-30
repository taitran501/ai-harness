# Contributing to ML-Harness

This repository contains the installable, task-neutral ML-Harness runtime and separately packaged task adapters. Competition datasets, split files, task-specific source code, score tables, and run artifacts belong in their competition workspaces, not in this repository.

## Repository boundaries

- Keep core behavior in harness/ and tools/.
- Put competition-specific behavior in task_plugins/<adapter>/ and publish it as a separate Python distribution.
- Keep generic task examples in configs/tasks/.
- Do not copy user task datasets, credentials, submissions, or generated experiment artifacts into this repository.
- Keep workspace state under the attached task's .ml-harness/ directory.

## Install for development

Use `uv` for environment management and install the ML runtime in editable mode:

~~~powershell
uv venv --python 3.11
uv pip install -e ".[ml]"
uv pip install -e task_plugins/stereoqueereval
ml-harness plugins
~~~

For provider development, install only the provider extras you are changing:

~~~powershell
uv pip install -e ".[ml,kaggle]"
# Or: uv pip install -e ".[ml,modal]"
~~~

## Contracts to preserve

- The IDE agent owns reasoning and native web search. The dashboard records state; it does not run an embedded LLM agent.
- Task-specific loading, splitting, metrics, and submission behavior must come from the selected adapter or task config.
- Baselines require completed runs and measured metrics.
- Data upload consent is scoped to task, provider, and dataset fingerprint; provider credentials must not enter remote bundles.
- Freeze and submission actions stay behind user approval.
- Do not claim submission readiness when an adapter does not implement the official builder and validator.

## Documentation and dashboard

The README is the installation entry point. Keep install and task attachment instructions in docs/installation.md, the agent workflow in docs/getting-started.md, provider setup in docs/cloud-execution.md, and adapter development in docs/task-adapters.md.

Build packaged dashboard assets from dashboard/:

~~~powershell
cd dashboard
npm install
npm run build:package
~~~

The build copies the static UI to tools/web/ for inclusion in the Python wheel.
