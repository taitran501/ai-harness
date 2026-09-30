# Architecture

ML-Harness is installed in a user's Python environment and attached to one competition workspace at a time. The competition workspace owns its task config, dataset, code, phase prompts, and artifacts.

## Components

~~~text
User + IDE agent
  | native research, web search, reasoning, response and artifact creation
  v
Agent Bridge CLI  <---->  Task workspace
                         task config and plugin selection
                         prompts/ml-harness/
                         .ml-harness/project.yaml
                         .ml-harness/artifacts/<task-id>/
  | typed actions, context packets, approval gates
  v
Harness runtime
  | plugin API, EDA, validation, experiment runner, job tracking
  +---- local Python execution
  +---- Kaggle adapter and CLI
  +---- Modal adapter and CLI

Local dashboard ---- reads and updates the selected task workspace
~~~

The dashboard is a local status and control surface. It does not run an LLM agent. The IDE agent uses its own available model and tools, then saves its phase response, inspected sources, and artifact paths through the workflow.

## Installation and task adapter boundary

The core Python distribution contains task-neutral CLI commands, runner code, state management, and packaged dashboard assets. A competition adapter is an independent Python distribution discovered through the ml_harness.task_plugins entry-point group.

The built-in declarative adapter handles tasks expressed through the task specification. A separately installed task adapter can supply custom data loading, model creation, validation split checks, metrics, EDA extensions, remote runtime dependencies, or official submission operations. The selected adapter and its capabilities are part of the task context.

## Workspace ownership

A task workspace stores:

- Its task specification and experiment code.
- An editable eight-phase prompt pack under prompts/ml-harness/.
- A local attachment manifest at .ml-harness/project.yaml.
- Phase responses, source records, run metadata, and artifacts under .ml-harness/artifacts/<task-id>/.

Attaching ML-Harness does not rewrite the task config or move the workspace's source files. Start one dashboard process per task workspace. Each server binds to loopback and validates Host and Origin requests.

## Experiment and phase lifecycle

The agent reads task context and phase prompts, inspects data, researches official competition information, and proposes experiments. The harness runs EDA, validation, local experiments, or tracked cloud jobs. Baseline locking requires a measured run and metric. Phase responses and sources become durable task history. Experiment, selection, and external evaluation phases can repeat after evidence changes.

Freeze and submission operations retain user-approval gates. An adapter without an official submission contract cannot be treated as submission-ready.

## Remote execution and privacy

Kaggle and Modal use a shared job state model. Remote code bundles include the selected harness runtime, adapter, task source files, and declared dependencies. Provider credentials are read from official local CLIs and excluded from code bundles. Task-data upload requires an explicit, provider-specific consent bound to the task and the current data fingerprint. Uploaded data is private by default.

The harness validates job, task, experiment, config, and artifact identity before importing completed remote outputs into the workspace's artifact tree.
