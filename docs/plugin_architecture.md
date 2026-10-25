# Task Plugins and Workspace Integration

AI-Harness is distributed as an installable Python package. Task-specific competition adapters are separate distributions discovered via Python entry points under the `ai_harness.task_plugins` group.

---

## 1. When to Use a Task Plugin

There are two ways to configure a task in AI-Harness:

1. **Declarative Task Specification (`configs/task.yaml`)**:
   - Ideal for standard benchmarks with tabular or textual datasets in standard formats (`.csv`, `.parquet`, `.tsv`).
   - Uses built-in generic metrics (accuracy, macro/micro F1, ROC-AUC, RMSE, MAE, logloss) and standard splitters.
   - Does not require writing Python adapter code.

2. **Custom Task Plugin (`TaskPlugin`)**:
   - Required when the competition defines custom multi-table datasets, specialized data loaders, domain-specific evaluation metrics, or strict submission packaging formats.
   - Implements validation split logic, exploratory data analysis hooks, and submission builders/validators.

---

## 2. Installation and Attachment

Install the core harness runtime, then install the competition adapter package:

```bash
# Install core runtime
pip install -e .

# List installed plugins
ai-harness plugins

# Attach the harness to your competition workspace
ai-harness attach --project-root /path/to/competition --task-config configs/task.yaml --plugin <plugin_id>
```

The `attach` command creates:
- A local attachment manifest: `.ai-harness/project.yaml`.
- An agent skill file tailored to the task: `.agents/skills/ai-harness/SKILL.md`.
- Isolated experiment state and artifact directories: `.ai-harness/artifacts/<task-id>/`.

The competition's own `configs/task.yaml` is preserved and not rewritten.

---

## 3. The Adapter Contract (`TaskPlugin`)

To create a competition adapter, subclass `harness.plugins.TaskPlugin` and register it in your package's `pyproject.toml`:

```toml
[project.entry-points."ai_harness.task_plugins"]
my_competition = "my_adapter_package.plugin:MyCompetitionPlugin"
```

### Core Adapter Hooks
`harness.plugins.TaskPlugin` is the **canonical interface**; the method names and signatures below are exactly what the runtime calls. Override only what your competition needs; every hook has a safe default.

| Method Hook | Signature (after `self`) | Responsibility |
|---|---|---|
| `load_training_data()` | `(context)` → `DataFrame \| None` | Ingests raw files, performs initial type casting, and returns the training DataFrame. `None` falls back to declarative loading. |
| `validate_training_data()` | `(context, data)` → `None` | Raises `TaskPluginError` if the data contract is invalid. |
| `build_validation_splits()` | `(context, data, strategy, validation_config)` → `[(fold_id, train, val)] \| None` | Produces deterministic splits enforcing anti-leakage invariants. `None` uses core defaults. |
| `validate_split()` | `(context, fold_id, train_data, validation_data)` → `None` | Raises when a split violates a task-specific contract. |
| `build_model()` | `(context, model_name, params)` → model `\| None` | Task-specific model; `None` uses the generic model factory. |
| `normalize_predictions()` | `(context, predictions)` → `dict` | Post-processes predictions before scoring. |
| `evaluate_predictions()` | `(context, ground_truth, predictions)` → `dict` | Computes competition-specific metrics. Return a mapping that includes `primary_score` (or the primary metric name). |
| `build_submission()` | `(context, predictions, output_path)` → `Path` | Formats predictions into the platform's required format (CSV, TSV, or ZIP). |
| `validate_submission()` | `(context, submission)` → `None` | Raises if the submission violates platform constraints. |
| `extend_eda_report()` | `(context, reports)` → `dict \| list \| None` | Adds domain-specific, value-safe checks to the generic EDA reports. |

Remote runtime hooks (`remote_runtime_requirements()`, `remote_bundle_files()`, `remote_runtime_manifest()`; each takes `(context)`) declare what Kaggle/Modal needs to run the adapter.

There is intentionally **no** `load_test_data()` or `run_eda()` hook: test-set inference is driven by your experiment script, and EDA is extended through `extend_eda_report()`. Methods with those names on a plugin are never called.

If an adapter does not implement submission hooks, generic metric evaluation remains functional, but submission packaging operations fail closed with a `PluginCapabilityError`.

### Reference Example
A complete, tested adapter lives at [`examples/task_plugin/example_adapter.py`](../examples/task_plugin/example_adapter.py) (see `tests/test_plugin_docs_and_example.py`). Copy it as the starting point for a new competition adapter instead of inferring method names.

---

## 4. Remote Execution and Runtime Manifests

When dispatching runs to remote GPU infrastructure (such as Modal or Kaggle):
- The plugin declares a runtime manifest specifying its exact package name, version, and dependencies.
- The manifest can specify pinned requirements, custom wheels, and task bundle files.
- Remote backends receive the selected code and dependencies; credentials and datasets are managed separately.
- Uploading local dataset files to cloud providers requires explicit user consent and verifies dataset SHA-256 fingerprints.

---

## 5. Workspace Ownership and Phase Prompt Packs

In an attached competition workspace:
- The workspace maintains editable phase instructions under `prompts/ai-harness/` with a manifest at `prompts/ai-harness/phase-pack.yaml`.
- The AI coding assistant reads rendered phase prompts, performs domain reasoning, and writes responses.
- AI-Harness persists recorded phase responses, evidence revisions, cited sources, and output artifacts in `.ai-harness/artifacts/<task-id>/phase_history.json`.

---

## 6. Workspace Layout

A configured competition repository follows this directory layout:

```text
competition-workspace/
|-- configs/
|   |-- task.yaml                 # Task specification contract
|   `-- exp000_baseline.yaml      # Baseline experiment configuration
|-- dataset/                      # Competition data files
|-- splits/
|   `-- fold_assignments.tsv      # Deterministic cross-validation splits
|-- prompts/
|   `-- ai-harness/
|       |-- phase-pack.yaml       # Phase workflow manifest
|       `-- phases/               # Editable phase prompt templates
|-- .ai-harness/
|   |-- project.yaml              # Local attachment manifest
|   `-- artifacts/<task-id>/      # Isolated run artifacts, manifests, and logs
`-- .agents/skills/               # Generated assistant skills
```
