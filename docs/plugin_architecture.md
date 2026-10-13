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
A `TaskPlugin` implementation may override the following methods:

| Method Hook | Responsibility |
|---|---|
| `load_train_data()` | Ingests raw files, performs initial type casting, and returns a clean training DataFrame. |
| `load_test_data()` | Ingests unlabelled evaluation data for test-set inference. |
| `build_validation_splits()` | Produces deterministic train/validation indices enforcing anti-leakage invariants. |
| `evaluate_predictions()` | Computes competition-specific evaluation metrics on prediction batches. |
| `build_submission()` | Formats test predictions into the competition platform's required format (CSV, TSV, or ZIP). |
| `validate_submission()` | Verifies that the generated submission package strictly satisfies platform constraints. |
| `run_eda()` | Generates domain-specific exploratory data analysis summaries. |

If an adapter does not implement submission hooks, generic metric evaluation remains functional, but submission packaging operations fail closed with a `PluginCapabilityError`.

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
