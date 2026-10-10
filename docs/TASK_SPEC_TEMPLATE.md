# Universal Task Specification Template (`UniversalTaskSpec v2.0`)

> [!IMPORTANT]
> **Verification Scope: Strictly NLP / Text Only**
> AI-Harness is currently **strictly verified and working well ONLY on Natural Language Processing (NLP / Text)** benchmarks (such as sequence classification, transformer fine-tuning, and prompt evaluation).
> Support for all other modalities (**Tabular, Computer Vision, Audio, Time-Series, Multimodal**) is **experimental work-in-progress (WIP / Roadmap)**, has NOT been verified in competitive conditions, and is NOT considered ready or working well in practice.
> The tabular schemas and examples in this document are provided for schema syntax reference only.
>
> This document defines the standard specification schema for onboarding arbitrary ML/DL/NLP competitions into **AI-Harness**.
> The harness consumes this specification to configure data loaders, establish leakage-proof validation folds, parameterize evaluation metrics, and inject agent skills.

---

## 1. Document Overview & Usage

| Attribute | Specification |
|---|---|
| **Schema Name** | `UniversalTaskSpec` |
| **Schema Version** | `2.0.0` |
| **Consuming Components** | `harness/task_spec.py`, `tools/bootstrap_task.py`, `harness/skill_injector.py`, `harness/diagnostics.py` |
| **Output Locations** | Raw Context: `docs/TASK.md` ➔ Declarative Config: `configs/task.yaml` |

### Onboarding Modes
1. **Zero-Touch Automated Mode (Recommended)**: Drop competition notes into `docs/TASK.md` and run:
   ```powershell
   ai-harness-bootstrap --project-root . --dataset dataset/ --context docs/TASK.md
   ```
2. **Declarative Mode**: Write `configs/task.yaml` directly using the schema below, then attach the installed harness and selected adapter:
   ```powershell
   ai-harness attach --project-root . --task-config configs/task.yaml --plugin <installed-plugin-id>
   ```

---

## 2. Specification Schema Reference

The specification consists of six orthogonal sections. Every field is strictly typed.

```text
UniversalTaskSpec
├── 1. Metadata (Platform, Benchmark Name, Modality)
├── 2. Data Ingestion (Paths, ID Column, Anti-Leakage Groups)
├── 3. Input Variants (Text templates, Feature subsets)
├── 4. Subtasks & Targets (Metrics, Classes, Regression bounds)
├── 5. Conditional Dependencies (Hierarchical routing, Oracle masks)
└── 6. Validation Protocol (K-Fold, Holdout, Splitting files)
```

### 2.1 Section 1: Task & Platform Metadata

| Field | Type | Required | Default | Allowed Values | Description |
|---|---|---|---|---|---|
| `task.name` | `string` | **Yes** | — | Any descriptive string | Identifier for competition or benchmark (e.g. `Kaggle-Titanic`). |
| `task.platform` | `string` | **Yes** | `generic` | `kaggle`, `codabench`, `codalab`, `huggingface`, `drivendata`, `generic` | Host competition platform. Dictates submission packaging format. |
| `task.modality` | `string` | **Yes** | `text` | `text` (Verified Core), `tabular` (WIP), `vision` (Roadmap), `time-series` (Roadmap), `multimodal` (Roadmap), `generic` (WIP) | Problem modality. Parameterizes Grounding Gate backbone recommendations. Only text/NLP is verified in practice. |
| `task.description` | `string` | No | `""` | Freeform text | Executive summary of domain objective and problem constraints. |
| `task.website` | `string` | No | `null` | Valid URI | Official benchmark URL or competition overview page. |

---

### 2.2 Section 2: Data Ingestion & Identifiers

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `data_config.train_path` | `string` | **Yes** | `dataset/` | File or directory path to training data (`.parquet`, `.csv`, `.tsv`, `.jsonl`, `.sqlite`, or `.zip`). |
| `data_config.test_path` | `string` | No | `null` | File or directory path to unlabelled test evaluation data. |
| `data_config.id_column` | `string` | **Yes** | `id` | Unique instance identifier column (e.g. `PassengerId`, `uid`, `row_id`). |
| `data_config.group_column` | `string` | No | `null` | **Anti-Leakage Group Column**. Identifiers (e.g. `user_id`, `patient_id`, `video_id`) that must never be split across folds. |
| `data_config.partition_column`| `string` | No | `null` | Partitioning attribute (e.g. `language`, `site`) for partitioned slicing. |

---

### 2.3 Section 3: Subtasks, Target Columns & Metrics

The harness models all competitions as one or more **subtasks**. A single target regression/classification is simply a 1-element subtask list.

| Field | Type | Required | Default | Allowed Values | Description |
|---|---|---|---|---|---|
| `subtasks[].id` | `string` | **Yes** | — | Identifier token | Internal subtask key (e.g. `target`, `toxicity`, `sales_price`). |
| `subtasks[].name` | `string` | No | `id` | Human-readable name | Display name in diagnostics dashboards and experiment cards. |
| `subtasks[].task_type` | `string` | **Yes** | — | `binary_classification`, `multiclass_classification`, `multilabel_classification`, `regression` | Mathematical objective of this target. |
| `subtasks[].column` | `string` | **Yes** | `id` | Column name in dataset | Column holding ground truth values in the training set. |
| `subtasks[].primary_metric`| `string` | **Yes** | `macro_f1` | `macro_f1`, `micro_f1`, `accuracy`, `roc_auc`, `rmse`, `mae`, `r2`, `logloss`, `quadratic_weighted_kappa` | Metric optimized for this subtask. |
| `subtasks[].direction` | `string` | **Yes** | `maximize` | `maximize`, `minimize` | Optimization trajectory. |
| `subtasks[].weight` | `float` | No | `1.0` | Positive float | Weight when aggregating composite score across multi-task setups. |
| `subtasks[].classes` | `list` | Cond. | `[]` | List of strings or ints | Exhaustive class labels (Required for classification subtasks). |

---

### 2.4 Section 4: Hierarchical & Conditional Dependencies (Optional)

Used for multi-stage pipelines where child predictions are conditional on parent labels (e.g., *predict category only if is_toxic == 1*).

```yaml
subtasks:
  - id: secondary_category
    column: secondary_category
    condition:
      parent_target: is_toxic
      operator: "=="
      values: 1
      fallback_value: "none"
      eval_condition_only: true
```

| Field | Type | Required | Description |
|---|---|---|---|
| `condition.parent_target` | `string` | **Yes** | The parent subtask ID that controls this subtask. |
| `condition.operator` | `string` | **Yes** | Logical comparator: `==`, `!=`, `in`, `not_in`, `gt`, `lt`. |
| `condition.values` | `any` | **Yes** | Parent values triggering active evaluation of this subtask. |
| `condition.fallback_value`| `any` | **Yes** | Static value assigned to child target when condition is FALSE. |
| `condition.eval_condition_only` | `bool` | No (`true`) | If `true`, validation metric is computed only over conditional subset. |

---

### 2.5 Section 5: Validation Protocol

| Field | Type | Required | Default | Allowed Values | Description |
|---|---|---|---|---|---|
| `validation.strategy` | `string` | **Yes** | `kfold` | `kfold`, `holdout`, `sample_probe` | Cross-validation architecture. |
| `validation.num_folds` | `int` | No | `5` | Integer $\ge 2$ | Number of cross-validation partitions. |
| `validation.split_file` | `string` | No | `splits/fold_assignments.tsv` | Relative path | Fixed fold assignment table mapping `id_column -> fold`. |
| `validation.dev_split_name` | `string` | No | `dev` | File name or split key | Dedicated development/validation split if `strategy == holdout`. |
| `validation.probe_samples` | `int` | No | `200` | Integer $\ge 10$ | Subsample ceiling when running smoke tests (`--debug`). |

---

## 3. Reference Implementation Configurations

### 3.1 Scenario A: Tabular Binary Classification with Group Invariant *(Schema Reference Only — Tabular is WIP)*

> [!NOTE]
> Tabular modeling is experimental work-in-progress (WIP). This scenario illustrates the declarative schema syntax for tabular tasks with group anti-leakage invariants.

```yaml
task:
  name: "HomeCredit-Default-Risk"
  platform: "kaggle"
  task_type: "binary_classification"
  modality: "tabular"
  primary_metric: "roc_auc"
  direction: "maximize"
  description: "Predict client repayment capabilities using historical credit telemetry."

  data_config:
    train_path: "dataset/train.parquet"
    test_path: "dataset/test.parquet"
    id_column: "SK_ID_CURR"
    group_column: "CLIENT_CLUSTER_ID"

  subtasks:
    - id: "target"
      name: "Loan Default"
      task_type: "binary_classification"
      column: "TARGET"
      primary_metric: "roc_auc"
      direction: "maximize"
      weight: 1.0
      classes: [0, 1]

  validation:
    strategy: "kfold"
    num_folds: 5
    split_file: "splits/fold_assignments.tsv"
    probe_samples: 500
```

---

### 3.2 Scenario B: Multi-Task Text Classification with Conditional Routing

```yaml
task:
  name: "MultiTaskTextBenchmark"
  platform: "custom"
  task_type: "multilabel_classification"
  modality: "text"
  primary_metric: "macro_f1"
  direction: "maximize"
  description: "Example text classification task with a conditional secondary label."

  data_config:
    train_path: "dataset/train.tsv"
    id_column: "item_id"

  inputs:
    - name: "comment"
      type: "text"
      source_columns: ["text"]

  subtasks:
    - id: "topic_presence"
      name: "Topic Presence"
      task_type: "binary_classification"
      column: "topic_presence"
      primary_metric: "macro_f1"
      direction: "maximize"
      weight: 1.0
      classes: ["yes", "no"]

    - id: "topic_kind"
      name: "Topic Kind"
      task_type: "multiclass_classification"
      column: "topic_kind"
      primary_metric: "macro_f1"
      direction: "maximize"
      weight: 1.0
      classes: ["type_a", "type_b", "type_c", "none"]
      condition:
        parent_target: "topic_presence"
        operator: "=="
        values: "yes"
        fallback_value: "none"
        eval_condition_only: true

  validation:
    strategy: "holdout"
    dev_split_name: "dataset/dev.tsv"
    probe_samples: 100
```

---

### 3.3 Scenario C: Tabular Continuous Regression *(Schema Reference Only — Tabular is WIP)*

> [!NOTE]
> Tabular modeling is experimental work-in-progress (WIP). This scenario illustrates the declarative schema syntax for regression tasks.

```yaml
task:
  name: "House-Prices-Regression"
  platform: "kaggle"
  task_type: "regression"
  modality: "tabular"
  primary_metric: "rmse"
  direction: "minimize"
  description: "Predict final residential sale prices from structured architectural features."

  data_config:
    train_path: "dataset/train.csv"
    test_path: "dataset/test.csv"
    id_column: "Id"

  subtasks:
    - id: "SalePrice"
      name: "SalePrice"
      task_type: "regression"
      column: "SalePrice"
      primary_metric: "rmse"
      direction: "minimize"
      weight: 1.0

  validation:
    strategy: "kfold"
    num_folds: 5
    split_file: "splits/fold_assignments.tsv"
    probe_samples: 100
```

---

## 4. Human Onboarding Context Template (`docs/TASK.md`)

When dropping raw context into `docs/TASK.md`, use this exact Markdown structure so `tools/bootstrap_task.py` can parse it deterministically:

```markdown
# Competition Context: [Insert Benchmark Name]

- **Competition Name**: [e.g. Kaggle - Housing Prices]
- **Platform**: [kaggle | codabench | huggingface | generic]
- **Modality**: [tabular | text | vision | time-series]
- **Task Type**: [binary_classification | multiclass_classification | regression]
- **Primary ID Column**: [e.g. Id, uuid, patient_id]
- **Anti-Leakage Group Column**: [e.g. user_id, group_id, or None if IID]
- **Target Columns**: [e.g. SalePrice or label]
- **Primary Metric**: [rmse | macro_f1 | roc_auc | logloss]
- **Optimization Direction**: [maximize | minimize]
- **Submission Output**: [submission.csv]
```
