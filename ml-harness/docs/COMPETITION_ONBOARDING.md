# Competition Onboarding Guide: 2-Stage Agentic Workflow

When joining a machine learning competition or benchmark (Kaggle, Codabench, DrivenData, or academic shared tasks), datasets rarely follow a uniform structure. Raw data often arrives as a `.zip` archive, multiple language-specific TSVs, separated feature/label tables, or nested media directories.

Rather than enforcing rigid dataset paths, **ML-Harness** pairs a declarative configuration (`configs/task.yaml`) with a **Two-Stage Agentic Workflow**. Modern AI coding assistants (such as Antigravity IDE, Claude Code, or OpenAI Codex) have deep reasoning, file inspection, and web research capabilities to navigate unfamiliar datasets and bind them into the harness cleanly.

---

## Workflow Overview

```text
Raw Competition Data (.zip / folder / URLs)
                  │
                  ▼
┌────────────────────────────────────────────────────────┐
│  STAGE 1: Exploration, Research & Harness Alignment    │
│  • Extract & inspect archive (columns, types, targets) │
│  • Web search competition rules & metric if low-context│
│  • Configure configs/task.yaml                         │
│  • Run python tools/inject_skill.py                    │
└─────────────────────────┬──────────────────────────────┘
                          │
                          ▼
┌────────────────────────────────────────────────────────┐
│  STAGE 2: Validation Strategy, Baseline & Execution    │
│  • Set up leak-free CV split (Stratified/Group/Time)   │
│  • Create configs/exp001_baseline.yaml                 │
│  • Fast smoke test (--debug --fold 0)                  │
│  • Full CV run & Solution Tree DAG anchor              │
└────────────────────────────────────────────────────────┘
```

---

## Stage 1: Exploration, Research & Harness Alignment

### When to use
Whenever you clone the repository for a new competition, drop a raw `.zip` archive or dataset folder into the project, and need the environment configured.

### Copy-Paste Prompt for Your Agent

```text
I am participating in the [Insert Competition/Benchmark Name] competition.
I have placed my raw dataset archive in `dataset/` (or `[insert path]`).

Please execute Stage 1 Onboarding:
1. Unpack & inspect: If it is a zip or tar archive, extract it. Inspect the extracted files, formats, column schemas, target labels, and missing data distributions.
2. Competition research: If context is minimal or missing, search the web for [Insert Competition Name] to identify the official evaluation metric, submission requirements, domain specifics, and any competition rules/constraints.
3. Harness alignment: Configure `configs/task.yaml` with the proper task type, target columns, primary metric, and evaluation direction. Then run `python tools/inject_skill.py` to regenerate the agent skill.
4. Report: Provide a summary of the dataset findings, metric alignment, and proposed validation strategy before proceeding to modeling.
```

### What the Agent Accomplishes in Stage 1:
- Extracts local archives (handles nested folders and diverse file extensions).
- Gathers domain context from competition guidelines, benchmark papers, or Kaggle overviews.
- Writes an accurate `configs/task.yaml` pointing to the actual data location (`data_config.train_path`).
- Runs `python tools/inject_skill.py` so the agent's internal skill in `.agents/skills/` is updated with competition-specific rules, model recipes, and evaluation constraints.

---

## Stage 2: Validation Strategy, Baseline & Execution

### When to use
After Stage 1 is complete and the task configuration and agent skill are aligned.

### Copy-Paste Prompt for Your Agent

```text
Stage 1 is complete. Now execute Stage 2 Modeling & Baseline:
1. Validation setup: Establish an appropriate, leak-free validation strategy (e.g. StratifiedKFold, GroupKFold, or TimeSeriesSplit) based on the dataset structure and task type.
2. Baseline formulation: Create `configs/exp001_baseline.yaml` using the epistemic template (a simple, fast, robust model such as TF-IDF/Linear for text or LightGBM for tabular).
3. Smoke test: Run a fast single-fold debug probe (`python train.py --config configs/exp001_baseline.yaml --debug --fold 0`).
4. Full validation: If the smoke probe converges cleanly, run the full cross-validation, log the metric to the solution tree (`artifacts/journal.json`), and propose the first set of UCB exploration branches.
```

### What the Agent Accomplishes in Stage 2:
- Protects against data leakage by establishing verified cross-validation splits.
- Formulates a reproducible, low-complexity baseline model.
- Verifies code execution and loss convergence rapidly with a sample probe before committing full resources.
- Registers the baseline score in `artifacts/journal.json` and `EXPERIMENTS.md`, creating the anchor node for Upper Confidence Bound (UCB) tree search.

---

## Manual Alternative (Terminal / Non-Agent)

If you prefer setting up the competition manually without an AI assistant:

```bash
# 1. Extract your dataset into your preferred location (e.g., data/ or dataset/)
tar -xf dataset/competition_data.zip -C dataset/

# 2. Select a preset or customize configs/task.yaml
cp configs/tasks/kaggle_tabular_regression.yaml configs/task.yaml

# 3. Inject the task configuration into the agent skill
python tools/inject_skill.py

# 4. Run a fast smoke test
python train.py --config configs/exp001_baseline.yaml --debug --fold 0

# 5. Run full cross-validation
python train.py --config configs/exp001_baseline.yaml
```
