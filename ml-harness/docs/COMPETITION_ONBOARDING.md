# Competition Onboarding: The Universal Agent Prompt Guide

When starting a machine learning competition or benchmark (Kaggle, Codabench, DrivenData, academic shared tasks), you typically have **at least one** of two things:

1. **A Dataset**: A local archive (`.zip`, `.tar.gz`), a raw folder, multiple TSV/CSV files, or a download URL.
2. **A Task Context**: A competition name, problem statement, competition link, academic paper, or evaluation guidelines.

You do **not** need to have both upfront, and you do **not** need manual CLI ingestion scripts. Modern AI coding assistants (Antigravity IDE, Claude Code, OpenAI Codex) possess deep reasoning, code execution, and web research capabilities to bridge the missing pieces autonomously.

---

## The Master Onboarding Prompt

Copy and paste this prompt to your AI assistant. Simply fill in whatever information you have (Dataset, Task Context, or both) and let the agent handle everything from extraction to the first verified baseline:

```text
You are an expert competitive ML engineer setting up this project using the ML-Harness framework.

Here is the initial information I have:
- Task Context: [Insert competition name, benchmark link, paper, or description — or state "None provided"]
- Dataset: [Insert path to local archive/folder, download URL, or files in workspace — or state "None provided"]

Your mission is to take this information, autonomously bridge any missing gaps, configure the harness, and establish a validated anchor baseline. Execute the following steps thoroughly:

1. Resolve the Missing Piece (At Least One Provided):
   - IF I only gave you the Dataset:
     Inspect the files and columns. Search the web, GitHub, or competition platforms using the file names, column headers, or domain text to identify the exact competition, official task name, primary evaluation metric, and rules.
   - IF I only gave you Task Context:
     Search and identify where the dataset is hosted (Kaggle, Hugging Face, Codabench, CodaLab, Zenodo, or direct URL). If public/accessible, download and unpack it into the project workspace (e.g., in `dataset/` or `data/`). If it requires private authentication, provide me the exact 1-line command or API step needed to download it.
   - IF I gave you Both:
     Cross-reference the dataset against the task specification to ensure all expected features, splits, and labels match.

2. Deep Data Hygiene & Structure Discovery:
   - Extract any archives (.zip, .tar, etc.) preserving folder hierarchy.
   - Scan all data files (CSV, TSV, Parquet, JSON, images, etc.). Determine shapes, data types, missing values, duplicates, and column roles (ID, features, target labels).
   - Detect task complexity: Is it single-table tabular, multi-table relational, multilingual text, multi-label, hierarchical targets, or multi-modal?
   - Identify grouping and leakage risks: Check for group columns (e.g. author ID, video key, patient ID, group cluster, timestamp/temporal sequence) that MUST NOT be split across train and validation sets.

3. Harness Alignment & Declarative Configuration:
   - Configure `configs/task.yaml`:
     - Set `task.name`, `task.platform`, `task.task_type`, and `task.modality`.
     - Set `task.primary_metric` with its exact optimization direction (`maximize` or `minimize`).
     - Point `data_config.train_path` to the real data location (file or directory).
     - Specify input feature definitions, subtasks, target columns, and group columns.
   - Run `python tools/inject_skill.py` to regenerate `.agents/skills/ml-experiment-harness/SKILL.md` so your internal agent skill is tailored to this competition's exact rules and metric.

4. Validation Strategy & Baseline Formulation:
   - Choose the leak-free cross-validation scheme (e.g., GroupKFold if grouping exists, StratifiedKFold if class imbalance, TimeSeriesSplit if temporal, or Holdout if an official dev set is provided).
   - Formulate a clean, atomic baseline in `configs/exp001_baseline.yaml` following the epistemic template (a fast, interpretable, robust model such as TF-IDF + LogisticRegression for text, LightGBM for tabular, or simple MLP/Linear).

5. Fast Smoke Probe & Full Baseline Execution:
   - First run a fast single-fold debug probe:
     `python train.py --config configs/exp001_baseline.yaml --debug --fold 0`
   - Inspect the logs: verify data loading, feature pipelines, loss convergence, and metric calculation. Fix any runtime or shape errors.
   - Once verified, execute the full cross-validation:
     `python train.py --config configs/exp001_baseline.yaml`
   - Record the baseline score in `artifacts/journal.json` and `EXPERIMENTS.md` as the anchor root of the solution tree.

6. Comprehensive Diagnostic Briefing:
   - Deliver a clear report covering:
     a) Task and dataset profile (dimensions, class distribution, anomalies).
     b) Leakage safeguards and validation protocol rationale.
     c) Baseline performance breakdown (overall primary metric and per-subtask/fold metrics).
     d) Initial error slice analysis (where the baseline fails most).
     e) Top 3 prioritized exploration hypotheses (feature engineering, model architecture, loss function) for the next iteration.
```

---

## Scenario Walkthroughs

### Scenario A: You only have a local `.zip` file (No context)
```text
- Task Context: None provided
- Dataset: dataset/StereoQueerEval_training.zip (or data.zip in workspace)
```
**What the Agent does:**
1. Unzips the file into a clean folder.
2. Reads the files (e.g. `StereoQueerEval_EN_training.tsv`).
3. Searches the web for `StereoQueerEval` and identifies:
   - Competition: SemEval-2027 Task 4.
   - Metric: Macro-F1 across subtasks (Stereotype, Hate Speech, Target).
   - Evaluation scheme: Multilingual evaluation.
4. Auto-generates `configs/task.yaml`, runs `tools/inject_skill.py`, and launches the baseline.

### Scenario B: You only have a competition link or name (No data downloaded)
```text
- Task Context: Kaggle "House Prices - Advanced Regression Techniques"
- Dataset: None provided
```
**What the Agent does:**
1. Searches the competition rules and data schema.
2. Identifies the primary metric: Log RMSE.
3. Downloads the data via `kaggle competitions download -c house-prices-advanced-regression-techniques` (or provides the Kaggle API command if credentials are required).
4. Unpacks `train.csv` and `test.csv`, writes `configs/task.yaml`, and trains a LightGBM/Ridge regression baseline.

### Scenario C: You have both data and task instructions
```text
- Task Context: Predicting customer churn with AUC-ROC optimization
- Dataset: data/raw/customer_churn.csv
```
**What the Agent does:**
1. Validates the columns against the problem description.
2. Checks for class imbalance and missing value patterns.
3. Sets up Stratified 5-Fold CV, configures `configs/task.yaml`, runs `tools/inject_skill.py`, and trains the baseline model.

---

## Manual Fallback (For Terminal Enthusiasts)

If you prefer executing the bootstrap manually without prompting an assistant:

```bash
# 1. Unzip your dataset archive into your chosen directory
tar -xf dataset/my_competition_data.zip -C dataset/

# 2. Pick a preset matching your task type
cp configs/tasks/kaggle_tabular_regression.yaml configs/task.yaml
# (Or kaggle_titanic.yaml, stereoqueereval.yaml, etc.)

# 3. Inject task specifications into the agent skill
python tools/inject_skill.py

# 4. Run a fast smoke test (single fold debug probe)
python train.py --config configs/exp001_tfidf_baseline.yaml --debug --fold 0

# 5. Run full cross-validation
python train.py --config configs/exp001_tfidf_baseline.yaml
```
