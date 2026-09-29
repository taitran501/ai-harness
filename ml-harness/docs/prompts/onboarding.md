# Master Onboarding Prompt

Use this prompt with any AI coding assistant (Antigravity IDE, Claude Code, OpenAI Codex) to onboard a new competition or benchmark into ML-Harness.

## The Prompt

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

### Scenario A: Dataset Only
When you have a local archive or dataset folder but no context:
```text
- Task Context: None provided
- Dataset: dataset/my_competition_training.zip
```
The agent inspects file names, reads schema headers, searches online for matches, discovers the competition rules, sets `configs/task.yaml`, and trains the baseline.

### Scenario B: Task Context Only
When you have a competition name or URL but have not downloaded the dataset:
```text
- Task Context: Kaggle "House Prices - Advanced Regression Techniques"
- Dataset: None provided
```
The agent searches competition guidelines, locates the data source, fetches/unpacks it into the workspace, creates the configuration, and launches the baseline.

### Scenario C: Both Provided
When you provide both the dataset location and problem statement:
```text
- Task Context: Multi-label toxic comment classification optimizing Macro-F1
- Dataset: data/train.tsv
```
The agent cross-validates schema consistency, establishes stratified cross-validation, and runs the baseline.
