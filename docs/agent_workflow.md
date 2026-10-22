# AI Assistant Workflow Guide

This guide describes how AI coding assistants (such as Antigravity, Claude Code, or OpenAI Codex) interact with the AI-Harness framework.

---

## 1. Core Operating Principles

1. **Context Economy**:
   Never dump raw datasets, large CSV tables, or multi-thousand-line logs into your context window. Always query task status and leaderboard standings using the Agent Bridge (`ai-harness-agent context`), which delivers a compressed Heads-Up Display (~1,000 tokens).

2. **Atomic Interventions**:
   Modify exactly one variable between parent and child experiments (such as a feature pipeline, a loss function, or a model architecture). Shotgun iterations that change multiple components simultaneously make performance deltas impossible to attribute.

3. **Fast Probes Before Full Runs**:
   Always execute a single-fold debug probe (`--debug --fold 0`) before allocating compute to full cross-validation. This catches syntax errors, data shape mismatches, and gradient issues immediately.

4. **Task-Owned Contracts**:
   The active task plugin or declarative task specification (`configs/task.yaml`) defines data ingestion, validation splits, metrics, and submission packaging. Always adhere to declared contracts rather than inventing ad-hoc evaluation logic.

5. **Artifact Immutability**:
   Folders under `.ai-harness/artifacts/<task-id>/<exp_id>` are strictly immutable once full validation completes. Never edit files inside completed run folders.

---

## 2. The 8-Phase Competition Journey

AI-Harness organizes competition work into eight sequential, evidence-gated lifecycle phases.

```text
1. setup        -> Configure task specifications, locate datasets, and verify dependencies.
2. understand   -> Run exploratory data analysis (EDA), inspect distributions, and verify anti-leakage invariants.
3. baseline     -> Establish a minimal working model, verify with a probe, and lock the baseline anchor.
4. experiment   -> Execute iterative AIDE UCB exploration cycles with atomic interventions.
5. select       -> Review diagnostic failure slices, error diversity, and candidate pool promotions.
6. public_lb    -> Track external competition scores (e.g., Kaggle public leaderboard).
7. freeze       -> Blend diverse models into ensembles and freeze final candidate slots.
8. finalization -> Perform deterministic pre-submission audits and package final deliverables.
```

The workspace prompt pack and phase transitions are managed via the `ai-harness phase` CLI:
```bash
# Initialize editable phase prompts under prompts/ai-harness/
ai-harness phase init

# Check phase completion status and evidence requirements
ai-harness phase status

# Render current task context into the active phase prompt
ai-harness phase render --phase experiment

# Record the assistant's phase findings and artifact paths
ai-harness phase record --phase experiment --response-file response.md
```

---

## 3. Onboarding a New Competition

There are two primary ways to configure a task workspace:

### Method A: Autonomous Bootstrapping
When given an unconfigured dataset directory and basic competition notes:
1. Provide competition details in `docs/TASK.md` (or pass a URL/description).
2. Run the bootstrapper:
   ```bash
   ai-harness-bootstrap --project-root . --dataset dataset/ --context docs/TASK.md
   ```
   This generates `configs/task.yaml`, `splits/fold_assignments.tsv`, and an initial baseline config `configs/exp000_baseline.yaml`.

### Method B: Declarative Attachment with a Task Plugin
When using a dedicated competition adapter package:
```bash
# List available adapter plugins
ai-harness plugins

# Attach the harness to the workspace
ai-harness attach --project-root . --task-config configs/task.yaml --plugin <plugin_id>
```

---

## 4. Agent Bridge CLI Reference

AI assistants interact with the harness through `tools/agent_bridge_cli.py` (`ai-harness-agent`):

### Querying Context HUD
```bash
# Retrieve compressed task summary, current stage, best CV score, and UCB recommendations
ai-harness-agent context

# Retrieve context in JSON format (useful for parsing recommended branches)
ai-harness-agent context --format json
```

### Managing Work Sessions
```bash
# Start a session with experiment and runtime budgets
ai-harness-agent start-session --goal "Improve primary validation metric" --max-experiments 3
```

### Executing Actions
All actions are executed using the `act` subcommand. Prefer **flat flags**
(`--exp-id`, `--config-path`, `--branch`, … — kebab-case maps to snake_case
params). On **PowerShell, never inline JSON** (`--params '{...}'` breaks under
PowerShell quoting): save params to a UTF-8 JSON file and pass `--param-file`.

```powershell
# 1. Create a child experiment from a parent node
ai-harness-agent act --action create_experiment --branch feat --descriptor stemming_clean --parent-id exp000_baseline

# 2. Inspect diagnostics and error slices of a parent run
ai-harness-agent act --action inspect_diagnostics --exp-id exp000_baseline

# 3. Promote an experiment to the candidate pool
ai-harness-agent act --action promote_candidate --exp-id exp001 --name candidate_v1

# 4. Synthesize an ensemble from candidates (complex params -> --param-file on PowerShell)
#    synthesize.json: {"name": "ens_v1", "candidates": ["candidate_v1", "candidate_v2"]}
ai-harness-agent act --action build_ensemble --param-file synthesize.json

# 5. Freeze a final candidate (requires user approval)
ai-harness-agent act --action freeze_candidate --user-approved --slot-1-id candidate_v1 --notes "Best validation score"
```

> Smoke tests always run locally (`run_smoke_test`, single fold) even in
> backend-locked workspaces; full validation dispatches through the task's
> declared execution backend (`run_full_validation`). Session budgets
> (`--max-experiments`, `--max-gpu-hours`, `--max-cloud-runs`) are tracked per
> `BridgeSession`.

---

## 5. Model Execution and Probes

Experiment pipelines are executed using `harness.runner` (`ai-harness-run`):

### Fast Single-Fold Debug Probe (Fold 0)
Always run a debug probe before launching full validation:
```bash
ai-harness-run --project-root . --config configs/exp001_feat_stemming_clean.yaml --debug --fold 0
```
This runs a fast check on fold 0 with a capped subsample to verify syntax, pipeline transformations, and loss stability. Probe outputs are stored under `probes/<exp_id>/` and never pollute the official leaderboard.

### Custom script and pipeline scope

Each child reads `harness.script_validation.load_validation_manifest()` (the path is
also in `HARNESS_VALIDATION_MANIFEST`). Its `partitions` list the exact `fold`,
`train_ids` and `validation_ids` to execute. Probe sampling is already applied.
Train only on these training IDs and output exactly one OOF row per validation ID.
This applies to single-fold smoke, bounded probes, full CV, external holdout and
expanding time-series validation. Load task-specific training/dev data as declared
by the task; a holdout dev dataset may be separate from train.

`HARNESS_RUN_KIND`, `HARNESS_SELECTED_FOLDS` (JSON) and `HARNESS_SAMPLE_LIMIT`
provide summary scope. `splits.tsv` retains the complete fold map for full-CV
compatibility; `-1` marks training-only rows. Do not use all its fold values as a
probe execution loop. Scripts that return full-CV output for a single-fold probe
are rejected rather than recorded as a successful smoke test.

### Full Cross-Validation
Once the probe passes cleanly, run full multi-fold validation:
```bash
ai-harness-run --project-root . --config configs/exp001_feat_stemming_clean.yaml
```
This computes full out-of-fold predictions, writes `metrics.json`, produces `DIAGNOSTICS.md`, generates `EXPERIMENT_CARD.md`, and updates the Solution Tree.

### Multi-Stage Pipeline Execution (train / blend / stack)
When a hypothesis needs several sequential subprocesses — e.g. train model A, train
model B, blend, then fit a stack — declare `execution.mode: pipeline` instead of a
single `entrypoint`. The harness runs each stage as an audited subprocess, in order,
sharing the experiment directory so later stages can read artifacts written by earlier
ones. This replaces the pattern of hand-writing one monolithic orchestration script
outside the harness, which bypassed the per-stage audit trail.

```yaml
experiment:
  approach: custom_script   # or "script"
execution:
  mode: pipeline
  pipeline:
    - name: train_a
      entrypoint: src/train_a.py
      args: [--fold, 0]
    - name: blend
      entrypoint: src/blend.py
```

A simpler form (names derived from the script stem, no per-stage args) is also
accepted:

```yaml
execution:
  mode: pipeline
  entrypoints: [src/train_a.py, src/blend.py]
```

Behavior and guarantees:
- **Sequential, shared dir.** Stages run in config order; each writes into the shared
  experiment directory (or `HARNESS_STAGE_DIR`). The last stage must produce
  `oof_predictions.csv` (and optionally `test_predictions.csv`), which the harness
  verifies exactly as for `mode: script` before finalizing.
- **Per-stage audit trail.** Every stage's stdout/stderr is captured to
  `stages/<name>.log`; per-stage exit code, duration, and log path are recorded in
  `stage_manifest.json` (persisted even when a stage fails).
- **Stage env vars.** In addition to the usual `HARNESS_*` env vars, each stage receives
  `HARNESS_STAGE_NAME`, `HARNESS_STAGE_INDEX`, `HARNESS_STAGE_TOTAL`, and
  `HARNESS_STAGE_DIR`.
- **Fail-fast.** The first non-zero stage exit stops the pipeline and raises a
  `RuntimeError` naming the failed stage and its stderr. Later stages do not run.
- **Back-compat.** `execution.mode: script` (single `entrypoint`) is unchanged.
- **Audit gate.** After the stages complete, the harness verifies
  `stage_manifest.json` is complete and consistent (stage count, all exit codes 0,
  index order 0..N-1, and every per-stage log present) before finalizing; any
  inconsistency raises an audit error.

**Valid `approach` × `mode` combos** (the runner rejects incompatible pairs at
dispatch — see `runner._execute_run`). The critical rule for pipelines:

| `execution.mode` | required `experiment.approach` |
|---|---|
| `pipeline` | `custom_script` or `script` (any other approach is rejected) |
| `script` | any approach |
| `built_in` | `ml_classical`, `ml`, `dl`, `llm_peft`, or `llm_prompt` |

So a pipeline config **MUST** set `experiment.approach: custom_script` (or `script`).

> [!TIP]
> Start from the **pipeline template** (`configs/pipeline_template.yaml`) — copy it
> to `configs/exp<NNN>_<branch>_<descriptor>.yaml`, replace every `[CUSTOMIZE]`
> marker, and write one script per stage under `src/` (scaffolds in the
> framework's `templates/pipeline/`). `ai-harness validate-task` warns if
> `train_path` /
> `target_column` still carry a placeholder token (uncustomized template).

---

## 6. Pre-Submission Audit Gate

Before submitting predictions to an external platform or client, run the deterministic pre-submission audit gate:

```bash
ai-harness-audit --submission-file .ai-harness/artifacts/<task-id>/<exp_id>/test_predictions.csv --task-config configs/task.yaml
```

The audit checks:
- Prediction file existence and non-zero byte size.
- Exact instance count matching the test dataset.
- Exact row ID alignment with zero null or duplicate keys.
- Proper probability range constraints $[0, 1]$ or class label validity.
- Complete absence of evaluation labels or leakages.
