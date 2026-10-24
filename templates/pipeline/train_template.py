"""[TEMPLATE] Pipeline stage -- train one model, write OOF predictions.

Copy this file into your workspace's  src/  and replace every  [CUSTOMIZE]
marker. The harness invokes each pipeline stage as a subprocess:

    python <stage_script> --exp-dir <artifact_dir> --config <run_config.yaml>

  - --exp-dir : the shared experiment dir (write artifacts here)
  - --config  : the merged run config (task spec + experiment config)
  - cwd       : the project (workspace) root

Env vars the harness sets for every stage:
    HARNESS_EXP_ID, HARNESS_EXP_DIR, HARNESS_PROJECT_ROOT, HARNESS_SEED,
    HARNESS_STAGE_NAME, HARNESS_STAGE_INDEX, HARNESS_STAGE_TOTAL,
    HARNESS_STAGE_DIR, HARNESS_VALIDATION_MANIFEST

Contract (the harness verifies AFTER the whole pipeline):
    - The LAST stage must write  oof_predictions.csv  into --exp-dir, with
      one column per subtask named  pred_<subtask_id>.
    - Optionally write  test_predictions.csv  (for submission) and
      metrics.json (primary metric key).
    - A non-zero exit from any stage halts the pipeline (fail-fast).
"""

import os
import sys
from pathlib import Path

import pandas as pd
import yaml

from harness.script_validation import load_validation_manifest


def _arg(name: str) -> str:
    if name not in sys.argv:
        raise SystemExit(f"missing {name} in argv: {sys.argv}")
    return sys.argv[sys.argv.index(name) + 1]


exp_dir = Path(_arg("--exp-dir"))
config = yaml.safe_load(Path(_arg("--config")).read_text(encoding="utf-8"))
stage = os.environ.get("HARNESS_STAGE_NAME", "train")
print(f"[{stage}] exp_dir={exp_dir}", flush=True)

task_doc = yaml.safe_load(Path(os.environ["HARNESS_TASK_CONFIG"]).read_text(encoding="utf-8"))
task = task_doc.get("task", task_doc)
data_cfg = task.get("data_config") or task.get("data") or {}
project_root = Path(os.environ.get("HARNESS_PROJECT_ROOT", "."))

# --- [CUSTOMIZE] Load your data ------------------------------------------
train_path = project_root / data_cfg["train_path"]
if str(train_path).endswith(".parquet"):
    df = pd.read_parquet(train_path)
else:
    df = pd.read_csv(train_path)

# --- [CUSTOMIZE] Load a separate dev dataset here when the task declares one.
# All training stages must honor this resolved execution scope.
scope = load_validation_manifest()
id_col = data_cfg["id_column"]
subtask_id = task["subtasks"][0]["id"]
records = []
for part in scope["partitions"]:
    train = df.set_index(id_col).loc[part["train_ids"]]
    dev = df.set_index(id_col).loc[part["validation_ids"]]
    # --- [CUSTOMIZE] Fit your model on train, then predict dev ---------------
    # model.fit(train[FEATURES], train[TARGET_COL])
    # prediction = model.predict(dev[FEATURES])
    raise NotImplementedError("[CUSTOMIZE] Implement fold training and validation inference")
    # records.extend({id_col: sample_id, f"pred_{subtask_id}": value,
    #                 "fold": part["fold"]}
    #                for sample_id, value in zip(dev.index, prediction))

# --- Write predictions for exactly the selected validation IDs ------------
# Earlier stages can use a stage-specific OUT_NAME (e.g. train_a_oof.csv).
OUT_NAME = "oof_predictions.csv"  # [CUSTOMIZE]
pd.DataFrame(records).to_csv(exp_dir / OUT_NAME, index=False)
print(f"[{stage}] wrote {len(records)} rows -> {OUT_NAME}", flush=True)
