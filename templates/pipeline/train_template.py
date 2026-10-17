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
    HARNESS_STAGE_DIR

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


def _arg(name: str) -> str:
    if name not in sys.argv:
        raise SystemExit(f"missing {name} in argv: {sys.argv}")
    return sys.argv[sys.argv.index(name) + 1]


exp_dir = Path(_arg("--exp-dir"))
config = yaml.safe_load(Path(_arg("--config")).read_text(encoding="utf-8"))
stage = os.environ.get("HARNESS_STAGE_NAME", "train")
print(f"[{stage}] exp_dir={exp_dir}", flush=True)

task = config["task"]
data_cfg = task["data_config"]
project_root = Path(os.environ.get("HARNESS_PROJECT_ROOT", "."))

# --- [CUSTOMIZE] Load your data ------------------------------------------
train_path = project_root / data_cfg["train_path"]
if str(train_path).endswith(".parquet"):
    df = pd.read_parquet(train_path)
else:
    df = pd.read_csv(train_path)

# --- [CUSTOMIZE] Build features + train your model -----------------------
# Replace with the modeling code for your hypothesis. Example:
#   import numpy as np
#   from sklearn.linear_model import LogisticRegression
#   X = df[FEATURES].fillna(0.0)
#   y = df[TARGET_COL]
#   model = LogisticRegression(C=0.01, max_iter=1000).fit(X, y)
#   oof_pred = model.predict(X)
oof_pred = df.iloc[:, 0]  # [CUSTOMIZE] replace with real out-of-fold predictions

# --- Write OOF predictions (verified by the harness) ---------------------
# If this is the LAST stage, write to "oof_predictions.csv" (the harness
# verifies that exact file). If a later blend stage combines your predictions,
# write to a stage-specific name (e.g. "train_a_oof.csv") instead.
OUT_NAME = "oof_predictions.csv"  # [CUSTOMIZE] or e.g. "train_a_oof.csv"
subtask_id = task["subtasks"][0]["id"]
oof = pd.DataFrame(
    {data_cfg["id_column"]: df[data_cfg["id_column"]], f"pred_{subtask_id}": oof_pred}
)
oof.to_csv(exp_dir / OUT_NAME, index=False)
print(f"[{stage}] wrote {len(oof)} rows -> {OUT_NAME}", flush=True)
