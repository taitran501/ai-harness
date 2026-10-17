"""[TEMPLATE] Pipeline stage -- blend OOF predictions from earlier stages.

Copy this file into your workspace's  src/  and replace every  [CUSTOMIZE]
marker. A blend stage reads the OOF predictions written by the train stages
(they share the experiment dir) and produces the FINAL  oof_predictions.csv
(and optionally  test_predictions.csv).

The harness invokes it the same way as any stage:

    python <stage_script> --exp-dir <artifact_dir> --config <run_config.yaml>

Make the train stages write to stage-specific names (e.g.  train_a_oof.csv)
so this stage can read them; then write the combined result to the standard
  oof_predictions.csv  that the harness verifies.
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
stage = os.environ.get("HARNESS_STAGE_NAME", "blend")
print(f"[{stage}] exp_dir={exp_dir}", flush=True)

task = config["task"]
data_cfg = task["data_config"]
subtask_id = task["subtasks"][0]["id"]
id_col = data_cfg["id_column"]
pred_col = f"pred_{subtask_id}"

# --- [CUSTOMIZE] Load the OOF predictions from the train stages ----------
# Replace the filenames below with what your train stages actually produce.
oof_a = pd.read_csv(exp_dir / "train_a_oof.csv")  # [CUSTOMIZE]
oof_b = pd.read_csv(exp_dir / "train_b_oof.csv")  # [CUSTOMIZE]

# --- [CUSTOMIZE] Blend ----------------------------------------------------
# Simple equal-weight average of the prediction columns. Replace with your
# blending logic (learned weights, stacking, rank averaging, ...).
blended = 0.5 * oof_a[pred_col] + 0.5 * oof_b[pred_col]

# --- Write the FINAL OOF predictions (verified by the harness) ------------
final = oof_a[[id_col]].copy()
final[pred_col] = blended
final.to_csv(exp_dir / "oof_predictions.csv", index=False)
print(f"[{stage}] wrote {len(final)} blended rows -> oof_predictions.csv", flush=True)
