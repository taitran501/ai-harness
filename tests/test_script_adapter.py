"""Unit tests for Script-Adapter Execution Protocol in UniversalRunner."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import yaml

from harness.runner import UniversalRunner
from tests.test_correctness_contracts import task


class TestScriptAdapterProtocol(unittest.TestCase):
    def test_script_adapter_success_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "configs").mkdir()
            (root / "dataset").mkdir()
            (root / "scripts").mkdir()

            spec = task()
            doc = {"task": spec.metadata}
            doc["task"]["data"]["train_path"] = "dataset/train.csv"
            doc["task"]["validation"] = {"strategy": "kfold", "num_folds": 2}
            (root / "configs/task.yaml").write_text(yaml.safe_dump(doc), encoding="utf-8")

            n = 12
            frame = pd.DataFrame({
                "id": range(n),
                "group": range(n),
                "x": range(n),
                "category": ["a", "b"] * 6,
                "target": [float(x * 2.0) for x in range(n)],
            })
            frame.to_csv(root / "dataset/train.csv", index=False)

            # Create a user custom script that writes oof_predictions.csv
            script_code = """
import os
import argparse
import pandas as pd
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--exp-dir", required=True)
parser.add_argument("--config", required=True)
args = parser.parse_args()

project_root = os.environ.get("HARNESS_PROJECT_ROOT", ".")
train_df = pd.read_csv(os.path.join(project_root, "dataset", "train.csv"))

# Produce dummy OOF predictions
oof = train_df[["id"]].copy()
oof["pred_target"] = train_df["target"].values + 0.1
oof["fold"] = 0
oof.to_csv(os.path.join(args.exp_dir, "oof_predictions.csv"), index=False)
print("Dummy script finished successfully")
"""
            script_file = root / "scripts" / "dummy_train.py"
            script_file.write_text(script_code, encoding="utf-8")

            args = SimpleNamespace(
                project_root=str(root),
                task_config=None,
                fold=None,
                debug=False,
                probe_samples=None,
            )
            cfg = {
                "experiment": {
                    "id": "exp000_baseline_script",
                    "branch": "baseline",
                    "parent_id": "none",
                    "problem_addressed": "Establish verified script-adapter baseline anchor",
                    "hypothesis": "Custom python script adapter integrates cleanly with harness gates",
                    "intervention": "Execute external training script producing valid OOF predictions",
                },
                "execution": {
                    "mode": "script",
                    "entrypoint": "scripts/dummy_train.py",
                },
            }

            runner = UniversalRunner(cfg, "config.yaml", args)
            result = runner.run()

            self.assertEqual(runner.manifest["status"], "COMPLETED")
            self.assertEqual(result["experiment_id"], "exp000_baseline_script")
            self.assertIn("primary_score", result)
            self.assertEqual(result["n_samples"], 12)
            self.assertTrue((runner.exp_dir / "artifact_manifest.json").exists())
            self.assertTrue((runner.exp_dir / "metrics.json").exists())
            self.assertTrue((runner.exp_dir / "oof_predictions.csv").exists())

    def test_script_adapter_error_handling(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "configs").mkdir()
            (root / "dataset").mkdir()
            (root / "scripts").mkdir()

            spec = task()
            doc = {"task": spec.metadata}
            doc["task"]["data"]["train_path"] = "dataset/train.csv"
            doc["task"]["validation"] = {"strategy": "kfold", "num_folds": 2}
            (root / "configs/task.yaml").write_text(yaml.safe_dump(doc), encoding="utf-8")

            frame = pd.DataFrame({
                "id": range(6),
                "group": range(6),
                "x": range(6),
                "category": ["a", "b"] * 3,
                "target": [float(x) for x in range(6)],
            })
            frame.to_csv(root / "dataset/train.csv", index=False)

            broken_script = """
import sys
sys.stderr.write("FATAL ERROR: Missing GPU device\\n")
sys.exit(42)
"""
            (root / "scripts" / "fail.py").write_text(broken_script, encoding="utf-8")

            args = SimpleNamespace(
                project_root=str(root),
                task_config=None,
                fold=None,
                debug=False,
                probe_samples=None,
            )
            cfg = {
                "experiment": {
                    "id": "exp_baseline_fail",
                    "branch": "baseline",
                    "parent_id": "none",
                    "problem_addressed": "Test custom script failure detection in baseline mode",
                    "hypothesis": "Subprocess failure is detected and reported accurately",
                    "intervention": "Execute failing python script",
                },
                "execution": {
                    "mode": "script",
                    "entrypoint": "scripts/fail.py",
                },
            }

            runner = UniversalRunner(cfg, "config.yaml", args)
            with self.assertRaises(RuntimeError) as ctx:
                runner.run()
            self.assertIn("FATAL ERROR", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
