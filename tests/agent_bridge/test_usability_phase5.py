"""Unit tests for agent usability Phase 5: CatBoost, diagnostics, project auto-resolution, and split robustness.

Covers:
1. Feature validation diagnostics (informative errors for missing/forbidden columns in validate_model_features).
2. Built-in CatBoostModel wrapper (create_model, fit, predict, predict_proba, num_parameters).
3. HarnessProject instantiation defaults and __post_init__ auto-discovery.
4. Runner split file handling: fold column aliases (kfold, split, fold_id), case-insensitive ID join, and detailed coverage error messages.
5. AgentBridge.act alias and harness.agent_bridge.main CLI forwarder.
6. JSON serialization robustness (_json_serial in agent_bridge_cli).
"""

import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from harness.agent_bridge import AgentBridge
from harness.models import (
    HAS_CATBOOST,
    CatBoostModel,
    create_model,
    validate_model_features,
)
from harness.project import HarnessProject
from harness.task_spec import UniversalTaskSpec
from tools.agent_bridge_cli import _json_serial


class TestAgentUsabilityP5(unittest.TestCase):
    """Test suite covering usability enhancements in Phase 5."""

    def test_feature_validation_diagnostics(self):
        """Verify validate_model_features raises informative, detailed errors."""
        df = pd.DataFrame({
            "id": [1, 2, 3],
            "feat_a": [0.1, 0.2, 0.3],
            "target": [0, 1, 0],
        })
        forbidden = {"id", "target", "fold"}

        # Case 1: Empty features
        with self.assertRaises(ValueError) as ctx:
            validate_model_features([], df, forbidden, variant="tab")
        self.assertIn("has no declared source_columns", str(ctx.exception))

        # Case 2: Missing features in DataFrame
        with self.assertRaises(ValueError) as ctx:
            validate_model_features(["feat_a", "feat_b", "feat_c"], df, forbidden, variant="tab")
        self.assertIn("missing in DataFrame", str(ctx.exception))
        self.assertIn("feat_b", str(ctx.exception))
        self.assertIn("feat_c", str(ctx.exception))

        # Case 3: Forbidden features declared
        with self.assertRaises(ValueError) as ctx:
            validate_model_features(["feat_a", "target", "id"], df, forbidden, variant="tab")
        self.assertIn("forbidden columns", str(ctx.exception))
        self.assertIn("target", str(ctx.exception))

        # Case 4: Valid features should succeed without error
        validate_model_features(["feat_a"], df, forbidden, variant="tab")

    @unittest.skipUnless(HAS_CATBOOST, "CatBoost is required for this test")
    def test_catboost_classification_and_regression(self):
        """Verify CatBoostModel fits, predicts probabilities/classes, and counts parameters."""
        clf_spec = UniversalTaskSpec.from_dict({
            "task": {
                "id_column": "id",
                "primary_metric": "accuracy",
                "direction": "maximize",
                "subtasks": {
                    "label": {
                        "task_type": "binary_classification",
                        "column": "label",
                        "classes": [0, 1],
                    }
                },
                "inputs": {
                    "tabular_raw": {
                        "type": "tabular",
                        "source_columns": ["feat_1", "feat_2"],
                    }
                },
            }
        })
        clf_df = pd.DataFrame({
            "id": list(range(20)),
            "feat_1": np.linspace(0, 1, 20),
            "feat_2": np.linspace(1, 0, 20),
            "label": [0, 1] * 10,
        })

        model = create_model(clf_spec, "catboost", iterations=5, depth=3)
        self.assertIsInstance(model, CatBoostModel)
        model.fit(clf_df, variant="tabular_raw")
        preds = model.predict(clf_df, variant="tabular_raw")

        self.assertIn("label", preds)
        self.assertIn("proba_label", preds)
        self.assertEqual(len(preds["label"]), 20)
        self.assertEqual(preds["proba_label"].shape, (20, 2))
        self.assertGreater(model.num_parameters(), 0)

        # Also test regression
        reg_spec = UniversalTaskSpec.from_dict({
            "task": {
                "id_column": "id",
                "primary_metric": "rmse",
                "direction": "minimize",
                "subtasks": {
                    "target": {
                        "task_type": "regression",
                        "column": "target",
                    }
                },
                "inputs": {
                    "tabular_raw": {
                        "type": "tabular",
                        "source_columns": ["feat_1", "feat_2"],
                    }
                },
            }
        })
        reg_df = pd.DataFrame({
            "id": list(range(20)),
            "feat_1": np.linspace(0, 1, 20),
            "feat_2": np.linspace(1, 0, 20),
            "target": np.linspace(10, 20, 20),
        })
        reg_model = create_model(reg_spec, "cb_regressor", iterations=5, depth=3)
        reg_model.fit(reg_df, variant="tabular_raw")
        reg_preds = reg_model.predict(reg_df, variant="tabular_raw")
        self.assertIn("target", reg_preds)
        self.assertEqual(len(reg_preds["target"]), 20)

    def test_harness_project_auto_resolution(self):
        """Verify HarnessProject can be instantiated directly with defaults and resolves paths."""
        project = HarnessProject(".")
        self.assertTrue(project.root.is_absolute())
        self.assertTrue(project.task_config_path.is_absolute())
        self.assertTrue(project.artifacts_dir.is_absolute())
        self.assertIsNotNone(project.manifest)

        # Default instantiation without arguments should also succeed
        default_proj = HarnessProject()
        self.assertEqual(default_proj.root, project.root)

    def test_runner_split_loading_aliases_and_errors(self):
        """Verify Runner split file loading handles fold aliases, case-insensitivity, and diagnostics."""
        import argparse

        from harness.runner import Runner

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cfg_dir = root / "configs"
            cfg_dir.mkdir(parents=True)
            task_yaml = cfg_dir / "task.yaml"
            task_yaml.write_text(
                """
task:
  name: SplitTest
  id: split_test
  id_column: row_id
  primary_metric: accuracy
  validation:
    split_file: splits.csv
  subtasks:
    y:
      task_type: binary_classification
      column: y
      classes: [0, 1]
  inputs:
    raw:
      type: tabular
      source_columns: [x]
""",
                encoding="utf-8",
            )
            # Create training data
            train_df = pd.DataFrame({"row_id": [101, 102, 103, 104], "x": [1, 2, 3, 4], "y": [0, 1, 0, 1]})
            train_df.to_csv(root / "train.csv", index=False)

            # Test 1: Split file using alias 'kfold' and uppercase 'ROW_ID'
            splits_df = pd.DataFrame({"ROW_ID": [101, 102, 103, 104], "kfold": [0, 1, 0, 1]})
            splits_df.to_csv(root / "splits.csv", index=False)

            exp_cfg = {
                "experiment": {"approach": "ml_classical"},
                "model": {"name": "baseline"},
                "input_variant": "raw",
                "paths": {"train_path": "train.csv", "split_path": "splits.csv"},
            }
            exp_file = cfg_dir / "exp001.yaml"
            exp_file.write_text("{}", encoding="utf-8")
            args = argparse.Namespace(
                project_root=str(root),
                task_config=str(task_yaml),
                debug=False,
                probe_samples=None,
                fold=None,
                job_id="test_run",
                force_import=False,
            )
            runner = Runner(exp_cfg, exp_file, args)
            loaded = runner._load_training_data()
            self.assertIn("fold", loaded.columns)
            self.assertEqual(loaded["fold"].tolist(), [0, 1, 0, 1])

            # Test 2: Split file missing some IDs -> raises informative error
            bad_splits = pd.DataFrame({"row_id": [101, 102], "fold": [0, 1]})
            bad_splits.to_csv(root / "bad_splits.csv", index=False)
            exp_cfg_bad = dict(exp_cfg, paths={"train_path": "train.csv", "split_path": "bad_splits.csv"})
            exp_bad_file = cfg_dir / "exp002.yaml"
            exp_bad_file.write_text("{}", encoding="utf-8")
            runner_bad = Runner(exp_cfg_bad, exp_bad_file, args)
            with self.assertRaises(ValueError) as ctx:
                runner_bad._load_training_data()
            self.assertIn("does not cover 2/4 training IDs", str(ctx.exception))
            self.assertIn("Sample uncovered IDs: [103, 104]", str(ctx.exception))

            # Test 3: Split file missing fold column
            no_fold_splits = pd.DataFrame({"row_id": [101, 102, 103, 104], "category": ["a", "b", "c", "d"]})
            no_fold_splits.to_csv(root / "no_fold.csv", index=False)
            exp_cfg_nofold = dict(exp_cfg, paths={"train_path": "train.csv", "split_path": "no_fold.csv"})
            exp_nofold_file = cfg_dir / "exp003.yaml"
            exp_nofold_file.write_text("{}", encoding="utf-8")
            runner_nofold = Runner(exp_cfg_nofold, exp_nofold_file, args)
            with self.assertRaises(ValueError) as ctx:
                runner_nofold._load_training_data()
            self.assertIn("missing required columns", str(ctx.exception))
            self.assertIn("fold column", str(ctx.exception))

    def test_agent_bridge_act_alias_and_main(self):
        """Verify AgentBridge.act is an exact alias for execute_action, and harness.agent_bridge.main exists."""
        from harness import agent_bridge

        self.assertTrue(callable(agent_bridge.main))
        self.assertIs(AgentBridge.act, AgentBridge.execute_action)

    def test_json_serialization_robustness(self):
        """Verify _json_serial correctly handles paths, datetimes, numpy scalars, arrays, and objects."""
        now = datetime.now()
        path = Path("configs/task.yaml")
        np_int = np.int64(42)
        np_float = np.float32(3.14)
        np_arr = np.array([1, 2, 3])

        class CustomObj:
            def __init__(self):
                self.val = "hello"

        payload = {
            "time": now,
            "path": path,
            "int_val": np_int,
            "float_val": np_float,
            "arr": np_arr,
            "custom": CustomObj(),
        }

        # Must serialize cleanly without TypeError
        dumped = json.dumps(payload, default=_json_serial)
        parsed = json.loads(dumped)
        self.assertEqual(parsed["path"], str(path))
        self.assertEqual(parsed["int_val"], 42)
        self.assertAlmostEqual(parsed["float_val"], 3.14, places=2)
        self.assertEqual(parsed["arr"], [1, 2, 3])
        self.assertEqual(parsed["custom"]["val"], "hello")


if __name__ == "__main__":
    unittest.main()
