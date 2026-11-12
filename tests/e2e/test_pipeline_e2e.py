from __future__ import annotations
# Consolidated Pipeline E2E Test Suite
# Tier: e2e
import json
import shutil
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# --- Imports from tests/test_boost_end_to_end.py ---
import zipfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import yaml
from harness.agent_bridge import AgentBridge, TypedActionName
from harness.artifacts import sha256, validate_test_prediction
from harness.candidates import build_candidate, validate_candidate
from harness.plugins import DeclarativeTaskPlugin, TaskPluginContext
from harness.runner import UniversalRunner
from harness.submission_manager import SubmissionManager
from harness.task_spec import UniversalTaskSpec
# --- Body from tests/test_boost_end_to_end.py ---




def make_task_spec_dict(has_submission_spec=True, has_test_path=True, archive_format="csv"):
    task_cfg = {
        "id": "test_task",
        "name": "Test Benchmark",
        "task_type": "regression",
        "primary_metric": "rmse",
        "direction": "minimize",
        "inputs": [
            {"name": "tabular", "type": "tabular", "source_columns": ["feat1", "feat2"]},
            {"name": "var1", "type": "tabular", "source_columns": ["feat1"]},
            {"name": "var2", "type": "tabular", "source_columns": ["feat2"]},
        ],
        "subtasks": [
            {
                "id": "target",
                "name": "Target Reg",
                "task_type": "regression",
                "column": "target",
                "metric": "rmse",
                "direction": "minimize",
                "weight": 1.0,
            }
        ],
        "validation": {
            "strategy": "kfold",
            "num_folds": 2,
        },
        "data_config": {
            "train_path": "dataset/train.csv",
            "id_column": "id",
            "target_column": "target",
        },
    }
    if has_test_path:
        task_cfg["data_config"]["test_path"] = "dataset/test.csv"
    if has_submission_spec:
        task_cfg["submission_spec"] = {
            "file_name": "submission.zip" if archive_format == "zip" else "submission.csv",
            "archive_format": archive_format,
            "required_columns": ["id", "target"],
        }
    return {"task": task_cfg}


class TestBoostEndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "configs").mkdir(parents=True)
        (self.root / "dataset").mkdir(parents=True)
        (self.root / ".ai-harness").mkdir(parents=True)

        # Train data (20 samples, 2 folds)
        np.random.seed(42)
        f1 = np.random.randn(20) * 5.0
        f2 = np.random.randn(20) * 5.0
        target = 2.0 * f1 + 2.0 * f2
        train_df = pd.DataFrame({
            "id": [f"train_{i}" for i in range(20)],
            "feat1": f1,
            "feat2": f2,
            "target": target,
            "part": [i % 2 for i in range(20)],
        })
        train_df.to_csv(self.root / "dataset/train.csv", index=False)

        # Test data (4 samples)
        f1_test = np.array([10.0, 11.0, 12.0, 13.0])
        f2_test = np.array([20.0, 22.0, 24.0, 26.0])
        self.test_df = pd.DataFrame({
            "id": [f"test_{i}" for i in range(4)],
            "feat1": f1_test,
            "feat2": f2_test,
            "part": [0, 1, 0, 1],
        })
        self.test_df.to_csv(self.root / "dataset/test.csv", index=False)

        task_yaml = make_task_spec_dict()
        (self.root / "configs/task.yaml").write_text(yaml.safe_dump(task_yaml), encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def test_item1_test_set_inference_and_manifest(self):
        """Hạng mục 1: Test Set Inference multi-fold K-Fold generates test_predictions.csv with manifest."""
        args = SimpleNamespace(
            project_root=str(self.root),
            task_config=str(self.root / "configs/task.yaml"),
            fold=None,
            debug=False,
            probe_samples=None,
            job_id=None,
        )
        cfg = {
            "experiment": {
                "id": "exp001",
                "branch": "baseline",
                "parent_id": "none",
                "problem_addressed": "Baseline anchor exploration for regression task",
                "hypothesis": "Test inference works across folds with target reduction of 0.05",
                "intervention": "Linear regression",
            },
            "model": {"name": "baseline"},
            "data": {
                "input_variants": ["tabular"],
                "selection_variant": "tabular",
            },
            "validation": {
                "strategy": "kfold",
                "num_folds": 2,
            },
        }

        runner = UniversalRunner(cfg, "config.yaml", args)
        runner.run()

        # Check test_predictions.csv exists
        test_pred_file = runner.exp_dir / "test_predictions.csv"
        self.assertTrue(test_pred_file.is_file())
        test_preds = pd.read_csv(test_pred_file)
        self.assertEqual(len(test_preds), len(self.test_df))
        self.assertEqual(list(test_preds["id"]), list(self.test_df["id"]))
        self.assertIn("pred_target", test_preds.columns)
        self.assertIn("target", test_preds.columns)

        # Check manifest
        manifest = json.loads((runner.exp_dir / "artifact_manifest.json").read_text(encoding="utf-8"))
        self.assertIn("prediction_splits", manifest)
        self.assertEqual(manifest["prediction_splits"].get("test_predictions.csv"), "test")
        self.assertIn("test_predictions.csv", manifest["files"])
        self.assertEqual(manifest["files"]["test_predictions.csv"], sha256(test_pred_file))

        # Check validate_test_prediction passes
        validated = validate_test_prediction(test_pred_file, task_spec=runner.task_spec, plugin=runner.task_plugin)
        self.assertEqual(validated["exp_id"], "exp001")

    def test_item1_partitioned_test_inference(self):
        """Hạng mục 1: Partitioned test inference generates full test predictions when partition_column is used."""
        args = SimpleNamespace(
            project_root=str(self.root),
            task_config=str(self.root / "configs/task.yaml"),
            fold=None,
            debug=False,
            probe_samples=None,
            job_id=None,
        )
        cfg = {
            "experiment": {
                "id": "exp_part",
                "branch": "baseline",
                "parent_id": "none",
                "problem_addressed": "Partitioned model evaluation across demographic slices",
                "hypothesis": "Partitioned models generate valid test predictions with target delta 0.05",
                "intervention": "Partitioned linear regression",
            },
            "model": {"name": "baseline"},
            "data": {
                "input_variants": ["tabular"],
                "selection_variant": "tabular",
                "partition_column": "part",
            },
            "validation": {
                "strategy": "kfold",
                "num_folds": 2,
            },
        }
        runner = UniversalRunner(cfg, "config.yaml", args)
        runner.run()

        test_pred_file = runner.exp_dir / "test_predictions.csv"
        self.assertTrue(test_pred_file.is_file())
        test_preds = pd.read_csv(test_pred_file)
        self.assertEqual(len(test_preds), len(self.test_df))
        self.assertEqual(list(test_preds["id"]), list(self.test_df["id"]))
        self.assertFalse(test_preds["pred_target"].isna().any())

    def test_item2_declarative_task_plugin_submission(self):
        """Hạng mục 2: Default Submission Builder & Validator for DeclarativeTaskPlugin."""
        # 1. Check capabilities when submission_spec is present
        spec = UniversalTaskSpec.from_yaml(self.root / "configs/task.yaml")
        ctx = TaskPluginContext(
            self.root,
            self.root / "configs/task.yaml",
            spec,
            "default",
        )
        plugin = DeclarativeTaskPlugin(ctx)
        caps = plugin.capabilities()
        self.assertTrue(caps["submission_builder"])
        self.assertTrue(caps["submission_validator"])

        # 2. Check capabilities when submission_spec is NOT present
        no_sub_yaml = make_task_spec_dict(has_submission_spec=False)
        (self.root / "configs/task_no_sub.yaml").write_text(yaml.safe_dump(no_sub_yaml), encoding="utf-8")
        spec_no_sub = UniversalTaskSpec.from_yaml(self.root / "configs/task_no_sub.yaml")
        ctx_no_sub = TaskPluginContext(
            self.root,
            self.root / "configs/task_no_sub.yaml",
            spec_no_sub,
            "default",
        )
        plugin_no_sub = DeclarativeTaskPlugin(ctx_no_sub)
        caps_no_sub = plugin_no_sub.capabilities()
        self.assertFalse(caps_no_sub["submission_builder"])
        self.assertFalse(caps_no_sub["submission_validator"])

        # 3. Test build_submission and validate_submission
        pred_df = pd.DataFrame({
            "id": ["test_0", "test_1", "test_2", "test_3"],
            "pred_target": [15.0, 16.5, 18.0, 19.5],
            "target": [15.0, 16.5, 18.0, 19.5],
        })
        out_path = self.root / "submissions/sub_test/submission.csv"
        built = plugin.build_submission(ctx, pred_df, out_path)
        self.assertTrue(built.is_file())
        sub_df = pd.read_csv(built)
        self.assertEqual(list(sub_df.columns), ["id", "target"])
        self.assertEqual(len(sub_df), 4)

        # Validate valid submission
        plugin.validate_submission(ctx, built)

        # Validate rejects invalid submission (null or duplicate IDs)
        bad_df = pd.DataFrame({"id": ["test_0", "test_0", "test_2", "test_3"], "target": [1.0, 2.0, 3.0, 4.0]})
        bad_path = self.root / "submissions/sub_bad/submission.csv"
        bad_path.parent.mkdir(parents=True, exist_ok=True)
        bad_df.to_csv(bad_path, index=False)
        with self.assertRaises(ValueError):
            plugin.validate_submission(ctx, bad_path)

    def test_item2_zip_archive_submission(self):
        """Hạng mục 2: DeclarativeTaskPlugin supports zip archive packaging when declared in submission_spec."""
        zip_task_yaml = make_task_spec_dict(has_submission_spec=True, archive_format="zip")
        (self.root / "configs/task_zip.yaml").write_text(yaml.safe_dump(zip_task_yaml), encoding="utf-8")
        spec = UniversalTaskSpec.from_yaml(self.root / "configs/task_zip.yaml")
        ctx = TaskPluginContext(
            self.root,
            self.root / "configs/task_zip.yaml",
            spec,
            "default",
        )
        plugin = DeclarativeTaskPlugin(ctx)
        pred_df = pd.DataFrame({
            "id": ["test_0", "test_1", "test_2", "test_3"],
            "pred_target": [15.0, 16.5, 18.0, 19.5],
        })
        out_zip = self.root / "submissions/sub_zip/submission.zip"
        built = plugin.build_submission(ctx, pred_df, out_zip)
        self.assertTrue(built.is_file())
        self.assertTrue(zipfile.is_zipfile(built))

        # Validate zipped submission passes
        plugin.validate_submission(ctx, built)

    def test_item2_agent_bridge_prepare_submission(self):
        """Hạng mục 2: prepare_submission unlocked in AgentBridge for default DeclarativeTaskPlugin."""
        args = SimpleNamespace(
            project_root=str(self.root),
            task_config=str(self.root / "configs/task.yaml"),
            fold=None,
            debug=False,
            probe_samples=None,
            job_id=None,
        )
        cfg = {
            "experiment": {
                "id": "exp001",
                "branch": "baseline",
                "parent_id": "none",
                "problem_addressed": "Baseline anchor exploration for regression task",
                "hypothesis": "Test inference works across folds with target reduction of 0.05",
                "intervention": "Linear regression",
            },
            "model": {"name": "baseline"},
            "data": {
                "input_variants": ["tabular"],
                "selection_variant": "tabular",
            },
            "validation": {
                "strategy": "kfold",
                "num_folds": 2,
            },
        }
        runner = UniversalRunner(cfg, "config.yaml", args)
        runner.run()

        cand_csv = str((runner.exp_dir / "test_predictions.csv").relative_to(self.root))
        bridge = AgentBridge(project_root=self.root)
        _, obs = bridge.execute_action(
            TypedActionName.PREPARE_SUBMISSION.value,
            params={
                "exp_id": "exp001",
                "candidate_csv": cand_csv,
            },
            user_approved=True,
        )
        self.assertEqual(obs.status, "SUCCESS")
        self.assertIn("submission.csv", obs.artifact_path)
        self.assertTrue(Path(obs.artifact_path).is_file())

    def test_item2_candidate_csv_omitted_auto_resolution(self):
        """Hạng mục 2: prepare_submission and SubmissionManager auto-resolve test_predictions.csv when candidate_csv is omitted."""
        args = SimpleNamespace(
            project_root=str(self.root),
            task_config=str(self.root / "configs/task.yaml"),
            fold=None,
            debug=False,
            probe_samples=None,
            job_id=None,
        )
        cfg = {
            "experiment": {
                "id": "exp001",
                "branch": "baseline",
                "parent_id": "none",
                "problem_addressed": "Baseline anchor exploration for regression task",
                "hypothesis": "Test inference works across folds with target reduction of 0.05",
                "intervention": "Linear regression",
            },
            "model": {"name": "baseline"},
            "data": {"input_variants": ["tabular"], "selection_variant": "tabular"},
            "validation": {"strategy": "kfold", "num_folds": 2},
        }
        runner = UniversalRunner(cfg, "config.yaml", args)
        runner.run()

        # 1. Test AgentBridge without candidate_csv param
        bridge = AgentBridge(project_root=self.root)
        _, obs = bridge.execute_action(
            TypedActionName.PREPARE_SUBMISSION.value,
            params={"exp_id": "exp001"},
            user_approved=True,
        )
        self.assertEqual(obs.status, "SUCCESS")
        self.assertTrue(Path(obs.artifact_path).is_file())

        # 2. Test SubmissionManager.register_submission without candidate_csv param
        mgr = SubmissionManager(project_root=self.root)
        entry = mgr.register_submission(exp_id="exp001")
        self.assertEqual(entry.status, "LOCAL_VERIFIED")
        self.assertTrue(Path(entry.archive_path).is_file())

    def test_item3_test_set_ensembling_in_candidates(self):
        """Hạng mục 3: Test Set Ensembling in build_candidate across constituents (single model and router)."""
        args = SimpleNamespace(
            project_root=str(self.root),
            task_config=str(self.root / "configs/task.yaml"),
            fold=None,
            debug=False,
            probe_samples=None,
            job_id=None,
        )
        # Create exp001
        cfg1 = {
            "experiment": {
                "id": "exp001",
                "branch": "baseline",
                "parent_id": "none",
                "problem_addressed": "Establish official baseline anchor for test suite",
                "hypothesis": "Baseline linear estimator establishes initial target score benchmark",
                "intervention": "Ridge alpha 1",
            },
            "model": {"name": "baseline"},
            "data": {"input_variants": ["tabular"], "selection_variant": "tabular"},
            "validation": {"strategy": "kfold", "num_folds": 2},
        }
        runner1 = UniversalRunner(cfg1, "cfg1.yaml", args)
        runner1.run()

        # Build single_model candidate
        cand_single = build_candidate(
            runner1.project,
            name="cand_single",
            mode="single_model",
            source_map={"source_exp": "exp001"},
        )
        self.assertTrue(cand_single["eligible"])
        cand_single_folder = runner1.project.artifacts_dir / cand_single["id"]
        self.assertTrue((cand_single_folder / "test_predictions.csv").is_file())
        self.assertEqual(cand_single.get("prediction_splits", {}).get("test_predictions.csv"), "test")
        validate_candidate(runner1.project, cand_single)

        # Validate test prediction on cand_single
        val_cand = validate_test_prediction(
            cand_single_folder / "test_predictions.csv",
            task_spec=runner1.task_spec,
            plugin=runner1.task_plugin,
        )
        self.assertEqual(val_cand["id"], cand_single["id"])

        # Build specialist_router candidate
        cand_route = build_candidate(
            runner1.project,
            name="cand_route",
            mode="specialist_router",
            source_map={"routing": {"target": "exp001"}},
        )
        self.assertTrue(cand_route["eligible"])
        cand_route_folder = runner1.project.artifacts_dir / cand_route["id"]
        self.assertTrue((cand_route_folder / "test_predictions.csv").is_file())
        self.assertEqual(cand_route.get("prediction_splits", {}).get("test_predictions.csv"), "test")
        validate_candidate(runner1.project, cand_route)

    def test_item3_weighted_blend_ensembling(self):
        """Hạng mục 3: build_candidate with weighted_blend correctly blends test predictions and passes guards."""
        args = SimpleNamespace(
            project_root=str(self.root),
            task_config=str(self.root / "configs/task.yaml"),
            fold=None,
            debug=False,
            probe_samples=None,
            job_id=None,
        )
        # Model 1 on feat1 (var1)
        cfg1 = {
            "experiment": {
                "id": "exp001",
                "branch": "baseline",
                "parent_id": "none",
                "problem_addressed": "Establish official baseline anchor for test suite",
                "hypothesis": "Baseline linear estimator establishes initial target score benchmark",
                "intervention": "Ridge alpha 1 on var1",
            },
            "model": {"name": "baseline"},
            "data": {"input_variants": ["var1"], "selection_variant": "var1"},
            "validation": {"strategy": "kfold", "num_folds": 2},
        }
        runner1 = UniversalRunner(cfg1, "cfg1.yaml", args)
        runner1.run()

        # Model 2 on feat2 (var2)
        cfg2 = {
            "experiment": {
                "id": "exp002",
                "branch": "baseline",
                "parent_id": "none",
                "problem_addressed": "Second baseline model exploration for test suite",
                "hypothesis": "Second linear estimator provides complementary features and targets",
                "intervention": "Ridge alpha 2 on var2",
            },
            "model": {"name": "baseline"},
            "data": {"input_variants": ["var2"], "selection_variant": "var2"},
            "validation": {"strategy": "kfold", "num_folds": 2},
        }
        runner2 = UniversalRunner(cfg2, "cfg2.yaml", args)
        runner2.run()

        # Synthesize real weighted_blend candidate (weights 0.5 and 0.5)
        cand_blend = build_candidate(
            runner1.project,
            name="cand_blend",
            mode="weighted_blend",
            source_map={"components": ["exp001", "exp002"], "weights": [0.5, 0.5]},
        )
        self.assertTrue(cand_blend["eligible"])
        cand_folder = runner1.project.artifacts_dir / cand_blend["id"]
        self.assertTrue((cand_folder / "test_predictions.csv").is_file())
        self.assertEqual(cand_blend.get("prediction_splits", {}).get("test_predictions.csv"), "test")
        validate_candidate(runner1.project, cand_blend)

        # Check blended predictions math
        t1 = pd.read_csv(runner1.exp_dir / "test_predictions.csv")
        t2 = pd.read_csv(runner2.exp_dir / "test_predictions.csv")
        tb = pd.read_csv(cand_folder / "test_predictions.csv")
        expected_pred = 0.5 * t1["pred_target"] + 0.5 * t2["pred_target"]
        np.testing.assert_allclose(tb["pred_target"].values, expected_pred.values, rtol=1e-5)

    def test_item3_probability_blend_classification_ensembling(self):
        """Hạng mục 3: build_candidate with probability_blend correctly blends classification probabilities on test set."""
        clf_root = self.root / "clf_task"
        (clf_root / "configs").mkdir(parents=True)
        (clf_root / "dataset").mkdir(parents=True)
        (clf_root / ".ai-harness").mkdir(parents=True)

        np.random.seed(42)
        f1 = np.random.randn(30)
        f2 = np.random.randn(30)
        y = ((f1 + f2) > 0).astype(int)
        train_df = pd.DataFrame({"id": [f"t_{i}" for i in range(30)], "feat1": f1, "feat2": f2, "target": y})
        train_df.to_csv(clf_root / "dataset/train.csv", index=False)
        test_df = pd.DataFrame({"id": [f"te_{i}" for i in range(4)], "feat1": [1.0, -1.0, 2.0, -2.0], "feat2": [-1.0, 1.0, 2.0, -2.0]})
        test_df.to_csv(clf_root / "dataset/test.csv", index=False)

        tcfg = {
            "id": "task_clf", "name": "TClf", "task_type": "binary_classification", "primary_metric": "macro_f1", "direction": "maximize",
            "inputs": [
                {"name": "var1", "type": "tabular", "source_columns": ["feat1"]},
                {"name": "var2", "type": "tabular", "source_columns": ["feat2"]},
            ],
            "subtasks": [{"id": "target", "name": "T", "task_type": "binary_classification", "column": "target", "metric": "macro_f1", "direction": "maximize", "classes": [0, 1], "weight": 1.0}],
            "validation": {"strategy": "kfold", "num_folds": 2},
            "data_config": {"train_path": "dataset/train.csv", "test_path": "dataset/test.csv", "id_column": "id", "target_column": "target"},
            "submission_spec": {"file_name": "submission.csv", "archive_format": "csv", "required_columns": ["id", "target"]},
        }
        (clf_root / "configs/task.yaml").write_text(yaml.safe_dump({"task": tcfg}), encoding="utf-8")
        args = SimpleNamespace(project_root=str(clf_root), task_config=str(clf_root / "configs/task.yaml"), fold=None, debug=False, probe_samples=None, job_id=None)

        cfg1 = {
            "experiment": {"id": "exp001", "branch": "baseline", "parent_id": "none", "problem_addressed": "Establish baseline anchor for classification task", "hypothesis": "Baseline logistic regression establishes initial macro_f1 benchmark", "intervention": "LogisticRegression var1"},
            "model": {"name": "baseline"}, "data": {"input_variants": ["var1"], "selection_variant": "var1"}, "validation": {"strategy": "kfold", "num_folds": 2},
        }
        r1 = UniversalRunner(cfg1, "c1.yaml", args)
        r1.run()

        cfg2 = {
            "experiment": {"id": "exp002", "branch": "baseline", "parent_id": "none", "problem_addressed": "Second baseline model for classification task", "hypothesis": "Second logistic regression provides complementary signal", "intervention": "LogisticRegression var2"},
            "model": {"name": "baseline"}, "data": {"input_variants": ["var2"], "selection_variant": "var2"}, "validation": {"strategy": "kfold", "num_folds": 2},
        }
        r2 = UniversalRunner(cfg2, "c2.yaml", args)
        r2.run()

        cand = build_candidate(r1.project, name="prob_blend", mode="probability_blend", source_map={"components": ["exp001", "exp002"], "weights": [0.5, 0.5]})
        validate_candidate(r1.project, cand)
        cand_dir = r1.project.artifacts_dir / cand["id"]
        test_pred_file = cand_dir / "test_predictions.csv"
        self.assertTrue(test_pred_file.is_file())
        test_p = pd.read_csv(test_pred_file)
        self.assertEqual(len(test_p), 4)
        self.assertIn("proba_target", test_p.columns)
        self.assertIn("classes_target", test_p.columns)
        self.assertIn("pred_target", test_p.columns)

        val_cand = validate_test_prediction(test_pred_file, task_spec=r1.task_spec, plugin=r1.task_plugin)
        self.assertEqual(val_cand["id"], cand["id"])


# --- Imports from tests/test_e2e_pipeline.py ---
import unittest
from pathlib import Path
from harness.io_utils import atomic_write_json
from harness.models import (
    HAS_LIGHTGBM,
    HAS_XGBOOST,
    LightGBMModel,
    XGBoostModel,
    create_model,
)
from harness.tree_journal import Node, TreeJournal
from tools.bootstrap_task import bootstrap_competition_task
# --- Body from tests/test_e2e_pipeline.py ---
"""End-to-End Pipeline, Serialization, GBDT Wrappers, and AIDE Tree Search Integration Tests."""






class TestJSONSerializationAndTypeSafety(unittest.TestCase):
    """Verifies R1: atomic_write_json gracefully serializes NumPy types, Paths, and non-string keys."""

    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp(prefix="test_json_ser_"))

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_atomic_write_json_with_numpy_and_paths(self):
        """Verify atomic_write_json handles all non-standard types without TypeError."""
        target_file = self.tmp_dir / "test_types.json"
        payload = {
            "int64_val": np.int64(42),
            "int32_val": np.int32(10),
            "float32_val": np.float32(3.1415),
            "float64_val": np.float64(2.71828),
            "bool_val": np.bool_(True),
            "1d_array": np.array([1, 2, 3]),
            "2d_array": np.array([[1.0, 2.0], [3.0, 4.0]]),
            "path_val": Path("data/splits/fold_0.tsv"),
            "nested_list": [np.int64(1), np.float32(2.5), Path("foo/bar")],
            "nested_dict": {
                "key_a": np.int64(99),
                "key_b": np.bool_(False),
            },
        }

        atomic_write_json(target_file, payload)
        self.assertTrue(target_file.exists())

        loaded = json.loads(target_file.read_text(encoding="utf-8"))
        self.assertEqual(loaded["int64_val"], 42)
        self.assertEqual(loaded["int32_val"], 10)
        self.assertAlmostEqual(loaded["float32_val"], 3.1415, places=3)
        self.assertAlmostEqual(loaded["float64_val"], 2.71828, places=4)
        self.assertIs(loaded["bool_val"], True)
        self.assertEqual(loaded["1d_array"], [1, 2, 3])
        self.assertEqual(loaded["2d_array"], [[1.0, 2.0], [3.0, 4.0]])
        self.assertIn("fold_0.tsv", loaded["path_val"])

    def test_atomic_write_json_with_non_string_keys(self):
        """Verify atomic_write_json sanitizes dictionaries with NumPy scalar or Path keys."""
        target_file = self.tmp_dir / "test_keys.json"
        payload = {
            np.int64(0): "fold_0_metric",
            np.int64(1): "fold_1_metric",
            Path("path/to/file"): "path_value",
        }

        atomic_write_json(target_file, payload)
        self.assertTrue(target_file.exists())

        loaded = json.loads(target_file.read_text(encoding="utf-8"))
        self.assertEqual(loaded["0"], "fold_0_metric")
        self.assertEqual(loaded["1"], "fold_1_metric")
        self.assertIn("path_value", loaded.values())


class TestGBDTModelsAndTreeSearch(unittest.TestCase):
    """Verifies R3: LightGBM and XGBoost wrappers and AIDE tree search multi-family classification."""

    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp(prefix="test_gbdt_"))
        self.task_cfg_file = self.tmp_dir / "task.yaml"
        task_data = {
            "task": {
                "name": "GBDT Unit Benchmark",
                "platform": "kaggle",
                "task_type": "binary_classification",
                "modality": "tabular",
                "primary_metric": "roc_auc",
                "direction": "maximize",
                "data_config": {
                    "train_path": "dataset/train.csv",
                    "id_column": "id",
                    "target_column": "target",
                },
                "inputs": [
                    {
                        "name": "features",
                        "type": "tabular",
                        "source_columns": ["feature_num", "feature_cat"],
                    }
                ],
                "subtasks": [
                    {
                        "id": "target",
                        "name": "target",
                        "task_type": "binary_classification",
                        "column": "target",
                        "primary_metric": "roc_auc",
                        "direction": "maximize",
                        "classes": ["0", "1"],
                    }
                ],
                "validation": {
                    "strategy": "kfold",
                    "num_folds": 2,
                    "split_file": "splits/fold_assignments.tsv",
                },
            }
        }
        with open(self.task_cfg_file, "w", encoding="utf-8") as f:
            yaml.safe_dump(task_data, f)
        self.task_spec = UniversalTaskSpec.from_yaml(self.task_cfg_file)

        self.df = pd.DataFrame({
            "id": [f"id_{i}" for i in range(40)],
            "feature_num": [float(i) * 0.5 for i in range(40)],
            "feature_cat": ["A" if i % 2 == 0 else "B" for i in range(40)],
            "target": [i % 2 for i in range(40)],
        })

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    @unittest.skipUnless(HAS_LIGHTGBM, "LightGBM not installed")
    def test_lightgbm_classifier_fit_predict(self):
        """Verify LightGBM classifier fits and outputs labels and probabilities."""
        model = create_model(self.task_spec, "lightgbm", n_estimators=10)
        self.assertIsInstance(model, LightGBMModel)
        model.fit(self.df, variant="features")
        pred = model.predict(self.df, variant="features")
        self.assertIn("target", pred)
        self.assertIn("proba_target", pred)
        self.assertIn("classes_target", pred)
        self.assertEqual(len(pred["target"]), len(self.df))
        self.assertGreater(model.num_parameters(), 0)

    @unittest.skipUnless(HAS_XGBOOST, "XGBoost not installed")
    def test_xgboost_classifier_fit_predict_with_string_labels(self):
        """Verify XGBoost classifier handles string labels seamlessly via LabelEncoder."""
        df_str = self.df.copy()
        df_str["target"] = ["class_0" if x == 0 else "class_1" for x in df_str["target"]]
        str_spec_data = yaml.safe_load(self.task_cfg_file.read_text(encoding="utf-8"))
        str_spec_data["task"]["subtasks"][0]["classes"] = ["class_0", "class_1"]
        str_cfg = self.tmp_dir / "str_task.yaml"
        with open(str_cfg, "w", encoding="utf-8") as f:
            yaml.safe_dump(str_spec_data, f)
        spec = UniversalTaskSpec.from_yaml(str_cfg)

        model = create_model(spec, "xgboost", n_estimators=10)
        self.assertIsInstance(model, XGBoostModel)
        model.fit(df_str, variant="features")
        pred = model.predict(df_str, variant="features")
        self.assertIn("target", pred)
        self.assertIn("proba_target", pred)
        self.assertIn("classes_target", pred)
        self.assertIn(pred["target"][0], ["class_0", "class_1"])
        self.assertGreater(model.num_parameters(), 0)

    def test_tree_journal_multi_family_unlocks_ensemble(self):
        """Verify TreeJournal classifies GBDT distinct from sparse_tfidf/linear, unlocking 'ens'."""
        journal_path = self.tmp_dir / "journal.json"
        journal = TreeJournal(storage_path=str(journal_path))

        # Node 1: Linear baseline
        node1 = Node(
            id="exp000_baseline",
            branch="baseline",
            model_family="linear",
            metrics={"overall_composite": 0.70, "model": "logistic_regression"},
            verified=True,
            eligible=True,
        )
        journal.append(node1)
        self.assertEqual(journal.count_distinct_model_families(), 1)
        rec1 = journal.recommend_next_branch()
        self.assertNotEqual(rec1["recommended_branch"], "ens")

        # Node 2: LightGBM (GBDT family)
        node2 = Node(
            id="exp001_lightgbm",
            branch="model",
            model_family="gbdt",
            metrics={"overall_composite": 0.75, "model": "lightgbm"},
            verified=True,
            eligible=True,
        )
        journal.append(node2)
        # 2 distinct families: sparse_tfidf/linear and gbdt
        self.assertEqual(journal.count_distinct_model_families(), 2)

        # Node 3: XGBoost (also GBDT family)
        node3 = Node(
            id="exp002_xgboost",
            branch="model",
            model_family="gbdt",
            metrics={"overall_composite": 0.76, "model": "xgboost"},
            verified=True,
            eligible=True,
        )
        journal.append(node3)
        # Still 2 distinct families because LightGBM and XGBoost both belong to GBDT
        self.assertEqual(journal.count_distinct_model_families(), 2)

        # Node 4: Vision model
        node4 = Node(
            id="exp003_convnext",
            branch="model",
            model_family="vision",
            metrics={"overall_composite": 0.81, "model": "convnext_small"},
            verified=True,
            eligible=True,
        )
        journal.append(node4)
        self.assertEqual(journal.count_distinct_model_families(), 3)

        # Node 5: Audio model
        node5 = Node(
            id="exp004_whisper",
            branch="model",
            model_family="audio",
            metrics={"overall_composite": 0.83, "model": "whisper-small"},
            verified=True,
            eligible=True,
        )
        journal.append(node5)
        self.assertEqual(journal.count_distinct_model_families(), 4)

    @unittest.skipUnless(HAS_LIGHTGBM, "LightGBM not installed")
    def test_lightgbm_custom_feature_transformer_passthrough(self):
        """Verify LightGBM runs with transform='passthrough' on agent-preprocessed features."""
        df_cat = self.df.copy()
        df_cat["feature_cat"] = df_cat["feature_cat"].astype("category")
        model = create_model(self.task_spec, "lightgbm", n_estimators=10, transform="passthrough")
        model.fit(df_cat, variant="features")
        pred = model.predict(df_cat, variant="features")
        self.assertIn("target", pred)
        self.assertEqual(len(pred["target"]), len(self.df))

    @unittest.skipUnless(HAS_XGBOOST, "XGBoost not installed")
    def test_xgboost_custom_feature_transformer_passthrough(self):
        """Verify XGBoost runs with feature_transformer='passthrough' on agent-preprocessed features."""
        df_cat = self.df.copy()
        df_cat["feature_cat"] = df_cat["feature_cat"].astype("category")
        model = create_model(self.task_spec, "xgboost", n_estimators=10, feature_transformer="passthrough")
        model.fit(df_cat, variant="features")
        pred = model.predict(df_cat, variant="features")
        self.assertIn("target", pred)
        self.assertEqual(len(pred["target"]), len(self.df))

    def test_tree_journal_sync_from_artifacts_gbdt(self):
        """Verify TreeJournal.sync_from_artifacts correctly infers 'gbdt' for LightGBM/XGBoost runs."""
        art_dir = self.tmp_dir / "artifacts"
        exp1_dir = art_dir / "exp000_baseline"
        exp1_dir.mkdir(parents=True)
        (exp1_dir / "run_config.yaml").write_text(yaml.safe_dump({"model": {"name": "baseline"}}), encoding="utf-8")
        (exp1_dir / "metrics.json").write_text(
            json.dumps({"model": "baseline", "model_family": "linear", "primary_metric": "roc_auc", "primary_score": 0.70, "overall_composite": 0.70}),
            encoding="utf-8",
        )

        exp2_dir = art_dir / "exp001_lightgbm"
        exp2_dir.mkdir(parents=True)
        (exp2_dir / "run_config.yaml").write_text(yaml.safe_dump({"model": {"name": "lightgbm"}}), encoding="utf-8")
        (exp2_dir / "metrics.json").write_text(
            json.dumps({"model": "lightgbm", "model_family": "gbdt", "primary_metric": "roc_auc", "primary_score": 0.75, "overall_composite": 0.75}),
            encoding="utf-8",
        )

        journal = TreeJournal(storage_path=str(self.tmp_dir / "journal_sync.json"))
        journal.sync_from_artifacts(str(art_dir))
        node_lgb = journal.get_node("exp001_lightgbm")
        self.assertIsNotNone(node_lgb)
        self.assertEqual(node_lgb.model_family, "gbdt")
        self.assertEqual(journal.count_distinct_model_families(), 2)


class TestEndToEndPipelineIntegration(unittest.TestCase):
    """Verifies R4: Full integration across bootstrap -> splits -> runner training -> metric serialization."""

    def setUp(self):
        self.workspace_dir = Path(tempfile.mkdtemp(prefix="test_e2e_ws_"))
        self.dataset_dir = self.workspace_dir / "dataset"
        self.dataset_dir.mkdir(parents=True)
        self.docs_dir = self.workspace_dir / "docs"
        self.docs_dir.mkdir(parents=True)
        self.configs_dir = self.workspace_dir / "configs"
        self.configs_dir.mkdir(parents=True)

        # Create tabular dataset
        df_train = pd.DataFrame({
            "patient_id": [f"PID_{i:03d}" for i in range(60)],
            "site_id": [f"SITE_{i % 3}" for i in range(60)],
            "feature_x": [float(i) * 1.2 for i in range(60)],
            "feature_y": [float(i % 5) for i in range(60)],
            "risk_label": [i % 2 for i in range(60)],
        })
        df_train.to_csv(self.dataset_dir / "train.csv", index=False)

        # Create TASK.md
        (self.docs_dir / "TASK.md").write_text(
            """
# Clinical Risk Benchmark
- **Competition Name**: Clinical Risk Benchmark
- **Platform**: kaggle
- **Modality**: tabular
- **Task Type**: binary_classification
- **Primary ID Column**: patient_id
- **Anti-Leakage Group Column**: site_id
- **Target Columns**: risk_label
- **Primary Metric**: roc_auc
- **Optimization Direction**: maximize
""",
            encoding="utf-8",
        )

    def tearDown(self):
        shutil.rmtree(self.workspace_dir, ignore_errors=True)

    def test_full_bootstrap_runner_and_agent_bridge_lifecycle(self):
        """Execute full cycle: bootstrap -> verify artifacts -> run probe -> inspect metrics."""
        task_cfg_path = self.configs_dir / "task.yaml"

        # 1. Bootstrap Task
        report = bootstrap_competition_task(
            dataset_dir=self.dataset_dir,
            context_file=self.docs_dir / "TASK.md",
            output_config=task_cfg_path,
            project_root=self.workspace_dir,
        )
        self.assertEqual(report["status"], "SUCCESS")
        self.assertTrue(task_cfg_path.is_file())

        # Verify splits exist and contain folds
        splits_file = self.workspace_dir / "splits" / "fold_assignments.tsv"
        self.assertTrue(splits_file.is_file())
        splits_df = pd.read_csv(splits_file, sep="\t")
        self.assertEqual(len(splits_df), 60)
        self.assertTrue(pd.api.types.is_integer_dtype(splits_df["fold"]))

        # Verify baseline config generated
        baseline_cfg = self.configs_dir / "exp000_baseline.yaml"
        self.assertTrue(baseline_cfg.is_file())

        # 2. Run universal runner smoke test (Fold 0)
        from argparse import Namespace
        args = Namespace(
            project_root=str(self.workspace_dir),
            task_config=str(task_cfg_path),
            job_id="test_probe_job_001",
            config=str(baseline_cfg),
            base_config=None,
            fold=0,
            debug=True,
            force=False,
            probe_samples=20,
            backend="local",
            skip_leaderboard=False,
        )

        with open(baseline_cfg, "r", encoding="utf-8") as f:
            cfg_data = yaml.safe_load(f)

        runner = UniversalRunner(cfg=cfg_data, config_path=baseline_cfg, args=args)
        result = runner.run()

        # 3. Verify metrics serialization
        self.assertIsNotNone(result)
        self.assertIn("primary_score", result)
        self.assertIn("fold_metrics", result)

        metrics_file = runner.exp_dir / "metrics.json"
        self.assertTrue(metrics_file.is_file())
        saved_metrics = json.loads(metrics_file.read_text(encoding="utf-8"))

        # Verify fold_metrics contains native int dev_fold without serialization failure
        fold_metrics = saved_metrics.get("fold_metrics", [])
        self.assertTrue(len(fold_metrics) > 0)
        self.assertEqual(fold_metrics[0]["dev_fold"], 0)
        self.assertIsInstance(fold_metrics[0]["dev_fold"], int)

        # 4. Verify artifact manifest
        manifest_file = runner.exp_dir / "artifact_manifest.json"
        self.assertTrue(manifest_file.is_file())
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "COMPLETED")

        # 5. Verify Agent Bridge Context Packet
        bridge = AgentBridge(project_root=self.workspace_dir)
        packet = bridge.get_context_packet()
        self.assertEqual(packet["runtime_owner"], "Antigravity (IDE)")
        self.assertIn("competition_hud", packet)
        self.assertIn("ucb_recommendation", packet)

        self.assertEqual(bridge.task_spec.name, "Clinical Risk Benchmark")
        markdown = bridge.render_context_markdown()
        self.assertIn("ANTIGRAVITY <-> HARNESS CONTEXT PACKET", markdown)

    @unittest.skipUnless(HAS_LIGHTGBM, "LightGBM not installed")
    def test_runner_lightgbm_probe_and_metrics_family(self):
        """Verify UniversalRunner executes LightGBM and sets model_family='gbdt' in metrics.json."""
        task_cfg_path = self.configs_dir / "task.yaml"
        bootstrap_competition_task(
            dataset_dir=self.dataset_dir,
            context_file=self.docs_dir / "TASK.md",
            output_config=task_cfg_path,
            project_root=self.workspace_dir,
        )
        lgb_cfg_path = self.configs_dir / "exp001_lgbm.yaml"
        base_cfg = self.configs_dir / "exp000_baseline.yaml"
        cfg_data = yaml.safe_load(base_cfg.read_text(encoding="utf-8"))
        cfg_data["experiment"]["id"] = "exp001_lgbm"
        cfg_data["model"] = {"name": "lightgbm", "params": {"n_estimators": 5}}
        lgb_cfg_path.write_text(yaml.safe_dump(cfg_data), encoding="utf-8")

        from argparse import Namespace
        args = Namespace(
            project_root=str(self.workspace_dir),
            task_config=str(task_cfg_path),
            job_id="test_lgb_job_001",
            config=str(lgb_cfg_path),
            base_config=None,
            fold=0,
            debug=True,
            force=False,
            probe_samples=20,
            backend="local",
            skip_leaderboard=False,
        )

        runner = UniversalRunner(cfg=cfg_data, config_path=lgb_cfg_path, args=args)
        result = runner.run()
        self.assertIsNotNone(result)
        metrics_file = runner.exp_dir / "metrics.json"
        self.assertTrue(metrics_file.is_file())
        saved = json.loads(metrics_file.read_text(encoding="utf-8"))
        self.assertEqual(saved["model"], "lightgbm")
        self.assertEqual(saved["model_family"], "gbdt")

    @unittest.skipUnless(HAS_XGBOOST, "XGBoost not installed")
    def test_runner_xgboost_probe_and_metrics_family(self):
        """Verify UniversalRunner executes XGBoost and sets model_family='gbdt' in metrics.json."""
        task_cfg_path = self.configs_dir / "task.yaml"
        bootstrap_competition_task(
            dataset_dir=self.dataset_dir,
            context_file=self.docs_dir / "TASK.md",
            output_config=task_cfg_path,
            project_root=self.workspace_dir,
        )
        xgb_cfg_path = self.configs_dir / "exp002_xgb.yaml"
        base_cfg = self.configs_dir / "exp000_baseline.yaml"
        cfg_data = yaml.safe_load(base_cfg.read_text(encoding="utf-8"))
        cfg_data["experiment"]["id"] = "exp002_xgb"
        cfg_data["model"] = {"name": "xgboost", "params": {"n_estimators": 5}}
        xgb_cfg_path.write_text(yaml.safe_dump(cfg_data), encoding="utf-8")

        from argparse import Namespace
        args = Namespace(
            project_root=str(self.workspace_dir),
            task_config=str(task_cfg_path),
            job_id="test_xgb_job_001",
            config=str(xgb_cfg_path),
            base_config=None,
            fold=0,
            debug=True,
            force=False,
            probe_samples=20,
            backend="local",
            skip_leaderboard=False,
        )

        runner = UniversalRunner(cfg=cfg_data, config_path=xgb_cfg_path, args=args)
        result = runner.run()
        self.assertIsNotNone(result)
        metrics_file = runner.exp_dir / "metrics.json"
        self.assertTrue(metrics_file.is_file())
        saved = json.loads(metrics_file.read_text(encoding="utf-8"))
        self.assertEqual(saved["model"], "xgboost")
        self.assertEqual(saved["model_family"], "gbdt")



if __name__ == '__main__':
    unittest.main()
