import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import yaml
import zipfile

from harness.runner import UniversalRunner
from harness.task_spec import UniversalTaskSpec
from harness.plugins import TaskPluginContext, DeclarativeTaskPlugin, load_task_plugin
from harness.artifacts import validate_run, validate_test_prediction, sha256
from harness.candidates import build_candidate, validate_candidate
from harness.submission_manager import SubmissionManager
from harness.agent_bridge import AgentBridge, TypedActionName


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


if __name__ == "__main__":
    unittest.main()
