"""End-to-End Pipeline, Serialization, GBDT Wrappers, and AIDE Tree Search Integration Tests."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd
import yaml

from harness.agent_bridge import AgentBridge
from harness.io_utils import atomic_write_json, json_default_serializer, sanitize_for_json
from harness.models import (
    GenericBaselineModel,
    LightGBMModel,
    XGBoostModel,
    create_model,
    HAS_LIGHTGBM,
    HAS_XGBOOST,
)
from harness.runner import UniversalRunner
from harness.task_spec import UniversalTaskSpec
from harness.tree_journal import Node, TreeJournal
from tools.bootstrap_task import bootstrap_competition_task


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


if __name__ == "__main__":
    unittest.main()
