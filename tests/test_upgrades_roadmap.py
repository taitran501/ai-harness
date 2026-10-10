"""Unit tests for items 4 to 8 of the harness technical upgrade roadmap.

1. Item 4: Extended competition metrics (qwk, ndcg, spearman, mape) and metrics_evaluator priority hook.
2. Item 5: Seamless bootstrap to attach and elimination of SKILL.md collision.
3. Item 6: Modal cloud runtime dependencies (lightgbm>=4.0.0, xgboost>=2.0.0).
4. Item 7: Tabular OOM protection with max_categories=30 in OneHotEncoder and ridge_regression model name.
5. Item 8: Modality-aware create_experiment in AgentBridge.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import types
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

from harness.evaluation import metric_name, evaluate, resolve_evaluation
from harness.task_spec import UniversalTaskSpec
from harness.remote_runtime import CORE_RUNTIME_DISTRIBUTIONS, resolve_remote_runtime
from harness.templates.modal_template import generate_modal_script
from harness.models import GenericBaselineModel, LightGBMModel, XGBoostModel
from harness.project import attach_project, HarnessProject
from harness.agent_bridge import AgentBridge, TypedActionName
from tools.bootstrap_task import bootstrap_competition_task

SOURCE_ROOT = Path(__file__).resolve().parent.parent


class TestExtendedMetricsAndRunnerHook(unittest.TestCase):
    def test_metric_name_aliases(self):
        self.assertEqual(metric_name("qwk"), "qwk")
        self.assertEqual(metric_name("quadratic_weighted_kappa"), "qwk")
        self.assertEqual(metric_name("cohen_kappa_score"), "qwk")
        self.assertEqual(metric_name("cohen_kappa"), "qwk")
        self.assertEqual(metric_name("kappa"), "qwk")
        self.assertEqual(metric_name("ndcg"), "ndcg")
        self.assertEqual(metric_name("ndcg_score"), "ndcg")
        self.assertEqual(metric_name("normalized_discounted_cumulative_gain"), "ndcg")
        self.assertEqual(metric_name("spearman"), "spearman")
        self.assertEqual(metric_name("spearmanr"), "spearman")
        self.assertEqual(metric_name("spearman_rho"), "spearman")
        self.assertEqual(metric_name("mape"), "mape")
        self.assertEqual(metric_name("mean_absolute_percentage_error"), "mape")

    def test_regression_extended_metrics(self):
        spec = UniversalTaskSpec.from_dict({
            "name": "test_reg",
            "task_type": "regression",
            "primary_metric": "mape",
            "direction": "minimize",
            "id_column": "id",
            "subtasks": [{
                "id": "target",
                "task_type": "regression",
                "primary_metric": "mape",
                "direction": "minimize",
                "weight": 1.0,
            }],
        })
        truth = pd.DataFrame({"id": [1, 2, 3, 4], "target": [10.0, 20.0, 30.0, 40.0]})
        preds = {"target": [11.0, 19.0, 31.0, 39.0]}
        res = evaluate(spec, truth, preds)
        self.assertEqual(res["primary_metric"], "mape")
        self.assertIn("mape", res["subtasks"]["target"])
        self.assertIn("spearman", res["subtasks"]["target"])
        self.assertIn("ndcg", res["subtasks"]["target"])
        self.assertIn("qwk", res["subtasks"]["target"])
        self.assertLess(res["primary_score"], 0.2)

    def test_classification_qwk_metric(self):
        spec = UniversalTaskSpec.from_dict({
            "name": "test_clf",
            "task_type": "multiclass_classification",
            "primary_metric": "qwk",
            "direction": "maximize",
            "id_column": "id",
            "subtasks": [{
                "id": "target",
                "task_type": "multiclass_classification",
                "primary_metric": "qwk",
                "classes": ["0", "1", "2"],
                "direction": "maximize",
                "weight": 1.0,
            }],
        })
        truth = pd.DataFrame({"id": [1, 2, 3, 4, 5], "target": ["0", "1", "2", "1", "0"]})
        preds = {"target": ["0", "1", "2", "1", "0"]}
        res = evaluate(spec, truth, preds)
        self.assertAlmostEqual(res["primary_score"], 1.0)
        self.assertIn("qwk", res["subtasks"]["target"])

    def test_runner_metrics_evaluator_priority(self):
        from harness.runner import UniversalRunner
        tmp = tempfile.TemporaryDirectory()
        root = Path(tmp.name)
        configs = root / "configs"
        configs.mkdir(parents=True)
        (root / "dataset").mkdir(parents=True)
        (root / "dataset" / "train.csv").write_text("id,val,target\n1,2,1.0\n2,3,2.0\n", encoding="utf-8")

        # Create custom scorer module
        src_dir = root / "custom_scoring"
        src_dir.mkdir(parents=True)
        (src_dir / "__init__.py").write_text("", encoding="utf-8")
        (src_dir / "scorer.py").write_text(
            "def score(truth, preds):\n"
            "    return {'custom_competition_metric': 0.987, 'primary_score': 0.987}\n",
            encoding="utf-8",
        )

        task_yaml = configs / "task.yaml"
        task_yaml.write_text(
            yaml.safe_dump({
                "task": {
                    "name": "custom_metric_task",
                    "task_type": "regression",
                    "primary_metric": "custom_competition_metric",
                    "direction": "maximize",
                    "id_column": "id",
                    "inputs": [{"name": "tabular", "type": "tabular", "source_columns": ["val"]}],
                    "subtasks": [{
                        "id": "target",
                        "task_type": "regression",
                        "primary_metric": "custom_competition_metric",
                        "direction": "maximize",
                        "weight": 1.0,
                    }],
                    "data_config": {"train_path": "dataset/train.csv"},
                }
            }),
            encoding="utf-8",
        )

        cfg_dict = {
            "experiment": {
                "id": "exp001_custom",
                "branch": "feat",
                "parent_id": "none",
                "problem_addressed": "Test custom metrics evaluator hook.",
                "hypothesis": "Custom metric hook evaluates before fallback to task spec.",
                "intervention": "Custom evaluator hook integration.",
            },
            "grounding": {
                "source_type": "verified_baseline",
                "source_id": "test_hook",
                "retrieval_summary": "Test metrics evaluator priority.",
                "verified_via": "local_registry",
            },
            "assumption_register": {
                "known_facts": ["Task uses custom metric."],
                "assumptions": ["Custom hook executes smoothly."],
                "uncertainty_level": "low",
            },
            "data": {
                "input_variants": ["tabular"],
                "selection_variant": "tabular",
            },
            "model": {"name": "baseline"},
            "paths": {
                "metrics_evaluator": "custom_scoring.scorer.score",
            },
        }
        exp_yaml = configs / "exp001_run.yaml"
        exp_yaml.write_text(yaml.safe_dump(cfg_dict), encoding="utf-8")

        import sys
        sys.path.insert(0, str(root))
        try:
            attach_project(root, task_yaml, force=True)
            args = types.SimpleNamespace(
                project_root=str(root),
                task_config=str(task_yaml),
                fold=None,
                debug=False,
                probe_samples=None,
                job_id=None,
            )
            runner = UniversalRunner(cfg_dict, str(exp_yaml), args)
            truth = pd.DataFrame({"id": [1, 2], "target": [1.0, 2.0]})
            preds = {"target": [1.0, 2.0]}
            eval_res = runner._evaluate(truth, preds)
            self.assertEqual(eval_res["primary_score"], 0.987)
            self.assertEqual(eval_res["primary_metric"], "custom_competition_metric")
        finally:
            if str(root) in sys.path:
                sys.path.remove(str(root))
            tmp.cleanup()


class TestBootstrapAttachSeamless(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "dataset").mkdir(parents=True)
        (self.root / "docs").mkdir(parents=True)
        (self.root / "configs").mkdir(parents=True)

        df = pd.DataFrame({
            "id": [1, 2, 3, 4, 5, 6],
            "feature1": [10, 20, 30, 40, 50, 60],
            "target": [0, 1, 0, 1, 0, 1],
        })
        df.to_csv(self.root / "dataset" / "train.csv", index=False)
        (self.root / "docs" / "TASK.md").write_text(
            "# Competition: Test Seamless Task\n"
            "**Primary Metric**: macro_f1\n"
            "**Task Type**: binary_classification\n"
            "**Primary ID Column**: id\n"
            "**Target Columns**: target\n",
            encoding="utf-8",
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_bootstrap_automatically_attaches_and_attach_after_bootstrap_no_crash(self):
        report = bootstrap_competition_task(
            dataset_dir=self.root / "dataset",
            context_file=self.root / "docs" / "TASK.md",
            output_config=self.root / "configs" / "task.yaml",
            project_root=self.root,
        )
        self.assertEqual(report["status"], "SUCCESS")
        self.assertTrue((self.root / ".ml-harness" / "project.yaml").is_file())
        self.assertTrue((self.root / ".agents" / "skills" / "ml-harness" / "SKILL.md").is_file())

        # Calling attach_project afterwards should NOT crash with FileExistsError
        attach_res = attach_project(
            project_root=self.root,
            task_config=self.root / "configs" / "task.yaml",
            force=False,
        )
        self.assertEqual(attach_res["status"], "ALREADY_ATTACHED")


class TestModalCloudRuntimeDependencies(unittest.TestCase):
    def test_core_distributions_contain_lightgbm_and_xgboost(self):
        self.assertIn("lightgbm", CORE_RUNTIME_DISTRIBUTIONS)
        self.assertIn("xgboost", CORE_RUNTIME_DISTRIBUTIONS)

    def test_generate_modal_script_includes_lightgbm_and_xgboost(self):
        tmp = tempfile.TemporaryDirectory()
        root = Path(tmp.name)
        configs = root / "configs"
        configs.mkdir(parents=True)
        (root / "dataset").mkdir(parents=True)
        (root / "dataset" / "train.csv").write_text("id,val\n1,2\n", encoding="utf-8")
        task_yaml = configs / "task.yaml"
        task_yaml.write_text("task:\n  name: modal_test\n  primary_metric: rmse\n", encoding="utf-8")
        attach_project(root, task_yaml, force=True)

        exp_yaml = configs / "exp001.yaml"
        exp_yaml.write_text("experiment:\n  id: exp001\n", encoding="utf-8")

        script = generate_modal_script(
            config_path=str(exp_yaml),
            project_root=root,
            volume_name="mock_vol",
        )
        self.assertIn("lightgbm>=4.0.0", script)
        self.assertIn("xgboost>=2.0.0", script)
        tmp.cleanup()


class TestTabularOOMGuard(unittest.TestCase):
    def test_onehotencoder_limits_max_categories(self):
        n_samples = 500
        df = pd.DataFrame({
            "id": range(n_samples),
            "num_feat": np.random.randn(n_samples),
            "high_card_cat": [f"cat_{i % 100}" for i in range(n_samples)],
            "target": np.random.randn(n_samples),
        })

        spec = UniversalTaskSpec.from_dict({
            "name": "test_high_card",
            "modality": "tabular",
            "task_type": "regression",
            "primary_metric": "rmse",
            "direction": "minimize",
            "id_column": "id",
            "inputs": [{"name": "tabular_features", "type": "tabular", "source_columns": ["num_feat", "high_card_cat"]}],
            "subtasks": [{"id": "target", "task_type": "regression", "primary_metric": "rmse", "direction": "minimize", "weight": 1.0}],
        })

        model = GenericBaselineModel(task_spec=spec, model_name="ridge")
        model.fit(df, variant="tabular_features")
        pipe, _, _ = model.fitted_models["target"]
        ohe = pipe.named_steps["features"].named_transformers_["category"].named_steps["encode"]
        self.assertEqual(ohe.max_categories, 30)

        transformed = pipe.named_steps["features"].transform(df[["num_feat", "high_card_cat"]])
        self.assertLessEqual(transformed.shape[1], 35)

    def test_ridge_regression_accepted_model_name(self):
        spec = UniversalTaskSpec.from_dict({
            "name": "test_ridge_alias",
            "modality": "tabular",
            "task_type": "regression",
            "primary_metric": "rmse",
            "direction": "minimize",
            "id_column": "id",
            "inputs": [{"name": "tabular_features", "type": "tabular", "source_columns": ["num_feat"]}],
            "subtasks": [{"id": "target", "task_type": "regression", "primary_metric": "rmse", "direction": "minimize", "weight": 1.0}],
        })
        model = GenericBaselineModel(task_spec=spec, model_name="ridge_regression")
        df = pd.DataFrame({"id": [1, 2], "num_feat": [1.0, 2.0], "target": [1.0, 2.0]})
        model.fit(df, variant="tabular_features")
        preds = model.predict(df, variant="tabular_features")
        self.assertIn("target", preds)


class TestModalityAwareCreateExperiment(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "configs").mkdir(parents=True)

    def tearDown(self):
        self.tmp.cleanup()

    def test_create_experiment_on_tabular_task(self):
        task_yaml = self.root / "configs" / "task.yaml"
        task_yaml.write_text(
            yaml.safe_dump({
                "task": {
                    "name": "tabular_house_prices",
                    "modality": "tabular",
                    "task_type": "regression",
                    "primary_metric": "rmse",
                    "direction": "minimize",
                    "id_column": "Id",
                    "inputs": [{"name": "tabular_features", "type": "tabular", "source_columns": ["GrLivArea"]}],
                    "subtasks": [{"id": "SalePrice", "task_type": "regression", "primary_metric": "rmse", "direction": "minimize", "weight": 1.0}],
                }
            }),
            encoding="utf-8",
        )
        attach_project(self.root, task_yaml, force=True)
        bridge = AgentBridge(self.root)
        sess = bridge.start_session(goal="Test tabular create_experiment")

        sess, obs = bridge.execute_action(
            action=TypedActionName.CREATE_EXPERIMENT.value,
            params={
                "exp_id": "exp002_feat_tabular_iter",
                "parent_id": "baseline",
                "branch": "feat",
                "descriptor": "tabular_iter",
            },
            session_id=sess.session_id,
        )
        self.assertEqual(obs.status, "SUCCESS")
        cfg_file = self.root / "configs" / "exp002_feat_tabular_iter.yaml"
        self.assertTrue(cfg_file.is_file())
        cfg_data = yaml.safe_load(cfg_file.read_text(encoding="utf-8"))

        # Verify modality-aware defaults:
        self.assertEqual(cfg_data["data"]["input_variants"], ["tabular_features"])
        self.assertEqual(cfg_data["model"]["name"], "lightgbm")
        self.assertIn("n_estimators", cfg_data["model"]["params"])

    def test_create_experiment_multi_variant_has_selection_variant_and_runs(self):
        (self.root / "dataset").mkdir(parents=True, exist_ok=True)
        rows = ["id,feat_a,feat_b,target"] + [f"{i},{i*10},{i*20},{float(i)}" for i in range(1, 11)]
        (self.root / "dataset" / "train.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
        task_yaml = self.root / "configs" / "task.yaml"
        task_yaml.write_text(
            yaml.safe_dump({
                "task": {
                    "name": "multi_variant_task",
                    "modality": "tabular",
                    "task_type": "regression",
                    "primary_metric": "rmse",
                    "direction": "minimize",
                    "id_column": "id",
                    "inputs": [
                        {"name": "var_a", "type": "tabular", "source_columns": ["feat_a"]},
                        {"name": "var_b", "type": "tabular", "source_columns": ["feat_b"]},
                    ],
                    "subtasks": [{"id": "target", "task_type": "regression", "primary_metric": "rmse", "direction": "minimize", "weight": 1.0}],
                    "data_config": {"train_path": "dataset/train.csv"},
                }
            }),
            encoding="utf-8",
        )
        attach_project(self.root, task_yaml, force=True)
        bridge = AgentBridge(self.root)
        sess = bridge.start_session(goal="Test multi-variant experiment creation")
        sess, obs = bridge.execute_action(
            action=TypedActionName.CREATE_EXPERIMENT.value,
            params={
                "exp_id": "exp003_multi",
                "parent_id": "baseline",
                "branch": "feat",
                "descriptor": "multi_variant_test",
            },
            session_id=sess.session_id,
        )
        self.assertEqual(obs.status, "SUCCESS")
        cfg_file = self.root / "configs" / "exp003_multi.yaml"
        cfg_data = yaml.safe_load(cfg_file.read_text(encoding="utf-8"))
        self.assertEqual(cfg_data["data"]["input_variants"], ["var_a", "var_b"])
        self.assertEqual(cfg_data["data"]["selection_variant"], "var_a")

        from harness.runner import UniversalRunner
        args = types.SimpleNamespace(
            project_root=str(self.root),
            task_config=str(task_yaml),
            fold=0,
            debug=True,
            probe_samples=None,
            job_id=None,
        )
        runner = UniversalRunner(cfg_data, str(cfg_file), args)
        res = runner.run()
        self.assertIn("primary_score", res)


class TestEdgeCasesAndRanking(unittest.TestCase):
    def test_extended_metrics_constant_and_single_sample_fallback(self):
        # 1. NDCG with single sample
        spec_ndcg = UniversalTaskSpec.from_dict({
            "name": "ndcg_single",
            "task_type": "regression",
            "primary_metric": "ndcg",
            "direction": "maximize",
            "id_column": "id",
            "subtasks": [{"id": "target", "task_type": "regression", "primary_metric": "ndcg", "direction": "maximize", "weight": 1.0}],
        })
        truth_1 = pd.DataFrame({"id": [1], "target": [5.0]})
        preds_1 = {"target": [5.0]}
        res_ndcg = evaluate(spec_ndcg, truth_1, preds_1)
        self.assertEqual(res_ndcg["primary_score"], 0.0)

        # 2. Spearman with constant target
        spec_spearman = UniversalTaskSpec.from_dict({
            "name": "spearman_const",
            "task_type": "regression",
            "primary_metric": "spearman",
            "direction": "maximize",
            "id_column": "id",
            "subtasks": [{"id": "target", "task_type": "regression", "primary_metric": "spearman", "direction": "maximize", "weight": 1.0}],
        })
        truth_const = pd.DataFrame({"id": [1, 2, 3], "target": [1.0, 1.0, 1.0]})
        preds_const = {"target": [1.0, 2.0, 3.0]}
        res_sp = evaluate(spec_spearman, truth_const, preds_const)
        self.assertEqual(res_sp["primary_score"], 0.0)
        self.assertFalse(np.isnan(res_sp["primary_score"]))

        # 3. QWK with constant target
        spec_qwk = UniversalTaskSpec.from_dict({
            "name": "qwk_const",
            "task_type": "multiclass_classification",
            "primary_metric": "qwk",
            "direction": "maximize",
            "id_column": "id",
            "subtasks": [{"id": "target", "task_type": "multiclass_classification", "primary_metric": "qwk", "classes": ["0", "1"], "direction": "maximize", "weight": 1.0}],
        })
        truth_qwk = pd.DataFrame({"id": [1, 2, 3], "target": ["0", "0", "0"]})
        preds_qwk = {"target": ["0", "0", "0"]}
        res_qwk = evaluate(spec_qwk, truth_qwk, preds_qwk)
        self.assertEqual(res_qwk["primary_score"], 0.0)
        self.assertFalse(np.isnan(res_qwk["primary_score"]))

    def test_ranking_task_type_support(self):
        spec = UniversalTaskSpec.from_dict({
            "name": "test_ranking",
            "task_type": "ranking",
            "primary_metric": "ndcg",
            "direction": "maximize",
            "id_column": "id",
            "inputs": [{"name": "tabular_features", "type": "tabular", "source_columns": ["score"]}],
            "subtasks": [{"id": "rank_score", "task_type": "ranking", "primary_metric": "ndcg", "direction": "maximize", "weight": 1.0}],
        })
        truth = pd.DataFrame({"id": [1, 2, 3, 4], "score": [1.0, 2.0, 3.0, 4.0], "rank_score": [1.0, 2.0, 3.0, 4.0]})
        preds = {"rank_score": [1.0, 2.0, 3.0, 4.0]}
        res = evaluate(spec, truth, preds)
        self.assertEqual(res["primary_metric"], "ndcg")
        self.assertGreater(res["primary_score"], 0.9)

        # Baseline model with ranking
        model = GenericBaselineModel(task_spec=spec, model_name="ridge")
        model.fit(truth, variant="tabular_features")
        pred_out = model.predict(truth, variant="tabular_features")
        self.assertIn("rank_score", pred_out)

    def test_runner_metrics_evaluator_colon_syntax_and_sys_path(self):
        from harness.runner import UniversalRunner
        tmp = tempfile.TemporaryDirectory()
        root = Path(tmp.name)
        configs = root / "configs"
        configs.mkdir(parents=True)
        (root / "dataset").mkdir(parents=True)
        (root / "dataset" / "train.csv").write_text("id,val,target\n1,2,1.0\n2,3,2.0\n", encoding="utf-8")

        # Create custom scorer module with colon syntax
        src_dir = root / "eval_pkg"
        src_dir.mkdir(parents=True)
        (src_dir / "__init__.py").write_text("", encoding="utf-8")
        (src_dir / "custom_eval.py").write_text(
            "def my_evaluator(truth, preds):\n"
            "    return {'colon_metric': 0.999, 'primary_score': 0.999}\n",
            encoding="utf-8",
        )

        task_yaml = configs / "task.yaml"
        task_yaml.write_text(
            yaml.safe_dump({
                "task": {
                    "name": "colon_task",
                    "task_type": "regression",
                    "primary_metric": "colon_metric",
                    "direction": "maximize",
                    "id_column": "id",
                    "inputs": [{"name": "tab", "type": "tabular", "source_columns": ["val"]}],
                    "subtasks": [{"id": "target", "task_type": "regression", "primary_metric": "colon_metric", "direction": "maximize", "weight": 1.0}],
                    "data_config": {"train_path": "dataset/train.csv"},
                }
            }),
            encoding="utf-8",
        )

        cfg_dict = {
            "experiment": {
                "id": "exp_colon",
                "branch": "feat",
                "parent_id": "none",
                "problem_addressed": "Test colon syntax evaluator hook without manual sys.path insertion.",
                "hypothesis": "Colon syntax works and runner automatically adds project root to sys.path.",
                "intervention": "Colon syntax evaluator hook test.",
            },
            "grounding": {
                "source_type": "verified_baseline",
                "source_id": "test_colon",
                "retrieval_summary": "Colon hook verification.",
                "verified_via": "local_registry",
            },
            "assumption_register": {
                "known_facts": ["Colon syntax used."],
                "assumptions": ["Imports successfully."],
                "uncertainty_level": "low",
            },
            "data": {
                "input_variants": ["tab"],
                "selection_variant": "tab",
            },
            "model": {"name": "baseline"},
            "paths": {
                "metrics_evaluator": "eval_pkg.custom_eval:my_evaluator",
            },
        }
        exp_yaml = configs / "exp_colon.yaml"
        exp_yaml.write_text(yaml.safe_dump(cfg_dict), encoding="utf-8")

        attach_project(root, task_yaml, force=True)
        args = types.SimpleNamespace(
            project_root=str(root),
            task_config=str(task_yaml),
            fold=None,
            debug=False,
            probe_samples=None,
            job_id=None,
        )
        runner = UniversalRunner(cfg_dict, str(exp_yaml), args)
        truth = pd.DataFrame({"id": [1, 2], "target": [1.0, 2.0]})
        preds = {"target": [1.0, 2.0]}
        eval_res = runner._evaluate(truth, preds)
        self.assertEqual(eval_res["primary_score"], 0.999)
        self.assertEqual(eval_res["primary_metric"], "colon_metric")
        tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
