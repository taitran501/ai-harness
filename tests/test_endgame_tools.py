"""Unit tests for Competitive Endgame Tools and AgentBridge Actions.

Tests:
1. ThresholdOptimizer: binary threshold search maximizing F1/Kappa.
2. QuantileMatcher: rank-quantile projection matching target CDF.
3. MicroDoseProbeGenerator: low-drift probing submission creation.
4. find_optimal_blend_weights: SLSQP blending optimization.
5. register_external_prediction: External Grandmaster candidate ingestion.
6. AgentBridge typed actions: optimize_blend, optimize_threshold, generate_microdose_probe.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import f1_score

from harness.agent_bridge import AgentBridge, TypedActionName
from harness.artifacts import sha256
from harness.candidates import find_optimal_blend_weights, register_external_prediction
from harness.endgame_submission import (
    MicroDoseProbeGenerator,
    QuantileMatcher,
    ThresholdOptimizer,
)
from harness.tree_journal import Node, TreeJournal
from tests.test_correctness_contracts import task


class TestThresholdOptimizer(unittest.TestCase):
    def test_imbalanced_f1_optimization(self):
        rng = np.random.default_rng(42)
        # 10% positive class
        y_true = rng.choice([0, 1], size=500, p=[0.9, 0.1])
        # Model predicts calibrated but lower raw probabilities
        y_prob = np.clip(y_true * 0.4 + rng.uniform(0.0, 0.25, size=500), 0.01, 0.99)

        score_05 = f1_score(y_true, (y_prob >= 0.5).astype(int))
        best_th, best_score = ThresholdOptimizer.optimize_threshold(
            y_true, y_prob, metric="f1", num_thresholds=181
        )

        self.assertGreaterEqual(best_score, score_05)
        self.assertLess(best_th, 0.5)  # Optimal threshold should be lower for imbalanced signal


class TestQuantileMatcher(unittest.TestCase):
    def test_quantile_matching_distribution(self):
        rng = np.random.default_rng(42)
        source = rng.normal(loc=0.0, scale=1.0, size=1000)
        target = rng.exponential(scale=2.0, size=1000)

        matched = QuantileMatcher.match_quantiles(source, target)
        self.assertEqual(len(matched), len(source))
        # Non-negative like target exponential distribution
        self.assertGreaterEqual(matched.min(), 0.0)
        # Median close to target median
        self.assertAlmostEqual(np.median(matched), np.median(target), places=1)


class TestMicroDoseProbeGenerator(unittest.TestCase):
    def test_generate_probe_logit_and_rank(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            base_csv = tmp / "base.csv"
            probe_csv = tmp / "probe.csv"
            out_csv = tmp / "probe_out.csv"

            n = 200
            df_base = pd.DataFrame({"id": range(n), "target": np.linspace(0.1, 0.9, n)})
            df_probe = pd.DataFrame({"id": range(n), "target": np.linspace(0.2, 0.8, n)})
            df_base.to_csv(base_csv, index=False)
            df_probe.to_csv(probe_csv, index=False)

            # 1. Logit space probe
            rep_logit = MicroDoseProbeGenerator.generate_probe(
                base_csv, probe_csv, out_csv, alpha=0.03, space="logit"
            )
            self.assertTrue(out_csv.exists())
            self.assertGreater(rep_logit["spearman_correlation_vs_base"], 0.99)
            self.assertLess(rep_logit["max_absolute_diff"], 0.05)

            # 2. Rank space probe
            rep_rank = MicroDoseProbeGenerator.generate_probe(
                base_csv, probe_csv, out_csv, alpha=0.05, space="rank"
            )
            self.assertGreater(rep_rank["spearman_correlation_vs_base"], 0.99)


class TestEndgameCandidatesAndBridge(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        (self.root / "configs").mkdir()
        (self.root / "dataset").mkdir()
        (self.root / "artifacts").mkdir()

        # Task contract
        spec = task()
        doc = {"task": spec.metadata}
        doc["task"]["data"]["train_path"] = "dataset/train.csv"
        doc["task"]["validation"] = {"strategy": "kfold", "num_folds": 2}
        (self.root / "configs/task.yaml").write_text(yaml.safe_dump(doc), encoding="utf-8")

        n = 20
        self.n = n
        self.train_df = pd.DataFrame({
            "id": range(n),
            "group": range(n),
            "x": range(n),
            "category": ["a", "b"] * (n // 2),
            "target": [float(x * 2.0) for x in range(n)],
        })
        self.train_df.to_csv(self.root / "dataset/train.csv", index=False)

        # Initialize TreeJournal
        self.journal_path = self.root / "artifacts" / "tree_journal.json"
        self.journal = TreeJournal(storage_path=self.journal_path)

        # Helper to create mock verified run artifact
        for exp_id, bias in [("exp001", 0.5), ("exp002", -0.5)]:
            exp_dir = self.root / "artifacts" / exp_id
            exp_dir.mkdir()

            oof = pd.DataFrame({
                "id": range(n),
                "fold": [i % 2 for i in range(n)],
                "input_variant": "features",
                "pred_target": [float(x * 2.0 + bias) for x in range(n)],
            })
            oof.to_csv(exp_dir / "oof_predictions.csv", index=False)
            oof.to_csv(exp_dir / "predictions.csv", index=False)
            self.train_df.to_csv(exp_dir / "evaluation_truth.csv", index=False)

            test_p = pd.DataFrame({
                "id": range(5),
                "target": [float(x * 2.0 + bias) for x in range(5)],
                "pred_target": [float(x * 2.0 + bias) for x in range(5)],
            })
            test_p.to_csv(exp_dir / "test_predictions.csv", index=False)

            metrics = {
                "schema_version": 2,
                "experiment_id": exp_id,
                "selection_variant": "features",
                "primary_score": abs(bias),
                "utility": -abs(bias),
            }
            (exp_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

            cfg = {
                "experiment": {"id": exp_id, "branch": "baseline"},
                "model": {"name": "ridge"},
            }
            (exp_dir / "run_config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")

            manifest = {
                "schema_version": 2,
                "exp_id": exp_id,
                "task_sha256": sha256(self.root / "configs/task.yaml"),
                "scorer_sha256": "default_scorer",
                "split_sha256": "split_hash_mock",
                "data_sha256": "data_hash_mock",
                "resolved_config_sha256": sha256(exp_dir / "run_config.yaml"),
                "prediction_splits": {"test_predictions.csv": "test"},
                "files": {
                    "run_config.yaml": sha256(exp_dir / "run_config.yaml"),
                    "metrics.json": sha256(exp_dir / "metrics.json"),
                    "oof_predictions.csv": sha256(exp_dir / "oof_predictions.csv"),
                    "evaluation_truth.csv": sha256(exp_dir / "evaluation_truth.csv"),
                    "test_predictions.csv": sha256(exp_dir / "test_predictions.csv"),
                },
            }
            (exp_dir / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

            # Add node to journal
            self.journal.append(Node(
                id=exp_id,
                branch="baseline",
                metrics={"primary_score": abs(bias), "score": abs(bias)},
                primary_metric_name="mse",
                direction="minimize",
                verified=True,
                eligible=True,
                status="COMPLETED",
                artifact_dir=str(exp_dir),
            ))

        self.bridge = AgentBridge(self.root)

    def tearDown(self):
        self._tmp.cleanup()

    def test_find_optimal_blend_weights(self):
        from harness.task_spec import load_active_task_spec
        from tests.evidence_fixtures import seal_fixture

        spec = load_active_task_spec(self.root / "configs/task.yaml")
        for exp_id in ("exp001", "exp002"):
            seal_fixture(self.root / "artifacts" / exp_id, spec, score=.5)
        opt = find_optimal_blend_weights(
            project=self.root,
            component_ids=["exp001", "exp002"],
            blend_mode="weighted_blend",
        )
        self.assertEqual(opt["component_ids"], ["exp001", "exp002"])
        self.assertEqual(len(opt["optimal_weights"]), 2)
        self.assertAlmostEqual(sum(opt["optimal_weights"]), 1.0, places=3)
        self.assertIn("best_score", opt)

    def test_register_external_prediction(self):
        ext_oof = pd.DataFrame({
            "id": range(self.n),
            "fold": 0,
            "pred_target": [float(x * 2.0 + 0.05) for x in range(self.n)],
        })
        ext_oof_path = self.root / "ext_oof.csv"
        ext_oof.to_csv(ext_oof_path, index=False)

        res = register_external_prediction(
            project=self.root,
            name="ext_gm_01",
            oof_path=ext_oof_path,
            author="Grandmaster_T",
            source="kaggle_kernel",
            notes="PyTorch CatBoost Stacking",
        )
        self.assertEqual(res["name"], "ext_gm_01")
        self.assertEqual(res["author"], "Grandmaster_T")
        self.assertTrue((self.root / "artifacts" / "ext_gm_01" / "artifact_manifest.json").exists())
        self.assertTrue((self.root / "artifacts" / "ext_gm_01" / "metrics.json").exists())

    def test_agent_bridge_endgame_actions(self):
        sess = self.bridge.start_session(goal="Test endgame actions")

        # 1. Action: optimize_threshold
        sess, obs_th = self.bridge.execute_action(
            action=TypedActionName.OPTIMIZE_THRESHOLD.value,
            params={"exp_id": "exp001", "metric": "f1"},
            session_id=sess.session_id,
        )
        self.assertEqual(obs_th.status, "SUCCESS")
        self.assertIn("best_threshold", obs_th.data)

        # 2. Action: generate_microdose_probe
        base_p = self.root / "artifacts" / "exp001" / "test_predictions.csv"
        probe_p = self.root / "artifacts" / "exp002" / "test_predictions.csv"
        sess, obs_probe = self.bridge.execute_action(
            action=TypedActionName.GENERATE_MICRODOSE_PROBE.value,
            params={
                "base": str(base_p),
                "probe": str(probe_p),
                "alpha": 0.03,
                "space": "probability",
            },
            session_id=sess.session_id,
        )
        self.assertEqual(obs_probe.status, "SUCCESS")
        self.assertGreater(obs_probe.data["spearman_correlation_vs_base"], 0.99)

        # 3. Action: register_external_prediction
        ext_oof = pd.DataFrame({
            "id": range(self.n),
            "pred_target": [float(x * 2.0) for x in range(self.n)],
        })
        ext_path = self.root / "ext_pred.csv"
        ext_oof.to_csv(ext_path, index=False)

        sess, obs_ext = self.bridge.execute_action(
            action=TypedActionName.REGISTER_EXTERNAL_PREDICTION.value,
            params={"name": "ext_bridge_01", "oof_file": str(ext_path), "author": "GM"},
            session_id=sess.session_id,
        )
        self.assertEqual(obs_ext.status, "SUCCESS")
        self.assertEqual(obs_ext.data["name"], "ext_bridge_01")


if __name__ == "__main__":
    unittest.main()
