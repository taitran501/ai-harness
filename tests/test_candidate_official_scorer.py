"""The agent's runner -> candidate -> optimizer chain must keep its official metric."""

import copy
import os
import shutil
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from harness.agent_bridge import AgentBridge
from harness.artifacts import sha256, validate_run
from harness.candidates import build_candidate, find_optimal_blend_weights
from harness.runner import UniversalRunner
from tests.fixtures import make_workspace
from tests.test_correctness_contracts import task


class CandidateOfficialScorerTests(unittest.TestCase):
    def setUp(self):
        doc = {"task": copy.deepcopy(task().metadata)}
        doc["task"]["data"]["train_path"] = "dataset/train.csv"
        doc["task"]["validation"] = {"strategy": "kfold", "num_folds": 2}
        doc["task"].update(primary_metric="official_score", direction="maximize")
        doc["task"]["subtasks"][0].update(primary_metric="official_score", direction="maximize")
        train = pd.DataFrame({"id": range(12), "group": range(12), "x": range(12),
                              "category": ["a", "b"] * 6, "target": [2.0 * i for i in range(12)]})
        self.root, _ = make_workspace(task=doc, train_csv=train.to_csv(index=False))
        self.addCleanup(shutil.rmtree, self.root)
        self.env = patch.dict(os.environ, {"HARNESS_PROJECT_ROOT": str(self.root)})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.addCleanup(lambda: sys.modules.pop("candidate_official_fixture", None))
        original_path = list(sys.path)
        self.addCleanup(lambda: sys.path.__setitem__(slice(None), original_path))
        (self.root / "candidate_official_fixture.py").write_text(
            "import numpy as np\n"
            "def score(truth, predictions):\n"
            "    values = predictions.get('pred_target', predictions.get('target'))\n"
            "    return {'official_score': -float(np.mean(np.abs(truth.target.to_numpy() - values)))}\n"
            "def rmse_score(truth, predictions):\n    return {'rmse': 42.0}\n",
            encoding="utf-8",
        )
        self.args = SimpleNamespace(project_root=str(self.root), task_config=None, debug=False,
                                    fold=None, probe_samples=None)

    def run_source(self, exp_id, alpha=1.0, hook="candidate_official_fixture:score"):
        cfg = {"experiment": {"id": exp_id, "branch": "baseline", "parent_id": None,
                              "problem_addressed": "Measure the official regression baseline.",
                              "hypothesis": "Regularization changes the official validation score.",
                              "intervention": "Train a ridge baseline with fixed splits."},
               "model": {"name": "ridge", "params": {"alpha": alpha}},
               "paths": {"metrics_evaluator": hook}}
        runner = UniversalRunner(cfg, "config.yaml", self.args)
        result = runner.run()
        return runner, result

    def test_bridge_promotes_custom_metric_without_changing_source_evidence(self):
        runner, result = self.run_source("exp000_official")
        before = sha256(runner.exp_dir / "artifact_manifest.json")
        bridge = AgentBridge(self.root)
        _, obs = bridge.execute_action("promote_candidate", {"exp_id": runner.exp_id})
        self.assertEqual(obs.status, "SUCCESS", obs.summary)
        self.assertEqual(obs.data["primary_metric"], "official_score")
        self.assertAlmostEqual(obs.composite_score, result["primary_score"])
        self.assertEqual(sha256(runner.exp_dir / "artifact_manifest.json"), before)

    def test_optimizer_uses_same_custom_metric_as_sources(self):
        runner, _ = self.run_source("exp001_official", .2)
        self.run_source("exp002_official", 2.0)
        result = find_optimal_blend_weights(runner.project, ["exp001_official", "exp002_official"], maxiter=2)
        self.assertEqual(result["evaluation_details"]["primary_metric"], "official_score")
        self.assertLessEqual(result["best_score"], 0)

    def test_different_custom_scorers_cannot_be_blended(self):
        runner, _ = self.run_source("exp001_official")
        self.addCleanup(lambda: sys.modules.pop("candidate_other_fixture", None))
        (self.root / "candidate_other_fixture.py").write_text(
            "def score(truth, predictions):\n    return {'official_score': -100.0}\n", encoding="utf-8")
        other, _ = self.run_source("exp002_official", 2.0, "candidate_other_fixture:score")
        for source in (runner, other):
            validate_run(source.exp_dir, task_spec=source.task_spec, plugin=source.task_plugin)
        with self.assertRaisesRegex(ValueError, "custom_scorer_sha256"):
            build_candidate(runner.project, "invalid", "weighted_blend",
                            {"components": ["exp001_official", "exp002_official"]})
        with self.assertRaisesRegex(ValueError, "custom_scorer_sha256"):
            find_optimal_blend_weights(runner.project, ["exp001_official", "exp002_official"])

    def test_named_hook_metric_overrides_generic_primary_score(self):
        from harness.oof_revalidation import score_predictions
        from harness.plugins import TaskPlugin

        truth = pd.DataFrame({"id": [1, 2], "target": [0., 1.]})
        result = score_predictions(task(), TaskPlugin(), SimpleNamespace(project_root=self.root),
                                   {"paths": {"metrics_evaluator": "candidate_official_fixture:rmse_score"}},
                                   truth, {"target": [0., 1.]})
        self.assertEqual(result["primary_score"], 42.)
        self.assertEqual(result["utility"], -42.)
