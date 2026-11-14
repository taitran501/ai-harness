"""Inspection observations preserve measured zeroes and unknown values."""

import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from harness.bridge_actions import ActionHandlerMixin


class TruthfulObservationTests(unittest.TestCase):
    def setUp(self):
        self.bridge = ActionHandlerMixin()
        self.bridge.comp_state = SimpleNamespace(
            spec={"primary_metric_baseline": 0.75},
            state={"gates": {}},
            compute_candidate_pool=Mock(),
        )
        self.candidate = {"id": "exp-zero", "score": 0.0}
        self.pool = {"all_candidates": [self.candidate], "overall_best": self.candidate}
        self.bridge.comp_state.compute_candidate_pool.return_value = self.pool

    def inspect(self, params=None):
        return self.bridge._handle_inspect_experiment(
            SimpleNamespace(step_index=1), None, None, None,
            "inspect_experiment", params or {}, time.time(),
        )

    def test_zero_score_is_preserved(self):
        for scores in ({"score": 0.0}, {"composite_score": 0.0, "score": 0.9},
                       {"composite_score": None, "score": 0.0}):
            with self.subTest(scores=scores):
                self.candidate.clear()
                self.candidate.update(id="exp-zero", **scores)
                obs = self.inspect({"exp_id": "exp-zero"})
                self.assertEqual(obs.status, "SUCCESS")
                self.assertEqual(obs.composite_score, 0.0)
                self.assertIn("score=0.0000", obs.summary)

    def test_missing_fold_dispersion_and_delta_are_unknown(self):
        obs = self.inspect()
        self.assertIsNone(obs.fold_variance)
        self.assertIsNone(obs.metric_delta)
        self.assertNotIn("0.015", obs.summary)

    def test_measured_fold_dispersion_and_delta_are_preserved(self):
        for std in (0.0, 0.02, None):
            with self.subTest(std=std):
                self.candidate.update(fold_std=std, delta_vs_baseline=0.0)
                obs = self.inspect()
                self.assertEqual(obs.fold_variance, std)
                self.assertEqual(obs.metric_delta, 0.0)

    def test_explicit_missing_experiment_fails_without_best_fallback(self):
        for candidates in ([self.candidate], []):
            with self.subTest(candidates=candidates):
                self.pool["all_candidates"] = candidates
                obs = self.inspect({"exp_id": "missing"})
                self.assertEqual(obs.status, "FAILED")
                self.assertEqual(obs.summary, "Experiment 'missing' not found in candidate pool.")
                self.assertEqual(obs.data, {"requested_id": "missing"})
                self.assertIsNone(obs.composite_score)
                self.assertIsNone(obs.fold_variance)

    def test_default_inspection_selects_overall_best(self):
        self.assertEqual(self.inspect().data["id"], "exp-zero")

    def test_missing_scores_use_only_declared_baseline(self):
        self.candidate.clear()
        self.candidate.update(id="exp-zero", composite_score=None, score=None)
        for baseline in (0.75, 0.0, None):
            with self.subTest(baseline=baseline):
                self.bridge.comp_state.spec["primary_metric_baseline"] = baseline
                obs = self.inspect()
                self.assertEqual(obs.composite_score, baseline)
                if baseline is None:
                    self.assertIn("score=unavailable", obs.summary)

    def test_missing_diagnostics_do_not_fabricate_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.bridge.project_root = Path(tmp)
            self.bridge.artifacts_dir = Path(tmp) / "artifacts"
            obs = self.bridge._handle_inspect_diagnostics(
                SimpleNamespace(step_index=1), None, SimpleNamespace(nodes={}), None,
                "inspect_diagnostics", {}, time.time(),
            )
        self.assertEqual(obs.status, "SUCCESS")
        self.assertIsNone(obs.composite_score)
        self.assertIsNone(obs.fold_variance)
        self.assertIsNone(obs.artifact_path)
        self.assertIn("missing diagnostics are not evidence of no errors", obs.summary)
