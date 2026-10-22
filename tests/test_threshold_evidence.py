"""Threshold recommendations must use the selected binary target and its probabilities."""

import copy
import json
import os
import shutil
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from harness.agent_bridge import AgentBridge
from harness.artifacts import sha256
from harness.candidates import register_external_prediction
from harness.endgame_submission import ThresholdOptimizer
from tests.evidence_fixtures import seal_fixture
from tests.fixtures import make_workspace
from tests.test_correctness_contracts import task


class ThresholdEvidenceTests(unittest.TestCase):
    def make_bridge(self, *, alias=False, strings=False, reverse=True, variants=False, regression=False, probabilities=True):
        spec = task() if regression else task("binary_classification", "accuracy", "maximize")
        doc = {"task": copy.deepcopy(spec.metadata)}
        doc["task"]["data"]["train_path"] = "dataset/train.csv"
        column = "label" if alias else "target"
        doc["task"]["subtasks"][0]["column"] = column
        labels = ["no", "yes"] if strings else [0, 1]
        if strings:
            doc["task"]["subtasks"][0]["classes"] = labels
        truth = pd.DataFrame({"id": range(4), "group": range(4), "x": range(4),
                              "category": ["a", "b"] * 2, column: [labels[0]] * 2 + [labels[1]] * 2})
        root, _ = make_workspace(task=doc, train_csv=truth.to_csv(index=False))
        self.addCleanup(shutil.rmtree, root)
        env = patch.dict(os.environ, {"HARNESS_PROJECT_ROOT": str(root)})
        env.start()
        self.addCleanup(env.stop)
        bridge = AgentBridge(root)
        values = [.1, .2, .8, .9]
        oof = pd.DataFrame({"id": truth.id, "pred_target": truth[column],
                            "proba_target": [json.dumps([v, 1-v] if reverse else [1-v, v]) for v in values],
                            "classes_target": json.dumps(labels[::-1] if reverse else labels)})
        if not probabilities:
            oof = oof.drop(columns=["proba_target", "classes_target"])
        if variants:
            folder = bridge.artifacts_dir / "A"
            folder.mkdir(parents=True)
            selected = oof.assign(fold=0, input_variant="features")
            decoy = selected.copy()
            decoy["input_variant"] = "bad_variant"
            decoy["proba_target"] = decoy.proba_target.iloc[::-1].to_numpy()
            pd.concat([decoy, selected]).to_csv(folder / "oof_predictions.csv", index=False)
            truth.to_csv(folder / "evaluation_truth.csv", index=False)
            seal_fixture(folder, bridge.task_spec, score=1.0)
        else:
            source = root / "external.csv"
            oof.iloc[::-1].to_csv(source, index=False)
            register_external_prediction(root, "A", source)
        return bridge

    def optimize(self, bridge, **params):
        return bridge.execute_action("optimize_threshold", {"exp_id": "A", "metric": "accuracy", **params})[1]

    def test_declared_positive_class_survives_reversed_probability_order(self):
        for reverse in (False, True):
            with self.subTest(reverse=reverse):
                bridge = self.make_bridge(reverse=reverse)
                before = sha256(bridge.artifacts_dir / "A/artifact_manifest.json")
                obs = self.optimize(bridge)
                self.assertEqual(obs.status, "SUCCESS", obs.summary)
                self.assertEqual(obs.composite_score, 1.0)
                self.assertEqual(obs.data["baseline_score_at_05"], 1.0)
                self.assertEqual(str(obs.data["positive_class"]), "1")
                self.assertEqual(sha256(bridge.artifacts_dir / "A/artifact_manifest.json"), before)

    def test_target_column_alias_and_string_labels_are_supported(self):
        for strings in (False, True):
            with self.subTest(strings=strings):
                bridge = self.make_bridge(alias=True, strings=strings)
                obs = self.optimize(bridge)
                self.assertEqual(obs.status, "SUCCESS", obs.summary)
                self.assertEqual(obs.composite_score, 1.0)
                self.assertEqual(obs.data["target"], "target")

    def test_only_the_selected_input_variant_is_used(self):
        obs = self.optimize(self.make_bridge(variants=True))
        self.assertEqual(obs.status, "SUCCESS", obs.summary)
        self.assertEqual(obs.composite_score, 1.0)
        self.assertEqual(obs.data["n_samples"], 4)
        self.assertEqual(obs.data["selection_variant"], "features")

    def test_regression_is_not_median_binarized(self):
        obs = self.optimize(self.make_bridge(regression=True))
        self.assertEqual(obs.status, "FAILED", obs.summary)
        self.assertIn("binary", obs.summary.lower())

    def test_modified_source_evidence_cannot_drive_a_threshold_recommendation(self):
        bridge = self.make_bridge()
        path = bridge.artifacts_dir / "A/oof_predictions.csv"
        path.write_text(path.read_text() + "\n")
        obs = self.optimize(bridge)
        self.assertEqual(obs.status, "FAILED", obs.summary)
        self.assertIn("hash mismatch", obs.summary)

    def test_hard_labels_are_not_used_as_if_they_were_probabilities(self):
        obs = self.optimize(self.make_bridge(probabilities=False))
        self.assertEqual(obs.status, "FAILED", obs.summary)
        self.assertIn("probabilities", obs.summary)

    def test_unknown_metrics_and_invalid_probabilities_fail_instead_of_using_f1(self):
        bridge = self.make_bridge()
        obs = self.optimize(bridge, metric="roc_auc")
        self.assertEqual(obs.status, "FAILED", obs.summary)
        self.assertIn("metric", obs.summary.lower())
        for prob in ([.1, float("nan")], [.1, 1.1], [.1], []):
            with self.subTest(prob=prob):
                with self.assertRaises(ValueError):
                    ThresholdOptimizer.optimize_threshold(np.array([0, 1]), np.array(prob))

    def test_bounded_threshold_grid_keeps_the_baseline_as_a_candidate(self):
        threshold, score = ThresholdOptimizer.optimize_threshold(np.array([0, 1]), np.array([.2, .8]), num_thresholds=2)
        self.assertEqual(threshold, .5)
        self.assertEqual(score, 1.0)
