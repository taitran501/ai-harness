"""Accepted external evidence must be readable by the agent's candidate chain."""

import copy
import os
import shutil
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from harness.agent_bridge import AgentBridge
from harness.artifacts import sha256, validate_run
from tests.fixtures import make_workspace
from tests.test_correctness_contracts import task


class ExternalCandidateMetadataTests(unittest.TestCase):
    def make_bridge(self, probability=False):
        spec = task("binary_classification", "roc_auc", "maximize") if probability else task()
        doc = {"task": copy.deepcopy(spec.metadata)}
        doc["task"]["data"]["train_path"] = "dataset/train.csv"
        truth = pd.DataFrame({"id": range(6), "group": range(6), "x": [.1, .9] * 3,
                              "category": ["a", "b"] * 3,
                              "target": [0, 1] * 3 if probability else list(range(6))})
        root, _ = make_workspace(task=doc, train_csv=truth.to_csv(index=False))
        self.addCleanup(shutil.rmtree, root)
        env = patch.dict(os.environ, {"HARNESS_PROJECT_ROOT": str(root)})
        env.start()
        self.addCleanup(env.stop)
        return AgentBridge(root), truth

    def register_and_promote(self, bridge, source, name):
        original_hash = sha256(source)
        _, registered = bridge.execute_action("register_external_prediction",
                                               {"name": name, "oof_path": str(source)})
        self.assertEqual(registered.status, "SUCCESS", registered.summary)
        folder = bridge.artifacts_dir / name
        validate_run(folder, task_spec=bridge.task_spec, plugin=bridge.task_plugin)
        _, promoted = bridge.execute_action("promote_candidate", {"exp_id": name})
        self.assertEqual(promoted.status, "FAILED", promoted.summary)
        self.assertIn("EXTERNAL_PREDICTION_NOT_SUBMITTABLE", promoted.summary)
        self.assertEqual(sha256(source), original_hash)
        return pd.read_csv(folder / "oof_predictions.csv")

    def test_minimal_shuffled_csv_and_target_alias_enter_candidate_pool(self):
        bridge, truth = self.make_bridge()
        output = truth[["id", "target"]].iloc[::-1].copy()
        output["input_variant"] = "exporter-specific-name"
        source = bridge.project_root / "external.csv"
        output.to_csv(source, index=False)
        managed = self.register_and_promote(bridge, source, "ext_csv")
        self.assertEqual(managed.input_variant.unique().tolist(), ["external"])
        self.assertEqual(managed.fold.unique().tolist(), [-1])
        self.assertEqual(managed.pred_target.tolist(), truth.target.tolist())

    def test_numpy_export_without_internal_metadata_can_be_promoted(self):
        bridge, truth = self.make_bridge()
        source = bridge.project_root / "external.npy"
        np.save(source, truth.target.to_numpy() + .1)
        managed = self.register_and_promote(bridge, source, "ext_numpy")
        self.assertEqual(managed.input_variant.unique().tolist(), ["external"])

    def test_probability_export_preserves_class_order_and_existing_folds(self):
        bridge, truth = self.make_bridge(probability=True)
        output = pd.DataFrame({"id": truth.id, "pred_target": truth.target,
                               "proba_target": ["[0.1, 0.9]", "[0.9, 0.1]"] * 3,
                               "classes_target": "[1, 0]", "fold": [0, 1] * 3})
        source = bridge.project_root / "probabilities.csv"
        output.to_csv(source, index=False)
        managed = self.register_and_promote(bridge, source, "ext_probability")
        self.assertEqual(managed.fold.tolist(), [0, 1] * 3)
        self.assertEqual(managed.classes_target.unique().tolist(), ["[1, 0]"])
