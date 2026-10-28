"""The agent's blend optimizer must honor the candidate builder's source contract."""

import copy
import json
import os
import shutil
import unittest
from unittest.mock import patch

import pandas as pd

from harness.agent_bridge import AgentBridge
from harness.artifacts import sha256, validate_run
from harness.candidates import find_optimal_blend_weights, register_external_prediction
from tests.fixtures import make_workspace
from tests.test_correctness_contracts import task


class BlendOptimizerContractTests(unittest.TestCase):
    def make_bridge(self, reverse_second=False):
        doc = {"task": copy.deepcopy(task("binary_classification", "accuracy", "maximize").metadata)}
        doc["task"]["data"]["train_path"] = "dataset/train.csv"
        truth = pd.DataFrame({"id": range(4), "group": range(4), "x": range(4),
                              "category": ["a", "b"] * 2, "target": [0, 0, 1, 1]})
        root, _ = make_workspace(task=doc, train_csv=truth.to_csv(index=False))
        self.addCleanup(shutil.rmtree, root)
        env = patch.dict(os.environ, {"HARNESS_PROJECT_ROOT": str(root)})
        env.start()
        self.addCleanup(env.stop)
        for name, values in (("A", [.1, .6, .7, .4]), ("B", [.6, .1, .4, .7])):
            reverse = name == "B" and reverse_second
            frame = pd.DataFrame({
                "id": truth.id, "pred_target": [int(v > .5) for v in values],
                "proba_target": [json.dumps([v, 1-v] if reverse else [1-v, v]) for v in values],
                "classes_target": json.dumps([1, 0] if reverse else [0, 1]),
            }).iloc[::-1]
            source = root / (name + ".csv")
            frame.to_csv(source, index=False)
            register_external_prediction(root, name, source)
        return AgentBridge(root)

    def test_incompatible_class_order_fails_before_reporting_an_optimization(self):
        bridge = self.make_bridge(reverse_second=True)
        with self.assertRaisesRegex(ValueError, "EXTERNAL_PREDICTION_NOT_SUBMITTABLE"):
            find_optimal_blend_weights(bridge.project, ["A", "B"])
        before = {}
        for name in ("A", "B"):
            folder = bridge.artifacts_dir / name
            validate_run(folder, task_spec=bridge.task_spec, plugin=bridge.task_plugin)
            before[name] = sha256(folder / "artifact_manifest.json")
        self.assertFalse(bridge.comp_state.state["candidate_pool"]["synthesized_candidates"])
        for name in before:
            self.assertEqual(sha256(bridge.artifacts_dir / name / "artifact_manifest.json"), before[name])

    def test_compatible_sources_optimize_and_build_the_reported_candidate(self):
        bridge = self.make_bridge()
        _, obs = bridge.execute_action("optimize_blend", {"components": ["A", "B"]})
        self.assertEqual(obs.status, "FAILED", obs.summary)
        self.assertIn("EXTERNAL_PREDICTION_NOT_SUBMITTABLE", obs.summary)

    def test_duplicate_components_and_unknown_modes_are_rejected(self):
        bridge = self.make_bridge()
        with self.assertRaisesRegex(ValueError, "distinct"):
            find_optimal_blend_weights(bridge.project, ["A", "A"])
        with self.assertRaisesRegex(ValueError, "Unsupported blend mode"):
            find_optimal_blend_weights(bridge.project, ["A", "B"], blend_mode="not_a_mode")
