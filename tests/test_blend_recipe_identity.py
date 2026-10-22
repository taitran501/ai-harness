"""Duplicate detection must compare source/weight mappings, including old recipes."""

import copy
import json
import hashlib
import os
import shutil
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml

from harness.agent_bridge import AgentBridge
from harness.artifacts import sha256
from harness.candidates import _candidate_recipe_fingerprint as fingerprint
from harness.task_spec import load_active_task_spec
from tests.evidence_fixtures import seal_fixture
from tests.fixtures import make_workspace
from tests.test_correctness_contracts import task


class BlendRecipeIdentityTests(unittest.TestCase):
    def fp(self, ids, weights=None):
        source_map = {"components": ids}
        if weights is not None:
            source_map["weights"] = weights
        return fingerprint("weighted_blend", ids, source_map)

    def test_source_order_and_weight_scale_are_canonical(self):
        self.assertEqual(self.fp(["A", "B"], [.9, .1]), self.fp(["B", "A"], [.1, .9]))
        self.assertEqual(self.fp(["A", "B"], [.1, .2]), self.fp(["A", "B"], [1, 2]))
        self.assertEqual(self.fp(["A", "B"]), self.fp(["A", "B"], [.5, .5]))

    def test_swapping_only_weights_changes_the_recipe(self):
        self.assertNotEqual(self.fp(["A", "B"], [.9, .1]), self.fp(["A", "B"], [.1, .9]))

    def test_bridge_accepts_distinct_blend_but_blocks_equivalent_legacy_recipe(self):
        doc = {"task": copy.deepcopy(task().metadata)}
        doc["task"]["data"]["train_path"] = "dataset/train.csv"
        truth = pd.DataFrame({"id": range(12), "group": range(12), "x": range(12),
                              "category": ["a", "b"] * 6, "target": [2. * i for i in range(12)]})
        root, _ = make_workspace(task=doc, train_csv=truth.to_csv(index=False))
        self.addCleanup(shutil.rmtree, root)
        with patch.dict(os.environ, {"HARNESS_PROJECT_ROOT": str(root)}):
            bridge = AgentBridge(root)
            spec = load_active_task_spec(root / "configs/task.yaml")
            error = 10 * np.sin(np.arange(len(truth)))
            for name, sign in (("A", 1), ("B", -1)):
                folder = bridge.artifacts_dir / name
                folder.mkdir(parents=True)
                pd.DataFrame({"id": truth.id, "fold": [0, 1] * 6, "input_variant": "features",
                              "pred_target": truth.target + sign * error}).to_csv(folder / "oof_predictions.csv", index=False)
                truth.to_csv(folder / "evaluation_truth.csv", index=False)
                (folder / "run_config.yaml").write_text(yaml.safe_dump({"experiment": {"id": name}}))
                seal_fixture(folder, spec, score=float(np.sqrt(np.mean(error ** 2))))
            def synth(ids, weights):
                return bridge.execute_action("synthesize_candidate", {
                    "mode": "weighted_blend", "source_map": {"components": ids, "weights": weights}})[1]

            first = synth(["A", "B"], [.9, .1])
            self.assertEqual(first.status, "SUCCESS", first.summary)
            candidate_id = first.data["id"]
            manifest = bridge.artifacts_dir / candidate_id / "candidate_manifest.json"
            original_hash = sha256(manifest)
            # Simulate the stored guard hash from the old release. Recipe
            # details remain available and should drive duplicate detection.
            stored = next(c for c in bridge.comp_state.state["candidate_pool"]["synthesized_candidates"]
                          if c["id"] == candidate_id)
            legacy_recipe = {"mode": "weighted_blend", "source_ids": ["A", "B"],
                             "weights": [.1, .9], "routing": None}
            stored["recipe_fingerprint"] = hashlib.sha256(
                json.dumps(legacy_recipe, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            bridge.comp_state.save()
            distinct = synth(["A", "B"], [.1, .9])
            self.assertEqual(distinct.status, "SUCCESS", distinct.summary)
            duplicate = synth(["B", "A"], [1, 9])
            self.assertEqual(duplicate.status, "FAILED")
            self.assertIn("recipe already exists", duplicate.summary)
            self.assertEqual(sha256(manifest), original_hash)
            self.assertEqual(json.loads(manifest.read_text())["details"]["weights"], [.9, .1])

    def test_invalid_weights_do_not_acquire_a_recipe_identity(self):
        for weights in ([], [1], [-1, 2], [0, 0], [float("nan"), 1], [float("inf"), 1]):
            with self.subTest(weights=weights):
                with self.assertRaisesRegex(ValueError, "Invalid blend weights"):
                    self.fp(["A", "B"], weights)
