import unittest
import json
import tempfile
from pathlib import Path

import pandas as pd

from harness.evaluation import PredictionBatch, PredictionContractError, align_prediction_frame, validate_prediction_batch
from harness.script_predictions import extract_test_predictions
from tests.fixtures import task


class PredictionBoundaryTests(unittest.TestCase):
    def test_in_memory_identity_is_sequence_exact(self):
        spec = task()
        for ids in ([2, 1], [1], [1, 1], [1, None], [1, 3]):
            with self.subTest(ids=ids), self.assertRaises(PredictionContractError):
                validate_prediction_batch(spec, PredictionBatch(ids, {"target": [1., 2.]}), expected_ids=[1, 2], purpose="test")

    def test_persisted_shuffle_aligns_and_does_not_coerce_ids(self):
        frame = pd.DataFrame({"id": ["1", "001"], "pred_target": [2., 1.]})
        result = align_prediction_frame(frame, expected_ids=["001", "1"], id_column="id")
        self.assertEqual(result.pred_target.tolist(), [1., 2.])
        with self.assertRaises(PredictionContractError):
            align_prediction_frame(frame, expected_ids=[1, 2], id_column="id")

    def test_test_probabilities_checked_for_accuracy(self):
        spec = task("binary_classification", "accuracy", "maximize")
        valid = {"ids": [1, 2], "pred_target": ["0", "1"], "proba_target": [[.8, .2], [.1, .9]], "classes_target": ["0", "1"]}
        validate_prediction_batch(spec, valid, expected_ids=[1, 2], purpose="test")
        for changes in ({"classes_target": ["1", "0"]}, {"proba_target": [[.8, .8], [.1, .9]]},
                        {"proba_target": [[float("nan"), .2], [.1, .9]]}, {"classes_target": None}):
            with self.subTest(changes=changes), self.assertRaises(PredictionContractError):
                validate_prediction_batch(spec, {**valid, **changes}, expected_ids=[1, 2], purpose="test")

    def test_unlabeled_persisted_test_payload_is_validated(self):
        spec = task()
        frame = pd.DataFrame({"id": [2, 1], "pred_target": [2., 1.]})
        self.assertEqual(extract_test_predictions(frame, spec, expected_ids=[1, 2])["ids"], [1, 2])
        frame.loc[0, "pred_target"] = float("inf")
        with self.assertRaises(PredictionContractError):
            extract_test_predictions(frame, spec, expected_ids=[1, 2])

    def test_candidate_sources_agreeing_on_wrong_test_ids_fail(self):
        from harness.artifacts import seal_file_map
        from harness.candidates import build_candidate
        from harness.project import HarnessProject
        from tests.fixtures import make_workspace
        from tests.evidence_fixtures import seal_fixture
        with tempfile.TemporaryDirectory() as tmp:
            spec = task()
            spec.metadata["data"]["test_path"] = "dataset/test.csv"
            root, _ = make_workspace(root_path=Path(tmp), task={"task": spec.metadata})
            pd.DataFrame({"id": [100, 101], "x": [0., 1.]}).to_csv(root / "dataset/test.csv", index=False)
            project = HarnessProject.discover(root)
            for name in ("a", "b"):
                folder = project.artifacts_dir / name
                folder.mkdir(parents=True)
                pd.DataFrame({"id": [1, 2], "pred_target": [0., 1.], "fold": [0, 1],
                              "input_variant": "features"}).to_csv(folder / "oof_predictions.csv", index=False)
                seal_fixture(folder, spec)
                pd.DataFrame({"id": [102, 103], "pred_target": [0., 1.]}).to_csv(folder / "test_predictions.csv", index=False)
                path = folder / "artifact_manifest.json"
                manifest = json.loads(path.read_text())
                manifest["files"] = seal_file_map(folder)
                path.write_text(json.dumps(manifest))
            with self.assertRaises(PredictionContractError):
                build_candidate(project, "bad", "weighted_blend", {"components": ["a", "b"]})
            self.assertFalse(list(project.artifacts_dir.glob("cand_*")))
