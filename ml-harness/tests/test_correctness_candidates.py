import unittest
import pandas as pd
from tools.audit_gate import audit_prediction_integrity, compute_diff_metrics
from tests.test_correctness_contracts import task


class CandidateContracts(unittest.TestCase):
    def test_audit_missing_duplicate_and_fractional_labels(self):
        spec = task("binary_classification", "accuracy", "maximize")
        self.assertTrue(
            audit_prediction_integrity(
                pd.DataFrame({"id": [1, 1], "pred_target": [0, 0.5]}), spec=spec
            )
        )
        self.assertTrue(
            audit_prediction_integrity(pd.DataFrame({"id": [1]}), spec=spec)
        )
        self.assertFalse(
            audit_prediction_integrity(
                pd.DataFrame({"id": [1, 2], "pred_target": [0, 1]}), spec=spec
            )
        )

    def test_metadata_diff_and_wrong_ids(self):
        a = pd.DataFrame({"id": [1, 2], "pred_target": [0, 1], "fold": [0, 0]})
        b = a.copy()
        b["fold"] = 1
        self.assertEqual(compute_diff_metrics(a, b)[0], 0)
        b["id"] = [1, 3]
        with self.assertRaises(ValueError):
            compute_diff_metrics(a, b)
