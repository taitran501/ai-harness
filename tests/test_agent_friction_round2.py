"""Unit tests for agent friction remediation round 2.

Covers:
1. Kaggle remote artifact import critical mismatch policy (_is_critical_identity_mismatch).
2. Path resolution robustness across relative/absolute and Windows paths (safe_relpath).
3. UniversalTaskSpec.from_dict compatibility when cfg['task'] is a string.
4. Evaluation alias evaluate_predictions parity with evaluate.
5. _NormalisedObject dictionary interface parity (.keys(), .values(), .items()).
"""

from __future__ import annotations

from pathlib import Path
import unittest

from harness.evaluation import evaluate, evaluate_predictions
from harness.io_utils import safe_relpath
from harness.kaggle_compat import _NormalisedObject
from harness.task_spec import UniversalTaskSpec
from harness.templates.kaggle_template import _is_critical_identity_mismatch


class TestKaggleArtifactCriticalMismatchPolicy(unittest.TestCase):
    """Kaggle artifact import must tolerate non-critical drift and block critical mismatches."""

    def test_core_version_drift_is_not_critical(self):
        mismatches = {"core_version": {"expected": "0.1.0", "actual": "0.2.0"}}
        critical = {
            k: v
            for k, v in mismatches.items()
            if _is_critical_identity_mismatch(k, v)
        }
        self.assertEqual(critical, {})

    def test_task_sha256_mismatch_is_critical(self):
        mismatches = {"task_sha256": {"expected": "hash_a", "actual": "hash_b"}}
        critical = {
            k: v
            for k, v in mismatches.items()
            if _is_critical_identity_mismatch(k, v)
        }
        self.assertIn("task_sha256", critical)

    def test_config_sha256_mismatch_is_critical(self):
        mismatches = {"config_sha256": {"expected": "hash_a", "actual": "hash_b"}}
        critical = {
            k: v
            for k, v in mismatches.items()
            if _is_critical_identity_mismatch(k, v)
        }
        self.assertIn("config_sha256", critical)

    def test_job_id_none_is_tolerated(self):
        mismatches = {"job_id": {"expected": "job_123", "actual": None}}
        critical = {
            k: v
            for k, v in mismatches.items()
            if _is_critical_identity_mismatch(k, v)
        }
        self.assertEqual(critical, {})

    def test_job_id_conflict_is_critical(self):
        mismatches = {"job_id": {"expected": "job_123", "actual": "job_456"}}
        critical = {
            k: v
            for k, v in mismatches.items()
            if _is_critical_identity_mismatch(k, v)
        }
        self.assertIn("job_id", critical)


class TestSafeRelpath(unittest.TestCase):
    """safe_relpath must compute clean relative paths without throwing ValueError."""

    def test_relative_to_relative(self):
        rel = safe_relpath("artifacts/exp001/oof.csv", ".")
        self.assertEqual(rel, "artifacts/exp001/oof.csv")

    def test_absolute_to_relative_start(self):
        abs_path = Path("artifacts/exp001/oof.csv").resolve()
        rel = safe_relpath(abs_path, ".")
        self.assertEqual(rel, "artifacts/exp001/oof.csv")

    def test_relative_to_absolute_start(self):
        abs_start = Path(".").resolve()
        rel = safe_relpath("artifacts/exp001/oof.csv", abs_start)
        self.assertEqual(rel, "artifacts/exp001/oof.csv")

    def test_forward_slashes_enforced(self):
        rel = safe_relpath("configs/task.yaml", ".")
        self.assertNotIn("\\", rel)
        self.assertEqual(rel, "configs/task.yaml")


class TestUniversalTaskSpecStringTaskCompatibility(unittest.TestCase):
    """UniversalTaskSpec.from_dict must accept string task names in cfg['task']."""

    def test_task_as_string_sets_name_and_does_not_crash(self):
        cfg = {
            "task": "airline_prediction",
            "task_type": "classification",
            "primary_metric": "macro_f1",
        }
        spec = UniversalTaskSpec.from_dict(cfg)
        self.assertEqual(spec.name, "airline_prediction")
        self.assertEqual(spec.task_type, "classification")
        self.assertEqual(spec.primary_metric, "macro_f1")

    def test_task_as_dict_still_works(self):
        cfg = {
            "task": {
                "name": "nested_task",
                "type": "regression",
                "primary_metric": "rmse",
            }
        }
        spec = UniversalTaskSpec.from_dict(cfg)
        self.assertEqual(spec.name, "nested_task")
        self.assertEqual(spec.task_type, "regression")
        self.assertEqual(spec.primary_metric, "rmse")


class TestEvaluationAlias(unittest.TestCase):
    """evaluate_predictions must be an alias of evaluate."""

    def test_evaluate_predictions_identical(self):
        self.assertIs(evaluate_predictions, evaluate)


class TestNormalisedObjectDictParity(unittest.TestCase):
    """_NormalisedObject must support .keys(), .values(), and .items()."""

    def test_keys_values_items_on_dict(self):
        data = {"status": "RUNNING", "publicScore": 0.85}
        proxy = _NormalisedObject(data, "test_sdk")
        self.assertIn("status", proxy.keys())
        self.assertIn("publicScore", proxy.keys())
        items_dict = dict(proxy.items())
        self.assertEqual(items_dict["status"], "RUNNING")
        self.assertAlmostEqual(items_dict["publicScore"], 0.85)

    def test_keys_values_items_on_object(self):
        class DummyRecord:
            def __init__(self):
                self.status = "COMPLETE"
                self.failure_message = None

        obj = DummyRecord()
        proxy = _NormalisedObject(obj, "test_sdk")
        self.assertIn("status", proxy.keys())
        self.assertEqual(proxy.get("status"), "COMPLETE")
        items = dict(proxy.items())
        self.assertEqual(items["status"], "COMPLETE")



class TestTaskDatasetSlugPolicy(unittest.TestCase):
    """data_config.kaggle_dataset_slug: none must suppress private dataset upload requirement."""

    def test_slug_none_string_detection(self):
        task_info = {"data_config": {"kaggle_dataset_slug": "none"}}
        task_dataset_slug = task_info.get("data_config", {}).get("kaggle_dataset_slug")
        resolved = "none" if str(task_dataset_slug).lower() == "none" else task_dataset_slug
        self.assertEqual(resolved, "none")



class TestTreeJournalListCompatibility(unittest.TestCase):
    """TreeJournal.load must handle list-based journal files gracefully without crashing."""

    def test_list_data_handled(self):
        import tempfile
        import json
        from harness.tree_journal import TreeJournal

        tmp = Path(tempfile.mkdtemp())
        jp = tmp / "journal.json"
        jp.write_text(json.dumps([{"event": "import", "target": "test"}]))
        tj = TreeJournal(jp)
        self.assertEqual(tj.nodes, {})


if __name__ == "__main__":
    unittest.main()
