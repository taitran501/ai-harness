"""Regression: ResearchMemory.sync() cache must never serve stale claims.

The sync cache previously keyed on (folder, mtime of manifest/config/metrics)
only, so:

1. tampering ANY other sealed artifact (e.g. OOF predictions) between two
   syncs left a cached ``verified_full`` record that a direct ``validate_run``
   rejects, and
2. a second task contract in the same process reused the first contract's
   cached record even though its own ``validate_run`` would reject the run.

Both behaviors made ``memory.sync()`` (the HUD/restart view of the evidence)
disagree with ``validate_run`` (the gate), so a tampered run appeared
``verified_full`` in the HUD. See docs/changes/test-isolation-and-sync-cache.md.
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

import harness.research_memory as research_memory
from harness.artifacts import validate_run
from harness.research_memory import ResearchMemory
from harness.task_spec import UniversalTaskSpec
from tests.evidence_fixtures import seal_fixture


def task_document(name: str = "sync-cache-text") -> dict:
    return {
        "task": {
            "name": name,
            "modality": "text",
            "primary_metric": "accuracy",
            "direction": "maximize",
            "data": {
                "id_column": "id",
                "train_path": "dataset/train.csv",
                "fold_column": "fold",
            },
            "validation": {"strategy": "kfold", "num_folds": 2},
            "inputs": [
                {"name": "text", "type": "text", "source_columns": ["text"]}
            ],
            "subtasks": [
                {
                    "id": "sentiment",
                    "column": "target",
                    "task_type": "binary_classification",
                    "classes": [0, 1],
                    "primary_metric": "accuracy",
                }
            ],
        }
    }


class ResearchSyncCache(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "configs").mkdir()
        (self.root / "configs/task.yaml").write_text(
            yaml.safe_dump(task_document())
        )
        self.spec = UniversalTaskSpec.from_dict(task_document())
        self.artifacts = self.root / "artifacts"
        self.artifacts.mkdir()
        self.memory = ResearchMemory(self.root, self.artifacts, self.spec)
        # One sealed, verified full run.
        self.folder = self.artifacts / "exp001"
        seal_fixture(self.folder, self.spec, score=0.81)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_tampered_oof_not_served_from_stale_cache(self) -> None:
        """Reviewer C3 scenario: HUD must not keep showing verified_full."""
        first = self.memory.sync()
        self.assertEqual(len(first), 1)
        self.assertEqual(first[0]["evidence_status"], "verified_full")

        # Tamper a non-manifest/config/metrics sealed artifact.
        oof = self.folder / "oof_predictions.csv"
        oof.write_text(oof.read_text(encoding="utf-8") + "3,1\n", encoding="utf-8")

        # The gate itself must reject the tampered run...
        with self.assertRaises(ValueError):
            validate_run(self.folder, task_spec=self.spec)
        # ...and sync() must not keep serving the cached verified_full record.
        second = self.memory.sync()
        self.assertNotIn(
            "exp001", [r["exp_id"] for r in second],
            "sync() served a stale verified_full record for a tampered run",
        )

    def test_second_contract_cannot_reuse_first_cache(self) -> None:
        """Cross-spec staleness: a new contract must revalidate, not inherit."""
        self.memory.sync()
        self.assertEqual(len(self.memory.sync()), 1)  # cache warm

        other_doc = json.loads(json.dumps(task_document()))
        other_doc["task"]["name"] = "a-different-contract"
        other_spec = UniversalTaskSpec.from_dict(other_doc)
        other = ResearchMemory(self.root, self.artifacts, other_spec)

        # Under its own contract, this run does not validate...
        with self.assertRaises(ValueError):
            validate_run(self.folder, task_spec=other_spec)
        # ...so its sync() must not inherit the first contract's cached claim.
        records = other.sync()
        self.assertEqual(records, [], "stale cross-spec cache record served")

    def test_cache_hit_when_unchanged(self) -> None:
        """The cache must still work: unchanged folder => one validate_run."""
        with patch.object(research_memory, "validate_run", wraps=research_memory.validate_run) as spy:
            self.memory.sync()
            self.assertEqual(len(self.memory.sync()), 1)
        self.assertEqual(
            spy.call_count, 1,
            "sync() revalidated an unchanged run; cache hit is gone",
        )

    def test_repaired_file_invalidation(self) -> None:
        """Any top-level file change (including metrics.json) invalidates the cache."""
        self.memory.sync()
        metrics = self.folder / "metrics.json"
        data = json.loads(metrics.read_text(encoding="utf-8"))
        data["primary_score"] = 0.99
        metrics.write_text(json.dumps(data), encoding="utf-8")

        with self.assertRaises(ValueError):
            validate_run(self.folder, task_spec=self.spec)
        self.assertEqual(self.memory.sync(), [])


if __name__ == "__main__":
    unittest.main()
