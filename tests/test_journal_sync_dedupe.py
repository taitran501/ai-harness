"""Regression tests for the TreeJournal sync-dedup optimization.

``TreeJournal.__init__`` previously fully validated every journal node
(``validate_run`` with task_spec) and then ``sync_from_artifacts()`` fully
re-validated every on-disk run directory and overwrote the same nodes.
``AgentBridge.execute_action`` additionally built two such journals per
action. The optimization skips the redundant pre-pass for on-disk nodes
(auto_sync=True) and reuses the action's journal for the post-action
snapshot.

These tests lock in the safety invariants of that change:

1. Every on-disk run is still fully validated once per auto-synced
   construction — the gate is intact: a tampered on-disk run is still
   rejected and marked ``LEGACY_UNVERIFIED``.
2. Orphan journal nodes (artifact dir gone from disk) are still marked
   ``LEGACY_UNVERIFIED``.
3. The ``auto_sync=False`` path behaves exactly as before (full per-node
   validation, no artifacts walk).
4. The post-action snapshot built from the reused journal equals the
   snapshot a brand-new ``TreeJournal`` built after the action would
   report (journal-reuse equivalence).
"""

from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from harness.agent_bridge import AgentBridge
from harness.plugins import TaskPluginContext, load_task_plugin
from harness.project import HarnessProject, attach_project
from harness.task_spec import load_active_task_spec
from harness.tree_journal import TreeJournal
from tests.evidence_fixtures import seal_fixture
from tests.fixtures import TRAIN_CSV_4ROW, task_document

SOURCE_ROOT = Path(__file__).resolve().parent.parent


def _spec_and_plugin(root: Path):
    task_config = root / "configs" / "task.yaml"
    spec = load_active_task_spec(task_config)
    plugin = load_task_plugin(
        "default",
        TaskPluginContext(
            project_root=root,
            task_config_path=task_config,
            task_spec=spec,
            plugin_id="default",
        ),
    )
    return spec, plugin


class JournalSyncDedupTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="aih_journal_dedupe_")
        self.root = Path(self._tmp.name)
        (self.root / "configs").mkdir()
        (self.root / "dataset").mkdir()
        (self.root / "dataset" / "train.csv").write_text(TRAIN_CSV_4ROW, encoding="utf-8")
        task_config = self.root / "configs" / "task.yaml"
        task_config.write_text(yaml.safe_dump(task_document(), sort_keys=False), encoding="utf-8")
        attach_project(self.root, task_config)
        self.project = HarnessProject.discover(self.root)
        self.artifacts = self.project.artifacts_dir
        self.journal_path = self.artifacts / "journal.json"
        self.spec, self.plugin = _spec_and_plugin(self.root)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _seal_run(self, exp_id: str, score: float = 0.8) -> Path:
        folder = self.artifacts / exp_id
        seal_fixture(folder, self.spec, score=score, job_id="job-" + exp_id)
        return folder

    def _write_journal(self, nodes: list) -> None:
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        self.journal_path.write_text(json.dumps({"nodes": nodes}), encoding="utf-8")

    def _journal_node(self, exp_id: str, artifact_dir: str, score: float = 0.84) -> dict:
        return {
            "id": exp_id,
            "parent_id": None,
            "branch": "feat",
            "model_family": "linear",
            "metrics": {"primary_score": score},
            "primary_metric_name": "accuracy",
            "direction": "maximize",
            "verified": True,
            "eligible": True,
            "status": "COMPLETED",
            "artifact_dir": artifact_dir,
        }

    def test_on_disk_run_still_validated_by_auto_sync(self):
        """A healthy on-disk run must come out verified after auto-sync."""
        self._seal_run("exp010", score=0.81)
        journal = TreeJournal(
            journal_path=self.journal_path,
            artifacts_dir=self.artifacts,
            auto_sync=True,
            task_spec=self.spec,
            task_plugin=self.plugin,
        )
        node = journal.nodes["exp010"]
        self.assertTrue(node.verified)
        self.assertTrue(node.eligible)
        self.assertEqual(node.status, "COMPLETED")

    def test_tampered_on_disk_run_still_rejected(self):
        """The dedup must not weaken the gate: a tampered on-disk run whose
        manifest hash no longer matches must be marked LEGACY_UNVERIFIED."""
        folder = self._seal_run("exp011", score=0.82)
        # Tamper after sealing: metrics.json no longer matches its sealed hash.
        (folder / "metrics.json").write_text(
            json.dumps({"primary_score": 0.99, "tampered": True}), encoding="utf-8"
        )
        journal = TreeJournal(
            journal_path=self.journal_path,
            artifacts_dir=self.artifacts,
            auto_sync=True,
            task_spec=self.spec,
            task_plugin=self.plugin,
        )
        node = journal.nodes["exp011"]
        self.assertFalse(node.verified)
        self.assertFalse(node.eligible)
        self.assertEqual(node.status, "LEGACY_UNVERIFIED")

    def test_orphan_journal_node_marked_unverified(self):
        """A journal node whose artifact dir is gone from disk must still be
        marked LEGACY_UNVERIFIED (the only nodes that keep the full check)."""
        self._seal_run("exp012", score=0.83)
        gone = self.artifacts / "exp013"
        seal_fixture(gone, self.spec, score=0.84)
        self._write_journal([self._journal_node("exp013", str(gone), score=0.84)])
        # Now delete the orphaned run dir from disk.
        shutil.rmtree(gone)
        journal = TreeJournal(
            journal_path=self.journal_path,
            artifacts_dir=self.artifacts,
            auto_sync=True,
            task_spec=self.spec,
            task_plugin=self.plugin,
        )
        orphan = journal.nodes["exp013"]
        self.assertFalse(orphan.verified)
        self.assertEqual(orphan.status, "LEGACY_UNVERIFIED")
        # The healthy on-disk run is unaffected and verified.
        self.assertTrue(journal.nodes["exp012"].verified)

    def test_auto_sync_false_keeps_full_per_node_check(self):
        """auto_sync=False must retain the pre-optimization behavior: full
        per-node validation without any artifacts-dir walk."""
        self._seal_run("exp014", score=0.85)
        orphan_dir = self.artifacts / "exp015"
        self._write_journal([self._journal_node("exp015", str(orphan_dir), score=0.85)])
        with patch(
            "harness.artifacts.validate_run", side_effect=ValueError("missing manifest v2")
        ) as mocked:
            journal = TreeJournal(
                journal_path=self.journal_path,
                artifacts_dir=self.artifacts,
                auto_sync=False,
                task_spec=self.spec,
                task_plugin=self.plugin,
            )
        # The journal node's (missing) artifact dir was fully checked...
        mocked.assert_called_once()
        self.assertFalse(journal.nodes["exp015"].verified)
        self.assertEqual(journal.nodes["exp015"].status, "LEGACY_UNVERIFIED")
        # ...and no on-disk discovery happened at all.
        self.assertNotIn("exp014", journal.nodes)


class ExecuteActionSnapshotReuseTests(unittest.TestCase):
    """The post-action HUD snapshot (reused journal + one sync) must equal
    what a fresh TreeJournal built after the action would report."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="aih_snap_reuse_")
        self.root = Path(self._tmp.name)
        shutil.copytree(SOURCE_ROOT / "configs", self.root / "configs")
        self.bridge = AgentBridge(self.root)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_final_state_matches_fresh_journal_snapshot(self):
        # One sealed baseline run so the snapshot has real content.
        spec, plugin = _spec_and_plugin(self.root)
        seal_fixture(
            self.bridge.project.artifacts_dir / "exp020",
            spec,
            score=0.88,
            job_id="job-exp020",
        )
        sess, obs = self.bridge.execute_action(
            action="record_research_decision",
            params={
                "outcome": "inconclusive",
                "conclusion": "Baseline verified on fixture workspace.",
            },
        )
        self.assertEqual(obs.status, "SUCCESS", obs.summary)
        fresh = TreeJournal(
            journal_path=self.bridge.project.artifacts_dir / "journal.json",
            artifacts_dir=self.bridge.project.artifacts_dir,
            auto_sync=True,
            task_spec=self.bridge.task_spec,
            task_plugin=self.bridge.task_plugin,
        )
        reference = self.bridge.comp_state.evaluate_journey_snapshot(
            fresh, self.bridge._new_submission_manager()
        )
        self.assertEqual(sess.final_state, reference.get("hud", {}))


if __name__ == "__main__":
    unittest.main()
