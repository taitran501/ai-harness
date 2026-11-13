"""Canonical research scope stability tests for the smoke gate."""
import json
import tempfile
import unittest
from pathlib import Path

import yaml

from harness.research_memory import ResearchMemory
from harness.task_spec import UniversalTaskSpec
from tests.fixtures import task_document


def _make_scope_memory(doc, workspace_dir):
    root = workspace_dir
    (root / "configs").mkdir(parents=True, exist_ok=True)
    (root / "configs" / "task.yaml").write_text(yaml.safe_dump(doc))
    spec = UniversalTaskSpec.from_dict(doc)
    return ResearchMemory(root, root / "artifacts", spec)


class ScopeStabilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.doc = task_document()
        self.memory = _make_scope_memory(self.doc, self.root)

    def tearDown(self) -> None:
        if self._tmp:
            self._tmp.cleanup()

    def _spec_with(self, **task_overrides):
        doc = json.loads(json.dumps(self.doc))
        doc["task"].update(task_overrides)
        return doc

    def test_scope_stable_across_benign_task_edits(self) -> None:
        """Editing data paths / hygiene / name must not orphan entries."""
        self.memory.record_insight(
            title="t", observation="o", mechanism="m", next_action="n"
        )
        doc = self._spec_with(
            name="Fixture Task v2, re-described",
            data_config={"train_path": "dataset/v2/train.csv",
                        "target_column": "label"},
        )
        doc["task"]["hygiene"] = {"allowed_top_dirs": ["baselines", "dist"]}
        edited = _make_scope_memory(doc, self.root)
        self.assertEqual(edited.scope, self.memory.scope)
        self.assertEqual(edited.context()["entry_count"], 1,
                         "benign task.yaml edit orphaned the recorded insight")

    def test_scope_changes_when_task_identity_changes(self) -> None:
        other = _make_scope_memory(self._spec_with(id="another-task-id"), self.root)
        self.assertNotEqual(other.scope, self.memory.scope)

    def test_legacy_file_entries_adopted_after_migration(self) -> None:
        """Entries written under an older scope key stay addressable."""
        self.memory.record_insight(title="t", observation="o",
                                   mechanism="m", next_action="n")
        data = json.loads(self.memory.path.read_text())
        old_scope = "deadbeef" * 8
        for entry in data["entries"]:
            entry["scope"] = old_scope
            if entry["kind"] == "insight":
                entry["next_action"] = "probe"
        data.pop("scope_history", None)
        self.memory.path.write_text(json.dumps(data))
        self.assertEqual(len(self.memory.entries()), 1,
                         "legacy-scope entry lost after scope-key migration")
        insight_id = self.memory.entries()[-1]["id"]
        self.memory.record_decision(
            outcome="inconclusive", conclusion="c", insight_id=insight_id,
            next_action="keep probing", lesson="legacy entries survive migration"
        )
        migrated = json.loads(self.memory.path.read_text())
        self.assertIn(old_scope, migrated["scope_history"]["fixture-task"])
        outsider = _make_scope_memory(self._spec_with(id="another-task-id"), self.root)
        self.assertEqual(outsider.context()["entry_count"], 0,
                         "legacy adoption leaked entries into another task")

    def test_cross_task_isolation_persists_after_migration(self) -> None:
        self.memory.record_insight(title="t", observation="o",
                                   mechanism="m", next_action="n")
        outsider = _make_scope_memory(self._spec_with(id="another-task-id"), self.root)
        self.assertEqual(outsider.context()["entry_count"], 0)
        self.assertEqual(len(self.memory.entries()), 1)
