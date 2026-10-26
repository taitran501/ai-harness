"""Regression tests for research-memory scope stability.

Real failure observed in the AIMO workspace (2026-10-16): editing
``configs/task.yaml`` (data path + hygiene allow-list) changed the
task-metadata digest used as the research-memory scope key, which
silently orphaned every insight/decision/checkpoint recorded by earlier
sessions — ``record_research_decision`` and ``record_research_checkpoint``
then rejected the real insight IDs with "must identify an insight in the
current task/scorer scope", and the context HUD reported 0 insights.

The scope key must therefore be stable across benign task-config edits
(task identity + scorer only), and previously-known scopes in the same
task's memory file must stay addressable (one-time legacy adoption),
while different tasks sharing one artifacts directory stay isolated.
"""

import json
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tests"))

from harness.research_memory import ResearchMemory  # noqa: E402
from harness.task_spec import UniversalTaskSpec  # noqa: E402
from tests.fixtures import task_document  # noqa: E402


def _make(doc, workspace_dir):
    root = workspace_dir
    (root / "configs").mkdir(parents=True, exist_ok=True)
    import yaml

    (root / "configs" / "task.yaml").write_text(yaml.safe_dump(doc))
    spec = UniversalTaskSpec.from_dict(doc)
    return ResearchMemory(root, root / "artifacts", spec)


class TestResearchMemoryScopeStability(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = None
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.doc = task_document()
        self.memory = _make(self.doc, self.root)

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
        edited = _make(doc, self.root)
        self.assertEqual(edited.scope, self.memory.scope)
        self.assertEqual(edited.context()["entry_count"], 1,
                         "benign task.yaml edit orphaned the recorded insight")

    def test_scope_changes_when_task_identity_changes(self) -> None:
        other = _make(self._spec_with(id="another-task-id"), self.root)
        self.assertNotEqual(other.scope, self.memory.scope)

    def test_legacy_file_entries_adopted_after_migration(self) -> None:
        """Entries written under an older scope key stay addressable."""
        self.memory.record_insight(title="t", observation="o",
                                   mechanism="m", next_action="n")
        data = json.loads(self.memory.path.read_text())
        # Simulate a pre-migration file: the entry carries an older scope
        # value and no scope_history exists.
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
        # After the first migrated write, the legacy scope is attributed to
        # the migrating task's bucket and cross-task isolation still holds.
        migrated = json.loads(self.memory.path.read_text())
        self.assertIn(old_scope, migrated["scope_history"]["fixture-task"])
        outsider = _make(self._spec_with(id="another-task-id"), self.root)
        self.assertEqual(outsider.context()["entry_count"], 0,
                         "legacy adoption leaked entries into another task")

    def test_cross_task_isolation_persists_after_migration(self) -> None:
        self.memory.record_insight(title="t", observation="o",
                                   mechanism="m", next_action="n")
        outsider = _make(self._spec_with(id="another-task-id"), self.root)
        self.assertEqual(outsider.context()["entry_count"], 0)
        # And the original task still sees its entry.
        self.assertEqual(len(self.memory.entries()), 1)


if __name__ == "__main__":
    unittest.main()
