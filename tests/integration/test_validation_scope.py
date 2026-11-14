import unittest
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from harness.run_data import RunDataMixin
from harness.validation_scope import scope_record
from tests.fixtures import task


class ValidationScopeTests(unittest.TestCase):
    def resolver(self, folds=None, complete=None, cli=None):
        runner = RunDataMixin()
        runner.task_spec = task()
        runner.task_spec.validation.complete_folds = complete
        runner.task_plugin = SimpleNamespace(build_validation_splits=lambda *a: None, validate_split=lambda *a: None)
        runner.plugin_context = None
        runner.seed = 13
        runner.cfg = {"data": {} if folds is None else {"folds": folds}}
        runner.args = SimpleNamespace(fold=cli)
        runner.manifest = {}
        return runner

    def frame(self):
        return pd.DataFrame({"id": range(20), "group": range(20), "target": range(20), "x": range(20), "fold": [i % 5 for i in range(20)]})

    def test_partial_selection_keeps_complete_protocol(self):
        for folds, cli in (([0, 1], None), (None, 0)):
            with self.subTest(folds=folds, cli=cli):
                runner = self.resolver(folds=folds, cli=cli)
                result = runner._resolve_validation_splits(self.frame(), "kfold")
                self.assertLess(len(result), len(runner._required_validation_splits))
                self.assertEqual(len(runner.required_validation_ids), 20)
                self.assertIs(result, runner._resolve_validation_splits(self.frame(), "kfold"))

    def test_complete_task_subset_and_reordered_full_selection(self):
        for folds, complete in (([0], [0]), ([4, 3, 2, 1, 0], None)):
            runner = self.resolver(folds=folds, complete=complete)
            result = runner._resolve_validation_splits(self.frame(), "kfold")
            self.assertEqual({p[0] for p in result}, {p[0] for p in runner._required_validation_splits})

    def test_unknown_duplicate_and_protocol_override_fail(self):
        for folds in ([0, 0], [8]):
            with self.assertRaises(ValueError):
                self.resolver(folds=folds)._resolve_validation_splits(self.frame(), "kfold")
        runner = self.resolver()
        runner.cfg["validation"] = {"complete_folds": [0]}
        with self.assertRaisesRegex(ValueError, "cannot redefine"):
            runner._resolve_validation_splits(self.frame(), "kfold")

    def test_completeness_is_derived_from_observations(self):
        parts = [{"fold": "0", "validation_ids": [1, 2]}, {"fold": "1", "validation_ids": [3]}]
        self.assertFalse(scope_record(parts, [0], [1, 2], [0])["complete"])
        self.assertFalse(scope_record(parts, [0, 1], [1, 2], [0, 1])["complete"])
        self.assertTrue(scope_record(parts, [1, 0], [3, 2, 1], [1, 0])["complete"])

    def run_workspace(self, tmp, folds=None):
        from harness.runner import UniversalRunner
        from tests.fixtures import make_workspace
        spec = task()
        spec.metadata["validation"] = {"strategy": "kfold", "num_folds": 5}
        frame = self.frame().assign(category="a")
        root, _ = make_workspace(root_path=Path(tmp), task={"task": spec.metadata}, train_csv=frame.to_csv(index=False))
        cfg = {"experiment": {"id": "baseline", "branch": "baseline", "parent_id": None,
                "problem_addressed": "Verify validation completeness", "hypothesis": "Scope remains separate from fold selection",
                "intervention": "Execute baseline under declared folds"}, "model": {"name": "baseline"}}
        if folds is not None:
            cfg["data"] = {"folds": folds}
        args = SimpleNamespace(project_root=str(root), task_config=None, fold=None, debug=False, probe_samples=None)
        return UniversalRunner(cfg, "baseline.yaml", args)

    def test_partial_run_is_probe_and_forged_full_tag_fails(self):
        from harness.artifacts import validate_run
        from harness.research_memory import ResearchMemory
        with tempfile.TemporaryDirectory() as tmp:
            runner = self.run_workspace(tmp, [0, 1])
            runner.run()
            self.assertEqual(runner.run_kind, "probe")
            self.assertFalse((runner.artifacts_dir / runner.exp_id).exists())
            self.assertFalse(ResearchMemory(runner.project_root).sync())
            path = runner.exp_dir / "artifact_manifest.json"
            manifest = json.loads(path.read_text())
            manifest["run_kind"] = "full"
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "VALIDATION_SCOPE_INCOMPLETE"):
                validate_run(runner.exp_dir, task_spec=runner.task_spec, plugin=runner.task_plugin)

    def test_post_publication_journal_failure_retains_completed_run(self):
        from harness.artifacts import validate_run
        with tempfile.TemporaryDirectory() as tmp:
            runner = self.run_workspace(tmp)
            with patch("harness.runner.TreeJournal.sync_from_artifacts", side_effect=RuntimeError("index unavailable")):
                with self.assertRaisesRegex(RuntimeError, "POST_PUBLICATION_INDEXING_FAILED"):
                    runner.run()
            manifest = validate_run(runner.exp_dir, task_spec=runner.task_spec, plugin=runner.task_plugin)
            self.assertEqual(manifest["status"], "COMPLETED")
