"""Docs <-> code consistency for TaskPlugin, plus the runnable reference adapter."""

from __future__ import annotations

import re
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "examples" / "task_plugin"))

import harness.plugins as plugins  # noqa: E402
from harness.plugins import TaskPlugin, TaskPluginContext, TaskPluginError, load_task_plugin  # noqa: E402

DOC = ROOT / "docs" / "plugin_architecture.md"
STALE_HOOKS = ("load_train_data", "load_test_data", "run_eda")


def _documented_hooks() -> list[str]:
    text = DOC.read_text(encoding="utf-8")
    return re.findall(r"^\| `(\w+)\(\)` \|", text, flags=re.M)


class TestPluginDocs(unittest.TestCase):
    def test_every_documented_hook_exists_on_taskplugin(self):
        hooks = _documented_hooks()
        self.assertTrue(hooks, "hook table not found in plugin_architecture.md")
        for hook in hooks:
            self.assertTrue(
                callable(getattr(TaskPlugin, hook, None)),
                f"docs mention `{hook}()` but TaskPlugin has no such hook",
            )

    def test_docs_do_not_mention_stale_hooks_as_hooks(self):
        for hook in _documented_hooks():
            self.assertNotIn(hook, STALE_HOOKS)

    def test_docs_reference_example_adapter(self):
        self.assertIn("examples/task_plugin/example_adapter.py", DOC.read_text(encoding="utf-8"))
        self.assertTrue((ROOT / "examples" / "task_plugin" / "example_adapter.py").is_file())


class TestExampleAdapter(unittest.TestCase):
    def setUp(self):
        import example_adapter

        self.mod = example_adapter
        self.root = Path(tempfile.mkdtemp(prefix="aih-example-plugin-"))
        (self.root / "dataset").mkdir()
        pd.DataFrame(
            {"id": range(8), "label": [0, 1, 0, 1, 0, 1, 0, 0]}
        ).to_csv(self.root / "dataset" / "train.csv", index=False)
        self.ctx = TaskPluginContext(
            self.root, self.root / "task.yaml", SimpleNamespace(), "example_adapter"
        )

    def _load(self):
        ep = mock.Mock()
        ep.name = "example_adapter"
        ep.load.return_value = self.mod.ExampleAdapterPlugin
        with mock.patch.object(plugins, "_plugin_entry_points", return_value=[ep]):
            return load_task_plugin("example_adapter", self.ctx)

    def test_loads_and_declares_expected_capabilities(self):
        caps = self._load().capabilities()
        for key in (
            "custom_data_loader",
            "custom_metrics",
            "custom_validation_split",
            "submission_builder",
            "submission_validator",
        ):
            self.assertTrue(caps[key], key)
        self.assertFalse(caps["custom_model_factory"])

    def test_data_splits_and_custom_evaluation(self):
        plugin = self._load()
        data = plugin.load_training_data(self.ctx)
        splits = plugin.build_validation_splits(self.ctx, data, "kfold", {"num_folds": 2})
        self.assertEqual(len(splits), 2)
        for _, train, val in splits:
            self.assertFalse(set(train["id"]) & set(val["id"]))
        # Defer unrecognized strategies to core harness defaults
        self.assertIsNone(plugin.build_validation_splits(self.ctx, data, "group_kfold", {}))

        # Perfect predictions -> 1.0; all-zero predictions -> recall (1.0, 0.0) -> 0.5
        perfect = plugin.evaluate_predictions(
            self.ctx, data, {"pred_label": data["label"].tolist()}
        )
        self.assertAlmostEqual(perfect["primary_score"], 1.0)
        zeros = plugin.evaluate_predictions(self.ctx, data, {"pred_label": [0] * len(data)})
        self.assertAlmostEqual(zeros["primary_score"], 0.5)

    def test_submission_roundtrip_and_validation_failure(self):
        plugin = self._load()
        preds = pd.DataFrame({"id": ["a", "b"], "pred_label": [0, 1]})
        out = plugin.build_submission(self.ctx, preds, self.root / "sub.csv")
        plugin.validate_submission(self.ctx, out)

        bad = self.root / "bad.csv"
        pd.DataFrame({"id": ["a"], "label": [7]}).to_csv(bad, index=False)
        with self.assertRaises(TaskPluginError):
            plugin.validate_submission(self.ctx, bad)


if __name__ == "__main__":
    unittest.main()
