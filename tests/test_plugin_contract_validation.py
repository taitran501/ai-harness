"""Tests for TaskPlugin capability and contract validation (P0-02)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import harness.plugins as plugins  # noqa: E402
from harness.plugins import (  # noqa: E402
    DeclarativeTaskPlugin,
    PluginCapabilityError,
    TaskPlugin,
    TaskPluginContext,
    load_task_plugin,
    validate_plugin_contract,
)


class DummyValidPlugin(TaskPlugin):
    plugin_id = "valid_dummy"

    def evaluate_predictions(self, context, ground_truth, predictions):
        return {"primary_score": 1.0}

    def load_training_data(self, context):
        return None


class BadSignaturePlugin(TaskPlugin):
    plugin_id = "bad_sig"

    # evaluate_predictions in TaskPlugin expects (self, context, ground_truth, predictions)
    # Here only 1 argument is accepted!
    def evaluate_predictions(self, context):
        return {}


class NonCallableHookPlugin(TaskPlugin):
    plugin_id = "non_callable"

    # evaluate_predictions is an int, not a method
    evaluate_predictions = 123  # type: ignore[assignment]


class BadCapabilitiesPlugin(TaskPlugin):
    plugin_id = "bad_caps"

    def capabilities(self):  # type: ignore[override]
        return "not a dict"


class TestPluginContractValidation(unittest.TestCase):
    def setUp(self):
        self.ctx = TaskPluginContext(
            Path("/mock"), Path("/mock/task.yaml"), SimpleNamespace(), "test_plugin"
        )

    def test_declarative_plugin_passes_contract(self):
        plugin = DeclarativeTaskPlugin(self.ctx)
        errors = validate_plugin_contract(plugin, raise_on_error=False)
        self.assertEqual(errors, [])

    def test_valid_custom_plugin_passes(self):
        plugin = DummyValidPlugin()
        errors = validate_plugin_contract(plugin, raise_on_error=True)
        self.assertEqual(errors, [])

    def test_bad_signature_detected_and_raised(self):
        plugin = BadSignaturePlugin()
        errors = validate_plugin_contract(plugin, raise_on_error=False)
        self.assertTrue(any("Hook 'evaluate_predictions' signature mismatch" in e for e in errors))

        with self.assertRaises(PluginCapabilityError) as cm:
            validate_plugin_contract(plugin, raise_on_error=True)
        self.assertIn("evaluate_predictions", str(cm.exception))

    def test_non_callable_hook_detected(self):
        plugin = NonCallableHookPlugin()
        errors = validate_plugin_contract(plugin, raise_on_error=False)
        self.assertTrue(any("is not callable" in e for e in errors))

    def test_invalid_capabilities_detected(self):
        plugin = BadCapabilitiesPlugin()
        errors = validate_plugin_contract(plugin, raise_on_error=False)
        self.assertTrue(any("capabilities() must return a dict" in e for e in errors))

    def test_load_task_plugin_enforces_contract(self):
        ep = mock.Mock()
        ep.name = "bad_sig"
        ep.load.return_value = BadSignaturePlugin
        with mock.patch.object(plugins, "_plugin_entry_points", return_value=[ep]):
            with self.assertRaises(PluginCapabilityError) as cm:
                load_task_plugin("bad_sig", self.ctx)
            self.assertIn("failed contract validation", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
