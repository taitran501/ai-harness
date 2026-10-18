"""Regression tests for the four F821 (undefined name) latent bugs.

All four pre-dated the Wave 1 cleanup and were invisible because:
- the dispatcher converts handler exceptions into FAILED observations
  (so the guaranteed NameError in _handle_inspect_experiment only surfaced
  as a vague "raised exception" summary), and
- the three missing imports were masked by lazy evaluation
  (PEP 563 annotations / local variable annotations / dead call paths).
"""

from __future__ import annotations

import shutil
import tempfile
import types
import unittest
from pathlib import Path
from typing import get_type_hints

import harness.cloud_runner as cloud_runner
import harness.status_report as status_report
from harness.cloud_inspector import CloudPreflightInspector
from harness.templates.modal_template import push_dataset_to_modal as _template_push

SOURCE_ROOT = Path(__file__).resolve().parent.parent


class TestInspectExperimentHandler(unittest.TestCase):
    """inspect_experiment must actually work, not fail on an undefined `pool`."""

    def setUp(self) -> None:
        self._ws = tempfile.TemporaryDirectory(prefix="f821_br_")
        self.root = Path(self._ws.name)
        shutil.copytree(SOURCE_ROOT / "configs", self.root / "configs", dirs_exist_ok=True)
        from harness.agent_bridge import AgentBridge

        self.bridge = AgentBridge(self.root)

    def tearDown(self) -> None:
        self._ws.cleanup()

    def _dispatch(self, action: str, params: dict) -> object:
        journal = self.bridge._new_journal(auto_sync=False)
        call = types.SimpleNamespace(
            step_index=1, action=action, params=params
        )
        return self.bridge._dispatch_typed_action(
            call, types.SimpleNamespace(), journal, None
        )

    def test_inspect_experiment_succeeds_with_explicit_id(self) -> None:
        obs = self._dispatch("inspect_experiment", {"exp_id": "baseline"})
        self.assertEqual(obs.status, "SUCCESS", msg=obs.summary)
        self.assertIn("baseline", obs.summary)

    def test_inspect_experiment_succeeds_with_default_id(self) -> None:
        # No exp_id -> defaults to the candidate pool's overall best
        # (or the baseline gate). Previously: NameError('pool') because the
        # pool was referenced before its assignment.
        obs = self._dispatch("inspect_experiment", {})
        self.assertEqual(obs.status, "SUCCESS", msg=obs.summary)
        self.assertIn("Inspected", obs.summary)


class TestMissingImportsBound(unittest.TestCase):
    def test_cloud_runner_push_dataset_to_modal_is_bound(self) -> None:
        # cloud_runner.run_via_modal's upload path calls this by bare name.
        self.assertIs(cloud_runner.push_dataset_to_modal, _template_push)

    def test_cloud_inspector_annotations_resolve(self) -> None:
        # get_type_hints forces evaluation of the signature annotations;
        # without `Optional` imported this raises NameError.
        hints = get_type_hints(CloudPreflightInspector.run_full_preflight_check)
        self.assertIn("project_root", hints)

    def test_status_report_List_name_resolves(self) -> None:
        self.assertIn("List", vars(status_report))
        self.assertIsNotNone(getattr(status_report, "List"))


if __name__ == "__main__":
    unittest.main()
