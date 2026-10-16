"""Regression tests for the ruff / B10-B16 cleanup PR.

- B10 — EMERGENCY_UNFREEZE audit event is self-contained
        (records the released slots + lock receipt, not only the reason).
- B12 — CloudJobManager.stream_logs resumes from a byte offset using a
        binary seek (text-mode seek with a non-zero offset is undefined
        behaviour on CPython).
- B15 — Dispatch preflight (full_data_check=False) must NOT load the
        training frame locally; explicit preflight still does.
- B16 — The generic model factory is competition-agnostic:
        DecomposedMappingEquityModel was moved to the zindi workspace
        task plugin (entry point ``zindi_bias_bounty``).
"""
from __future__ import annotations

import json
import shutil
import unittest

from tests.fixtures import make_workspace


# ---------------------------------------------------------------------------
# B10: emergency_unfreeze audit event is self-contained
# ---------------------------------------------------------------------------
class TestB10UnfreezeAuditSelfContained(unittest.TestCase):
    def test_unfreeze_event_records_released_slots(self):
        from harness.competition_state import CompetitionState

        root, _ = make_workspace()
        try:
            comp = CompetitionState(project_root=root)
            comp.state["freeze_selection"] = {
                "selected_slots": {"slot_1": "cand_alpha", "slot_2": "cand_beta"},
                "slot_1_safe_cv": "cand_alpha",
                "slot_2_aggressive_lb": "cand_beta",
                "frozen_at": "2026-10-17T00:00:00+00:00",
                "freeze_notes": "regression fixture",
                "lock_receipt_path": ".ai-harness/freeze_lock.json",
                "git_commit": "abc1234",
            }
            comp.emergency_unfreeze(reason="regression test unfreeze")

            events = [
                e for e in comp.state["audit_trail"]
                if e["event_type"] == "EMERGENCY_UNFREEZE"
            ]
            self.assertEqual(len(events), 1)
            details = events[0]["details"]
            # The released slots must be recorded on the unfreeze event
            # itself, so the entry stands alone in the audit trail.
            self.assertEqual(
                details["released_slots"],
                {"slot_1": "cand_alpha", "slot_2": "cand_beta"},
            )
            self.assertEqual(
                details["lock_receipt_path"], ".ai-harness/freeze_lock.json"
            )
            self.assertEqual(details["frozen_at_git_commit"], "abc1234")
            self.assertEqual(
                details["previous_freeze"], "2026-10-17T00:00:00+00:00"
            )
            self.assertEqual(details["reason"], "regression test unfreeze")
        finally:
            shutil.rmtree(root, ignore_errors=True)


# ---------------------------------------------------------------------------
# B12: stream_logs binary-seek resume
# ---------------------------------------------------------------------------
class TestB12StreamLogsBinarySeek(unittest.TestCase):
    def test_resume_from_byte_offset(self):
        from harness.cloud_job_runner import CloudJob, CloudJobManager

        root, _ = make_workspace()
        try:
            mgr = CloudJobManager(project_root=root)
            log_rel = "b12_stream.log"
            content = "line one alpha\nline two beta\nline three gamma\n"
            # Binary write: keeps byte offsets exact (write_text would
            # translate \n to CRLF on Windows and shift the offsets).
            (mgr.project_root / log_rel).write_bytes(content.encode("utf-8"))
            job = CloudJob(
                job_id="job_b12_stream",
                backend="modal",
                config_path="configs/exp001.yaml",
                exp_id="exp001",
                status="COMPLETED",
                log_file=log_rel,
            )
            mgr._jobs[job.job_id] = job

            offset = content.index("line two")
            chunks = list(mgr.stream_logs(job.job_id, start_byte=offset))
            text = "".join(
                json.loads(chunk[len("data: "):])["text"]
                for chunk in chunks
                if chunk.startswith("data: ")
            )
            # Resume must start exactly at the byte offset:
            self.assertTrue(text.startswith("line two"))
            self.assertNotIn("line one", text)
            self.assertIn("line three", text)
            # Completion event must still be emitted.
            self.assertTrue(any(chunk.startswith("event: end") for chunk in chunks))
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_resume_with_offset_past_eof_is_safe(self):
        """seek past EOF must not raise and must simply end the stream."""
        from harness.cloud_job_runner import CloudJob, CloudJobManager

        root, _ = make_workspace()
        try:
            mgr = CloudJobManager(project_root=root)
            log_rel = "b12_eof.log"
            content = "only line\n"
            (mgr.project_root / log_rel).write_bytes(content.encode("utf-8"))
            job = CloudJob(
                job_id="job_b12_eof",
                backend="modal",
                config_path="configs/exp001.yaml",
                exp_id="exp001",
                status="COMPLETED",
                log_file=log_rel,
            )
            mgr._jobs[job.job_id] = job

            chunks = list(mgr.stream_logs(job.job_id, start_byte=10_000))
            self.assertTrue(any(chunk.startswith("event: end") for chunk in chunks))
        finally:
            shutil.rmtree(root, ignore_errors=True)


# ---------------------------------------------------------------------------
# B15: dispatch preflight skips the training-frame load
# ---------------------------------------------------------------------------
class TestB15PreflightFastMode(unittest.TestCase):
    def _workspace_with_config(self):
        root, task_config = make_workspace()
        (root / "configs" / "exp001.yaml").write_text(
            "experiment:\n  id: exp001\n  branch: main\nmodel:\n  name: lightgbm\n",
            encoding="utf-8",
        )
        return root, task_config

    def test_full_data_check_flag_controls_frame_load(self):
        import harness.plugins as plugins_mod
        from harness.cloud_inspector import CloudPreflightInspector
        from harness.plugins import TaskPlugin

        root, _ = self._workspace_with_config()
        calls = []
        try:
            class StubPlugin(TaskPlugin):
                def load_training_data(self, context):
                    calls.append(1)
                    return None

            def _stub_loader(plugin_id, context):
                return StubPlugin()

            original = plugins_mod.load_task_plugin
            plugins_mod.load_task_plugin = _stub_loader
            try:
                cfg = root / "configs" / "exp001.yaml"
                # Dispatch mode: the frame must NOT be loaded locally.
                CloudPreflightInspector.run_full_preflight_check(
                    cfg, project_root=root, full_data_check=False
                )
                self.assertEqual(calls, [], "dispatch preflight must not load the frame")
                # Explicit preflight mode: the frame load is kept.
                CloudPreflightInspector.run_full_preflight_check(
                    cfg, project_root=root, full_data_check=True
                )
                self.assertEqual(calls, [1], "explicit preflight must load the frame")
            finally:
                plugins_mod.load_task_plugin = original
        finally:
            shutil.rmtree(root, ignore_errors=True)


# ---------------------------------------------------------------------------
# B16: generic model factory is competition-agnostic
# ---------------------------------------------------------------------------
class TestB16ModelFactoryCompetitionAgnostic(unittest.TestCase):
    def test_zindi_model_names_no_longer_in_generic_factory(self):
        from harness import models
        from harness.task_spec import load_active_task_spec

        root, task_config = make_workspace()
        try:
            spec = load_active_task_spec(task_config)
            zindi_names = [
                "decomposed_mapping_equity",
                "decomposed_gbdt",
                "analytic_decomposed",
                "exact_decomposed",
                "perfect_analytic_equity",
                "calibrated_decomposed_equity",
                "calibrated_equity_disparity",
                "dynamic_divisor_equity",
            ]
            for name in zindi_names:
                with self.assertRaisesRegex(ValueError, "Unsupported built-in model"):
                    models.create_model(spec, name)
            # The class itself is gone from the framework module.
            self.assertFalse(
                hasattr(models, "DecomposedMappingEquityModel"),
                "competition-specific model must not live in harness/models.py",
            )
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_zindi_plugin_serves_model_if_installed(self):
        """Cross-workspace wiring: the zindi entry point serves the model.

        Skipped in environments where the zindi plugin package is not
        installed (e.g. fresh CI); it is installed in the operator env.
        """
        from harness.plugins import list_task_plugins

        ids = {p["id"] for p in list_task_plugins()}
        if "zindi_bias_bounty" not in ids:
            self.skipTest("zindi task plugin not installed in this environment")

        from harness.plugins import TaskPluginContext, load_task_plugin
        from harness.task_spec import load_active_task_spec

        root, task_config = make_workspace()
        try:
            spec = load_active_task_spec(task_config)
            ctx = TaskPluginContext(
                project_root=root,
                task_config_path=task_config,
                task_spec=spec,
                plugin_id="zindi_bias_bounty",
            )
            plugin = load_task_plugin("zindi_bias_bounty", ctx)
            model = plugin.build_model(ctx, "analytic_decomposed", {})
            self.assertIsNotNone(model)
            self.assertEqual(
                type(model).__name__, "DecomposedMappingEquityModel"
            )
            # Out-of-domain names still fall through to the generic factory.
            self.assertIsNone(plugin.build_model(ctx, "lightgbm", {}))
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
