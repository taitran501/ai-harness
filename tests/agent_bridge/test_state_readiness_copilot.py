"""PR #C regression tests — state & agent-friction fixes (narrow scope).

Each test locks in one item from the 2026-10-16 module review wave that was
re-verified against the live source before implementation:

- B1  competition_state: a missing/empty declared dataset can no longer read
      as READY with 0 rows (phantom readiness); it is BLOCKED with a
      DATA_MISSING critical warning.
- B2  competition_state: every HUD recommendation id maps to a real typed
      action (no silent fallback to inspect_diagnostics); composite
      ESTABLISH_VERIFIED_BASELINE returns an explicit FAILED naming the
      required sequence.
- B3  agent_bridge: run_smoke_test / run_probe validate config_path the same
      way run_full_validation does (no traversal outside the project root).
- B4  phase_workflow: numbered Markdown headings (## 1. Summary) and bold
      pseudo-headings (**Summary:**) satisfy required_sections.
- B5  kaggle_inspect: kernel_output surfaces kaggle CLI failures (rc != 0)
      as an error instead of silently returning an empty file list.
- B6  competition_state: save() stamps updated_at on every persist; dead
      _save_raw is removed.
- B7  agent_bridge: session actions/observations are capped so long sessions
      do not serialize MB-scale JSON on every save.
- B8  agent_bridge: smoke_runs counts only PASSED smoke tests.
- B9  competition_state/agent_bridge: AgentContextConfig is built and
      updated with a known-field filter (unknown keys no longer crash
      priming render).
- B17 runner: --force help text no longer claims overwrite is possible.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tests.fixtures import make_workspace as _make_workspace

# ---------------------------------------------------------------------------
# B1: data readiness must not read READY on a missing/empty dataset
# ---------------------------------------------------------------------------

class TestDataReadinessMissingDataset(unittest.TestCase):
    def test_missing_dataset_is_blocked_not_ready(self):
        from harness.competition_state import CompetitionState

        root, _ = _make_workspace(with_train_file=False, train_path="dataset/missing.csv")
        comp = CompetitionState(project_root=root)
        report = comp.run_data_readiness_audit()
        self.assertEqual(report["dataset_overview"]["total_rows"], 0)
        self.assertEqual(report["status"], "BLOCKED")
        ids = [w.get("id") for w in report["warnings"]]
        self.assertIn("DATA_MISSING", ids)
        data_missing = next(w for w in report["warnings"] if w["id"] == "DATA_MISSING")
        self.assertEqual(data_missing["severity"], "critical")

    def test_present_dataset_is_not_blocked_by_guard(self):
        from harness.competition_state import CompetitionState

        root, _ = _make_workspace(with_train_file=True)
        comp = CompetitionState(project_root=root)
        report = comp.run_data_readiness_audit()
        self.assertGreater(report["dataset_overview"]["total_rows"], 0)
        self.assertNotEqual(report["status"], "BLOCKED")
        self.assertNotIn(
            "DATA_MISSING", [w.get("id") for w in report["warnings"]]
        )


# ---------------------------------------------------------------------------
# B2: co-pilot action mapping
# ---------------------------------------------------------------------------

_MAPPINGS = {
    "CONFIRM_TASK_SPEC": "confirm_task_spec",
    "CONFIRM_DATA_RISKS": "inspect_data_readiness",
    "LOCK_BASELINE_ANCHOR": "establish_baseline_anchor",
    "BUILD_SPECIALIST_ROUTER": "inspect_diagnostics",
    "FREEZE_FOR_PRIVATE_LB": "freeze_candidate",
    "GENERATE_FINAL_PAPER": "finalize_competition",
}


class TestCopilotActionMapping(unittest.TestCase):
    def _snapshot_for(self, action_id: str) -> dict:
        return {
            "hud": {"what_should_i_do_next": {"action_id": action_id}},
            "candidate_pool": {"baseline_id": "exp001", "baseline_score": 0.5},
        }

    def test_all_hud_recommendation_ids_map_to_real_actions(self):
        from harness.competition_state import CompetitionState

        root, _ = _make_workspace()
        comp = CompetitionState(project_root=root)

        calls = []

        class FakeBridge:
            def __init__(self, *a, **k):
                pass

            def execute_action(self, action, params=None):
                calls.append((action, dict(params or {})))
                return (
                    SimpleNamespace(),
                    SimpleNamespace(
                        __dict__={"status": "SUCCESS", "action": action}
                    ),
                )

        for action_id, expected_action in _MAPPINGS.items():
            calls.clear()
            with (
                patch(
                    "harness.agent_bridge.AgentBridge", FakeBridge
                ),
                patch.object(
                    CompetitionState,
                    "evaluate_journey_snapshot",
                    return_value=self._snapshot_for(action_id),
                ),
            ):
                result = comp.execute_agent_copilot_action(None, None)
            self.assertNotEqual(result.get("status"), "FAILED",
                               f"unexpected FAILED for {action_id}: {result}")
            self.assertIn("observation", result)
            expected_params = (
                {"exp_id": "baseline"}
                if expected_action == "establish_baseline_anchor"
                else {}
            )
            self.assertEqual(calls, [(expected_action, expected_params)])

    def test_composite_baseline_establishment_is_named_not_substituted(self):
        from harness.competition_state import CompetitionState

        root, _ = _make_workspace()
        comp = CompetitionState(project_root=root)

        with (
            patch(
                "harness.agent_bridge.AgentBridge",
                side_effect=AssertionError("bridge must not be used for composite workflow"),
            ),
            patch.object(
                CompetitionState,
                "evaluate_journey_snapshot",
                return_value=self._snapshot_for("ESTABLISH_VERIFIED_BASELINE"),
            ),
        ):
            result = comp.execute_agent_copilot_action(None, None)

        self.assertEqual(result["status"], "FAILED")
        self.assertEqual(result["action_id"], "ESTABLISH_VERIFIED_BASELINE")
        self.assertIn("composite workflow", result["reason"])
        self.assertEqual(result["observation"]["status"], "FAILED")
        self.assertIn(
            "create_experiment -> run_full_validation -> establish_baseline_anchor",
            result["observation"]["summary"],
        )


# ---------------------------------------------------------------------------
# B3 + B8: smoke/probe config validation and smoke_runs counting
# ---------------------------------------------------------------------------

class TestSmokeProbeFriction(unittest.TestCase):
    def _bridge(self, root: Path):
        from harness.agent_bridge import AgentBridge

        return AgentBridge(root)

    def _call(self, bridge, act="run_smoke_test"):
        return SimpleNamespace(step_index=1, action=act, rationale="test")

    def test_smoke_test_rejects_traversal_config_path(self):
        root, _ = _make_workspace()
        bridge = self._bridge(root)
        session = bridge.get_active_or_create_session()
        with self.assertRaises(ValueError):
            bridge._handle_run_smoke_test(
                self._call(bridge),
                session,
                None,
                None,
                "run_smoke_test",
                {"config_path": "../../etc/passwd.yaml"},
                0.0,
            )

    def test_probe_rejects_traversal_config_path(self):
        root, _ = _make_workspace()
        bridge = self._bridge(root)
        session = bridge.get_active_or_create_session()
        with self.assertRaises(ValueError):
            bridge._handle_run_probe(
                self._call(bridge, "run_probe"),
                session,
                None,
                None,
                "run_probe",
                {"config_path": "configs/../../../evil.yaml"},
                0.0,
            )

    def test_smoke_runs_only_counts_passed_runs(self):
        root, _ = _make_workspace()
        bridge = self._bridge(root)
        session = bridge.get_active_or_create_session()
        session.compute_usage = {"smoke_runs": 0}

        with patch.object(
            bridge, "_run_subprocess_safe", return_value=(True, 0, "ok", 0.1)
        ):
            obs = bridge._handle_run_smoke_test(
                self._call(bridge), session, None, None, "run_smoke_test", {}, 0.0
            )
        self.assertEqual(obs.status, "SUCCESS")
        self.assertEqual(session.compute_usage["smoke_runs"], 1)

        with patch.object(
            bridge, "_run_subprocess_safe", return_value=(False, 1, "boom", 0.1)
        ):
            obs = bridge._handle_run_smoke_test(
                self._call(bridge), session, None, None, "run_smoke_test", {}, 0.0
            )
        self.assertEqual(obs.status, "FAILED")
        self.assertEqual(session.compute_usage["smoke_runs"], 1, "FAILED smoke must not consume budget")


# ---------------------------------------------------------------------------
# B4: loose phase heading matching
# ---------------------------------------------------------------------------

class TestPhaseHeadingMatching(unittest.TestCase):
    def test_numbered_and_bold_headings_accepted(self):
        from harness.phase_workflow import _phase_heading_matches

        self.assertTrue(_phase_heading_matches("## Summary", "Summary"))
        self.assertTrue(_phase_heading_matches("## 1. Summary", "Summary"))
        self.assertTrue(_phase_heading_matches("### 2) Analysis - details", "Analysis"))
        self.assertTrue(_phase_heading_matches("**Summary:**", "Summary"))
        self.assertTrue(_phase_heading_matches("#### iii. Evidence", "Evidence"))
        self.assertFalse(_phase_heading_matches("## Data Leakage", "Summary"))
        self.assertFalse(_phase_heading_matches("", "Summary"))

    def test_record_accepts_numbered_sections(self):
        from harness.phase_workflow import PhaseWorkflow

        root, _ = _make_workspace()
        flow = PhaseWorkflow(root)
        response = (
            "# setup\n\n"
            "## 1. Summary\n\nSummary body.\n\n"
            "## 2) Evidence\n\nEvidence body.\n\n"
            "## 3. Decisions\n\nDecision body.\n\n"
            "**Next steps:**\n\nNext step body.\n"
        )
        flow.record("setup", response)
        self.assertTrue(flow.phase_status()["phases"][0]["completed"])


# ---------------------------------------------------------------------------
# B5: kaggle inspect surfaces CLI failures
# ---------------------------------------------------------------------------

class TestKaggleInspectKernelOutput(unittest.TestCase):
    def test_nonzero_returncode_reports_error(self):
        from harness.kaggle_inspect import kernel_output

        with (
            patch(
                "harness.templates.kaggle_template.KaggleKernelManager.get_kernel_status",
                return_value={"kernel_slug": "user/kern", "status": "complete"},
            ),
            patch(
                "subprocess.run",
                return_value=SimpleNamespace(returncode=1, stderr="quota exceeded"),
            ),
        ):
            result = kernel_output(kernel_slug="user/kern")
        self.assertIn("error", result)
        self.assertIn("rc=1", result["error"])
        self.assertIn("quota exceeded", result["error"])

    def test_zero_returncode_has_no_error(self):
        from harness.kaggle_inspect import kernel_output

        with (
            patch(
                "harness.templates.kaggle_template.KaggleKernelManager.get_kernel_status",
                return_value={"kernel_slug": "user/kern", "status": "complete"},
            ),
            patch(
                "subprocess.run",
                return_value=SimpleNamespace(returncode=0, stderr=""),
            ),
        ):
            result = kernel_output(kernel_slug="user/kern")
        self.assertNotIn("error", result)
        self.assertEqual(result["kernel_slug"], "user/kern")


# ---------------------------------------------------------------------------
# B6: save() stamps updated_at; _save_raw removed
# ---------------------------------------------------------------------------

class TestStateSaveStamping(unittest.TestCase):
    def test_save_stamps_updated_at(self):
        from harness.competition_state import CompetitionState

        root, _ = _make_workspace()
        comp = CompetitionState(project_root=root)
        comp.state["mode"] = "competition"
        comp.save()
        self.assertIsNotNone(comp.state.get("updated_at"))
        persisted = json.loads(comp.state_file.read_text(encoding="utf-8"))
        self.assertEqual(persisted.get("updated_at"), comp.state["updated_at"])

    def test_save_raw_dead_code_removed(self):
        from harness.competition_state import CompetitionState

        self.assertFalse(hasattr(CompetitionState, "_save_raw"))


# ---------------------------------------------------------------------------
# B7: session history caps
# ---------------------------------------------------------------------------

class TestSessionHistoryCaps(unittest.TestCase):
    def test_save_session_caps_actions_and_observations(self):
        from harness.agent_bridge import (
            MAX_SESSION_ACTIONS,
            MAX_SESSION_OBSERVATIONS,
            AgentBridge,
        )

        root, _ = _make_workspace()
        bridge = AgentBridge(root)
        session = bridge.get_active_or_create_session()
        for i in range(MAX_SESSION_ACTIONS + 50):
            session.actions.append({"step_index": i, "action": "create_experiment"})
            session.observations.append(
                {
                    "step_index": i,
                    "action": "create_experiment",
                    "status": "SUCCESS",
                    "summary": "ok",
                }
            )

        bridge.save_session(session)

        payload = json.loads(
            (bridge.sessions_dir / f"{session.session_id}.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertLessEqual(len(payload["actions"]), MAX_SESSION_ACTIONS)
        self.assertLessEqual(
            len(payload["observations"]), MAX_SESSION_OBSERVATIONS
        )
        # Most recent entries survive (the tail is kept).
        self.assertEqual(payload["actions"][-1]["step_index"], MAX_SESSION_ACTIONS + 49)
        self.assertEqual(
            payload["observations"][-1]["step_index"], MAX_SESSION_ACTIONS + 49
        )


# ---------------------------------------------------------------------------
# B9: AgentContextConfig unknown-field filtering
# ---------------------------------------------------------------------------

class TestAgentContextConfigFiltering(unittest.TestCase):
    def _comp(self, root):
        from harness.competition_state import CompetitionState

        return CompetitionState(project_root=root)

    def test_render_tolerates_unknown_persisted_keys(self):
        root, _ = _make_workspace()
        comp = self._comp(root)
        comp.state["agent_context_config"]["bogus_unknown_key"] = "x"
        with patch.object(
            type(comp),
            "evaluate_journey_snapshot",
            return_value={
                "hud": {
                    "where_am_i": {
                        "phase_title": "t",
                        "mode": "competition",
                        "active_tracks": {},
                        "phase_question": "q?",
                    },
                    "what_is_current_best": {
                        "baseline": {"id": "exp000", "score": 0.5},
                        "cv_best": {"id": "exp001", "score": 0.6, "delta": 0.1},
                        "public_lb_best": {"label": "Kaggle", "sub_id": None, "score": None},
                        "private_frozen": {"max_candidates": 3, "is_frozen": False},
                    },
                    "what_is_blocking_me": [],
                    "what_should_i_do_next": {
                        "title": "t",
                        "detail": "d",
                        "agent_plan": ["step"],
                    },
                }
            },
        ):
            block = comp.render_agent_context_block(None, None)
        self.assertIsInstance(block, str)
        self.assertGreater(len(block), 0)

    def test_update_workspace_settings_filters_unknown_keys(self):
        from harness.agent_bridge import AgentBridge

        root, _ = _make_workspace()
        bridge = AgentBridge(root)
        session = bridge.get_active_or_create_session()
        call = SimpleNamespace(step_index=1, action="update_workspace_settings", rationale="t")

        obs = bridge._handle_update_workspace_settings(
            call,
            session,
            None,
            None,
            "update_workspace_settings",
            {"agent_context_config": {"inject_phase_hud": False, "evil_key": 1}},
            0.0,
        )
        self.assertEqual(obs.status, "SUCCESS")
        stored = bridge.comp_state.state["agent_context_config"]
        self.assertFalse(stored["inject_phase_hud"])
        self.assertNotIn("evil_key", stored)


# ---------------------------------------------------------------------------
# B17: --force help text
# ---------------------------------------------------------------------------

class TestForceHelpText(unittest.TestCase):
    def test_force_help_does_not_claim_overwrite(self):
        from harness.runner import build_parser

        help_text = build_parser().format_help()
        self.assertIn("NEVER overwritten", help_text)
        self.assertNotIn("Force overwrite existing experiment artifacts", help_text)


if __name__ == "__main__":
    unittest.main()
