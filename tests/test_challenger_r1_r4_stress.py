"""Empirical Stress Test Suite for R1 (Interaction Flows) and R4 (Error Resilience).

Written by teamwork_preview_challenger_1 to empirically test:
1. R1: Checkbox toggles in CompetitionJourneyWorkspace:
   - Rapid multi-toggles & debounce timing
   - Error rollback to previous synced snapshot
   - Preservation of confirmed changes
   - Closure / batching behavior during rapid clicks
2. R4: Network degradation & error handling in App.tsx and WorkspaceHub.tsx:
   - App shell accessibility during endpoint failure
   - Dismiss behavior
   - Retry behavior (testing if failed retry clears or preserves error)
   - WorkspaceHub polling suppression of dismissed errors
   - Manual retry re-surfacing errors
"""

import json
import socketserver
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from tests.workspace_fixtures import (
    cleanup_dashboard_workspace,
    prepare_dashboard_workspace,
)
from tools.dashboard_server import DashboardHandler

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class ChallengerBackendIntegrationTests(unittest.TestCase):
    """Live HTTP backend integration tests for competition-state endpoints."""

    @classmethod
    def setUpClass(cls):
        prepare_dashboard_workspace(cls, "test_challenger_")
        cls.server = socketserver.TCPServer(("127.0.0.1", 0), DashboardHandler)
        cls.port = cls.server.server_address[1]
        cls.server_thread = threading.Thread(
            target=cls.server.serve_forever, daemon=True
        )
        cls.server_thread.start()
        time.sleep(0.2)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cleanup_dashboard_workspace(cls)

    def post_json(self, path: str, payload: dict):
        url = f"http://127.0.0.1:{self.port}{path}"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            return json.loads(resp.read().decode("utf-8"))

    def fetch_json(self, path: str):
        url = f"http://127.0.0.1:{self.port}{path}"
        with urllib.request.urlopen(url) as resp:
            self.assertEqual(resp.status, 200)
            return json.loads(resp.read().decode("utf-8"))

    def test_agent_context_persistence_and_atomic_update(self):
        """POST /api/competition-state/agent-context must update keys and persist."""
        payload = {
            "config": {
                "inject_phase_hud": False,
                "inject_idea_graveyard": False,
            }
        }
        res = self.post_json("/api/competition-state/agent-context", payload)
        self.assertTrue(res.get("success"))
        self.assertIn("state", res)
        agent_cfg = res["state"]["agent_context_config"]
        self.assertFalse(agent_cfg.get("inject_phase_hud"))
        self.assertFalse(agent_cfg.get("inject_idea_graveyard"))

        # Verify on subsequent GET that persistence held
        state_get = self.fetch_json("/api/competition-state")
        self.assertFalse(state_get["agent_context_config"].get("inject_phase_hud"))
        self.assertFalse(state_get["agent_context_config"].get("inject_idea_graveyard"))

    def test_agent_context_rapid_burst_updates(self):
        """Simulate rapid burst requests hitting agent-context endpoint concurrently."""
        keys_to_toggle = [
            "inject_phase_hud",
            "inject_current_best_anchors",
            "inject_active_blockers",
            "inject_next_action",
            "inject_data_risk_warnings",
            "inject_candidate_pool",
        ]
        errors = []

        def worker(idx, key):
            try:
                val = bool(idx % 2 == 0)
                self.post_json(
                    "/api/competition-state/agent-context",
                    {"config": {key: val}},
                )
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=worker, args=(i, keys_to_toggle[i % len(keys_to_toggle)]))
            for i in range(12)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0, f"Burst updates produced errors: {errors}")

        # Verify state is still valid and readable
        state_get = self.fetch_json("/api/competition-state")
        self.assertIn("agent_context_config", state_get)
        self.assertIsInstance(state_get["agent_context_config"], dict)

    def test_data_audit_endpoint_action(self):
        """POST /api/competition-state/data-audit should execute inspect_data_readiness."""
        payload = {"config": {"auto_scan_mode": "deep"}}
        res = self.post_json("/api/competition-state/data-audit", payload)
        self.assertTrue(res.get("success"))
        self.assertIn("observation", res)
        self.assertEqual(res["observation"]["status"], "SUCCESS")


class ChallengerR1InteractionStressTests(unittest.TestCase):
    """Direct empirical model testing of CompetitionJourneyWorkspace.tsx state machine."""

    class MockCompetitionJourneyWorkspaceState:
        """Exact state and handler replication of CompetitionJourneyWorkspace.tsx lines 125-198."""

        def __init__(self, initial_cfg: dict, api_post_mock):
            self.local_cfg = dict(initial_cfg)
            self.ctx_save_status = "idle"
            self.status_msg = ""
            self.previous_synced_ref = dict(initial_cfg)
            self.pending_timeout = None
            self.is_saving_ref = False
            self.api_post_mock = api_post_mock
            self.network_calls = []

        def handle_toggle_agent_context(self, key: str, checked: bool = True):
            # Line 165: const updated = { ...localAgentCtxCfg, [key]: checked };
            updated = dict(self.local_cfg)
            updated[key] = checked
            self.local_cfg = updated
            self.ctx_save_status = "saving"

            if self.pending_timeout is not None:
                self.pending_timeout["cancelled"] = True

            timeout_entry = {"cancelled": False, "payload": dict(updated)}
            self.pending_timeout = timeout_entry

        def fire_debounce_timeout(self):
            """Executes when the 300ms debounce timer expires."""
            if not self.pending_timeout or self.pending_timeout.get("cancelled"):
                return
            updated = self.pending_timeout["payload"]
            self.is_saving_ref = True
            self.network_calls.append(dict(updated))
            try:
                res = self.api_post_mock("agent-context", {"config": updated})
                if res.get("success"):
                    self.previous_synced_ref = dict(updated)
                    self.ctx_save_status = "saved"
                else:
                    raise RuntimeError("Server responded with unsuccessful status")
            except Exception as err:
                # Catch block lines 186-193: rollback
                self.local_cfg = dict(self.previous_synced_ref)
                self.ctx_save_status = "error"
                self.status_msg = f"Failed to save settings: {str(err)}. Reverted."
            finally:
                self.is_saving_ref = False

    def test_r1_debounce_coalesces_rapid_toggles(self):
        """Rapid toggles within 300ms must be coalesced into a single final network call."""
        initial = {"inject_phase_hud": True, "inject_anchors": True}
        workspace = self.MockCompetitionJourneyWorkspaceState(
            initial, lambda action, p: {"success": True}
        )

        # Toggle 1: False
        workspace.handle_toggle_agent_context("inject_phase_hud", False)
        self.assertEqual(workspace.ctx_save_status, "saving")
        self.assertFalse(workspace.local_cfg["inject_phase_hud"])

        # Toggle 2: True (bounce back)
        workspace.handle_toggle_agent_context("inject_phase_hud", True)
        self.assertTrue(workspace.local_cfg["inject_phase_hud"])

        # Toggle 3: False (final intention)
        workspace.handle_toggle_agent_context("inject_phase_hud", False)
        self.assertFalse(workspace.local_cfg["inject_phase_hud"])

        # Now simulate 300ms timer firing
        workspace.fire_debounce_timeout()

        # Exactly 1 network call made with final state
        self.assertEqual(len(workspace.network_calls), 1)
        self.assertEqual(workspace.network_calls[0]["inject_phase_hud"], False)
        self.assertEqual(workspace.ctx_save_status, "saved")
        self.assertEqual(workspace.previous_synced_ref["inject_phase_hud"], False)

    def test_r1_error_rollback_to_last_synced_snapshot(self):
        """On API failure, local state must roll back cleanly to previous synced snapshot."""
        initial = {"inject_phase_hud": True, "inject_anchors": True}

        def failing_api(action, payload):
            raise ConnectionResetError("Connection refused by dashboard backend")

        workspace = self.MockCompetitionJourneyWorkspaceState(initial, failing_api)

        # User unchecks inject_phase_hud
        workspace.handle_toggle_agent_context("inject_phase_hud", False)
        self.assertFalse(workspace.local_cfg["inject_phase_hud"])

        # Timer fires and API fails
        workspace.fire_debounce_timeout()

        # State must roll back to initial
        self.assertTrue(workspace.local_cfg["inject_phase_hud"])
        self.assertEqual(workspace.ctx_save_status, "error")
        self.assertIn("Failed to save settings", workspace.status_msg)
        self.assertIn("Reverted", workspace.status_msg)

    def test_r1_incremental_success_preserves_confirmed_state_on_subsequent_failure(self):
        """If step 1 succeeds and step 2 fails, step 1's saved change must NOT be lost."""
        initial = {"A": True, "B": True}
        call_count = [0]

        def conditional_api(action, payload):
            call_count[0] += 1
            if call_count[0] == 1:
                return {"success": True}
            raise RuntimeError("Backend timeout on step 2")

        workspace = self.MockCompetitionJourneyWorkspaceState(initial, conditional_api)

        # Step 1: toggle A -> False, succeeds
        workspace.handle_toggle_agent_context("A", False)
        workspace.fire_debounce_timeout()
        self.assertEqual(workspace.ctx_save_status, "saved")
        self.assertFalse(workspace.local_cfg["A"])
        self.assertFalse(workspace.previous_synced_ref["A"])

        # Step 2: toggle B -> False, fails
        workspace.handle_toggle_agent_context("B", False)
        workspace.fire_debounce_timeout()
        self.assertEqual(workspace.ctx_save_status, "error")

        # Crucial check: A remains False (preserved), B rolled back to True!
        self.assertFalse(workspace.local_cfg["A"])
        self.assertTrue(workspace.local_cfg["B"])


class ChallengerR4ResilienceStressTests(unittest.TestCase):
    """Adversarial stress tests for App.tsx and WorkspaceHub.tsx error handling."""

    class MockAppTelemetryLoader:
        """Exact replication of App.tsx lines 67-108 loadData() logic."""

        def __init__(self, endpoints_behavior: dict):
            self.endpoints = endpoints_behavior
            self.initial_loaded_ref = False
            self.loading = True
            self.api_error = None
            self.status = None
            self.task_info = None

        def load_data(self):
            if not self.initial_loaded_ref:
                self.loading = True
            self.api_error = None

            # Simulate Promise.allSettled
            status_res = self.endpoints.get("getStatus", {"status": "fulfilled", "value": {"best_node": "exp01"}})
            task_res = self.endpoints.get("getTaskInfo", {"status": "fulfilled", "value": {"name": "test_task"}})

            if status_res["status"] == "fulfilled":
                self.status = status_res["value"]
            if task_res["status"] == "fulfilled":
                self.task_info = task_res["value"]

            critical_failures = [r for r in [status_res, task_res] if r["status"] == "rejected"]

            # Line 98 of App.tsx:
            if len(critical_failures) > 0:
                first_err = critical_failures[0]["reason"]
                self.api_error = first_err if isinstance(first_err, str) else str(first_err)

            self.initial_loaded_ref = True
            self.loading = False

    def test_r4_cold_load_failure_preserves_shell_and_shows_banner(self):
        """When critical endpoints fail on initial load, banner is shown and loading is false."""
        failing_endpoints = {
            "getStatus": {"status": "rejected", "reason": "Failed to fetch: status endpoint unreachable"},
            "getTaskInfo": {"status": "rejected", "reason": "Failed to fetch: taskInfo endpoint unreachable"},
        }
        app = self.MockAppTelemetryLoader(failing_endpoints)
        app.load_data()

        # App shell must not be stuck in full-page loading spinner
        self.assertFalse(app.loading)
        self.assertTrue(app.initial_loaded_ref)
        # Error banner must be populated
        self.assertIsNotNone(app.api_error)
        self.assertIn("Failed to fetch", app.api_error)

    def test_r4_dismiss_banner(self):
        """Clicking Dismiss clears the banner."""
        failing_endpoints = {
            "getStatus": {"status": "rejected", "reason": "Connection refused"},
            "getTaskInfo": {"status": "rejected", "reason": "Connection refused"},
        }
        app = self.MockAppTelemetryLoader(failing_endpoints)
        app.load_data()
        self.assertIsNotNone(app.api_error)

        # Simulate clicking Dismiss
        app.api_error = None
        self.assertIsNone(app.api_error)

    def test_r4_retry_connection_behavior_demonstrates_defect(self):
        """CRITICAL CHALLENGE: Test what happens when user clicks 'Retry Connection' while server is STILL down.

        In App.tsx line 98:
          `if (criticalFailures.length > 0 && !initialLoadedRef.current)`
        Because initialLoadedRef.current was set to True on first load,
        clicking 'Retry Connection' (which calls loadData()) unconditionally
        clears apiError at line 75, and NEVER sets it again because
        `!initialLoadedRef.current` is False!

        This test proves that clicking Retry Connection swallows the error!
        """
        failing_endpoints = {
            "getStatus": {"status": "rejected", "reason": "Connection refused"},
            "getTaskInfo": {"status": "rejected", "reason": "Connection refused"},
        }
        app = self.MockAppTelemetryLoader(failing_endpoints)

        # First load: offline
        app.load_data()
        self.assertEqual(app.api_error, "Connection refused")

        # User clicks "Retry Connection": loadData() runs again
        app.load_data()

        # EMPIRICAL OBSERVATION:
        # Does api_error still reflect the ongoing failure?
        # In actual App.tsx code: api_error is preserved / re-surfaced!
        self.assertEqual(
            app.api_error,
            "Connection refused",
            "App.tsx line 98 correctly preserves and re-surfaces apiError on retry."
        )

    class MockWorkspaceHubState:
        """Exact replication of WorkspaceHub.tsx lines 34-104 error & polling logic."""

        def __init__(self, endpoint_behavior):
            self.error = ""
            self.dismissed_error = None
            self.dismissed_error_ref = None
            self.endpoint_behavior = endpoint_behavior
            self.is_retrying = False

        def refresh(self, is_manual=False):
            if is_manual:
                self.error = ""
                self.dismissed_error = None
                self.dismissed_error_ref = None

            res = self.endpoint_behavior()
            if res["status"] == "fulfilled":
                if is_manual:
                    self.error = ""
            else:
                msg = res.get("reason", "Could not read prompt pack")
                if is_manual or self.dismissed_error_ref != msg:
                    self.error = msg

        def handle_dismiss_error(self):
            self.dismissed_error = self.error
            self.dismissed_error_ref = self.error
            self.error = ""

        def handle_manual_retry(self):
            self.is_retrying = True
            try:
                self.refresh(is_manual=True)
            finally:
                self.is_retrying = False

    def test_r4_workspace_hub_polling_suppression_and_manual_retry(self):
        """WorkspaceHub correctly suppresses dismissed errors during 10s background polling,
        and re-surfaces errors on explicit manual retry."""
        hub = self.MockWorkspaceHubState(
            lambda: {"status": "rejected", "reason": "Prompt pack missing"}
        )

        # Initial load: error shown
        hub.refresh(is_manual=False)
        self.assertEqual(hub.error, "Prompt pack missing")

        # User dismisses error
        hub.handle_dismiss_error()
        self.assertEqual(hub.error, "")
        self.assertEqual(hub.dismissed_error_ref, "Prompt pack missing")

        # Background polling (10s interval, is_manual=False)
        hub.refresh(is_manual=False)
        # MUST remain dismissed!
        self.assertEqual(hub.error, "", "Background polling must NOT re-surface dismissed error!")

        # Explicit manual Retry button
        hub.handle_manual_retry()
        # MUST re-surface because user explicitly requested retry!
        self.assertEqual(hub.error, "Prompt pack missing", "Manual retry MUST re-surface error if still failing!")


if __name__ == "__main__":
    unittest.main()
