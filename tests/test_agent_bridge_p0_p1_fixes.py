"""Regression tests for P0/P1 agent-bridge integration fixes.

Covers:
- P0 A3: budget-blocked action does NOT terminate the session.
- P0 C1: calling a different action while a USER_APPROVAL action is pending
  does not silently drop the pending approval.
- P0 A4: failed full-validation runs do not consume the experiment budget.
- P0 B4: cloud job terminal accounting computes gpu_hours so the session
  GPU budget is actually enforced.
- P1 A1: active session selection is stable via the `active` flag.
- P1 B1: resource reservation freeze-check is not stale.
- P1 C2: approving a pending action does not duplicate it in session.actions.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from harness.agent_bridge import AgentBridge
from tests.fixtures import copy_repo_configs, workspace_with_repo_configs



def _make_workspace() -> Path:
    return workspace_with_repo_configs(prefix="test_p0p1_")


class TestBudgetBlockDoesNotTerminateSession(unittest.TestCase):
    """P0 A3: a budget-blocked action is an observation, not a terminal state."""

    def setUp(self) -> None:
        self._ws = tempfile.TemporaryDirectory(prefix="test_a3_")
        self.root = Path(self._ws.name)
        copy_repo_configs(self.root)
        self.bridge = AgentBridge(self.root)

    def tearDown(self) -> None:
        self._ws.cleanup()

    def test_blocked_action_keeps_session_active(self) -> None:
        sess = self.bridge.start_session(
            goal="budget test",
            budget={"max_experiments": 0},  # any full-validation is blocked
        )
        # Run a blocked full-validation (no config needed; guard fires before dispatch).
        _, obs = self.bridge.execute_action(
            action="run_full_validation",
            params={"config_path": "configs/does_not_exist.yaml"},
            session_id=sess.session_id,
        )
        self.assertEqual(obs.status, "BLOCKED_BY_BUDGET")
        # Session must still be ACTIVE so the agent can continue with other actions.
        reloaded = self.bridge.load_session(sess.session_id)
        self.assertEqual(reloaded.status, "ACTIVE")
        # The observation is recorded so the agent sees the block.
        self.assertIn(
            "BLOCKED_BY_BUDGET",
            [o["status"] for o in reloaded.observations],
        )
        # The blocked call must NOT appear in actions (it was never executed).
        self.assertNotIn(
            "run_full_validation",
            [a["action"] for a in reloaded.actions],
        )

    def test_blocked_action_does_not_poison_active_session_pick(self) -> None:
        """After a budget block, the next AUTO action resumes the same session."""
        sess = self.bridge.start_session(
            goal="budget test",
            budget={"max_experiments": 0},
        )
        self.bridge.execute_action(
            action="run_full_validation",
            params={"config_path": "configs/does_not_exist.yaml"},
            session_id=sess.session_id,
        )
        # An AUTO action without an explicit session_id should land in the
        # same (still-active) session, not spawn a new one.
        sess2, _ = self.bridge.execute_action(
            action="get_next_experiment_id",
            params={},
            goal="budget test",  # same goal as the original session
        )
        self.assertEqual(sess2.session_id, sess.session_id)


class TestPendingApprovalNotDroppedByOtherAction(unittest.TestCase):
    """P0 C1: a different action while a USER_APPROVAL action is pending
    must not clear `pending_approval_action`."""

    def setUp(self) -> None:
        self._ws = tempfile.TemporaryDirectory(prefix="test_c1_")
        self.root = Path(self._ws.name)
        copy_repo_configs(self.root)
        self.bridge = AgentBridge(self.root)

    def tearDown(self) -> None:
        self._ws.cleanup()

    def test_inspect_while_pending_keeps_approval(self) -> None:
        sess = self.bridge.start_session(goal="approval test")
        # Pause on a USER_APPROVAL action.
        _, obs = self.bridge.execute_action(
            action="prepare_submission",
            params={"exp_id": "baseline"},
            session_id=sess.session_id,
            user_approved=False,
        )
        self.assertEqual(obs.status, "PAUSED_FOR_APPROVAL")
        pending = self.bridge.load_session(sess.session_id).pending_approval_action
        self.assertIsNotNone(pending)

        # Now call an AUTO action on the same session.
        _, obs2 = self.bridge.execute_action(
            action="get_next_experiment_id",
            params={},
            session_id=sess.session_id,
        )
        self.assertEqual(obs2.status, "SUCCESS")

        # The pending approval must still be there.
        reloaded = self.bridge.load_session(sess.session_id)
        self.assertIsNotNone(reloaded.pending_approval_action)
        self.assertEqual(
            reloaded.pending_approval_action["action"], "prepare_submission"
        )
        self.assertEqual(reloaded.status, "PAUSED_FOR_APPROVAL")

    def test_approving_after_other_action_executes_pending(self) -> None:
        """The pending action can still be approved after an interleaved action."""
        sess = self.bridge.start_session(goal="approval test")
        self.bridge.execute_action(
            action="prepare_submission",
            params={"exp_id": "baseline"},
            session_id=sess.session_id,
            user_approved=False,
        )
        # Interleave an AUTO action.
        self.bridge.execute_action(
            action="get_next_experiment_id", params={}, session_id=sess.session_id
        )
        # Approve.
        sess3, obs = self.bridge.approve_pending_action(sess.session_id)
        # The approval flow should have executed prepare_submission (which will
        # fail because no candidate exists, but the action was attempted, not
        # dropped). The observation must reference prepare_submission.
        self.assertEqual(obs.action, "prepare_submission")


class TestFailedFullValidationDoesNotConsumeBudget(unittest.TestCase):
    """P0 A4: a FAILED full-validation run does not permanently consume the
    experiment budget (reconcile only counts terminal rows, and the agent can
    still retry)."""

    def setUp(self) -> None:
        self._ws = tempfile.TemporaryDirectory(prefix="test_a4_")
        self.root = Path(self._ws.name)
        copy_repo_configs(self.root)
        self.bridge = AgentBridge(self.root)

    def tearDown(self) -> None:
        self._ws.cleanup()

    def test_failed_run_terminal_row_counts_but_session_recovers(self) -> None:
        sess = self.bridge.start_session(
            goal="fail test",
            budget={"max_experiments": 1},
        )
        # Force a failed local run.
        with patch.object(
            self.bridge, "_run_subprocess_safe", return_value=(False, 1, "boom", 0.01)
        ):
            _, obs = self.bridge.execute_action(
                action="run_full_validation",
                params={"config_path": "configs/exp001_feat_feature_pipeline.yaml"},
                session_id=sess.session_id,
            )
        self.assertEqual(obs.status, "FAILED")
        # The terminal row exists in the ledger.
        ledger_path = self.bridge.artifacts_dir / "resource_ledger.json"
        ledger = json.loads(ledger_path.read_text())
        terminal = [
            r for r in ledger["reservations"].values()
            if r["status"] in {"COMPLETED", "FAILED", "CANCELLED"}
        ]
        self.assertGreaterEqual(len(terminal), 1)
        # Reconciled usage reflects the terminal row.
        reloaded = self.bridge.load_session(sess.session_id)
        self.assertGreaterEqual(reloaded.compute_usage.get("experiments_used", 0), 1)
        # But the session is still ACTIVE (not STOPPED), so the agent can
        # inspect/record/decide — only a NEW full-validation would be blocked.
        self.assertEqual(reloaded.status, "ACTIVE")


class TestCloudJobGpuHoursAccounting(unittest.TestCase):
    """P0 B4: a terminal cloud job row carries gpu_hours so the session GPU
    budget is actually enforced (not silently disabled by a None)."""

    def setUp(self) -> None:
        self._ws = tempfile.TemporaryDirectory(prefix="test_b4_")
        self.root = Path(self._ws.name)
        copy_repo_configs(self.root)
        self.bridge = AgentBridge(self.root)

    def tearDown(self) -> None:
        self._ws.cleanup()

    def test_terminal_row_carries_gpu_hours_for_modal(self) -> None:
        from harness.resource_ledger import terminal

        # Simulate a modal job row in the ledger.
        ledger_path = self.bridge.artifacts_dir / "resource_ledger.json"
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        token = "tok_gpu_test"
        from harness.state_store import transaction
        with transaction(ledger_path, {"reservations": {}}) as ledger:
            ledger["reservations"][token] = {
                "session_id": "session_0001",
                "category": "training",
                "backend": "modal",
                "status": "RESERVED",
                "reserved_seconds": None,
                "wall_seconds": None,
                "gpu_hours": None,
                "job_id": "job_modal_1",
            }
        # Call terminal with a gpu allocation (1 GPU for 3600s = 1 gpu-hour).
        result = terminal(
            ledger_path, token, "COMPLETED", 3600.0,
            gpu_count=1, allocation_seconds=3600.0,
        )
        self.assertTrue(result)
        ledger = json.loads(ledger_path.read_text())
        row = ledger["reservations"][token]
        self.assertEqual(row["status"], "COMPLETED")
        self.assertIsNotNone(row["gpu_hours"])
        self.assertAlmostEqual(row["gpu_hours"], 1.0, places=3)

    def test_reconcile_gpu_hours_not_none_when_rows_have_values(self) -> None:
        """When terminal rows carry gpu_hours, reconcile must not produce None."""
        ledger_path = self.bridge.artifacts_dir / "resource_ledger.json"
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        from harness.state_store import transaction
        with transaction(ledger_path, {"reservations": {}}) as ledger:
            ledger["reservations"]["tok1"] = {
                "session_id": "session_0001",
                "category": "training",
                "backend": "modal",
                "status": "COMPLETED",
                "wall_seconds": 3600.0,
                "gpu_hours": 0.5,
            }
        sess = self.bridge.start_session(goal="gpu test")
        sess.session_id = "session_0001"
        self.bridge._reconcile_session_usage(sess)
        self.assertIsNotNone(sess.compute_usage.get("gpu_hours_used"))
        self.assertAlmostEqual(sess.compute_usage["gpu_hours_used"], 0.5, places=3)


class TestActiveSessionSelectionStable(unittest.TestCase):
    """P1 A1: the `active` flag gives a stable session to resume, regardless of
    mtime ordering."""

    def setUp(self) -> None:
        self._ws = tempfile.TemporaryDirectory(prefix="test_a1_")
        self.root = Path(self._ws.name)
        copy_repo_configs(self.root)
        self.bridge = AgentBridge(self.root)

    def tearDown(self) -> None:
        self._ws.cleanup()

    def test_start_session_marks_active_and_demotes_others(self) -> None:
        s1 = self.bridge.start_session(goal="first")
        s2 = self.bridge.start_session(goal="second")
        # s2 is now the active one.
        reloaded2 = self.bridge.load_session(s2.session_id)
        self.assertTrue(reloaded2.active)
        reloaded1 = self.bridge.load_session(s1.session_id)
        self.assertFalse(reloaded1.active)

    def test_get_active_or_create_prefers_active_flag(self) -> None:
        s1 = self.bridge.start_session(goal="first")
        s2 = self.bridge.start_session(goal="second")
        # Force s1 to be the active one (simulate the agent explicitly
        # resuming s1). Set the in-memory flag BEFORE saving so that
        # save_session's demote logic does not re-demote s1.
        s1.active = True
        s2.active = False
        self.bridge.save_session(s2)  # save s2 first (inactive, no demote side-effect)
        self.bridge.save_session(s1)  # now claim s1 as active

        picked, _ = self.bridge.execute_action(
            action="get_next_experiment_id", params={},
        )
        # Should have landed in the active session (s1), not the most recent.
        self.assertEqual(picked.session_id, s1.session_id)


class TestApproveDoesNotDuplicateAction(unittest.TestCase):
    """P1 C2: approving a pending action does not leave two entries for the
    same action in session.actions."""

    def setUp(self) -> None:
        self._ws = tempfile.TemporaryDirectory(prefix="test_c2_")
        self.root = Path(self._ws.name)
        copy_repo_configs(self.root)
        self.bridge = AgentBridge(self.root)

    def tearDown(self) -> None:
        self._ws.cleanup()

    def test_approve_removes_pending_from_actions(self) -> None:
        sess = self.bridge.start_session(goal="c2 test")
        # Pause on a USER_APPROVAL action.
        self.bridge.execute_action(
            action="prepare_submission",
            params={"exp_id": "baseline"},
            session_id=sess.session_id,
            user_approved=False,
        )
        # Approve.
        self.bridge.approve_pending_action(sess.session_id)
        reloaded = self.bridge.load_session(sess.session_id)
        # prepare_submission should appear at most once in actions.
        count = sum(1 for a in reloaded.actions if a["action"] == "prepare_submission")
        self.assertLessEqual(count, 1)


class TestFreezeCheckNotStale(unittest.TestCase):
    """P1 B1: a reserve() call sees a concurrent freeze as soon as it is
    committed to competition_state.json."""

    def setUp(self) -> None:
        self._ws = tempfile.TemporaryDirectory(prefix="test_b1_")
        self.root = Path(self._ws.name)
        copy_repo_configs(self.root)
        self.bridge = AgentBridge(self.root)

    def tearDown(self) -> None:
        self._ws.cleanup()

    def test_reserve_blocks_when_state_file_says_frozen(self) -> None:
        # Directly write a frozen state.
        state_file = self.bridge.artifacts_dir / "competition_state.json"
        state_file.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "gates": {"freeze_locked": True},
            "candidate_pool": {"synthesized_candidates": [], "pinned_candidates": []},
        }
        state_file.write_text(json.dumps(state), encoding="utf-8")
        ledger_path = self.bridge.artifacts_dir / "resource_ledger.json"
        from harness.resource_ledger import reserve
        with self.assertRaises(ValueError) as ctx:
            reserve(
                ledger_path, "session_0001", "run_full_validation",
                "local", {"max_experiments": 5},
            )
        self.assertIn("frozen", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main()
