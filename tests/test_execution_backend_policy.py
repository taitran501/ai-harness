"""Unit tests for the task-contract execution backend policy.

Covers: UniversalTaskSpec.execution_* properties, AgentBridge backend
resolution (param > contract > local), lock enforcement, HUD surfacing, and
budget-guard canonicalization.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from harness.agent_bridge import AgentBridge
from harness.task_spec import UniversalTaskSpec, load_active_task_spec

SOURCE_ROOT = Path(__file__).resolve().parent.parent


def _task_spec_dict(name: str = "BackendTest", execution: dict | None = None) -> dict:
    cfg: dict = {"name": name, "primary_metric": "rmsle"}
    if execution is not None:
        cfg["execution"] = execution
    return cfg


class TestExecutionConfigProperties(unittest.TestCase):
    def test_execution_block_parsed_from_metadata(self) -> None:
        spec = UniversalTaskSpec.from_dict(_task_spec_dict(execution={"backend": "Kaggle", "locked": True}))
        self.assertEqual(spec.execution_backend, "kaggle")
        self.assertTrue(spec.execution_locked)
        self.assertEqual(spec.execution_config, {"backend": "Kaggle", "locked": True})

    def test_no_execution_block_defaults_empty(self) -> None:
        spec = UniversalTaskSpec.from_dict(_task_spec_dict())
        self.assertIsNone(spec.execution_backend)
        self.assertFalse(spec.execution_locked)
        self.assertEqual(spec.execution_config, {})

    def test_malformed_execution_block_ignored(self) -> None:
        spec = UniversalTaskSpec.from_dict(_task_spec_dict(execution="kaggle"))  # type: ignore[typeddict-item]
        self.assertIsNone(spec.execution_backend)
        self.assertFalse(spec.execution_locked)


def _make_bridge(task_yaml: str) -> tuple[AgentBridge, Path]:
    ws = Path(tempfile.mkdtemp(prefix="test_exec_backend_"))
    shutil.copytree(SOURCE_ROOT / "configs", ws / "configs")
    (ws / "configs" / "task.yaml").write_text(task_yaml, encoding="utf-8")
    return AgentBridge(ws), ws


LOCKED_TASK = (
    "task:\n"
    "  name: BackendTest\n"
    "  primary_metric: rmsle\n"
    "  execution:\n"
    "    backend: kaggle\n"
    "    locked: true\n"
)
UNLOCKED_TASK = (
    "task:\n"
    "  name: BackendTest\n"
    "  primary_metric: rmsle\n"
    "  execution:\n"
    "    backend: kaggle\n"
)
PLAIN_TASK = "task:\n  name: BackendTest\n  primary_metric: rmsle\n"


class TestBackendResolution(unittest.TestCase):
    def setUp(self) -> None:
        self.bridge, self.ws = _make_bridge(LOCKED_TASK)
        self.addCleanup(shutil.rmtree, self.ws, True)

    def test_contract_default_applied_without_param(self) -> None:
        backend, locked = self.bridge._resolve_execution_backend({})
        self.assertEqual(backend, "kaggle")
        self.assertTrue(locked)

    def test_explicit_matching_param_accepted(self) -> None:
        backend, _ = self.bridge._resolve_execution_backend({"backend": "KAGGLE"})
        self.assertEqual(backend, "kaggle")

    def test_lock_violation_rejected(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            self.bridge._resolve_execution_backend({"backend": "local"})
        self.assertIn("Backend lock violation", str(ctx.exception))
        self.assertIn("kaggle", str(ctx.exception))

    def test_invalid_backend_rejected_when_unlocked(self) -> None:
        # swap to plain task (no lock)
        (self.ws / "configs" / "task.yaml").write_text(PLAIN_TASK, encoding="utf-8")
        self.bridge.task_spec = load_active_task_spec(
            self.ws / "configs" / "task.yaml", force_reload=True
        )
        with self.assertRaises(ValueError) as ctx:
            self.bridge._resolve_execution_backend({"backend": "lambda"})
        self.assertIn("Unsupported execution backend", str(ctx.exception))

    def test_unlocked_contract_allows_override(self) -> None:
        (self.ws / "configs" / "task.yaml").write_text(UNLOCKED_TASK, encoding="utf-8")
        self.bridge.task_spec = load_active_task_spec(
            self.ws / "configs" / "task.yaml", force_reload=True
        )
        backend, locked = self.bridge._resolve_execution_backend({})
        self.assertEqual((backend, locked), ("kaggle", False))
        backend, _ = self.bridge._resolve_execution_backend({"backend": "local"})
        self.assertEqual(backend, "local")

    def test_plain_task_defaults_to_local(self) -> None:
        (self.ws / "configs" / "task.yaml").write_text(PLAIN_TASK, encoding="utf-8")
        self.bridge.task_spec = load_active_task_spec(
            self.ws / "configs" / "task.yaml", force_reload=True
        )
        backend, locked = self.bridge._resolve_execution_backend({})
        self.assertEqual((backend, locked), ("local", False))


class TestHudAndGuard(unittest.TestCase):
    def test_hud_surfaces_locked_backend(self) -> None:
        bridge, ws = _make_bridge(LOCKED_TASK)
        self.addCleanup(shutil.rmtree, ws, True)
        md = bridge.render_context_markdown()
        self.assertIn("**Execution Backend**", md)
        self.assertIn("`KAGGLE`", md)
        self.assertIn("LOCKED", md)

    def test_hud_omits_backend_when_undeclared(self) -> None:
        bridge, ws = _make_bridge(PLAIN_TASK)
        self.addCleanup(shutil.rmtree, ws, True)
        md = bridge.render_context_markdown()
        self.assertNotIn("Execution Backend", md)

    def test_guard_canonicalizes_run_full_validation_backend(self) -> None:
        bridge, ws = _make_bridge(LOCKED_TASK)
        self.addCleanup(shutil.rmtree, ws, True)
        session = bridge.start_session(goal="backend test")
        params: dict = {"config_path": "configs/exp001_feat_feature_pipeline.yaml"}
        violation = bridge._check_budget_and_stop_guards(session, "run_full_validation", params)
        # No session budgets declared, so no block — but the backend must be canonicalized
        self.assertIsNone(violation)
        self.assertEqual(params.get("backend"), "kaggle")

    def test_guard_lock_violation_surfaced_for_full_validation(self) -> None:
        bridge, ws = _make_bridge(LOCKED_TASK)
        self.addCleanup(shutil.rmtree, ws, True)
        session = bridge.start_session(goal="backend test")
        params = {"config_path": "configs/exp001_feat_feature_pipeline.yaml", "backend": "local"}
        try:
            violation = bridge._check_budget_and_stop_guards(session, "run_full_validation", params)
        except ValueError as exc:
            self.assertIn("Backend lock violation", str(exc))
            return
        self.assertIsNone(violation, "lock violation must be raised or reported")

    def test_execute_action_records_lock_violation_as_failed_observation(self) -> None:
        bridge, ws = _make_bridge(LOCKED_TASK)
        self.addCleanup(shutil.rmtree, ws, True)
        sess, obs = bridge.execute_action(
            "run_full_validation",
            params={"config_path": "configs/exp001_feat_feature_pipeline.yaml", "backend": "local"},
            goal="lock e2e",
        )
        self.assertEqual(obs.status, "FAILED")
        self.assertIn("Backend lock violation", obs.summary)
        self.assertIn("Guard rejected", obs.summary)
        self.assertIn(sess.session_id, sess.session_id)
        self.assertTrue(any("Backend lock violation" in str(o) for o in sess.observations))


if __name__ == "__main__":
    unittest.main()
