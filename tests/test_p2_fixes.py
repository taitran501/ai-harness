"""Regression tests for P2 defense-in-depth fixes.

1. `_validate_config_path` rejects config paths that resolve outside the
   project root (defense-in-depth for `run_full_validation`, `run_smoke_test`,
   `run_probe`).
2. `compute_candidate_pool` uses per-subtask direction for specialist ranking,
   not just the aggregate spec direction.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from harness.agent_bridge import AgentBridge

SOURCE_ROOT = Path(__file__).resolve().parent.parent


class TestValidateConfigPath(unittest.TestCase):
    def setUp(self) -> None:
        self._ws = tempfile.TemporaryDirectory(prefix="test_p2_")
        self.root = Path(self._ws.name)
        shutil.copytree(SOURCE_ROOT / "configs", self.root / "configs")
        self.bridge = AgentBridge(self.root)
        # Ensure a configs/ dir with a dummy file exists.
        configs_dir = self.root / "configs"
        configs_dir.mkdir(parents=True, exist_ok=True)
        self.valid_cfg = configs_dir / "test_exp.yaml"
        self.valid_cfg.write_text("experiment:\n  id: test_exp\n")

    def tearDown(self) -> None:
        self._ws.cleanup()

    def test_valid_config_path_ok(self) -> None:
        resolved = self.bridge._validate_config_path("configs/test_exp.yaml")
        self.assertEqual(resolved, self.valid_cfg.resolve())

    def test_traversal_rejected(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            self.bridge._validate_config_path("../../etc/passwd.yaml")
        self.assertIn("outside the project root", str(ctx.exception))

    def test_absolute_outside_rejected(self) -> None:
        outside = self._ws_path_outside()
        with self.assertRaises(ValueError):
            self.bridge._validate_config_path(str(outside))

    def _ws_path_outside(self) -> Path:
        # A path outside the project root (sibling of the temp dir).
        return Path(tempfile.gettempdir()) / "outside_test.yaml"


class TestPerSubtaskDirection(unittest.TestCase):
    def setUp(self) -> None:
        self._ws = tempfile.TemporaryDirectory(prefix="test_p2dir_")
        self.root = Path(self._ws.name)
        shutil.copytree(SOURCE_ROOT / "configs", self.root / "configs")
        self.bridge = AgentBridge(self.root)

    def tearDown(self) -> None:
        self._ws.cleanup()

    def test_specialist_uses_per_subtask_direction(self) -> None:
        """A minimized subtask should rank lower candidates as 'better'."""
        comp = self.bridge.comp_state
        # Spec direction is maximize, but the subtask is minimized.
        comp.spec["direction"] = "maximize"
        comp.spec["subtasks"] = [
            {"id": "err", "direction": "minimize", "primary_metric": "mae"},
        ]
        comp.spec["evaluation_criteria"] = {"err": {"direction": "minimize"}}

        # Build a minimal journal with two completed nodes that differ on
        # the "err" subtask. Lower "err" should be preferred (minimize).
        from harness.tree_journal import Node, TreeJournal

        journal = TreeJournal(auto_sync=False)

        def make_node(node_id: str, err_score: float, primary: float) -> Node:
            node = Node(
                id=node_id,
                parent_id="baseline",
                branch="exp",
                direction="maximize",
                metrics={
                    "primary_metric": primary,
                    "overall_composite": primary,
                    "subtasks": {"err": {"score": err_score}},
                },
                status="COMPLETED",
                verified=True,
                eligible=True,
            )
            journal.append(node)
            return node

        make_node("baseline", 3.0, 3.0)
        make_node("expA", 5.0, 5.0)  # worse for minimize
        make_node("expB", 1.0, 1.0)  # better for minimize

        comp.state["gates"]["baseline_exp_id"] = "baseline"
        pool = comp.compute_candidate_pool(journal)
        specialist = pool.get("subtask_specialists", {}).get("err")
        self.assertIsNotNone(specialist, "Expected a specialist for 'err'")
        # For a minimized subtask, the lower score (expB) should win.
        self.assertEqual(specialist["id"], "expB")


if __name__ == "__main__":
    unittest.main()
