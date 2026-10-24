"""Unit tests for developer ergonomics, facade parity, and runtime resilience."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from harness.agent_bridge import AgentBridge, HANDLERS, _ACTION_REGISTRY, typed_actions
from harness.artifacts import contract_identity, repair_run, repair_run_dir, run_repair
from harness.competition_state import CompetitionStateMachine
from harness.evaluation import validate_run
from harness.kaggle_compat import KaggleClient, KaggleLeaderboardTracker
from harness.kaggle_runtime import find_dataset_mount
from harness.repair import repair_folder
from harness.repair import repair_run as repair_run_alias
from harness.research_memory import ResearchMemory
from harness.task_spec import load_task_spec


class ErgonomicsAndResilienceTests(unittest.TestCase):
    def test_research_memory_defaults_and_load(self):
        # Auto-discovers project root and defaults without requiring positional arguments
        rm = ResearchMemory(".")
        self.assertIsNotNone(rm.root)
        self.assertIsNotNone(rm.artifacts)

        rm_loaded = ResearchMemory.load(".")
        self.assertEqual(rm.root, rm_loaded.root)

    def test_research_memory_frontier_access_styles(self):
        rm = ResearchMemory(".")
        # As property
        frontier_prop = rm.frontier
        self.assertIsInstance(frontier_prop, list)

        # As callable
        frontier_call = rm.frontier()
        self.assertEqual(frontier_prop, frontier_call)

        # Aliases
        self.assertEqual(rm.research_frontier, frontier_prop)
        self.assertEqual(rm.research_frontier(), frontier_prop)

        # Classmethod access styles
        class_frontier = ResearchMemory.frontier()
        self.assertIsInstance(class_frontier, list)
        class_rf = ResearchMemory.research_frontier()
        self.assertEqual(class_frontier, class_rf)

    def test_competition_state_machine_load(self):
        sm = CompetitionStateMachine.load(".")
        self.assertIsInstance(sm, CompetitionStateMachine)
        self.assertEqual(sm.project_root, Path(".").resolve())

    def test_artifacts_and_repair_cross_exports(self):
        # Functions must be callable and identical
        self.assertIs(repair_run_dir, repair_run)
        self.assertIs(repair_run_alias, repair_folder)
        self.assertTrue(callable(run_repair))
        self.assertTrue(callable(validate_run))

    def test_agent_bridge_action_registry_and_dynamic_handlers(self):
        self.assertIn("record_research_insight", typed_actions)
        self.assertEqual(HANDLERS, typed_actions)
        self.assertEqual(_ACTION_REGISTRY, typed_actions)
        self.assertEqual(AgentBridge._ACTION_REGISTRY, typed_actions)

        bridge = AgentBridge(".")
        self.assertTrue(hasattr(bridge, "_action_record_research_insight"))
        self.assertTrue(hasattr(bridge, "record_research_insight"))

    def test_contract_identity_with_dict_task_spec(self):
        # Must not raise AttributeError: 'dict' object has no attribute 'metadata'
        dict_spec = {"metadata": {"id": "test_comp", "name": "Test"}}
        ident = contract_identity(dict_spec)
        self.assertIn("task_sha256", ident)
        self.assertIn("scorer_sha256", ident)

        raw_dict = {"id": "raw_task", "name": "Raw"}
        ident_raw = contract_identity(raw_dict)
        self.assertIn("task_sha256", ident_raw)

    def test_load_task_spec_export(self):
        self.assertTrue(callable(load_task_spec))
        spec = load_task_spec(".")
        self.assertIsNotNone(spec)

    def test_kaggle_client_facade_methods_and_aliases(self):
        # KaggleLeaderboardTracker must be importable from kaggle_compat
        self.assertIsNotNone(KaggleLeaderboardTracker)

        client = KaggleClient()
        self.assertTrue(hasattr(client, "api"))
        self.assertTrue(hasattr(client, "_api"))
        self.assertTrue(hasattr(client, "leaderboard"))
        self.assertTrue(hasattr(client, "get_leaderboard"))
        self.assertTrue(hasattr(client, "list_topics"))
        self.assertTrue(hasattr(client, "topics"))
        self.assertTrue(hasattr(client, "discussions"))
        self.assertTrue(hasattr(client, "list_discussions"))
        self.assertTrue(hasattr(client, "topic_messages"))
        self.assertTrue(hasattr(client, "list_topic_messages"))
        self.assertTrue(hasattr(client, "kernels_pull"))
        self.assertTrue(hasattr(client, "kernel_pull"))
        self.assertTrue(hasattr(client, "pull_kernel"))
        self.assertTrue(hasattr(client, "get_kernel"))

    def test_find_dataset_mount_casing_and_resilience(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td)
            # Create a dataset directory with lowercase or camelCase
            ds_dir = base / "aih-kaggle-test-dataset"
            ds_dir.mkdir(parents=True)

            # 1. Exact match
            found = find_dataset_mount(base, "user/aih-kaggle-test-dataset")
            self.assertEqual(found, ds_dir)

            # 2. Case-insensitive match
            found_ci = find_dataset_mount(base, "user/AIH-KAGGLE-TEST-DATASET")
            self.assertEqual(found_ci, ds_dir)

            # 3. Nested layout without owner direct parent
            nested = base / "datasets" / "some_hash" / "nested-ds"
            nested.mkdir(parents=True)
            found_nested = find_dataset_mount(base, "user/nested-ds")
            self.assertEqual(found_nested, nested)


if __name__ == "__main__":
    unittest.main()
