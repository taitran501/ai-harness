"""Tests for Environment Detection, Cloud Job Runner, and Task Switcher."""

import json
import unittest
from pathlib import Path

from harness.environment_detector import EnvironmentDetector
from harness.cloud_job_runner import CloudJobManager, CloudJob
from tools.dashboard_server import (
    get_next_experiment_id,
    list_available_tasks,
    switch_task,
)


class TestEnvironmentAndCloudJobs(unittest.TestCase):
    def setUp(self):
        self.task_file = Path("configs/task.yaml")
        self.original_task_yaml = self.task_file.read_text(encoding="utf-8") if self.task_file.exists() else None

    def tearDown(self):
        if self.original_task_yaml and self.task_file.exists():
            self.task_file.write_text(self.original_task_yaml, encoding="utf-8")

    def test_environment_detector_full_status(self):
        status = EnvironmentDetector.get_full_status()
        self.assertIn("modal", status)
        self.assertIn("kaggle", status)
        self.assertIn("docker", status)
        self.assertIn("gpu", status)
        self.assertTrue(status["modal"]["installed"])
        self.assertTrue(status["kaggle"]["installed"])
        self.assertTrue(status["docker"]["installed"])
        self.assertTrue(status["gpu"]["detected"])

    def test_get_next_experiment_id(self):
        res = get_next_experiment_id(branch="feat", descriptor="test_cleaning")
        self.assertIn("next_id", res)
        self.assertIn("next_number", res)
        self.assertTrue(res["next_number"] >= 1)
        self.assertTrue(res["next_id"].startswith(f"exp{res['next_number']:03d}_feat_test_cleaning"))

    def test_list_and_switch_tasks(self):
        tasks = list_available_tasks()
        self.assertTrue(len(tasks) >= 1)
        active = [t for t in tasks if t.get("is_active")]
        self.assertTrue(len(active) >= 1)

        # Test task switch
        first_id = tasks[0]["id"]
        switch_res = switch_task(first_id)
        self.assertTrue(switch_res["success"])
        self.assertEqual(switch_res["task_id"], first_id)

        # Restore to stereoqueereval
        switch_task("stereoqueereval")

    def test_cloud_job_manager_lifecycle(self):
        manager = CloudJobManager()
        # Test start dummy local job (e.g. invalid config or debug)
        configs = list(Path("configs").glob("*.yaml"))
        self.assertTrue(len(configs) > 0)
        cfg_path = str(configs[0].relative_to(Path("."))).replace("\\", "/")

        job = manager.start_job(
            backend="local",
            config_path=cfg_path,
            auto_audit=False,
            auto_sync=False,
        )
        self.assertIsNotNone(job.job_id)
        self.assertEqual(job.backend, "local")
        self.assertIn(job.status, ("PENDING", "RUNNING", "COMPLETED", "FAILED"))

        fetched = manager.get_job(job.job_id)
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched["job_id"], job.job_id)

        jobs_list = manager.list_jobs(limit=5)
        self.assertTrue(len(jobs_list) >= 1)

    def test_modal_and_kaggle_auth_helpers(self):
        # 1. Parse token inputs
        raw_cmd = "modal token set --token-id ak-1234567890abcdef --token-secret as-abcdef1234567890"
        t_id, t_sec = EnvironmentDetector.parse_modal_token_input(raw_cmd)
        self.assertEqual(t_id, "ak-1234567890abcdef")
        self.assertEqual(t_sec, "as-abcdef1234567890")

        # 2. Launch auth returns valid URL structure
        m_res = EnvironmentDetector.launch_modal_auth()
        self.assertTrue(m_res["success"])
        self.assertIn("modal.com/settings/tokens", m_res["auth_url"])

        k_res = EnvironmentDetector.launch_kaggle_auth()
        self.assertTrue(k_res["success"])
        self.assertIn("kaggle.com/settings", k_res["auth_url"])


if __name__ == "__main__":
    unittest.main()
