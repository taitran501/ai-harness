"""Tests for environment detection, cloud jobs, and per-task workspace isolation."""

import json
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from harness.environment_detector import EnvironmentDetector
from harness.cloud_job_runner import CloudJobManager, CloudJob
from harness.project import HarnessProject
from tools.dashboard_server import (
    get_next_experiment_id,
    list_available_tasks,
    switch_task,
)
from tests.workspace_fixtures import (
    cleanup_dashboard_workspace,
    prepare_dashboard_workspace,
)


class TestEnvironmentAndCloudJobs(unittest.TestCase):
    def setUp(self):
        self.root = prepare_dashboard_workspace(self, "test_cloud_jobs_")
        project = HarnessProject.discover(self.root)
        self._module_patches = [
            patch("harness.cloud_job_runner.PROJECT_ROOT", self.root),
            patch("harness.cloud_job_runner.HARNESS_PROJECT", project),
            patch("harness.cloud_job_runner.ARTIFACTS_DIR", project.artifacts_dir),
            patch("harness.environment_detector.PROJECT_ROOT", self.root),
        ]
        for module_patch in self._module_patches:
            module_patch.start()

    def tearDown(self):
        for module_patch in reversed(self._module_patches):
            module_patch.stop()
        cleanup_dashboard_workspace(self)

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

        # A dashboard remains attached to one task; switching requires attaching a different workspace.
        first_id = tasks[0]["id"]
        switch_res = switch_task(first_id)
        self.assertFalse(switch_res["success"])
        self.assertIn("own workspace", switch_res["error"])

    def test_cloud_job_manager_lifecycle(self):
        manager = CloudJobManager(self.root)
        cfg_path = "configs/exp000_template.yaml"
        finished = threading.Event()
        process = Mock(pid=12345)
        process.wait.side_effect = lambda: (finished.wait(2), 1)[1]

        with patch("harness.cloud_job_runner.subprocess.Popen", return_value=process):
            job = manager.start_job(
                backend="local",
                config_path=cfg_path,
                auto_audit=False,
                auto_sync=False,
            )
            finished.set()

        self.assertIsNotNone(job.job_id)
        self.assertEqual(job.backend, "local")

        deadline = time.monotonic() + 2
        terminal_statuses = {"FAILED", "COMPLETED", "CANCELLED"}
        fetched = manager.get_job(job.job_id)
        while (
            not fetched
            or fetched["status"] not in terminal_statuses
            or job.job_id in manager._processes
        ):
            if time.monotonic() >= deadline:
                self.fail("Mocked local job monitor did not finish cleanly.")
            time.sleep(0.01)
            fetched = manager.get_job(job.job_id)
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched["job_id"], job.job_id)
        self.assertEqual(fetched["status"], "FAILED")

        jobs_list = manager.list_jobs(limit=5)
        self.assertTrue(len(jobs_list) >= 1)
        self.assertEqual(jobs_list[0]["job_id"], job.job_id)

    def test_modal_and_kaggle_auth_helpers(self):
        # 1. Parse token inputs
        raw_cmd = "modal token set --token-id ak-1234567890abcdef --token-secret as-abcdef1234567890"
        t_id, t_sec = EnvironmentDetector.parse_modal_token_input(raw_cmd)
        self.assertEqual(t_id, "ak-1234567890abcdef")
        self.assertEqual(t_sec, "as-abcdef1234567890")

        # Auth tests must never open browser tabs or start a real OAuth process.
        process = Mock()
        process.communicate.return_value = ("fixture login", "")
        with patch("webbrowser.open", return_value=True) as browser, patch(
            "harness.environment_detector.subprocess.Popen", return_value=process
        ) as launch:
            m_res = EnvironmentDetector.launch_modal_auth()
            self.assertTrue(m_res["success"])
            self.assertIn("modal.com/settings/tokens", m_res["auth_url"])
            k_res = EnvironmentDetector.launch_kaggle_auth()
            self.assertTrue(k_res["success"])
            self.assertIn("kaggle.com/settings", k_res["auth_url"])
            self.assertEqual(browser.call_count, 2)
            self.assertEqual(launch.call_args.args[0], ["kaggle", "auth", "login"])


if __name__ == "__main__":
    unittest.main()
