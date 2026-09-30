"""Tests for environment detection, cloud jobs, and per-task workspace isolation."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

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

    def test_kaggle_oauth_status_reads_active_config_dir_and_username(self):
        with tempfile.TemporaryDirectory(prefix="mlh-kaggle-auth-") as config_dir:
            (Path(config_dir) / "access_token").write_text("placeholder-token", encoding="utf-8")
            with patch.dict(
                os.environ,
                {"KAGGLE_CONFIG_DIR": config_dir, "KAGGLE_API_TOKEN": "", "KAGGLE_USERNAME": "", "KAGGLE_KEY": ""},
                clear=False,
            ), patch(
                "harness.environment_detector.shutil.which", return_value="kaggle"
            ), patch(
                "harness.environment_detector.subprocess.run",
                side_effect=[
                    Mock(returncode=0, stdout="Kaggle CLI 2.2.3", stderr=""),
                    Mock(returncode=0, stdout="Usage: kaggle auth login [OPTIONS]\n  --force", stderr=""),
                    Mock(returncode=0, stdout="username: fixture-user\nauth_method: OAuth", stderr=""),
                ],
            ):
                status = EnvironmentDetector.detect_kaggle()

        self.assertTrue(status["authenticated"])
        self.assertEqual(status["username"], "fixture-user")
        self.assertEqual(status["authentication_method"], "OAuth")
        self.assertEqual(status["cli_version"], "Kaggle CLI 2.2.3")
        self.assertTrue(status["oauth_supported"])
        self.assertEqual(status["config_dir"], str(Path(config_dir).resolve()))

    def test_kaggle_connect_starts_forced_official_oauth_command(self):
        process = Mock()
        process.poll.return_value = None
        with patch("harness.environment_detector.shutil.which", return_value="/fake/bin/kaggle"), patch(
            "harness.environment_detector.subprocess.run",
            return_value=Mock(returncode=0, stdout="--force", stderr=""),
        ), patch(
            "harness.environment_detector.subprocess.Popen", return_value=process
        ) as popen:
            result = EnvironmentDetector.launch_kaggle_auth()

        try:
            self.assertTrue(result["success"])
            self.assertEqual(result["status"], "running")
            self.assertEqual(
                popen.call_args.args[0],
                ["/fake/bin/kaggle", "auth", "login", "--force"],
            )
            self.assertNotIn("auth_url", result)
        finally:
            if EnvironmentDetector._kaggle_auth_log is not None:
                EnvironmentDetector._kaggle_auth_log.close()
            EnvironmentDetector._kaggle_auth_log = None
            EnvironmentDetector._kaggle_auth_process = None
            EnvironmentDetector._kaggle_auth_state = {
                "status": "idle",
                "command": None,
                "started_at": None,
                "finished_at": None,
                "return_code": None,
            }

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

    def test_modal_token_input_parser(self):
        # 1. Parse token inputs
        raw_cmd = "modal token set --token-id ak-1234567890abcdef --token-secret as-abcdef1234567890"
        t_id, t_sec = EnvironmentDetector.parse_modal_token_input(raw_cmd)
        self.assertEqual(t_id, "ak-1234567890abcdef")
        self.assertEqual(t_sec, "as-abcdef1234567890")


if __name__ == "__main__":
    unittest.main()
