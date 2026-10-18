"""Tests for environment detection, cloud jobs, and per-task workspace isolation."""

import threading
import time
import unittest
from unittest.mock import Mock, patch

# Pre-import the heavy sklearn/scipy chain OUTSIDE any test so that
# first-time import side effects (scipy runs subprocess.run on some
# platforms at import time) never land inside the Popen patch window of
# test_cloud_job_manager_lifecycle. This made that test order-dependent:
# green in the full suite, red when the file ran alone.
import harness.evaluation  # noqa: F401
from harness.cloud_job_runner import CloudJobManager
from harness.environment_detector import EnvironmentDetector
from harness.project import HarnessProject
from tests.workspace_fixtures import (
    cleanup_dashboard_workspace,
    prepare_dashboard_workspace,
)
from tools.dashboard_server import (
    get_next_experiment_id,
    list_available_tasks,
    switch_task,
)


class TestEnvironmentAndCloudJobs(unittest.TestCase):
    def setUp(self):
        self.root = prepare_dashboard_workspace(self, "test_cloud_jobs_")
        project = HarnessProject.discover(self.root)
        # isolate the process-wide env cache between tests
        from harness import env_cache

        env_cache.invalidate()
        self._module_patches = [
            patch("harness.cloud_job_runner.PROJECT_ROOT", self.root),
            patch("harness.cloud_job_runner.HARNESS_PROJECT", project),
            patch("harness.cloud_job_runner.ARTIFACTS_DIR", project.artifacts_dir),
            patch("harness.environment_detector.PROJECT_ROOT", self.root),
        ]
        for module_patch in self._module_patches:
            module_patch.start()

    def tearDown(self):
        from harness import env_cache

        env_cache.invalidate()
        for module_patch in reversed(self._module_patches):
            module_patch.stop()
        cleanup_dashboard_workspace(self)

    def test_environment_detector_full_status(self):
        with patch.object(EnvironmentDetector, "detect_modal", return_value={"installed": True, "profiles": []}), \
             patch.object(EnvironmentDetector, "detect_kaggle", return_value={"installed": True, "authenticated": True}), \
             patch.object(EnvironmentDetector, "detect_docker", return_value={"installed": True, "running": True}), \
             patch.object(EnvironmentDetector, "detect_gpu", return_value={"detected": True, "count": 1}), \
             patch.object(EnvironmentDetector, "detect_llm_status", return_value={"providers": {}, "total_configured": 0}):
            status = EnvironmentDetector.get_full_status()
            self.assertIn("modal", status)
            self.assertIn("kaggle", status)
            self.assertIn("docker", status)
            self.assertIn("gpu", status)
            self.assertIn("llm", status)
            self.assertTrue(status["modal"]["installed"])
            self.assertTrue(status["kaggle"]["installed"])
            self.assertTrue(status["docker"]["installed"])
            self.assertTrue(status["gpu"]["detected"])
            # patched values must flow through get_full_status (cache included)
            self.assertEqual(status["gpu"], {"detected": True, "count": 1})

    def test_get_full_status_uses_cache_on_second_call(self):
        # Patch the PROBE (not the cached wrapper) so the wrapper's caching
        # is what is under test.
        calls = {"n": 0}

        def fake_probe_gpu():
            calls["n"] += 1
            return {"detected": True, "gpu_name": "cached-gpu"}

        with patch.object(EnvironmentDetector, "_probe_gpu", side_effect=fake_probe_gpu):
            first = EnvironmentDetector.get_full_status()
            second = EnvironmentDetector.get_full_status()
        self.assertEqual(first["gpu"], {"detected": True, "gpu_name": "cached-gpu"})
        self.assertEqual(second["gpu"], first["gpu"])
        self.assertEqual(calls["n"], 1, "second call must be served from the TTL cache")

    def test_get_full_status_force_bypasses_cache(self):
        calls = {"n": 0}

        def fake_probe_gpu():
            calls["n"] += 1
            return {"detected": True, "gpu_name": f"gpu-{calls['n']}"}

        with patch.object(EnvironmentDetector, "_probe_gpu", side_effect=fake_probe_gpu):
            first = EnvironmentDetector.get_full_status()
            second = EnvironmentDetector.get_full_status(force=True)
        self.assertNotEqual(first["gpu"], second["gpu"])
        self.assertEqual(calls["n"], 2)

    def test_gpu_probe_skips_torch_when_smi_reports(self):
        # nvidia-smi found a GPU and torch is NOT imported -> no torch import,
        # cuda_available is None ("not probed"), never a false negative.
        import sys as _sys

        saved_torch = _sys.modules.get("torch")
        _sys.modules.pop("torch", None)
        try:
            with patch("shutil.which", side_effect=lambda n: "/fake/nvidia-smi" if n == "nvidia-smi" else None), \
                 patch("subprocess.run") as run:
                run.return_value = Mock(
                    returncode=0,
                    stdout="FAKE-GPU, 4096, 2048, 550.54\n",
                )
                result = EnvironmentDetector._probe_gpu()
        finally:
            if saved_torch is not None:
                _sys.modules["torch"] = saved_torch
        self.assertTrue(result["detected"])
        self.assertEqual(result["gpu_name"], "FAKE-GPU")
        self.assertIsNone(result["cuda_available"])
        self.assertIn("skipped", result.get("cuda_probe", ""))
        # the probe must not import torch itself: presence is unchanged
        self.assertIs(_sys.modules.get("torch"), saved_torch)

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
