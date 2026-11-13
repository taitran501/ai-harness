# Consolidated Cloud Runners E2E Test Suite
# Tier: e2e
from __future__ import annotations
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[2]

# --- Imports from tests/test_cloud_jobs_and_env.py ---
import threading
import time
from unittest.mock import Mock
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
# --- Body from tests/test_cloud_jobs_and_env.py ---
"""Tests for environment detection, cloud jobs, and per-task workspace isolation."""


# Pre-import the heavy sklearn/scipy chain OUTSIDE any test so that
# first-time import side effects (scipy runs subprocess.run on some
# platforms at import time) never land inside the Popen patch window of
# test_cloud_job_manager_lifecycle. This made that test order-dependent:
# green in the full suite, red when the file ran alone.


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


# --- Imports from tests/test_modal_cloud_modes.py ---
import unittest
from pathlib import Path
from harness.cloud_job_runner import CloudJob
from harness.templates.modal_template import (
    ModalAppManager,
    generate_modal_script,
    push_dataset_to_modal,
)
from tests.fixtures import make_workspace
# --- Body from tests/test_modal_cloud_modes.py ---
"""Tests for Modal attached (live-stream) and cloud (detached) modes."""




class TestModalAppManager(unittest.TestCase):
    def test_get_app_status_running(self):
        sample_apps = [
            {
                "app_id": "ap-123456",
                "description": "test-app",
                "state": "running",
                "created_at": "2026-11-01 00:00:00",
                "stopped_at": None,
            }
        ]
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(
                returncode=0,
                stdout=json.dumps(sample_apps),
                stderr="",
            )
            res = ModalAppManager.get_app_status("ap-123456")
            self.assertTrue(res["success"])
            self.assertEqual(res["status"], "RUNNING")
            self.assertEqual(res["state"], "running")

    def test_get_app_status_completed(self):
        sample_apps = [
            {
                "app_id": "ap-123456",
                "description": "test-app",
                "state": "stopped",
                "created_at": "2026-11-01 00:00:00",
                "stopped_at": "2026-11-01 00:10:00",
            }
        ]
        def fake_run(cmd, **kwargs):
            if cmd[:3] == ["modal", "app", "list"]:
                return Mock(returncode=0, stdout=json.dumps(sample_apps), stderr="")
            if cmd[:3] == ["modal", "app", "logs"]:
                return Mock(returncode=0, stdout="Clean exit, all done.", stderr="")
            return Mock(returncode=1)

        with patch("subprocess.run", side_effect=fake_run):
            res = ModalAppManager.get_app_status("ap-123456")
            self.assertTrue(res["success"])
            self.assertEqual(res["status"], "COMPLETED")

    def test_get_app_status_failed(self):
        sample_apps = [
            {
                "app_id": "ap-123456",
                "description": "test-app",
                "state": "stopped",
                "created_at": "2026-11-01 00:00:00",
                "stopped_at": "2026-11-01 00:05:00",
            }
        ]
        def fake_run(cmd, **kwargs):
            if cmd[:3] == ["modal", "app", "list"]:
                return Mock(returncode=0, stdout=json.dumps(sample_apps), stderr="")
            if cmd[:3] == ["modal", "app", "logs"]:
                return Mock(
                    returncode=0,
                    stdout="Traceback (most recent call last):\nRuntimeError: Out of memory",
                    stderr="",
                )
            return Mock(returncode=1)

        with patch("subprocess.run", side_effect=fake_run):
            res = ModalAppManager.get_app_status("ap-123456")
            self.assertTrue(res["success"])
            self.assertEqual(res["status"], "FAILED")

    def test_pull_app_output_verifies_manifest_and_hashes(self):
        import shutil

        from tests.evidence_fixtures import seal_fixture
        from tests.fixtures import task

        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / "source" / "exp001"
            seal_fixture(source, task())
            dest = Path(td) / "artifacts" / "exp001"

            def fake_volume_get(cmd, **kwargs):
                shutil.copytree(source, Path(cmd[5]) / "exp001")
                return Mock(returncode=0, stdout="", stderr="")

            with patch("subprocess.run", side_effect=fake_volume_get):
                res = ModalAppManager.pull_app_output(
                    exp_id="exp001", volume_name="test-volume", artifact_dir=dest,
                    expected_identity={"exp_id": "exp001"})
                self.assertTrue(res["success"], res.get("error"))
                self.assertEqual((dest / "metrics.json").read_bytes(), (source / "metrics.json").read_bytes())
                (source / "metrics.json").write_text("{}")
                res = ModalAppManager.pull_app_output(
                    exp_id="exp001", volume_name="test-volume", artifact_dir=dest,
                    expected_identity={"exp_id": "exp001"})
                self.assertFalse(res["success"])
                self.assertIn("hash mismatch", res["error"])


class TestModalScriptAndVolumeLogic(unittest.TestCase):
    def test_excluded_data_roots_are_relative_paths(self):
        root, _ = make_workspace(prefix="aih-modal-test-")
        try:
            (root / "configs" / "exp_test.yaml").write_text("model:\n  name: dummy\n", encoding="utf-8")
            script = generate_modal_script(
                "configs/exp_test.yaml",
                project_root=root,
                task_config_path=root / "configs" / "task.yaml",
                exp_id="exp_test",
                volume_name="test-vol",
            )
            # Ensure no absolute paths are leaked into excluded_data_roots
            self.assertNotIn(str(root).replace("\\", "/"), script)
            # Ensure relative paths like 'dataset' or 'splits' are used
            self.assertIn("dataset", script)
        finally:
            import shutil
            shutil.rmtree(root, ignore_errors=True)

    def test_push_dataset_skips_when_bundle_exists(self):
        root, _ = make_workspace(prefix="aih-modal-skip-")
        try:
            from harness.cloud_policy import UploadConsentStore, dataset_fingerprint
            from harness.project import HarnessProject

            p = HarnessProject.discover(root)
            fp = dataset_fingerprint(p)
            UploadConsentStore(p).grant("modal", fp)

            with patch("subprocess.run") as mock_run:
                mock_run.return_value = Mock(
                    returncode=0,
                    stdout="task-data/data\ntask-data/aih-data-bundle.json\n",
                    stderr="",
                )
                res = push_dataset_to_modal(project_root=root)
                self.assertTrue(res["success"])
                self.assertTrue(res.get("skipped_upload"))
                # modal volume put must NOT have been called
                for call in mock_run.call_args_list:
                    cmd = call[0][0]
                    self.assertNotIn("put", cmd)
        finally:
            import shutil
            shutil.rmtree(root, ignore_errors=True)


class TestModalCloudRecovery(unittest.TestCase):
    def test_recover_modal_job_running(self):
        root, _ = make_workspace(prefix="aih-rec-modal-")
        try:
            manager = CloudJobManager(root)
            job = CloudJob(
                job_id="job_modal_test",
                backend="modal",
                config_path="configs/exp.yaml",
                exp_id="exp_test",
                task_id="task_test",
                config_sha256="abcdef123456",
                artifact_dir=".ai-harness/artifacts/exp_test",
                remote_id="ap-running123",
                status="RUNNING",
            )
            manager._persist_job(job)

            with patch.object(
                ModalAppManager,
                "get_app_status",
                return_value={"success": True, "status": "RUNNING", "state": "running"},
            ):
                recovered = manager._recover_modal_job(job)
                self.assertTrue(recovered)
                self.assertEqual(job.status, "RUNNING")
                self.assertEqual(job.provider_status, "RUNNING")
        finally:
            import shutil
            shutil.rmtree(root, ignore_errors=True)

    def test_recover_modal_job_completed_with_pull(self):
        root, _ = make_workspace(prefix="aih-rec-modal-done-")
        try:
            manager = CloudJobManager(root)
            job = CloudJob(
                job_id="job_modal_test",
                backend="modal",
                config_path="configs/exp.yaml",
                exp_id="exp_test",
                task_id="task_test",
                config_sha256="abcdef123456",
                artifact_dir=".ai-harness/artifacts/exp_test",
                remote_id="ap-done123",
                status="RUNNING",
            )
            manager._persist_job(job)

            with patch.object(
                ModalAppManager,
                "get_app_status",
                return_value={"success": True, "status": "COMPLETED", "state": "stopped"},
            ), patch.object(
                ModalAppManager,
                "pull_app_output",
                return_value={"success": True, "migrated_files": ["metrics.json"]},
            ), patch.object(
                manager,
                "_verify_remote_artifacts",
                return_value=None,
            ), patch.object(
                manager,
                "_handle_post_run",
                return_value=None,
            ):
                recovered = manager._recover_modal_job(job)
                self.assertTrue(recovered)
                self.assertEqual(job.status, "COMPLETED")
                self.assertEqual(job.provider_status, "COMPLETED")
        finally:
            import shutil
            shutil.rmtree(root, ignore_errors=True)


# --- Imports from tests/test_modal_runtime_requirements.py ---
import unittest
from pathlib import Path
from harness.cli import attach_project
from harness.remote_runtime import resolve_remote_runtime
from harness.task_spec import load_active_task_spec
# --- Body from tests/test_modal_runtime_requirements.py ---
"""Tests for dynamic task runtime requirements and fast uv image building for Modal."""




class TestModalRuntimeRequirements(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="aih-test-reqs-")
        self.root = Path(self.tmp.name)
        self.configs = self.root / "configs"
        self.configs.mkdir(parents=True)
        (self.root / "dataset").mkdir(parents=True)
        (self.root / "dataset" / "train.csv").write_text("id,val\n1,2\n", encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def test_task_spec_loads_runtime_requirements(self):
        task_yaml = self.configs / "task.yaml"
        task_yaml.write_text(
            """task:
  name: llm_task
  primary_metric: accuracy
  runtime:
    requirements:
      - "torch>=2.5.0"
      - "transformers>=4.51.0"
      - "peft>=0.14.0"
""",
            encoding="utf-8",
        )
        spec = load_active_task_spec(task_yaml)
        self.assertEqual(
            spec.runtime_requirements,
            ["torch>=2.5.0", "transformers>=4.51.0", "peft>=0.14.0"],
        )

    def test_resolve_remote_runtime_includes_task_requirements(self):
        task_yaml = self.configs / "task.yaml"
        task_yaml.write_text(
            """task:
  name: llm_task
  primary_metric: accuracy
  runtime:
    requirements:
      - "torch>=2.5.0"
      - "transformers>=4.51.0"
""",
            encoding="utf-8",
        )
        attach_project(self.root, task_yaml, force=True)
        from harness.project import HarnessProject

        project = HarnessProject.discover(self.root)
        runtime = resolve_remote_runtime(project)
        self.assertIn("torch>=2.5.0", runtime["requirements"])
        self.assertIn("transformers>=4.51.0", runtime["requirements"])
        # Check deduplication: no requirement appears more than once
        self.assertEqual(len(runtime["requirements"]), len(set(runtime["requirements"])))

    def test_generate_modal_script_uses_uv_and_deduplicates(self):
        task_yaml = self.configs / "task.yaml"
        task_yaml.write_text(
            """task:
  name: llm_task
  primary_metric: accuracy
  runtime:
    requirements:
      - "torch>=2.5.0"
      - "transformers>=4.51.0"
      - "numpy>=1.24.0"
""",
            encoding="utf-8",
        )
        attach_project(self.root, task_yaml, force=True)

        exp_yaml = self.configs / "exp001.yaml"
        exp_yaml.write_text("experiment:\n  id: exp001\n", encoding="utf-8")

        script = generate_modal_script(
            config_path=str(exp_yaml),
            project_root=self.root,
            volume_name="test_vol",
        )

        # Check uv pip install is used
        self.assertIn(".pip_install(\"uv\")", script)
        self.assertIn("uv pip install --system", script)
        self.assertIn("'torch>=2.5.0'", script)
        self.assertIn("'transformers>=4.51.0'", script)

        # Compiles valid python code
        compile(script, "test_modal_script.py", "exec")



if __name__ == '__main__':
    unittest.main()
