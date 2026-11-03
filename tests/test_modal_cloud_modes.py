"""Tests for Modal attached (live-stream) and cloud (detached) modes."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from harness.cloud_job_runner import CloudJob, CloudJobManager
from harness.templates.modal_template import (
    ModalAppManager,
    generate_modal_script,
    push_dataset_to_modal,
)
from tests.fixtures import make_workspace


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
        import hashlib

        with tempfile.TemporaryDirectory() as td:
            dest_dir = Path(td) / "artifacts" / "exp001"
            exp_id = "exp001"
            volume_name = "test-volume"
            content = b"metric: 0.95\n"
            content_hash = hashlib.sha256(content).hexdigest()
            manifest = {
                "schema_version": 2,
                "job_id": "job_123",
                "task_id": "task_abc",
                "exp_id": exp_id,
                "config_sha256": "abcdef123456",
                "files": {
                    "metrics.json": content_hash,
                },
            }

            def fake_volume_get(cmd, **kwargs):
                # cmd: ['modal', 'volume', 'get', volume_name, f'/artifacts/{exp_id}', target_dir, '--force']
                target = Path(cmd[5])
                exp_sub = target / exp_id
                exp_sub.mkdir(parents=True, exist_ok=True)
                (exp_sub / "metrics.json").write_bytes(content)
                (exp_sub / "artifact_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
                return Mock(returncode=0, stdout="", stderr="")

            with patch("subprocess.run", side_effect=fake_volume_get):
                res = ModalAppManager.pull_app_output(
                    exp_id=exp_id,
                    volume_name=volume_name,
                    artifact_dir=dest_dir,
                    expected_identity={
                        "job_id": "job_123",
                        "task_id": "task_abc",
                        "exp_id": exp_id,
                        "config_sha256": "abcdef123456",
                    },
                )
                self.assertTrue(res["success"])
                self.assertTrue((dest_dir / "metrics.json").is_file())
                self.assertEqual((dest_dir / "metrics.json").read_bytes(), content)


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


if __name__ == "__main__":
    unittest.main()
