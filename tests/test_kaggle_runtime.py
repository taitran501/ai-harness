"""Kaggle contracts exercised without an account, upload, or GPU job."""

import ast
import base64
import copy
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
import zipfile

import yaml

from harness.artifacts import validate_run
from harness.cloud_policy import UploadConsentStore, dataset_fingerprint
from harness.data_bundle import stage_declared_data
from harness.kaggle_runtime import ensure_dependencies, restore_dataset, export_run, find_dataset_mount
from harness.project import HarnessProject, attach_project
from harness.templates.kaggle_template import KaggleKernelManager
from harness.task_spec import load_active_task_spec
from tests.evidence_fixtures import seal_fixture
from tests.test_research_continuity import config, task_document


class KaggleRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="mlh-kaggle-contract-")
        self.root = Path(self.temp.name)
        (self.root / "configs").mkdir()
        (self.root / "dataset").mkdir()
        document = task_document()
        document["task"]["name"] = "Explicit Kaggle Task"
        (self.root / "configs/task.yaml").write_text(yaml.safe_dump(document), encoding="utf-8")
        (self.root / "dataset/train.csv").write_text(
            "id,text,target,fold\n" + "".join(
                f"{i},{'nice happy dog' if i % 2 == 0 else 'awful angry cat'},{i % 2},{(i // 2) % 2}\n"
                for i in range(12)), encoding="utf-8")
        self.exp_id = "exp001_kaggle"
        (self.root / "configs/exp.yaml").write_text(yaml.safe_dump(config(self.exp_id)), encoding="utf-8")
        attach_project(self.root, self.root / "configs/task.yaml")
        self.project = HarnessProject.discover(self.root)
        self.runtime = {"core": {"version": "0.2.0"},
                        "plugin": {"id": "default", "install": None},
                        "requirements": [], "bundle_files": []}

    def tearDown(self):
        self.temp.cleanup()

    def package(self, **kwargs):
        with patch.object(KaggleKernelManager, "get_authenticated_username", return_value="fixture-user"):
            return KaggleKernelManager.setup_kernel_package(
                self.root / "stage", "configs/exp.yaml", kwargs.pop("exp_id", self.exp_id),
                "fixture-user/fixture-data", project_root=self.root,
                runtime_manifest=self.runtime, **kwargs)

    def test_distribution_names_are_not_import_names_and_pins_are_enforced(self):
        with patch("harness.kaggle_runtime.metadata.version", return_value="99.0"), patch(
            "harness.kaggle_runtime.subprocess.run") as run:
            ensure_dependencies(self.runtime)
            run.assert_not_called()  # scikit-learn/PyYAML are installed distributions.
            runtime = copy.deepcopy(self.runtime)
            runtime["requirements"] = ["numpy==1.24.0"]
            with self.assertRaisesRegex(RuntimeError, "internet is disabled.*numpy==1.24.0"):
                ensure_dependencies(runtime)
            run.assert_not_called()

    def test_failed_install_stops_before_training_and_versions_are_rechecked(self):
        runtime = copy.deepcopy(self.runtime)
        runtime["requirements"] = ["numpy==1.24.0"]
        with patch("harness.kaggle_runtime.metadata.version", return_value="99.0"), patch(
            "harness.kaggle_runtime.subprocess.run", side_effect=subprocess.CalledProcessError(1, "pip")):
            with self.assertRaises(subprocess.CalledProcessError):
                ensure_dependencies(runtime, enable_internet=True)
        with patch("harness.kaggle_runtime.metadata.version", return_value="99.0"), patch(
            "harness.kaggle_runtime.subprocess.run") as run:
            with self.assertRaisesRegex(RuntimeError, "remain unsatisfied"):
                ensure_dependencies(runtime, enable_internet=True)
            self.assertIn("numpy==1.24.0", run.call_args.args[0])

    def test_directory_archives_restore_only_manifest_files_and_missing_mount_fails(self):
        mount = self.root / "mount"
        stage_declared_data(self.project, mount)
        shutil.make_archive(str(mount / "dataset"), "zip", root_dir=mount / "dataset")
        shutil.rmtree(mount / "dataset")
        destination = self.root / "restored"
        restore_dataset(mount, destination)
        self.assertEqual((destination / "dataset/train.csv").read_bytes(), (self.root / "dataset/train.csv").read_bytes())
        with self.assertRaisesRegex(RuntimeError, "selected Kaggle dataset mount is missing"):
            restore_dataset(self.root / "not-the-selected-dataset", self.root / "other")

    def test_export_preserves_all_sealed_files_and_import_from_another_cwd(self):
        run = self.root / "remote" / self.exp_id
        run.mkdir(parents=True)
        (run / "plugin_details.json").write_text("{}")
        seal_fixture(run, load_active_task_spec(self.project.task_config_path))
        output = self.root / "output"
        output.mkdir()
        identity = {"job_id": "job-fixture", "task_id": "task-fixture", "exp_id": self.exp_id}
        export_run(run, output, identity)
        self.assertEqual(len(list(output.rglob("artifact_manifest.json"))), 1)
        self.assertIn("plugin_details.json", validate_run(output)["files"])
        def download(command, **kwargs):
            shutil.copytree(output, Path(command[-1]), dirs_exist_ok=True)
            return Mock(returncode=0, stdout="downloaded", stderr="")
        with patch.object(KaggleKernelManager, "_download_output_api", side_effect=ImportError("CLI-only fixture")), patch(
            "harness.templates.kaggle_template.subprocess.run", side_effect=download):
            result = KaggleKernelManager.pull_kernel_output(
                self.exp_id, "fixture-user/kernel", project_root=self.root,
                expected_identity=identity)
        self.assertTrue(result["success"], result)
        validate_run(self.project.artifacts_dir / self.exp_id)
        self.assertEqual((self.project.artifacts_dir / self.exp_id / "plugin_details.json").read_bytes(), b"{}")

    def test_metadata_limits_include_experiment_and_use_explicit_task(self):
        package = self.package(exp_id="exp001_" + "a" * 100, job_id="a-dispatch")
        meta = json.loads((package / "kernel-metadata.json").read_text())
        self.assertTrue(5 <= len(meta["title"]) <= 50)
        self.assertEqual(meta["title"].lower().replace(" ", "-"), meta["id"].split("/")[-1])
        self.assertTrue(meta["id"].startswith("fixture-user/explicit-kaggle-task-"))
        first = meta["id"]
        meta = json.loads((self.package(exp_id="exp001_" + "a" * 100, job_id="another-dispatch") / "kernel-metadata.json").read_text())
        self.assertNotEqual(first, meta["id"])
        # Quoted filenames/IDs must remain valid Python literals.
        source = (self.package(exp_id="exp'quoted", job_id=None) / "run_kaggle.py").read_text()
        compile(source, "run_kaggle.py", "exec")

    def test_generated_script_runs_full_validation_and_exports_one_manifest(self):
        package = self.package(job_id="job-fixture")
        source = (package / "run_kaggle.py").read_text()
        bundle = next(node.args[0].value for node in ast.walk(ast.parse(source))
                      if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                      and node.func.attr == "b64decode")
        with zipfile.ZipFile(io.BytesIO(base64.b64decode(bundle))) as archive:
            self.assertIn("harness/runner.py", archive.namelist())
            self.assertFalse(any(name.startswith("dataset/") for name in archive.namelist()))
        mount = self.root / "mounted-input" / "datasets" / "fixture-user" / "fixture-data"
        stage_declared_data(self.project, mount)
        shutil.make_archive(str(mount / "dataset"), "zip", root_dir=mount / "dataset")
        shutil.rmtree(mount / "dataset")
        # Substitute the provider mount only; execute the generated program and actual runner.
        source = source.replace('Path("/kaggle/input")', f'Path({str(self.root / "mounted-input")!r})')
        script = self.root / "execute.py"
        script.write_text(source, encoding="utf-8")
        output = self.root / "kaggle-working"
        output.mkdir()
        env = os.environ.copy()
        env.pop("HARNESS_PROJECT_ROOT", None)
        env.pop("HARNESS_TASK_CONFIG", None)
        result = subprocess.run([sys.executable, str(script)], cwd=output, env=env,
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        manifest = validate_run(output)
        self.assertEqual(manifest["job_id"], "job-fixture")
        self.assertEqual(manifest["run_kind"], "full")
        self.assertEqual(len(list(output.rglob("artifact_manifest.json"))), 1)
        self.assertFalse((output / "dataset").exists())
        self.assertFalse((output / "harness").exists())

    def test_upload_auth_error_does_not_trigger_dataset_version(self):
        document = yaml.safe_load(self.project.task_config_path.read_text())
        document["task"]["data"]["license"] = "CC0-1.0"
        self.project.task_config_path.write_text(yaml.safe_dump(document))
        fingerprint = dataset_fingerprint(self.project)
        UploadConsentStore(self.project).grant("kaggle", fingerprint)
        with patch.object(KaggleKernelManager, "get_authenticated_username", return_value="fixture-user"), patch(
            "harness.templates.kaggle_template.subprocess.run", return_value=Mock(returncode=1, stdout="", stderr="403 Forbidden")) as run:
            result = KaggleKernelManager.push_dataset_to_kaggle(project_root=self.root)
        self.assertFalse(result["success"])
        self.assertEqual(run.call_count, 1)
        self.assertIn("--keep-tabular", run.call_args.args[0])

    def test_upload_printed_error_is_rejected_even_with_zero_exit(self):
        document = yaml.safe_load(self.project.task_config_path.read_text())
        document["task"]["data"]["license"] = "CC0-1.0"
        self.project.task_config_path.write_text(yaml.safe_dump(document))
        UploadConsentStore(self.project).grant("kaggle", dataset_fingerprint(self.project))
        with patch.object(KaggleKernelManager, "get_authenticated_username", return_value="fixture-user"), patch(
            "harness.templates.kaggle_template.subprocess.run", return_value=Mock(returncode=0, stdout="Dataset creation error: invalid license", stderr="")):
            result = KaggleKernelManager.push_dataset_to_kaggle(project_root=self.root)
        self.assertFalse(result["success"])

    def test_dataset_processing_is_waited_for_and_failures_stop_dispatch(self):
        states = [Mock(returncode=0, stdout="processing", stderr=""), Mock(returncode=0, stdout="ready", stderr="")]
        with patch("harness.templates.kaggle_template.subprocess.run", side_effect=states) as run, patch(
            "harness.templates.kaggle_template.time.sleep"):
            KaggleKernelManager.wait_for_dataset("fixture-user/dataset", poll_interval=0)
        self.assertEqual(run.call_count, 2)
        with patch("harness.templates.kaggle_template.subprocess.run", return_value=Mock(returncode=0, stdout="error", stderr="")):
            with self.assertRaisesRegex(RuntimeError, "processing failed"):
                KaggleKernelManager.wait_for_dataset("fixture-user/dataset")
        with self.assertRaises(TimeoutError):
            KaggleKernelManager.wait_for_dataset("fixture-user/dataset", timeout_seconds=0)

    def test_credentials_in_config_dir_resolve_real_username(self):
        credentials = self.root / "credentials"
        credentials.mkdir()
        (credentials / "credentials.json").write_text(json.dumps({"username": "oauth-fixture", "access_token": "dummy"}))
        with patch.dict(os.environ, {"KAGGLE_USERNAME": "", "KAGGLE_CONFIG_DIR": str(credentials)}):
            self.assertEqual(KaggleKernelManager.get_authenticated_username(), "oauth-fixture")

    def test_status_ignores_kernel_name_and_cli_failure_is_not_completion(self):
        slug = "fixture-user/exp-error-analysis-running"
        for raw, expected in [("complete", "COMPLETED"), ("KernelWorkerStatus.RUNNING", "RUNNING"), ("error", "FAILED")]:
            with self.subTest(raw=raw), patch("harness.templates.kaggle_template.subprocess.run",
                return_value=Mock(returncode=0, stdout=f'{slug} has status "{raw}"', stderr="")):
                self.assertEqual(KaggleKernelManager.get_kernel_status(self.exp_id, slug)["status"], expected)
        with patch("harness.templates.kaggle_template.subprocess.run",
            return_value=Mock(returncode=1, stdout='has status "complete"', stderr="403 Forbidden")):
            self.assertEqual(KaggleKernelManager.get_kernel_status(self.exp_id, slug)["status"], "ERROR")

    def test_dispatch_uses_submitted_id_and_cli_can_be_in_a_separate_environment(self):
        from harness.cloud_runner import run_via_kaggle
        context = {"project": self.project, "exp_id": self.exp_id, "runtime": self.runtime,
                   "config_rel": "configs/exp.yaml", "task_id": "task-fixture",
                   "config_sha256": "sha-fixture", "task": {},
                   "external_kaggle_dataset": "fixture-user/fixture-data",
                   "fingerprint": {"fingerprint": "fingerprint-fixture"}}
        with patch("harness.cloud_runner._prepare_remote_run", return_value=context), patch.object(
            KaggleKernelManager, "get_authenticated_username", return_value="fixture-user"), patch(
            "harness.cloud_runner.subprocess.run", return_value=Mock(returncode=0)) as push, patch.object(
            KaggleKernelManager, "wait_for_kernel", return_value={"status": "COMPLETED"}) as wait, patch.object(
            KaggleKernelManager, "pull_kernel_output", return_value={"success": True}) as pull, patch(
            "harness.cloud_runner._sync_completed_run", return_value={}), patch.dict(sys.modules, {"kaggle": None}):
            result = run_via_kaggle("configs/exp.yaml", project_root=self.root, job_id="job_20261010_exp001_fixture")
        package = Path(push.call_args.args[0][-1])
        try:
            submitted = json.loads((package / "kernel-metadata.json").read_text())["id"]
            self.assertEqual(result["kernel_slug"], submitted)
            self.assertEqual(wait.call_args.args[1], submitted)
            self.assertEqual(pull.call_args.args[1], submitted)
            self.assertEqual(pull.call_args.kwargs["project_root"], self.root)
        finally:
            # This directory was returned by the dispatch mock, not the real checkout.
            self.assertTrue(package.name.startswith("mlh-kaggle-"))
            self.assertEqual(package.parent, Path(tempfile.gettempdir()))
            shutil.rmtree(package)

    def test_failed_output_download_never_imports_partial_files(self):
        with patch.object(KaggleKernelManager, "_download_output_api", side_effect=ImportError("CLI-only fixture")), patch("harness.templates.kaggle_template.subprocess.run",
                   return_value=Mock(returncode=1, stdout="", stderr="download interrupted")):
            result = KaggleKernelManager.pull_kernel_output(self.exp_id, "fixture-user/kernel", project_root=self.root)
        self.assertFalse(result["success"])
        self.assertIn("download interrupted", result["raw_output"])
        self.assertFalse((self.project.artifacts_dir / self.exp_id).exists())

    def test_mount_resolution_matches_selected_owner_and_rejects_unrelated_data(self):
        base = self.root / "input"
        selected = base / "datasets" / "fixture-user" / "fixture-data"
        selected.mkdir(parents=True)
        other = base / "datasets" / "other-user" / "fixture-data"
        other.mkdir(parents=True)
        self.assertEqual(find_dataset_mount(base, "fixture-user/fixture-data"), selected)
        with self.assertRaisesRegex(RuntimeError, "missing"):
            find_dataset_mount(base, "fixture-user/absent")
        with self.assertRaisesRegex(RuntimeError, "missing"):
            find_dataset_mount(base, "absent-owner/fixture-data")
        with self.assertRaisesRegex(RuntimeError, "Ambiguous"):
            find_dataset_mount(base, "fixture-data")

    def test_api_download_imports_flat_output_without_a_cli_process(self):
        output = self.root / "api-output" / self.exp_id
        output.mkdir(parents=True)
        identity = {"job_id": "fixture-job", "exp_id": self.exp_id}
        seal_fixture(output, load_active_task_spec(self.project.task_config_path), **identity)
        api = Mock()
        def api_download(slug, *, path, file_pattern, quiet):
            self.assertEqual(slug, "fixture-user/kernel")
            self.assertEqual(file_pattern, r"^[^/\\]+$")
            shutil.copytree(output, Path(path), dirs_exist_ok=True)
        api.kernels_output.side_effect = api_download
        module = Mock(KaggleApi=Mock(return_value=api))
        with patch.dict(sys.modules, {"kaggle": Mock(), "kaggle.api": Mock(),
                                     "kaggle.api.kaggle_api_extended": module}), patch(
            "harness.templates.kaggle_template.subprocess.run") as cli, patch.object(
            KaggleKernelManager, "get_authenticated_username", return_value="fixture-user"):
            result = KaggleKernelManager.pull_kernel_output(self.exp_id, "fixture-user/kernel",
                project_root=self.root, expected_identity=identity)
        self.assertTrue(result["success"], result)
        cli.assert_not_called()
        api.authenticate.assert_called_once()
        validate_run(self.project.artifacts_dir / self.exp_id)


if __name__ == "__main__":
    unittest.main()
