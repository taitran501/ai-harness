import hashlib
import ast
import base64
import io
import json
import tempfile
import unittest
from email.message import Message
from pathlib import Path
from unittest.mock import Mock, patch
import zipfile

import yaml

from harness.baseline import validate_baseline_artifact
from harness.cloud_policy import UploadConsentStore, dataset_fingerprint, private_upload_resource_names, provider_preflight
from harness.cloud_job_runner import CloudJob, CloudJobManager
from harness.data_bundle import DataBundleError, stage_declared_data
from harness.eda import run_eda
from harness.job_state import update_worker_state
from harness.phase_workflow import PHASE_IDS, PhaseWorkflow, PromptPackError
from harness.project import HarnessProject, attach_project
from harness.templates.kaggle_template import KaggleKernelManager
from harness.templates.modal_template import generate_modal_script, push_dataset_to_modal


class WorkspaceFixture(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="mlh_workspace_")
        self.root = Path(self.temp.name)
        (self.root / "configs").mkdir()
        (self.root / "dataset").mkdir()
        (self.root / "splits").mkdir()
        (self.root / "dataset" / "train.csv").write_text(
            "text,label\nnice dog,0\nkind dog,0\nawful cat,1\nmean cat,1\n",
            encoding="utf-8",
        )
        (self.root / "splits" / "folds.csv").write_text(
            "id,fold\n0,0\n1,0\n2,1\n3,1\n", encoding="utf-8"
        )
        self.task_config = self.root / "configs" / "task.yaml"
        self.task_config.write_text(
            yaml.safe_dump(
                {
                    "task": {
                        "id": "fixture-task",
                        "name": "Fixture Competition",
                        "task_type": "classification",
                        "type": "classification",
                        "primary_metric": "accuracy",
                        "direction": "maximize",
                        "data_config": {
                            "train_path": "dataset/train.csv",
                            "target_column": "label",
                            "split_file": "splits/folds.csv",
                        },
                        "validation": {
                            "strategy": "kfold",
                            "num_folds": 2,
                            "split_file": "splits/folds.csv",
                        },
                    }
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        attach_project(self.root, self.task_config)
        self.project = HarnessProject.discover(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def _phase_response(self, phase_id):
        return (
            f"# {phase_id}\n\n"
            "## Summary\n\nSummary.\n\n"
            "## Evidence\n\nEvidence and checked artifacts.\n\n"
            "## Decisions\n\nDecision.\n\n"
            "## Next steps\n\nNext step.\n"
        )

    def test_phase_order_baseline_evidence_and_revisit_loop(self):
        flow = PhaseWorkflow(self.root)
        self.assertEqual(tuple(item["id"] for item in flow.phase_status()["phases"]), PHASE_IDS)
        self.assertEqual(flow.phase_status()["next_phase"], "setup")
        with self.assertRaises(PromptPackError):
            flow.record("understand", self._phase_response("understand"))

        flow.record("setup", self._phase_response("setup"))
        flow.record("understand", self._phase_response("understand"))
        with self.assertRaisesRegex(PromptPackError, "baseline anchor"):
            flow.record("baseline", self._phase_response("baseline"))

        baseline_dir = self.project.artifacts_dir / "exp000"
        baseline_dir.mkdir(parents=True)
        (baseline_dir / "metrics.json").write_text(
            json.dumps({"experiment_id": "exp000", "accuracy": 0.75}), encoding="utf-8"
        )
        self.project.artifacts_dir.joinpath("competition_state.json").write_text(
            json.dumps(
                {
                    "gates": {
                        "baseline_locked": True,
                        "baseline_exp_id": "exp000",
                        "freeze_locked": True,
                    }
                }
            ),
            encoding="utf-8",
        )
        from tests.evidence_fixtures import seal_fixture
        from harness.task_spec import load_active_task_spec
        seal_fixture(baseline_dir,load_active_task_spec(self.project.task_config_path))
        validate_baseline_artifact(self.project.artifacts_dir, "exp000", "accuracy")

        for phase_id in PHASE_IDS[2:]:
            flow.record(phase_id, self._phase_response(phase_id))
        state = flow.phase_status()
        experiment = next(item for item in state["phases"] if item["id"] == "experiment")
        self.assertFalse(experiment["can_revisit"])
        self.assertIsNone(state["next_phase"])

        second_run = self.project.artifacts_dir / "exp001"
        second_run.mkdir()
        (second_run / "metrics.json").write_text(
            json.dumps({"experiment_id": "exp001", "accuracy": 0.80}), encoding="utf-8"
        )
        state = flow.phase_status()
        experiment = next(item for item in state["phases"] if item["id"] == "experiment")
        self.assertTrue(experiment["can_revisit"])
        self.assertIn("Task context", flow.render("experiment"))
        flow.record("experiment", self._phase_response("experiment"))

    def test_data_bundle_consent_fingerprint_and_eda_privacy(self):
        initial = dataset_fingerprint(self.project)
        store = UploadConsentStore(self.project)
        self.assertFalse(store.is_granted("kaggle", initial["fingerprint"]))
        store.grant("kaggle", initial)
        self.assertTrue(store.is_granted("kaggle", initial["fingerprint"]))

        with tempfile.TemporaryDirectory(prefix="mlh_bundle_") as bundle_dir:
            result = stage_declared_data(self.project, bundle_dir)
            manifest = json.loads((Path(bundle_dir) / "mlh-data-bundle.json").read_text(encoding="utf-8"))
            self.assertEqual(result["file_count"], 2)
            self.assertIn("dataset/train.csv", manifest["files"])
            self.assertIn("splits/folds.csv", manifest["files"])
            self.assertNotIn("nice dog", json.dumps(manifest))

        report = run_eda(self.root)
        serialized = json.dumps(report)
        self.assertNotIn("nice dog", serialized)
        self.assertIn("No raw row values", report["reports"][0]["privacy"])

        (self.root / "dataset" / "train.csv").write_text(
            "text,label\nchanged value,1\n", encoding="utf-8"
        )
        changed = dataset_fingerprint(self.project)
        self.assertNotEqual(initial["fingerprint"], changed["fingerprint"])
        self.assertFalse(store.is_granted("kaggle", changed["fingerprint"]))
        task_document = yaml.safe_load(self.task_config.read_text(encoding="utf-8"))
        task_document["task"]["data_config"]["train_path"] = "../outside.csv"
        self.task_config.write_text(yaml.safe_dump(task_document), encoding="utf-8")
        with self.assertRaises(DataBundleError):
            dataset_fingerprint(self.project)

    def test_provider_preflight_reads_official_quota_and_billing_json(self):
        from harness.cloud_policy import _USAGE_CACHE
        _USAGE_CACHE.clear()
        with patch(
            "harness.environment_detector.EnvironmentDetector.detect_kaggle",
            return_value={"installed": True, "authenticated": True, "username": "fixture-user", "executable": "kaggle"},
        ), patch(
            "harness.environment_detector.EnvironmentDetector.detect_modal",
            return_value={"installed": True, "has_credentials": True, "active_profile": "fixture-profile", "executable": "modal"},
        ), patch(
            "harness.cloud_policy.subprocess.run",
            side_effect=[
                Mock(returncode=0, stdout="Kaggle CLI 2.2.0", stderr=""),
                Mock(returncode=0, stdout='{"gpu_hours": {"remaining": 20}}', stderr=""),
                Mock(returncode=0, stdout="Modal 1.5.0", stderr=""),
                Mock(returncode=0, stdout='{"billed_cost": 0.0}', stderr=""),
            ],
        ) as run:
            kaggle = provider_preflight(self.project, "kaggle")
            modal = provider_preflight(self.project, "modal")

        self.assertEqual(kaggle["quota"]["status"], "available")
        self.assertEqual(kaggle["quota"]["data"]["gpu_hours"]["remaining"], 20)
        self.assertEqual(modal["billing"]["status"], "available")
        self.assertIn("not a remaining quota", modal["billing"]["note"])
        self.assertEqual(modal["quota"]["status"], "not_exposed")
        self.assertEqual(run.call_args_list[1].args[0], ["kaggle", "quota", "--json"])
        self.assertEqual(run.call_args_list[3].args[0], ["modal", "billing", "summary", "--json"])

    def test_private_provider_resource_names_follow_task_data_fingerprint(self):
        first = dataset_fingerprint(self.project)
        names = private_upload_resource_names(self.project, fingerprint_data=first, kaggle_username="fixture-user")
        self.assertTrue(names["kaggle_dataset_slug"].startswith("fixture-user/mlh-fixture-task-"))
        self.assertIn(first["fingerprint"][:12], names["modal_volume_name"])
        self.assertNotIn("public", names["kaggle_dataset_slug"])
        (self.root / "dataset" / "train.csv").write_text("text,label\nnew,1\n", encoding="utf-8")
        changed = private_upload_resource_names(self.project, kaggle_username="fixture-user")
        self.assertNotEqual(names["kaggle_dataset_slug"], changed["kaggle_dataset_slug"])

    def test_direct_upload_helpers_require_consent_and_reject_shared_resource_names(self):
        from harness.cloud_policy import private_upload_resource_names
        from harness.templates.kaggle_template import KaggleKernelManager

        task_doc = yaml.safe_load(self.task_config.read_text(encoding="utf-8"))
        task_doc["task"]["data_config"]["license"] = "CC0-1.0"
        self.task_config.write_text(yaml.safe_dump(task_doc), encoding="utf-8")
        fingerprint = dataset_fingerprint(self.project)
        kaggle = KaggleKernelManager.push_dataset_to_kaggle(
            dataset_slug="someone/existing-public-dataset", project_root=self.root
        )
        modal = push_dataset_to_modal(volume_name="shared-volume", project_root=self.root)
        self.assertFalse(kaggle["success"])
        self.assertFalse(modal["success"])
        self.assertIn("explicit consent", kaggle["error"])
        self.assertIn("explicit consent", modal["error"])

        store = UploadConsentStore(self.project)
        store.grant("kaggle", fingerprint)
        store.grant("modal", fingerprint)
        private_names = private_upload_resource_names(self.project, fingerprint_data=fingerprint, kaggle_username="fixture-user")
        with patch("harness.data_bundle.stage_declared_data", return_value={"file_count": 2}), patch(
            "harness.templates.kaggle_template.subprocess.run", return_value=Mock(returncode=0, stdout="created", stderr="")
        ), patch.object(KaggleKernelManager, "get_authenticated_username", return_value="fixture-user"):
            uploaded = KaggleKernelManager.push_dataset_to_kaggle(
                dataset_slug="someone/existing-public-dataset", project_root=self.root
            )
        self.assertTrue(uploaded["success"])
        self.assertEqual(uploaded["dataset_slug"], private_names["kaggle_dataset_slug"])
        self.assertEqual(uploaded["visibility"], "private")

        modal_shared = push_dataset_to_modal(volume_name="shared-volume", project_root=self.root)
        self.assertFalse(modal_shared["success"])
        self.assertEqual(modal_shared["expected_volume_name"], private_names["modal_volume_name"])

    def test_dashboard_upload_ignores_user_resource_name_and_generates_private_target(self):
        from tools.dashboard_server import DashboardHandler
        handler = object.__new__(DashboardHandler)
        handler.send_json = Mock()
        fingerprint = dataset_fingerprint(self.project)
        with patch("tools.dashboard_server.HARNESS_PROJECT", self.project), patch(
            "tools.dashboard_server.PROJECT_ROOT", self.root
        ), patch("harness.cloud_policy.dataset_fingerprint", return_value=fingerprint), patch(
            "harness.cloud_policy.UploadConsentStore.is_granted", return_value=True
        ), patch.object(KaggleKernelManager, "get_authenticated_username", return_value="fixture-user"), patch.object(
            KaggleKernelManager,
            "push_dataset_to_kaggle",
            return_value={"success": True, "dataset_slug": "fixture-user/generated", "logs": "private upload"},
        ) as push:
            handler.handle_api_cloud_push_dataset(
                {"platform": "kaggle", "dataset_slug": "someone/shared-public-dataset"}
            )

        self.assertEqual(push.call_args.kwargs["dataset_slug"], "fixture-user/mlh-fixture-task-" + fingerprint["fingerprint"][:12] + "-data")
        response = handler.send_json.call_args.args[0]
        self.assertEqual(response["dataset_fingerprint"], fingerprint["fingerprint"])
        self.assertEqual(response["visibility"], "private")

    def test_baseline_requires_finite_metric_from_matching_run(self):
        with self.assertRaisesRegex(ValueError, "LEGACY_UNVERIFIED"):
            validate_baseline_artifact(self.project.artifacts_dir, "exp000", "accuracy")
        exp_dir = self.project.artifacts_dir / "exp000"
        exp_dir.mkdir(parents=True)
        (exp_dir / "metrics.json").write_text(
            json.dumps(
                [
                    {"experiment_id": "other", "accuracy": 0.99},
                    {"experiment_id": "exp000", "accuracy": 0.75},
                ]
            ),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError,"LEGACY_UNVERIFIED"):
            validate_baseline_artifact(self.project.artifacts_dir,"exp000","accuracy")

    def test_generated_cloud_bundles_compile_and_exclude_data(self):
        config = self.root / "configs" / "exp001.yaml"
        config.write_text("experiment:\n  id: exp001\n", encoding="utf-8")
        runtime = {
            "schema_version": 1,
            "core": {"distribution": "ml-harness", "version": "0.2.0"},
            "plugin": {"id": "default", "version": "1", "install": None},
            "requirements": [],
            "bundle_files": [],
        }
        modal_source = generate_modal_script(
            "configs/exp001.yaml",
            project_root=self.root,
            task_config_path=self.task_config,
            exp_id="exp001",
            task_id="fixture-task",
            job_id="job_fixture",
            runtime_manifest=runtime,
            config_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),
        )
        compile(modal_source, "modal_fixture.py", "exec")
        self.assertIn("block_network=True", modal_source)
        self.assertNotIn("modal.Mount.from_local_dir", modal_source)

        with tempfile.TemporaryDirectory(prefix="mlh_kaggle_bundle_") as stage_dir:
            package = KaggleKernelManager.setup_kernel_package(
                stage_dir,
                "configs/exp001.yaml",
                "exp001",
                "user/fixture-data",
                project_root=self.root,
                task_config_path=self.task_config,
                task_id="fixture-task",
                job_id="job_fixture",
                runtime_manifest=runtime,
            )
            source = (package / "run_kaggle.py").read_text(encoding="utf-8")
            compile(source, "run_kaggle.py", "exec")
            bundle = next(node.args[0].value for node in ast.walk(ast.parse(source))
                          if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                          and node.func.attr == "b64decode")
            with zipfile.ZipFile(io.BytesIO(base64.b64decode(bundle))) as archive:
                names = archive.namelist()
                self.assertFalse(any(name.startswith("dataset/") or name.startswith("splits/") for name in names))

    def test_kaggle_polling_waits_for_terminal_provider_state(self):
        states = [
            {"success": True, "status": "QUEUED"},
            {"success": True, "status": "RUNNING"},
            {"success": True, "status": "COMPLETED"},
        ]
        callbacks = []
        with patch.object(KaggleKernelManager, "get_kernel_status", side_effect=states), patch(
            "harness.templates.kaggle_template.time.sleep"
        ):
            result = KaggleKernelManager.wait_for_kernel(
                "exp001", poll_interval=0, timeout_seconds=1, status_callback=callbacks.append
            )
        self.assertEqual(result["status"], "COMPLETED")
        self.assertEqual([item["status"] for item in callbacks], ["QUEUED", "RUNNING", "COMPLETED"])

        for terminal in ("FAILED", "CANCELLED"):
            with self.subTest(status=terminal), patch.object(
                KaggleKernelManager, "get_kernel_status", return_value={"success": True, "status": terminal}
            ):
                result = KaggleKernelManager.wait_for_kernel("exp001", poll_interval=0, timeout_seconds=1)
                self.assertEqual(result["status"], terminal)

    def test_remote_result_recovers_after_dashboard_restart(self):
        manager = CloudJobManager(self.root)
        job_id = "job_20260930_exp900_a1b2c3d4"
        exp_id = "exp900"
        config_sha = "config-sha"
        artifact_dir = self.project.artifacts_dir / exp_id
        artifact_dir.mkdir()
        metrics = {"experiment_id": exp_id, "accuracy": 0.84}
        metrics_bytes = json.dumps(metrics).encode("utf-8")
        (artifact_dir / "metrics.json").write_bytes(metrics_bytes)
        manifest = {
            "job_id": job_id,
            "task_id": self.project.manifest["task_id"],
            "exp_id": exp_id,
            "config_sha256": config_sha,
            "files": {"metrics.json": hashlib.sha256(metrics_bytes).hexdigest()},
        }
        (artifact_dir / "artifact_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        from tests.evidence_fixtures import seal_fixture
        from harness.task_spec import load_active_task_spec
        metrics=seal_fixture(artifact_dir,load_active_task_spec(self.project.task_config_path),.84,job_id=job_id,task_id=self.project.manifest["task_id"],config_sha256=config_sha,run_id=job_id)
        log_file = self.project.artifacts_dir / "jobs" / f"{job_id}.log"
        log_file.write_text("", encoding="utf-8")
        job = CloudJob(
            job_id=job_id,
            backend="modal",
            config_path="configs/exp900.yaml",
            exp_id=exp_id,
            status="RUNNING",
            log_file=log_file.relative_to(self.root).as_posix(),
            artifact_dir=artifact_dir.relative_to(self.root).as_posix(),
            task_id=self.project.manifest["task_id"],
            config_sha256=config_sha,
            auto_audit=False,
            auto_sync=False,
        )
        manager._persist_job(job)
        update_worker_state(
            self.root,
            job_id,
            status="COMPLETED",
            exp_id=exp_id,
            config_sha256=config_sha,
            result_metrics=metrics,
            artifact_paths=["metrics.json"],
        )

        restarted = CloudJobManager(self.root)
        recovered = restarted.get_job(job_id)
        self.assertEqual(recovered["status"], "COMPLETED")
        self.assertTrue(recovered["artifact_identity_verified"])
        self.assertEqual(recovered["result_metrics"]["accuracy"], 0.84)

    def test_dashboard_rejects_cross_origin_local_api_requests(self):
        from tools.dashboard_server import DashboardHandler

        handler = object.__new__(DashboardHandler)
        handler.headers = Message()
        handler.headers["Host"] = "127.0.0.1:8080"
        handler.headers["Origin"] = "http://attacker.example"
        handler.send_error = Mock()
        self.assertTrue(handler._reject_nonlocal_request())
        handler.send_error.assert_called_once()

        handler.headers = Message()
        handler.headers["Host"] = "127.0.0.1:8080"
        handler.headers["Origin"] = "http://127.0.0.1:8080"
        handler.send_error.reset_mock()
        self.assertFalse(handler._reject_nonlocal_request())
        handler.send_error.assert_not_called()


if __name__ == "__main__":
    unittest.main()
