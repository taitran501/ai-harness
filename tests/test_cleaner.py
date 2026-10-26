"""Safety contracts use only synthetic workspaces and isolated temp roots."""

import contextlib
import dataclasses
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
import zipfile
from pathlib import Path
from unittest.mock import Mock, patch

import yaml

from harness import cleaner
from harness.cli import main
from harness.project import HarnessProject
from tests.fixtures import make_workspace


class CleanerTests(unittest.TestCase):
    def setUp(self):
        self.root, self.config = make_workspace()
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.legacy = self.root / ".ml-harness"
        self.legacy.mkdir()
        (self.legacy / "old.bin").write_bytes(b"legacy evidence")
        self.probe = self.root / ".ai-harness/artifacts/task/probes/exp001/run1"
        self.probe.mkdir(parents=True)
        (self.probe / "probe.bin").write_bytes(b"probe")
        self.age(self.legacy)
        self.age(self.probe)

    @staticmethod
    def age(path, hours=3):
        stamp = time.time() - hours * 3600
        for entry in [*path.rglob("*"), path]:
            os.utime(entry, (stamp, stamp))

    def plan(self):
        return cleaner.scan_workspace_clutter(self.root)

    def test_default_dry_run_and_exact_scopes(self):
        official = self.root / ".ai-harness/artifacts/task/exp001"
        official.mkdir()
        (official / "keep.bin").write_bytes(b"official")
        plan = self.plan()
        self.assertEqual({i.path for i in plan}, {self.legacy, self.probe})
        self.assertTrue(all(i["status"] == "would_delete" for i in cleaner.execute_cleanup(plan)))
        self.assertTrue((self.legacy / "old.bin").exists())
        self.assertTrue(all(i["status"] == "deleted" for i in cleaner.execute_cleanup(plan, apply=True)))
        self.assertTrue((official / "keep.bin").exists())

    def test_forged_paths_refused(self):
        template = self.plan()[0]
        outside = self.root.parent / (self.root.name + "-outside")
        outside.mkdir()
        self.addCleanup(shutil.rmtree, outside)
        (outside / "keep").write_text("safe")
        for path in [self.root, outside, self.root / ".ml-harness/../.ml-harness",
                     self.root / "dataset", self.root / "configs",
                     self.root / ".ai-harness/artifacts/task/exp001"]:
            path.mkdir(exist_ok=True, parents=True)
            with self.subTest(path=path):
                item = dataclasses.replace(template, path=path)
                self.assertEqual(cleaner.execute_cleanup([item], apply=True)[0]["status"], "refused")
                self.assertTrue(path.exists())

    def test_nested_probe_location_forgery_refused(self):
        child = self.probe / "nested"
        child.mkdir()
        self.age(self.probe)
        item = next(i for i in self.plan() if i.kind == "probe")
        item = dataclasses.replace(item, path=child, snapshot=cleaner._inventory(child))
        self.assertEqual(cleaner.execute_cleanup([item], apply=True)[0]["status"], "refused")

    def test_protected_content_inside_legacy(self):
        for name in cleaner.PROTECTED_DIRS:
            with self.subTest(name=name):
                protected = self.legacy / name
                protected.mkdir()
                (protected / "keep").write_text("keep")
                self.age(self.legacy)
                self.assertNotIn(self.legacy, [i.path for i in self.plan()])
                shutil.rmtree(protected)

    def test_declared_task_data_inside_legacy_and_probe(self):
        for path in [self.legacy / "old.bin", self.probe / "probe.bin"]:
            doc = yaml.safe_load(self.config.read_text())
            doc["task"]["data_config"]["train_path"] = str(path.relative_to(self.root))
            self.config.write_text(yaml.safe_dump(doc))
            plan = self.plan()
            self.assertFalse(any(path.is_relative_to(i.path) for i in plan))

    def test_changed_task_config_rechecked_at_apply(self):
        plan = self.plan()
        doc = yaml.safe_load(self.config.read_text())
        doc["task"]["data_config"]["train_path"] = ".ml-harness/old.bin"
        self.config.write_text(yaml.safe_dump(doc))
        results = cleaner.execute_cleanup(plan, apply=True)
        self.assertEqual(next(i for i in results if i["kind"] == "legacy")["status"], "refused")
        self.assertTrue(self.legacy.exists())

    def test_active_legacy_manifest_protected(self):
        manifest = self.root / ".ai-harness/project.yaml"
        shutil.copyfile(manifest, self.legacy / "project.yaml")
        manifest.unlink()
        self.age(self.legacy)
        self.assertNotIn(self.legacy, [i.path for i in self.plan()])

    def test_active_legacy_artifacts_protected(self):
        manifest = self.root / ".ai-harness/project.yaml"
        doc = yaml.safe_load(manifest.read_text())
        doc["artifacts_dir"] = ".ml-harness/artifacts"
        manifest.write_text(yaml.safe_dump(doc))
        self.assertNotIn(self.legacy, [i.path for i in self.plan()])

    def test_recent_descendant_and_changed_plan_retained(self):
        plan = self.plan()
        (self.legacy / "old.bin").write_bytes(b"new")
        self.assertNotIn(self.legacy, [i.path for i in self.plan()])
        result = cleaner.execute_cleanup(plan, apply=True)
        self.assertEqual(next(i for i in result if i["kind"] == "legacy")["status"], "refused")
        self.assertTrue(self.legacy.exists())

    def test_archive_round_trip_and_dry_run(self):
        plan = [i for i in self.plan() if i.kind == "legacy"]
        cleaner.execute_cleanup(plan, archive=True)
        self.assertFalse((self.root / ".ai-harness/archives").exists())
        results = cleaner.execute_cleanup(plan, apply=True, archive=True)
        self.assertEqual(results[0]["status"], "deleted")
        with zipfile.ZipFile(results[0]["archive"]) as archive:
            self.assertEqual(archive.read(".ml-harness/old.bin"), b"legacy evidence")
        self.assertFalse(self.legacy.exists())

    def test_archive_failure_retains_source(self):
        plan = [i for i in self.plan() if i.kind == "legacy"]
        with patch("harness.cleaner.zipfile.ZipFile", side_effect=OSError("disk full")):
            result = cleaner.execute_cleanup(plan, apply=True, archive=True)
        self.assertEqual(result[0]["status"], "refused")
        self.assertTrue((self.legacy / "old.bin").exists())

    def test_archive_validation_failure_and_source_change_retain_source(self):
        plan = [i for i in self.plan() if i.kind == "legacy"]
        with patch("harness.cleaner.zipfile.ZipFile.testzip", return_value="bad.bin"):
            self.assertEqual(cleaner.execute_cleanup(plan, apply=True, archive=True)[0]["status"], "refused")
        real_archive = cleaner._archive
        def changing_archive(item):
            target = real_archive(item)
            (self.legacy / "old.bin").write_bytes(b"changed during archive")
            return target
        with patch("harness.cleaner._archive", side_effect=changing_archive):
            self.assertEqual(cleaner.execute_cleanup(plan, apply=True, archive=True)[0]["status"], "refused")
        self.assertEqual((self.legacy / "old.bin").read_bytes(), b"changed during archive")

    def test_malformed_or_missing_attached_config_fails_closed(self):
        plan = self.plan()
        for contents in ["task: [invalid]", "[", "[]"]:
            self.config.write_text(contents)
            self.assertEqual(self.plan(), [])
            self.assertTrue(all(i["status"] == "refused" for i in cleaner.execute_cleanup(plan, apply=True)))
        self.config.unlink()
        self.assertEqual(self.plan(), [])

    def test_candidate_link_swapped_after_scan_is_refused(self):
        plan = [i for i in self.plan() if i.kind == "legacy"]
        with tempfile.TemporaryDirectory() as outside:
            target = Path(outside)
            (target / "keep").write_text("keep")
            shutil.rmtree(self.legacy)
            try:
                self.legacy.symlink_to(target, target_is_directory=True)
            except OSError:
                if os.name != "nt":
                    self.skipTest("Symlink unavailable")
                subprocess.run(["cmd", "/c", "mklink", "/J", str(self.legacy), str(target)],
                               check=True, capture_output=True)
            try:
                result = cleaner.execute_cleanup(plan, apply=True)
                self.assertEqual(result[0]["status"], "refused")
                self.assertEqual((target / "keep").read_text(), "keep")
            finally:
                if self.legacy.is_symlink():
                    self.legacy.unlink()
                else:
                    self.legacy.rmdir()

    def test_archive_destination_link_refused_before_writing(self):
        plan = [i for i in self.plan() if i.kind == "legacy"]
        with tempfile.TemporaryDirectory() as outside:
            target = Path(outside)
            link = self.root / ".ai-harness/archives"
            try:
                link.symlink_to(target, target_is_directory=True)
            except OSError:
                if os.name != "nt":
                    self.skipTest("Symlink unavailable")
                subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)],
                               check=True, capture_output=True)
            try:
                self.assertEqual(cleaner.execute_cleanup(plan, apply=True, archive=True)[0]["status"], "refused")
                self.assertEqual(list(target.iterdir()), [])
                self.assertTrue(self.legacy.exists())
            finally:
                if link.is_symlink():
                    link.unlink()
                else:
                    link.rmdir()

    def test_symlink_or_junction_and_ancestor_refused(self):
        with tempfile.TemporaryDirectory() as outside:
            target = Path(outside)
            (target / "keep").write_text("keep")
            link = self.legacy / "linked"
            try:
                link.symlink_to(target, target_is_directory=True)
            except OSError:
                if os.name != "nt":
                    self.skipTest("Symlink creation unavailable")
                subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], check=True,
                               capture_output=True)
            try:
                self.assertNotIn(self.legacy, [i.path for i in self.plan()])
                with self.assertRaises(ValueError):
                    cleaner.scan_workspace_clutter(link)
                self.assertEqual((target / "keep").read_text(), "keep")
            finally:
                if link.is_symlink():
                    link.unlink()
                else:
                    link.rmdir()

    def test_cli_preview_confirmation_and_yes(self):
        args = ["clean", "--legacy", "--project-root", str(self.root)]
        with contextlib.redirect_stdout(io.StringIO()), patch("builtins.input") as prompt:
            self.assertEqual(main(args), 0)
            prompt.assert_not_called()
        with contextlib.redirect_stdout(io.StringIO()), patch("builtins.input", return_value="no"):
            self.assertEqual(main([*args, "--apply"]), 1)
        self.assertTrue(self.legacy.exists())
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main([*args, "--apply", "-y", "--archive"]), 0)
        self.assertFalse(self.legacy.exists())
        self.assertTrue(self.probe.exists())

    def test_cli_requires_category_and_rejects_conflicting_flags(self):
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(["clean"]), 2)
            with self.assertRaises(SystemExit):
                main(["clean", "--dry-run", "--apply"])

    def test_invalid_age(self):
        for age in [-1, float("nan"), float("inf")]:
            with self.assertRaises(ValueError):
                cleaner.scan_workspace_clutter(self.root, age)
            with self.assertRaises(ValueError):
                cleaner.scan_system_temp(age)

    def test_legacy_hygiene_advisory(self):
        from harness.hygiene import ALLOWED_TOP_DIRS, scan_workspace_hygiene

        self.assertNotIn(".ml-harness", ALLOWED_TOP_DIRS)
        findings = scan_workspace_hygiene(self.root)["findings"]
        self.assertTrue(any(i["kind"] == "legacy_harness_dir" and "clean --legacy" in i["detail"] for i in findings))

    def test_active_probe_run_refused(self):
        # RUNNING status in artifact_manifest.json
        (self.probe / "artifact_manifest.json").write_text(json.dumps({"status": "RUNNING"}), encoding="utf-8")
        self.age(self.probe)
        plan = self.plan()
        self.assertNotIn(self.probe, {i.path for i in plan})

        # Lock file presence also refuses
        (self.probe / "artifact_manifest.json").write_text(json.dumps({"status": "COMPLETED"}), encoding="utf-8")
        (self.probe / ".state.lock").write_text("lock", encoding="utf-8")
        self.age(self.probe)
        plan = self.plan()
        self.assertNotIn(self.probe, {i.path for i in plan})


class TempCleanerTests(unittest.TestCase):
    def test_declared_data_in_system_temp_protected_and_rechecked(self):
        workspace, config = make_workspace()
        self.addCleanup(shutil.rmtree, workspace, ignore_errors=True)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "aih-kaggle-job_data"
            path.mkdir()
            (path / "train.csv").write_text("important data")
            CleanerTests.age(path)
            with patch("harness.cleaner.tempfile.gettempdir", return_value=temp):
                plan = cleaner.scan_system_temp(project_root=workspace)
                self.assertEqual(len(plan), 1)
                doc = yaml.safe_load(config.read_text())
                doc["task"]["data_config"]["train_path"] = str(path / "train.csv")
                config.write_text(yaml.safe_dump(doc))
                self.assertEqual(cleaner.scan_system_temp(project_root=workspace), [])
                self.assertEqual(cleaner.execute_cleanup(plan, apply=True)[0]["status"], "refused")
            self.assertTrue((path / "train.csv").exists())

    def test_temp_recent_descendant_and_active_marker_added_after_scan(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "harness_kernel_fixture"
            path.mkdir()
            (path / "bundle").write_text("data")
            CleanerTests.age(path)
            with patch("harness.cleaner.tempfile.gettempdir", return_value=temp):
                plan = cleaner.scan_system_temp()
                self.assertEqual(len(plan), 1)
                (path / "bundle").write_text("recent data")
                self.assertEqual(cleaner.scan_system_temp(), [])
                CleanerTests.age(path)
                (path / cleaner.STAGING_MARKER).write_text(str(os.getpid()))
                CleanerTests.age(path)
                self.assertEqual(cleaner.execute_cleanup(plan, apply=True)[0]["status"], "refused")
            self.assertTrue(path.exists())

    def test_exact_prefix_age_floor_and_live_pid(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for prefix in (*cleaner.TEMP_PREFIXES, "aih-kaggle-other", "other", "aih-modal-other"):
                path = root / (prefix + "fixture")
                path.mkdir()
                (path / "bundle").write_bytes(b"data")
                CleanerTests.age(path)
            recent = root / "aih-kaggle-job_recent"
            recent.mkdir()
            CleanerTests.age(recent, hours=1)
            active = root / "aih-modal-job_active"
            active.mkdir()
            (active / cleaner.STAGING_MARKER).write_text(str(os.getpid()))
            CleanerTests.age(active)
            with patch("harness.cleaner.tempfile.gettempdir", return_value=temp):
                plan = cleaner.scan_system_temp(older_than_hours=0)
                self.assertEqual(len(plan), 4)
                cleaner.execute_cleanup(plan)
                self.assertTrue(all(i.path.exists() for i in plan))
                results = cleaner.execute_cleanup(plan, apply=True)
                self.assertTrue(all(i["status"] == "deleted" for i in results))
            self.assertTrue(recent.exists())
            self.assertTrue(active.exists())
            self.assertTrue((root / "otherfixture").exists())

    def test_temp_plan_cannot_redirect_root_or_prefix(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "aih-kaggle-job_fixture"
            path.mkdir()
            CleanerTests.age(path)
            with patch("harness.cleaner.tempfile.gettempdir", return_value=temp):
                item = cleaner.scan_system_temp()[0]
                for forged in [dataclasses.replace(item, path=root),
                               dataclasses.replace(item, root=root.parent),
                               dataclasses.replace(item, path=root / "other")]:
                    self.assertEqual(cleaner.execute_cleanup([forged], apply=True)[0]["status"], "refused")
            self.assertTrue(path.exists())


class StagingLifecycleTests(unittest.TestCase):
    def test_runner_refuses_staging_outside_temp_sandbox(self):
        from harness.cloud_runner import _release_staging

        with tempfile.TemporaryDirectory() as stage, tempfile.TemporaryDirectory() as sandbox:
            path = Path(stage)
            (path / "keep").write_text("keep")
            with patch("harness.cloud_runner.tempfile.gettempdir", return_value=sandbox), contextlib.redirect_stderr(io.StringIO()):
                _release_staging(path, False)
            self.assertTrue((path / "keep").exists())

    def setUp(self):
        self.root, _ = make_workspace()
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.project = HarnessProject.discover(self.root)
        (self.root / "configs/exp.yaml").write_text("experiment:\n  id: exp001\n", encoding="utf-8")
        self.context = {"project": self.project, "exp_id": "exp001", "runtime": {
            "core": {"version": "test"}, "plugin": {"id": "test"}},
            "config_rel": "configs/exp.yaml", "task_id": "fixture-task",
            "config_sha256": "sha", "task": {}, "external_kaggle_dataset": "user/data",
            "fingerprint": {"fingerprint": "fixture"}}

    def test_kaggle_success_failure_dry_run_and_keep(self):
        from harness.cloud_runner import run_via_kaggle

        for mode in ["success", "package_failure", "push_failure", "pull_failure", "dry_run", "keep"]:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temp, contextlib.ExitStack() as stack:
                # Only stage allocation is redirected, never scan the actual system temp.
                stage = Path(temp) / "aih-kaggle-job_fixture"
                stage.mkdir()
                stack.enter_context(patch("harness.cloud_runner.tempfile.mkdtemp", return_value=str(stage)))
                stack.enter_context(patch("harness.cloud_runner._prepare_remote_run", return_value=self.context))
                stack.enter_context(patch("harness.cloud_runner.KaggleKernelManager.ensure_cli_ready"))
                def package(*args, **kwargs):
                    if mode == "package_failure":
                        raise RuntimeError("package failed")
                    (stage / "kernel-metadata.json").write_text(json.dumps({"id": "user/kernel"}))
                stack.enter_context(patch("harness.cloud_runner.KaggleKernelManager.setup_kernel_package", side_effect=package))
                stack.enter_context(patch("harness.cloud_runner.KaggleKernelManager.push_code_bundle",
                                          side_effect=RuntimeError("push failed") if mode == "push_failure" else None))
                stack.enter_context(patch("harness.cloud_runner.subprocess.run", return_value=Mock(returncode=0)))
                stack.enter_context(patch("harness.cloud_runner.KaggleKernelManager.wait_for_kernel", return_value={"status": "COMPLETED"}))
                stack.enter_context(patch("harness.cloud_runner.KaggleKernelManager.pull_kernel_output",
                                          return_value={"success": mode != "pull_failure"}))
                stack.enter_context(patch("harness.cloud_runner._sync_completed_run", return_value={}))
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                if "failure" in mode:
                    with self.assertRaises(RuntimeError):
                        run_via_kaggle("configs/exp.yaml")
                else:
                    run_via_kaggle("configs/exp.yaml", dry_run=mode == "dry_run", keep_staging=mode == "keep")
                self.assertEqual(stage.exists(), mode == "keep")
                if stage.exists():
                    self.assertFalse((stage / cleaner.STAGING_MARKER).exists())

    def test_modal_success_failure_and_keep(self):
        from harness.cloud_runner import run_via_modal

        for mode in ["success", "failure", "keep"]:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temp, contextlib.ExitStack() as stack:
                stage = Path(temp) / "aih-modal-job_fixture"
                stage.mkdir()
                artifact = self.project.artifacts_dir / "exp001"
                artifact.mkdir(exist_ok=True, parents=True)
                (artifact / "artifact_manifest.json").write_text(json.dumps({
                    "job_id": "modal_exp001", "task_id": "fixture-task", "exp_id": "exp001", "config_sha256": "sha"}))
                stack.enter_context(patch.dict(sys.modules, {"modal": Mock()}))
                stack.enter_context(patch("harness.cloud_runner.tempfile.mkdtemp", return_value=str(stage)))
                stack.enter_context(patch("harness.cloud_runner._prepare_remote_run", return_value=self.context))
                stack.enter_context(patch("harness.cloud_runner.private_upload_resource_names", return_value={"modal_volume_name": "test"}))
                stack.enter_context(patch("harness.cloud_runner.push_dataset_to_modal", return_value={"success": True}))
                stack.enter_context(patch("harness.cloud_runner.generate_modal_script", return_value="# fixture"))
                stack.enter_context(patch("harness.cloud_runner.subprocess.run", return_value=Mock(returncode=1 if mode == "failure" else 0)))
                stack.enter_context(patch("harness.cloud_runner._sync_completed_run", return_value={}))
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                if mode == "failure":
                    with self.assertRaises(RuntimeError):
                        run_via_modal("configs/exp.yaml")
                else:
                    run_via_modal("configs/exp.yaml", keep_staging=mode == "keep")
                self.assertEqual(stage.exists(), mode == "keep")

    def test_kernel_dispatch_writes_staging_marker(self):
        from harness.kernel_dispatch import dispatch_kernel
        nb = self.root / "notebook.ipynb"
        nb.write_text("{}", encoding="utf-8")
        client = Mock()
        client.kernels_push.return_value = Mock(ref="user/test-kernel", url=None, kernel_id=1, version_number=1)
        res = dispatch_kernel(
            notebook_path=str(nb),
            kernel_id="user/test-kernel",
            title="Test Kernel",
            client=client,
            keep_staging=True,
            watch=False,
        )
        staging_dir = Path(res["staging_dir"])
        self.addCleanup(shutil.rmtree, staging_dir, ignore_errors=True)
        marker = staging_dir / cleaner.STAGING_MARKER
        self.assertTrue(marker.is_file())
        self.assertEqual(int(marker.read_text().strip()), os.getpid())

if __name__ == "__main__":
    unittest.main()
