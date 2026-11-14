"""Integration test suite consolidating workspace cleaner, env cache, repair, and status report.

Consolidates:
- tests/test_cleaner.py
- tests/test_env_cache.py
- tests/test_repair.py
- tests/test_repair_integrity.py
- tests/test_status_report.py
"""

from __future__ import annotations

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

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from harness.artifacts import sha256, validate_run
from harness import cleaner
from harness.cli import build_parser as cli_build_parser, main as cli_main
from harness import env_cache
from harness.project import HarnessProject
from harness import repair as repair_mod
from harness.repair import (
    EvidenceDriftError,
    build_parser as repair_build_parser,
    classify_run,
    main as repair_cli_main,
    repair_run_dir,
    scan_artifacts,
)
from harness.status_report import collect_status, render_markdown
from tests.fixtures import make_workspace
from tests.evidence_fixtures import drift_metrics, make_project, seal_into


# ===========================================================================
# 1. Cleaner tests (from tests/test_cleaner.py)
# ===========================================================================


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
        for path in [
            self.root,
            outside,
            self.root / ".ml-harness/../.ml-harness",
            self.root / "dataset",
            self.root / "configs",
            self.root / ".ai-harness/artifacts/task/exp001",
        ]:
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
            doc = yaml.safe_load(self.config.read_text(encoding="utf-8"))
            doc["task"]["data_config"]["train_path"] = str(path.relative_to(self.root))
            self.config.write_text(yaml.safe_dump(doc), encoding="utf-8")
            plan = self.plan()
            self.assertFalse(any(path.is_relative_to(i.path) for i in plan))

    def test_changed_task_config_rechecked_at_apply(self):
        plan = self.plan()
        doc = yaml.safe_load(self.config.read_text(encoding="utf-8"))
        doc["task"]["data_config"]["train_path"] = ".ml-harness/old.bin"
        self.config.write_text(yaml.safe_dump(doc), encoding="utf-8")
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
        doc = yaml.safe_load(manifest.read_text(encoding="utf-8"))
        doc["artifacts_dir"] = ".ml-harness/artifacts"
        manifest.write_text(yaml.safe_dump(doc), encoding="utf-8")
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
            self.config.write_text(contents, encoding="utf-8")
            self.assertEqual(self.plan(), [])
            self.assertTrue(all(i["status"] == "refused" for i in cleaner.execute_cleanup(plan, apply=True)))
        self.config.unlink()
        self.assertEqual(self.plan(), [])

    def test_candidate_link_swapped_after_scan_is_refused(self):
        plan = [i for i in self.plan() if i.kind == "legacy"]
        with tempfile.TemporaryDirectory() as outside:
            target = Path(outside)
            (target / "keep").write_text("keep", encoding="utf-8")
            shutil.rmtree(self.legacy)
            try:
                self.legacy.symlink_to(target, target_is_directory=True)
            except OSError:
                if os.name != "nt":
                    self.skipTest("Symlink unavailable")
                subprocess.run(
                    ["cmd", "/c", "mklink", "/J", str(self.legacy), str(target)],
                    check=True,
                    capture_output=True,
                )
            try:
                result = cleaner.execute_cleanup(plan, apply=True)
                self.assertEqual(result[0]["status"], "refused")
                self.assertEqual((target / "keep").read_text(encoding="utf-8"), "keep")
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
                subprocess.run(
                    ["cmd", "/c", "mklink", "/J", str(link), str(target)],
                    check=True,
                    capture_output=True,
                )
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
            (target / "keep").write_text("keep", encoding="utf-8")
            link = self.legacy / "linked"
            try:
                link.symlink_to(target, target_is_directory=True)
            except OSError:
                if os.name != "nt":
                    self.skipTest("Symlink creation unavailable")
                subprocess.run(
                    ["cmd", "/c", "mklink", "/J", str(link), str(target)],
                    check=True,
                    capture_output=True,
                )
            try:
                self.assertNotIn(self.legacy, [i.path for i in self.plan()])
                with self.assertRaises(ValueError):
                    cleaner.scan_workspace_clutter(link)
                self.assertEqual((target / "keep").read_text(encoding="utf-8"), "keep")
            finally:
                if link.is_symlink():
                    link.unlink()
                else:
                    link.rmdir()

    def test_cli_preview_confirmation_and_yes(self):
        args = ["clean", "--legacy", "--project-root", str(self.root)]
        with contextlib.redirect_stdout(io.StringIO()), patch("builtins.input") as prompt:
            self.assertEqual(cli_main(args), 0)
            prompt.assert_not_called()
        with contextlib.redirect_stdout(io.StringIO()), patch("builtins.input", return_value="no"):
            self.assertEqual(cli_main([*args, "--apply"]), 1)
        self.assertTrue(self.legacy.exists())
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli_main([*args, "--apply", "-y", "--archive"]), 0)
        self.assertFalse(self.legacy.exists())
        self.assertTrue(self.probe.exists())

    def test_cli_requires_category_and_rejects_conflicting_flags(self):
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(cli_main(["clean"]), 2)
            with self.assertRaises(SystemExit):
                cli_main(["clean", "--dry-run", "--apply"])

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
        (self.probe / "artifact_manifest.json").write_text(json.dumps({"status": "RUNNING"}), encoding="utf-8")
        self.age(self.probe)
        plan = self.plan()
        self.assertNotIn(self.probe, {i.path for i in plan})

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
            (path / "train.csv").write_text("important data", encoding="utf-8")
            CleanerTests.age(path)
            with patch("harness.cleaner.tempfile.gettempdir", return_value=temp):
                plan = cleaner.scan_system_temp(project_root=workspace)
                self.assertEqual(len(plan), 1)
                doc = yaml.safe_load(config.read_text(encoding="utf-8"))
                doc["task"]["data_config"]["train_path"] = str(path / "train.csv")
                config.write_text(yaml.safe_dump(doc), encoding="utf-8")
                self.assertEqual(cleaner.scan_system_temp(project_root=workspace), [])
                self.assertEqual(cleaner.execute_cleanup(plan, apply=True)[0]["status"], "refused")
            self.assertTrue((path / "train.csv").exists())

    def test_temp_recent_descendant_and_active_marker_added_after_scan(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "harness_kernel_fixture"
            path.mkdir()
            (path / "bundle").write_text("data", encoding="utf-8")
            CleanerTests.age(path)
            with patch("harness.cleaner.tempfile.gettempdir", return_value=temp):
                plan = cleaner.scan_system_temp()
                self.assertEqual(len(plan), 1)
                (path / "bundle").write_text("recent data", encoding="utf-8")
                self.assertEqual(cleaner.scan_system_temp(), [])
                CleanerTests.age(path)
                (path / cleaner.STAGING_MARKER).write_text(str(os.getpid()), encoding="utf-8")
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
            (active / cleaner.STAGING_MARKER).write_text(str(os.getpid()), encoding="utf-8")
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
                for forged in [
                    dataclasses.replace(item, path=root),
                    dataclasses.replace(item, root=root.parent),
                    dataclasses.replace(item, path=root / "other"),
                ]:
                    self.assertEqual(cleaner.execute_cleanup([forged], apply=True)[0]["status"], "refused")
            self.assertTrue(path.exists())


class StagingLifecycleTests(unittest.TestCase):
    def test_runner_refuses_staging_outside_temp_sandbox(self):
        from harness.cloud_runner import _release_staging

        with tempfile.TemporaryDirectory() as stage, tempfile.TemporaryDirectory() as sandbox:
            path = Path(stage)
            (path / "keep").write_text("keep", encoding="utf-8")
            with patch("harness.cloud_runner.tempfile.gettempdir", return_value=sandbox), contextlib.redirect_stderr(io.StringIO()):
                _release_staging(path, False)
            self.assertTrue((path / "keep").exists())

    def setUp(self):
        self.root, _ = make_workspace()
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)
        self.project = HarnessProject.discover(self.root)
        (self.root / "configs/exp.yaml").write_text("experiment:\n  id: exp001\n", encoding="utf-8")
        self.context = {
            "project": self.project,
            "cfg": {"experiment": {"id": "exp001"}},
            "exp_id": "exp001",
            "runtime": {"core": {"version": "test"}, "plugin": {"id": "test"}},
            "config_rel": "configs/exp.yaml",
            "task_id": "fixture-task",
            "config_sha256": "sha",
            "task": {},
            "external_kaggle_dataset": "user/data",
            "fingerprint": {"fingerprint": "fixture"},
        }

    def test_kaggle_success_failure_dry_run_and_keep(self):
        from harness.cloud_runner import run_via_kaggle

        for mode in ["success", "package_failure", "push_failure", "pull_failure", "dry_run", "keep"]:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as temp, contextlib.ExitStack() as stack:
                stage = Path(temp) / "aih-kaggle-job_fixture"
                stage.mkdir()
                stack.enter_context(patch("harness.cloud_runner.tempfile.mkdtemp", return_value=str(stage)))
                stack.enter_context(patch("harness.cloud_runner._prepare_remote_run", return_value=self.context))
                stack.enter_context(patch("harness.cloud_runner.KaggleKernelManager.ensure_cli_ready"))

                def package(*args, **kwargs):
                    if mode == "package_failure":
                        raise RuntimeError("package failed")
                    (stage / "kernel-metadata.json").write_text(json.dumps({"id": "user/kernel"}), encoding="utf-8")

                stack.enter_context(patch("harness.cloud_runner.KaggleKernelManager.setup_kernel_package", side_effect=package))
                stack.enter_context(
                    patch(
                        "harness.cloud_runner.KaggleKernelManager.push_code_bundle",
                        side_effect=RuntimeError("push failed") if mode == "push_failure" else None,
                    )
                )
                stack.enter_context(patch("harness.cloud_runner.subprocess.run", return_value=Mock(returncode=0)))
                stack.enter_context(patch("harness.cloud_runner.KaggleKernelManager.wait_for_kernel", return_value={"status": "COMPLETED"}))
                stack.enter_context(
                    patch(
                        "harness.cloud_runner.KaggleKernelManager.pull_kernel_output",
                        return_value={"success": mode != "pull_failure"},
                    )
                )
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
                (artifact / "artifact_manifest.json").write_text(
                    json.dumps({
                        "job_id": "modal_exp001_fixjobid1234",
                        "task_id": "fixture-task",
                        "exp_id": "exp001",
                        "config_sha256": "sha",
                    }),
                    encoding="utf-8",
                )
                stack.enter_context(patch("harness.cloud_runner.uuid.uuid4", return_value=Mock(hex="fixjobid1234567890")))
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
        self.assertEqual(int(marker.read_text(encoding="utf-8").strip()), os.getpid())


# ===========================================================================
# 2. Environment cache tests (from tests/test_env_cache.py)
# ===========================================================================


class EnvCacheTests(unittest.TestCase):
    def setUp(self):
        env_cache.invalidate()

    def tearDown(self):
        env_cache.invalidate()
        for var in ("AI_HARNESS_ENV_CACHE", "AI_HARNESS_ENV_TTL_GPU"):
            os.environ.pop(var, None)

    def test_set_and_get(self):
        env_cache.set_cached("gpu", {"detected": True})
        self.assertEqual(env_cache.get_cached("gpu"), {"detected": True})

    def test_expired_entry_returns_none(self):
        env_cache.set_cached("gpu", {"detected": True}, ttl=0.01)
        time.sleep(0.03)
        self.assertIsNone(env_cache.get_cached("gpu"))

    def test_ttl_override_env_var(self):
        os.environ["AI_HARNESS_ENV_TTL_GPU"] = "0.01"
        env_cache.set_cached("gpu", {"a": 1})
        time.sleep(0.03)
        self.assertIsNone(env_cache.get_cached("gpu"))

    def test_invalid_ttl_override_falls_back_to_default(self):
        os.environ["AI_HARNESS_ENV_TTL_GPU"] = "not-a-number"
        self.assertEqual(env_cache._ttl_for("gpu"), env_cache.DEFAULT_TTLS["gpu"])

    def test_disabled_cache_never_caches(self):
        os.environ["AI_HARNESS_ENV_CACHE"] = "off"
        env_cache.set_cached("gpu", {"detected": True})
        self.assertIsNone(env_cache.get_cached("gpu"))

    def test_cached_probe_runs_probe_once(self):
        calls = {"n": 0}

        def probe():
            calls["n"] += 1
            return {"n": calls["n"]}

        first = env_cache.cached_probe("kaggle", probe)
        second = env_cache.cached_probe("kaggle", probe)
        self.assertEqual(first, {"n": 1})
        self.assertEqual(second, {"n": 1})
        self.assertEqual(calls["n"], 1)

    def test_invalidate_clears_everything(self):
        env_cache.set_cached("modal", 1)
        env_cache.set_cached("kaggle", 2)
        env_cache.invalidate()
        self.assertIsNone(env_cache.get_cached("modal"))
        self.assertIsNone(env_cache.get_cached("kaggle"))


class DiskCacheTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.cache = Path(self._tmp.name) / "env_probes.json"

    def tearDown(self):
        env_cache.invalidate()
        for var in ("AI_HARNESS_ENV_CACHE", "AI_HARNESS_ENV_TTL_GPU"):
            os.environ.pop(var, None)
        self._tmp.cleanup()

    def test_roundtrip(self):
        env_cache.disk_store(self.cache, {"gpu": {"detected": True}})
        loaded = env_cache.disk_load(self.cache)
        self.assertEqual(loaded, {"gpu": {"detected": True}})

    def test_expired_record_returns_none(self):
        self.cache.parent.mkdir(parents=True, exist_ok=True)
        record = {"generated_at": time.time() - (env_cache.DISK_TTL + 1), "data": {"gpu": {}}}
        self.cache.write_text(json.dumps(record), encoding="utf-8")
        self.assertIsNone(env_cache.disk_load(self.cache))

    def test_corrupt_record_returns_none(self):
        self.cache.parent.mkdir(parents=True, exist_ok=True)
        self.cache.write_text("{not json", encoding="utf-8")
        self.assertIsNone(env_cache.disk_load(self.cache))

    def test_missing_file_returns_none(self):
        self.assertIsNone(env_cache.disk_load(self.cache))

    def test_disabled_cache_skips_disk(self):
        os.environ["AI_HARNESS_ENV_CACHE"] = "off"
        env_cache.disk_store(self.cache, {"gpu": {}})
        self.assertFalse(self.cache.exists())
        self.assertIsNone(env_cache.disk_load(self.cache))

    def test_get_full_status_uses_disk_cache_across_processes(self):
        from harness.environment_detector import EnvironmentDetector

        calls = {"n": 0}

        def fake_probe():
            calls["n"] += 1
            return {"fake": calls["n"], "detected": True}

        with unittest.mock.patch.object(
            EnvironmentDetector, "detect_modal", classmethod(lambda cls: fake_probe())
        ), unittest.mock.patch.object(
            EnvironmentDetector, "detect_kaggle", classmethod(lambda cls: fake_probe())
        ), unittest.mock.patch.object(
            EnvironmentDetector, "detect_docker", classmethod(lambda cls: fake_probe())
        ), unittest.mock.patch.object(
            EnvironmentDetector, "detect_gpu", classmethod(lambda cls: fake_probe())
        ), unittest.mock.patch.object(
            EnvironmentDetector, "detect_llm_status", classmethod(lambda cls: fake_probe())
        ):
            first = EnvironmentDetector.get_full_status(disk_cache_dir=self.cache.parent)
        self.assertEqual(calls["n"], 5)

        env_cache.invalidate()
        with unittest.mock.patch.object(
            EnvironmentDetector,
            "_probe_gpu",
            side_effect=AssertionError("probe must not re-run when the disk record is fresh"),
        ), unittest.mock.patch.object(
            EnvironmentDetector,
            "_probe_llm_status",
            side_effect=AssertionError("probe must not re-run when the disk record is fresh"),
        ):
            second = EnvironmentDetector.get_full_status(disk_cache_dir=self.cache.parent)
        self.assertEqual(calls["n"], 5, "no probe may re-run when the disk record is fresh")
        self.assertEqual(first["gpu"], second["gpu"])

    def test_force_bypasses_disk_cache(self):
        from harness.environment_detector import EnvironmentDetector

        calls = {"n": 0}

        def fake_probe():
            calls["n"] += 1
            return {"n": calls["n"]}

        for _ in range(2):
            with unittest.mock.patch.object(
                EnvironmentDetector, "detect_modal", classmethod(lambda cls: fake_probe())
            ), unittest.mock.patch.object(
                EnvironmentDetector, "detect_kaggle", classmethod(lambda cls: fake_probe())
            ), unittest.mock.patch.object(
                EnvironmentDetector, "detect_docker", classmethod(lambda cls: fake_probe())
            ), unittest.mock.patch.object(
                EnvironmentDetector, "detect_gpu", classmethod(lambda cls: fake_probe())
            ), unittest.mock.patch.object(
                EnvironmentDetector, "detect_llm_status", classmethod(lambda cls: fake_probe())
            ), unittest.mock.patch.object(
                env_cache, "get_cached", return_value=None
            ):
                EnvironmentDetector.get_full_status(
                    disk_cache_dir=self.cache.parent, force=True
                )
        self.assertEqual(calls["n"], 10, "force=True must re-run every probe")


class ModelRegistryTests(unittest.TestCase):
    def tearDown(self):
        from harness import model_registry

        try:
            model_registry._CACHE_FILE.unlink()
        except FileNotFoundError:
            pass

    def test_serves_names_without_importing_models_module(self):
        from harness import model_registry

        model_registry.builtin_model_names()

        sys_mods = {k: v for k, v in sys.modules.items() if k == "harness.models"}
        sys.modules.pop("harness.models", None)
        try:
            names = model_registry.builtin_model_names()
            self.assertNotIn("harness.models", sys.modules, "registry must not import harness.models on a cache hit")
        finally:
            sys.modules.update(sys_mods)
        self.assertTrue(len(names) >= 10)
        self.assertIn("baseline", names)

    def test_cache_invalidates_on_models_py_change(self):
        from harness import model_registry

        names_before = model_registry.builtin_model_names()
        record = json.loads(model_registry._CACHE_FILE.read_text(encoding="utf-8"))
        record["mtime"] = 12345.0
        model_registry._CACHE_FILE.write_text(json.dumps(record), encoding="utf-8")

        with unittest.mock.patch(
            "harness.models.available_model_names", return_value=list(names_before) + ["__sentinel__"]
        ) as patched:
            names_after = model_registry.builtin_model_names()
        self.assertIn("__sentinel__", names_after)
        patched.assert_called_once()
        record2 = json.loads(model_registry._CACHE_FILE.read_text(encoding="utf-8"))
        self.assertEqual(record2["mtime"], model_registry._MODELS_PY.stat().st_mtime)

    def test_corrupt_cache_falls_back_to_import(self):
        from harness import model_registry

        model_registry._CACHE_FILE.write_text("{broken", encoding="utf-8")
        with unittest.mock.patch("harness.models.available_model_names", return_value=["x"]):
            names = model_registry.builtin_model_names()
        self.assertEqual(names, ["x"])


class LocalEndpointNormalizationTests(unittest.TestCase):
    def test_localhost_normalized_to_127_0_0_1(self):
        import urllib.request as urlreq

        from harness.llm_engine import LLMEngine

        seen = {}

        def fake_urlopen(req, timeout=None):
            seen["url"] = req.full_url
            seen["timeout"] = timeout
            raise urlreq.URLError("connection refused")

        with unittest.mock.patch.object(urlreq, "urlopen", side_effect=fake_urlopen):
            result = LLMEngine.check_local_endpoint("http://localhost:11434/v1")
        self.assertFalse(result)
        self.assertIn("127.0.0.1", seen["url"])
        self.assertNotIn("localhost", seen["url"])

    def test_explicit_host_untouched(self):
        import urllib.request as urlreq

        from harness.llm_engine import LLMEngine

        seen = {}

        def fake_urlopen(req, timeout=None):
            seen["url"] = req.full_url
            raise urlreq.URLError("nope")

        with unittest.mock.patch.object(urlreq, "urlopen", side_effect=fake_urlopen):
            LLMEngine.check_local_endpoint("http://192.168.1.5:8000/v1")
        self.assertIn("192.168.1.5", seen["url"])

    def test_dead_localhost_port_fails_within_timeout(self):
        from harness.llm_engine import LLMEngine

        started = time.perf_counter()
        LLMEngine.check_local_endpoint("http://localhost:11434/v1", timeout=0.5)
        elapsed = time.perf_counter() - started
        self.assertLess(elapsed, 1.0)


# ===========================================================================
# 3. Artifact repair tests (from tests/test_repair.py)
# ===========================================================================


class TestClassifyRun(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="aih-repair-")
        self.root, self.spec, self.artifacts = make_project(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def _seal(self, exp_id="exp_a"):
        return seal_into(self.artifacts / exp_id, self.spec, 0.8)

    def test_verified_run_is_verified(self):
        folder = self._seal()
        cls = classify_run(folder)
        self.assertEqual(cls["state"], "verified")

    def test_drifted_file_detected(self):
        folder = self._seal()
        drift_metrics(folder, self.spec, 0.9)
        cls = classify_run(folder)
        self.assertEqual(cls["state"], "drifted")
        self.assertIn("metrics.json", cls["drifted"])
        self.assertEqual(cls["missing_on_disk"], [])

    def test_missing_file_detected(self):
        folder = self._seal()
        (folder / "oof_predictions.csv").unlink()
        cls = classify_run(folder)
        self.assertEqual(cls["state"], "drifted")
        self.assertIn("oof_predictions.csv", cls["missing_on_disk"])

    def test_unverifiable_not_drift(self):
        folder = self._seal()
        manifest_path = folder / "artifact_manifest.json"
        m = json.loads(manifest_path.read_text(encoding="utf-8"))
        m["status"] = "RUNNING"
        manifest_path.write_text(json.dumps(m), encoding="utf-8")
        cls = classify_run(folder)
        self.assertEqual(cls["state"], "unverifiable")
        self.assertIn("not a file-hash drift", cls["note"])


class TestRepair(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="aih-repair-fix-")
        self.root, self.spec, self.artifacts = make_project(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def _seal(self, exp_id="exp_a"):
        return seal_into(self.artifacts / exp_id, self.spec, 0.8)

    def test_repair_refreshes_hash_and_restores_validation(self):
        folder = self._seal()
        drift_metrics(folder, self.spec, 0.85)
        self.assertEqual(classify_run(folder)["state"], "drifted")
        with self.assertRaises(EvidenceDriftError):
            repair_run_dir(folder)
        res = repair_run_dir(folder, force=True)
        self.assertTrue(res["now_verified"], res)
        self.assertTrue(res["forced"])
        manifest = json.loads((folder / "artifact_manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["run_id"], "test-exp_a")
        self.assertEqual(manifest["status"], "COMPLETED")
        self.assertTrue(manifest["repaired"])
        validate_run(folder)

    def test_repair_does_not_add_undeclared_files(self):
        folder = self._seal()
        (folder / "stray_file.txt").write_text("noise", encoding="utf-8")
        drift_metrics(folder, self.spec, 0.85)
        repair_run_dir(folder, force=True)
        manifest = json.loads((folder / "artifact_manifest.json").read_text(encoding="utf-8"))
        self.assertNotIn("stray_file.txt", manifest["files"])
        self.assertIn("metrics.json", manifest["files"])

    def test_repair_drop_missing_removes_deleted_files(self):
        folder = self._seal()
        (folder / "oof_predictions.csv").unlink()
        res = repair_run_dir(folder, drop_missing=True, force=True)
        manifest = json.loads((folder / "artifact_manifest.json").read_text(encoding="utf-8"))
        self.assertNotIn("oof_predictions.csv", manifest["files"])
        self.assertIn("oof_predictions.csv", res["dropped_missing"])

    def test_repair_missing_folder_raises(self):
        with self.assertRaises(FileNotFoundError):
            repair_run_dir(self.artifacts / "nope")

    def test_repair_no_manifest_raises(self):
        folder = self.artifacts / "exp_nomanifest"
        folder.mkdir()
        (folder / "metrics.json").write_text("{}", encoding="utf-8")
        with self.assertRaises(ValueError):
            repair_run_dir(folder)


class TestScanAndCli(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="aih-repair-cli-")
        self.root, self.spec, self.artifacts = make_project(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def _seal(self, exp_id):
        return seal_into(self.artifacts / exp_id, self.spec, 0.8)

    def test_scan_counts_states(self):
        self._seal("good")
        bad = self._seal("bad")
        drift_metrics(bad, self.spec, 0.99)
        scan = scan_artifacts(self.artifacts)
        self.assertEqual(scan["runs_total"], 2)
        self.assertEqual(len(scan["drifted"]), 1)
        self.assertEqual(scan["drifted"][0]["exp_dir"], "bad")
        self.assertIn("good", scan["verified"])

    def test_check_reports_without_writing(self):
        bad = self._seal("bad")
        before = (bad / "artifact_manifest.json").read_text(encoding="utf-8")
        drift_metrics(bad, self.spec, 0.99)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = repair_cli_main(["--project-root", str(self.root), "--check", "--json"])
        self.assertEqual(code, 0)
        payload = json.loads(buf.getvalue())
        self.assertEqual(payload["drifted"], 1)
        self.assertEqual(before, (bad / "artifact_manifest.json").read_text(encoding="utf-8"))

    def test_repair_all_blocks_evidence_drift_and_force_restores(self):
        bad = self._seal("bad")
        drift_metrics(bad, self.spec, 0.99)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = repair_cli_main(["--project-root", str(self.root), "--all", "--json"])
        self.assertEqual(code, 3, buf.getvalue())
        payload = json.loads(buf.getvalue())
        self.assertEqual(payload["repaired"], 0)
        self.assertEqual(payload["blocked"], 1)
        self.assertEqual(payload["blocked_details"][0]["status"], "blocked_evidence_drift")

        with self.assertRaises(Exception):
            validate_run(bad)

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = repair_cli_main(["--project-root", str(self.root), "--all", "--json", "--force"])
        self.assertEqual(code, 0, buf.getvalue())
        payload = json.loads(buf.getvalue())
        self.assertEqual(payload["repaired"], 1)
        self.assertTrue(payload["details"][0]["now_verified"])
        self.assertTrue(payload["details"][0]["forced"])
        validate_run(bad)

    def test_repair_specific_exp_blocks_evidence_drift(self):
        self._seal("good")
        bad = self._seal("bad")
        drift_metrics(bad, self.spec, 0.99)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = repair_cli_main(["--project-root", str(self.root), "--exp", "bad", "--json"])
        self.assertEqual(code, 3, buf.getvalue())
        payload = json.loads(buf.getvalue())
        self.assertEqual(payload["repaired"], 0)
        self.assertEqual(payload["blocked"], 1)
        self.assertEqual(payload["blocked_details"][0]["exp_dir"], "bad")

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = repair_cli_main(["--project-root", str(self.root), "--exp", "bad", "--json", "--force"])
        self.assertEqual(code, 0, buf.getvalue())
        payload = json.loads(buf.getvalue())
        self.assertEqual(payload["repaired"], 1)
        self.assertEqual(payload["details"][0]["exp_dir"], "bad")

    def test_repair_nothing_to_do(self):
        self._seal("good")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = repair_cli_main(["--project-root", str(self.root), "--all", "--json"])
        self.assertEqual(code, 0)
        payload = json.loads(buf.getvalue())
        self.assertEqual(payload["repaired"], 0)

    def test_not_attached_returns_2(self):
        with tempfile.TemporaryDirectory() as empty:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                code = repair_cli_main(["--project-root", empty, "--json"])
            self.assertEqual(code, 2)
            self.assertIn("error", json.loads(buf.getvalue()))


class TestRepairParser(unittest.TestCase):
    def test_parser_flags(self):
        parser = repair_build_parser()
        args = parser.parse_args(["--check", "--drop-missing", "--json", "--exp", "x"])
        self.assertTrue(args.check)
        self.assertTrue(args.drop_missing)
        self.assertTrue(args.json)
        self.assertEqual(args.exp, ["x"])
        self.assertFalse(args.all)

    def test_cli_registers_repair_subcommand(self):
        p = cli_build_parser()
        args = p.parse_args(["repair"])
        self.assertEqual(args.command, "repair")


# ===========================================================================
# 4. Repair integrity gate tests (from tests/test_repair_integrity.py)
# ===========================================================================

TRUTH_CSV = "id,y\n0,0.0\n1,1.0\n2,2.0\n3,3.0\n"
OOF_CSV = "id,pred_y,input_variant\n0,0.5,v1\n1,1.5,v1\n2,1.5,v1\n3,3.5,v1\n"
EXPECTED_SCORE = 0.5


def _regression_task_doc():
    return {
        "task": {
            "id": "repair-integrity-task",
            "name": "Repair Integrity Task",
            "task_type": "regression",
            "type": "regression",
            "primary_metric": "rmse",
            "direction": "minimize",
            "data_config": {
                "train_path": "dataset/train.csv",
                "target_column": "y",
                "id_column": "id",
            },
            "validation": {"strategy": "kfold", "num_folds": 2},
            "subtasks": [
                {
                    "id": "target",
                    "name": "target",
                    "task_type": "regression",
                    "column": "y",
                    "primary_metric": "rmse",
                    "direction": "minimize",
                    "weight": 1.0,
                }
            ],
        }
    }


def _write_run(root: Path, exp_id: str) -> Path:
    artifacts_dir = root / ".ai-harness" / "artifacts" / "repair-integrity-task"
    run_dir = artifacts_dir / exp_id
    run_dir.mkdir(parents=True, exist_ok=True)

    run_config = run_dir / "run_config.yaml"
    run_config.write_text(
        yaml.safe_dump({"model": {"name": "fixture-lr"}, "estimator": "linear"}, sort_keys=False),
        encoding="utf-8",
    )
    (run_dir / "evaluation_truth.csv").write_text(TRUTH_CSV, encoding="utf-8")
    (run_dir / "oof_predictions.csv").write_text(OOF_CSV, encoding="utf-8")
    card = run_dir / "EXPERIMENT_CARD.md"
    card.write_text(f"# {exp_id}\nfixture evidence card\n", encoding="utf-8")

    metrics = {
        "primary_score": EXPECTED_SCORE,
        "primary_metric": "rmse",
        "utility": -EXPECTED_SCORE,
        "direction": "minimize",
        "experiment_id": exp_id,
        "selection_variant": "v1",
        "subtask_scores": {"target": EXPECTED_SCORE},
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    manifest = {
        "schema_version": 2,
        "status": "COMPLETED",
        "run_kind": "full",
        "run_id": f"run-{exp_id}",
        "exp_id": exp_id,
        "task_sha256": "a" * 64,
        "scorer_sha256": "b" * 64,
        "data_sha256": "c" * 64,
        "split_sha256": "d" * 64,
        "resolved_config_sha256": sha256(run_config),
        "files": {
            p.name: sha256(p)
            for p in (
                run_config,
                run_dir / "metrics.json",
                run_dir / "oof_predictions.csv",
                run_dir / "evaluation_truth.csv",
                card,
            )
        },
    }
    (run_dir / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    from tests.evidence_fixtures import seal_scope_fixture
    from tests.fixtures import task
    seal_scope_fixture(run_dir, task())
    return run_dir


class RepairIntegrityGateTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._out_buf = io.StringIO()
        self._out = contextlib.redirect_stdout(self._out_buf)
        self._out.__enter__()
        root, _task_config = make_workspace(task=_regression_task_doc())
        self.workspace = root

    def tearDown(self):
        self._out.__exit__(None, None, None)
        self._tmp.cleanup()

    def _captured_output(self) -> str:
        return self._out_buf.getvalue()

    def test_clean_run_verifies(self):
        run_dir = _write_run(self.workspace, "exp001_clean")
        validate_run(run_dir)
        self.assertEqual(repair_mod.classify_run(run_dir)["state"], "verified")

    def test_rederivation_matches_recorded_score(self):
        run_dir = _write_run(self.workspace, "exp002_rederive")
        from harness.oof_revalidation import rederive_primary_score
        from harness.plugins import TaskPluginContext, load_task_plugin
        from harness.project import HarnessProject
        from harness.task_spec import load_active_task_spec

        project = HarnessProject.discover(self.workspace)
        spec = load_active_task_spec(project.task_config_path)
        plugin_context = TaskPluginContext(
            project.root, project.task_config_path, spec, project.plugin_id or "default"
        )
        plugin = load_task_plugin(project.plugin_id, plugin_context)

        report = rederive_primary_score(
            run_dir, task_spec=spec, plugin=plugin, plugin_context=plugin_context
        )
        self.assertTrue(report["verifiable"])
        self.assertTrue(report["ok"])
        self.assertEqual(report["recorded"], EXPECTED_SCORE)
        self.assertAlmostEqual(report["recomputed"], EXPECTED_SCORE, places=6)

    def test_tampered_metrics_blocked_by_default(self):
        run_dir = _write_run(self.workspace, "exp003_tampered")
        metrics_path = run_dir / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics["primary_score"] = 9.99
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        rc = repair_mod.run_repair(project_root=str(self.workspace), exp=["exp003_tampered"])
        self.assertEqual(rc, 3)

        out = self._captured_output()
        self.assertIn("blocked_evidence_drift", out)

        with self.assertRaises(ValueError):
            validate_run(run_dir)
        manifest = json.loads((run_dir / "artifact_manifest.json").read_text(encoding="utf-8"))
        self.assertIsNone(manifest.get("repaired"))
        original = json.loads(metrics_path.read_text(encoding="utf-8"))
        self.assertEqual(original["primary_score"], 9.99)
        self.assertNotEqual(manifest["files"]["metrics.json"], sha256(metrics_path))

        log_path = (
            self.workspace / ".ai-harness" / "artifacts" / "repair-integrity-task" / "repair_log.jsonl"
        )
        self.assertTrue(log_path.is_file())
        entries = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
        self.assertTrue(any(e["status"] == "blocked_evidence_drift" for e in entries))
        blocked = next(e for e in entries if e["status"] == "blocked_evidence_drift")
        self.assertIn("metrics.json", blocked["evidence_files"])

    def test_benign_drift_still_repairs(self):
        run_dir = _write_run(self.workspace, "exp004_benign")
        card = run_dir / "EXPERIMENT_CARD.md"
        card.write_text("# updated card\n", encoding="utf-8")

        rc = repair_mod.run_repair(project_root=str(self.workspace), exp=["exp004_benign"])
        self.assertEqual(rc, 0)
        out = self._captured_output()
        self.assertIn("Repaired 1", out)

        validate_run(run_dir)
        manifest = json.loads((run_dir / "artifact_manifest.json").read_text(encoding="utf-8"))
        self.assertTrue(manifest["repaired"])
        self.assertTrue(manifest["repaired_at"])
        self.assertEqual(manifest["files"]["EXPERIMENT_CARD.md"], sha256(card))

        log_path = (
            self.workspace / ".ai-harness" / "artifacts" / "repair-integrity-task" / "repair_log.jsonl"
        )
        entries = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["exp_dir"], "exp004_benign")
        self.assertEqual(entries[0]["mode"], "resave")
        self.assertIn("EXPERIMENT_CARD.md", entries[0]["files_changed"])

    def test_verify_reseals_when_score_matches(self):
        run_dir = _write_run(self.workspace, "exp005_verify_ok")
        metrics_path = run_dir / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics_path.write_text(json.dumps(metrics, indent=4, sort_keys=True), encoding="utf-8")

        rc = repair_mod.run_repair(
            project_root=str(self.workspace), exp=["exp005_verify_ok"], verify=True
        )
        self.assertEqual(rc, 0)
        out = self._captured_output()
        self.assertIn("Repaired 1", out)
        self.assertIn("recorded=", out)

        validate_run(run_dir)
        manifest = json.loads((run_dir / "artifact_manifest.json").read_text(encoding="utf-8"))
        self.assertTrue(manifest["repaired"])
        self.assertEqual(manifest["files"]["metrics.json"], sha256(metrics_path))

        log_path = (
            self.workspace / ".ai-harness" / "artifacts" / "repair-integrity-task" / "repair_log.jsonl"
        )
        entries = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
        check = entries[0]["metric_check"]
        self.assertTrue(check["verifiable"])
        self.assertTrue(check["ok"])
        self.assertEqual(check["recorded"], EXPECTED_SCORE)
        self.assertAlmostEqual(check["recomputed"], EXPECTED_SCORE, places=6)

    def test_verify_blocks_metric_mismatch(self):
        run_dir = _write_run(self.workspace, "exp006_mismatch")
        metrics_path = run_dir / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics["primary_score"] = 0.1
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        rc = repair_mod.run_repair(
            project_root=str(self.workspace), exp=["exp006_mismatch"], verify=True
        )
        self.assertEqual(rc, 3)
        out = self._captured_output()
        self.assertIn("metric_mismatch", out)
        self.assertIn("recorded=0.1", out)
        self.assertIn(f"recomputed={EXPECTED_SCORE}", out)

        with self.assertRaises(ValueError):
            validate_run(run_dir)
        manifest = json.loads((run_dir / "artifact_manifest.json").read_text(encoding="utf-8"))
        self.assertIsNone(manifest.get("repaired"))

        log_path = (
            self.workspace / ".ai-harness" / "artifacts" / "repair-integrity-task" / "repair_log.jsonl"
        )
        entries = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
        self.assertTrue(any(e["status"] == "metric_mismatch" for e in entries))
        mismatch = next(e for e in entries if e["status"] == "metric_mismatch")
        self.assertEqual(mismatch["metric_check"]["recorded"], 0.1)
        self.assertAlmostEqual(mismatch["metric_check"]["recomputed"], EXPECTED_SCORE, places=6)

    def test_force_reseals_but_is_audited(self):
        run_dir = _write_run(self.workspace, "exp007_force")
        metrics_path = run_dir / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics["primary_score"] = 0.1
        metrics["utility"] = -0.1
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        rc = repair_mod.run_repair(
            project_root=str(self.workspace), exp=["exp007_force"], force=True
        )
        self.assertEqual(rc, 0)
        validate_run(run_dir)

        log_path = (
            self.workspace / ".ai-harness" / "artifacts" / "repair-integrity-task" / "repair_log.jsonl"
        )
        entries = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
        self.assertTrue(entries[0]["forced"])
        self.assertIn("exp007_force", entries[0]["exp_dir"])

    def test_missing_evidence_file_blocked_and_not_verifiable(self):
        run_dir = _write_run(self.workspace, "exp008_missing")
        (run_dir / "oof_predictions.csv").unlink()

        rc = repair_mod.run_repair(project_root=str(self.workspace), exp=["exp008_missing"])
        self.assertEqual(rc, 3)

        rc = repair_mod.run_repair(
            project_root=str(self.workspace), exp=["exp008_missing"], verify=True
        )
        out = self._captured_output()
        self.assertEqual(rc, 3)
        self.assertIn("not_verifiable", out)

        with self.assertRaises(ValueError):
            validate_run(run_dir)

    def test_repair_log_lives_outside_every_sealed_bundle(self):
        run_dir = _write_run(self.workspace, "exp009_log")
        card = run_dir / "EXPERIMENT_CARD.md"
        card.write_text("v2\n", encoding="utf-8")
        repair_mod.run_repair(project_root=str(self.workspace), exp=["exp009_log"])
        self._captured_output()

        manifest = json.loads((run_dir / "artifact_manifest.json").read_text(encoding="utf-8"))
        self.assertNotIn("repair_log.jsonl", manifest["files"])

        card.write_text("v3\n", encoding="utf-8")
        repair_mod.run_repair(project_root=str(self.workspace), exp=["exp009_log"])
        self._captured_output()
        log_path = (
            self.workspace / ".ai-harness" / "artifacts" / "repair-integrity-task" / "repair_log.jsonl"
        )
        lines = log_path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 2)
        self.assertEqual(json.loads(lines[0])["exp_dir"], "exp009_log")
        self.assertEqual(json.loads(lines[1])["exp_dir"], "exp009_log")


# ===========================================================================
# 5. Status report tests (from tests/test_status_report.py)
# ===========================================================================


def _run_cli_main(argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = cli_main(argv)
    return code, buf.getvalue()


class TestStatusCommand(unittest.TestCase):
    def test_parser_registers_status(self):
        parser = cli_build_parser()
        args = parser.parse_args(["status"])
        self.assertEqual(args.command, "status")
        self.assertFalse(args.markdown)
        self.assertFalse(args.no_cloud)

    def test_collect_never_raises_without_project(self):
        with tempfile.TemporaryDirectory() as tmp:
            status = collect_status(project_root=tmp, include_cloud=False)
            self.assertIn("python", status)
            self.assertIn("install", status)
            self.assertIn("project", status)

    def test_collect_reports_sections_on_fresh_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            status = collect_status(project_root=tmp, include_cloud=False)
            self.assertTrue(status["install"].get("found"), status["install"])
            self.assertEqual(status["install"]["version"], "0.2.0")
            self.assertEqual(status["artifacts"]["verified"], 0)
            self.assertEqual(status["artifacts"]["broken"], [])

    def test_markdown_render_contains_markers(self):
        with tempfile.TemporaryDirectory() as tmp:
            status = collect_status(project_root=tmp, include_cloud=False)
            md = render_markdown(status)
            self.assertIn("# ai-harness status", md)
            self.assertIn("install:", md)
            self.assertIn("project:", md)

    def test_main_json_output_is_valid_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, out = _run_cli_main(["status", "--project-root", tmp, "--no-cloud"])
            self.assertIn(code, (0, 2))
            payload = json.loads(out)
            self.assertIn("install", payload)

    def test_main_markdown_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, out = _run_cli_main(["status", "--project-root", tmp, "--no-cloud", "--markdown"])
            self.assertIn("ai-harness status", out)

    def test_exit_code_2_when_not_attached(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, _ = _run_cli_main(["status", "--project-root", tmp, "--no-cloud", "--markdown"])
            self.assertEqual(code, 2)


class TestArtifactDriftSurfacing(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="aih-status-")
        self.root, self.spec, self.artifacts = make_project(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_all_verified_no_drift(self):
        seal_into(self.artifacts / "exp001_ok", self.spec, 0.8)
        status = collect_status(project_root=str(self.root), include_cloud=False)
        art = status["artifacts"]
        self.assertEqual(art["verified"], 1)
        self.assertEqual(art["drifted_repairable"], 0)
        self.assertIsNone(art["repair_hint"])

    def test_drifted_run_is_repairable_with_hint(self):
        folder = seal_into(self.artifacts / "exp002_ok", self.spec, 0.8)
        drift_metrics(folder, self.spec, 0.9)
        status = collect_status(project_root=str(self.root), include_cloud=False)
        art = status["artifacts"]
        self.assertEqual(art["verified"], 0)
        self.assertEqual(art["drifted_repairable"], 1)
        self.assertIsNotNone(art["repair_hint"])
        self.assertIn("repair --check", art["repair_hint"])
        md = render_markdown(status)
        self.assertIn("1 drift-repairable", md)
        self.assertIn("repair --check", md)

    def test_mixed_verified_and_drifted(self):
        seal_into(self.artifacts / "exp001_ok", self.spec, 0.8)
        bad = seal_into(self.artifacts / "exp002_bad", self.spec, 0.8)
        drift_metrics(bad, self.spec, 0.9)
        status = collect_status(project_root=str(self.root), include_cloud=False)
        art = status["artifacts"]
        self.assertEqual(art["verified"], 1)
        self.assertEqual(art["drifted_repairable"], 1)
        self.assertEqual(art["runs_total"], 2)

    def test_unverifiable_not_counted_as_repairable(self):
        folder = seal_into(self.artifacts / "exp003_bad", self.spec, 0.8)
        manifest = json.loads((folder / "artifact_manifest.json").read_text(encoding="utf-8"))
        manifest["exp_id"] = "tampered-not-a-real-hash"
        (folder / "artifact_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        status = collect_status(project_root=str(self.root), include_cloud=False)
        art = status["artifacts"]
        self.assertEqual(art["drifted_repairable"], 0)
        self.assertGreaterEqual(art["unverifiable"], 1)
        self.assertIsNone(art["repair_hint"])


class TestTaskCapabilitiesDiscovery(unittest.TestCase):
    def test_unattached_reports_missing_attachment(self):
        with tempfile.TemporaryDirectory() as tmp:
            status = collect_status(project_root=tmp, include_cloud=False)
            self.assertIn("task_capabilities", status)
            caps = status["task_capabilities"]
            self.assertFalse(caps.get("attached"))
            self.assertIn("task_attachment", caps.get("capabilities", {}))
            self.assertEqual(caps["capabilities"]["task_attachment"]["state"], "missing")

    def test_attached_reports_capabilities(self):
        root, task_cfg = make_workspace(prefix="aih-cap-")
        try:
            status = collect_status(project_root=str(root), include_cloud=False)
            caps = status.get("task_capabilities", {})
            self.assertTrue(caps.get("attached"))
            c = caps.get("capabilities", {})
            self.assertIn("data_loading", c)
            self.assertEqual(c["data_loading"]["state"], "ready")
            self.assertIn("custom_scoring", c)
            self.assertEqual(c["custom_scoring"]["state"], "supported")
            self.assertIn("local_execution", c)
            self.assertEqual(c["local_execution"]["state"], "ready")
            self.assertIn("submission", c)
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
