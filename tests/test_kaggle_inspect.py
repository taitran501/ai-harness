"""Tests for the Kaggle kernel-inspect commands and the dispatch --dry-run plan."""

import json
import unittest
from pathlib import Path
from unittest import mock

from harness.cli import build_parser
from harness import kaggle_inspect
from harness import cloud_runner


class TestKaggleInspectCli(unittest.TestCase):
    def test_parser_registers_kaggle(self):
        parser = build_parser()
        args = parser.parse_args(["kaggle", "status", "user/title"])
        self.assertEqual(args.command, "kaggle")
        self.assertEqual(args.args, ["status", "user/title"])

    def test_kernel_status_delegates_to_manager(self):
        fake = {"kernel_slug": "user/title", "status": "FAILED", "raw_output": "has status 'failed'"}
        with mock.patch("harness.templates.kaggle_template.KaggleKernelManager.get_kernel_status", return_value=fake) as m:
            result = kaggle_inspect.kernel_status("user/title")
        self.assertEqual(result["status"], "FAILED")
        m.assert_called_once_with("", "user/title")

    def test_kernel_log_returns_error_log(self):
        status_fake = {"kernel_slug": "user/title", "status": "FAILED", "raw_output": ""}
        log_fake = "Traceback ... ModuleNotFoundError: No module named 'lightgbm'"
        with mock.patch("harness.templates.kaggle_template.KaggleKernelManager.get_kernel_status", return_value=status_fake), \
             mock.patch("harness.templates.kaggle_template.KaggleKernelManager.get_kernel_error_log", return_value=log_fake) as m:
            result = kaggle_inspect.kernel_log("user/title")
        self.assertEqual(result["status"], "FAILED")
        self.assertIn("lightgbm", result["error_log"])
        m.assert_called_once_with("user/title", max_lines=60)


class TestDryRunPlan(unittest.TestCase):
    def test_dry_run_plan_captures_metadata_and_bundle(self):
        stage = Path(__file__).parent / "tmp_dryrun_stage"
        stage.mkdir(exist_ok=True)
        (stage / "kernel-metadata.json").write_text(json.dumps({"id": "user/title", "enable_internet": "true"}), encoding="utf-8")
        (stage / "code" / "train.py").parent.mkdir(parents=True, exist_ok=True)
        (stage / "code" / "train.py").write_text("print('hi')", encoding="utf-8")
        context = {
            "exp_id": "exp001",
            "task_id": "task-x",
            "config_rel": "configs/exp001.yaml",
            "config_sha256": "abc123",
            "task": {"runtime": {"enable_internet": True, "kaggle_bundle_mode": "dataset"}},
        }
        runtime = {"core": {"version": "0.2.0"}, "plugin": {"id": "default"}, "requirements": ["lightgbm>=4"]}
        plan = cloud_runner._kaggle_dry_run_plan(context, mock.Mock(root=stage), "exp001", "user/ds", stage, runtime)
        self.assertEqual(plan["exp_id"], "exp001")
        self.assertEqual(plan["dataset_slug"], "user/ds")
        self.assertTrue(plan["enable_internet"])
        self.assertEqual(plan["bundle_mode"], "dataset")
        self.assertEqual(plan["kernel_metadata"]["id"], "user/title")
        self.assertIn("code/train.py", plan["bundle_files"])
        self.assertTrue(any("kaggle kernels push" in s for s in plan["would_run"]))
        # cleanup
        import shutil
        shutil.rmtree(stage, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
