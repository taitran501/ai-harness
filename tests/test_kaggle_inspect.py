"""Tests for the Kaggle kernel-inspect commands and the dispatch --dry-run plan."""

import json
import unittest
from pathlib import Path
from unittest import mock

from harness import cloud_runner, kaggle_inspect
from harness.cli import build_parser


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

    def test_kernel_watch_delegates_to_manager(self):
        wait_fake = {"kernel_slug": "user/title", "status": "COMPLETED", "success": True, "raw_output": "done"}
        with mock.patch("harness.templates.kaggle_template.KaggleKernelManager.wait_for_kernel", return_value=wait_fake) as m:
            result = kaggle_inspect.kernel_watch("user/title", poll_interval=5.0, timeout_seconds=120.0)
        self.assertEqual(result["status"], "COMPLETED")
        self.assertTrue(result["success"])
        m.assert_called_once_with(exp_id="", kernel_slug="user/title", poll_interval=5.0, timeout_seconds=120.0)

    def test_kernel_list_delegates_to_client(self):
        from types import SimpleNamespace
        k1 = SimpleNamespace(ref="user/k1", title="Kernel 1", author="alice", total_votes=10, last_run_time="2026-10-24", url=None)
        k2 = SimpleNamespace(ref="user/k2", title="Kernel 2", author="bob", votes=5, last_run_time=None, url=None)
        mock_client = mock.MagicMock()
        mock_client.list_kernels.return_value = [k1, k2]
        with mock.patch("harness.kaggle_compat.KaggleClient", return_value=mock_client):
            res = kaggle_inspect.kernel_list(competition="comp-x", sort_by="hot", page_size=10)
        self.assertEqual(res["count"], 2)
        self.assertEqual(res["kernels"][0]["ref"], "user/k1")
        self.assertEqual(res["kernels"][0]["total_votes"], 10)
        self.assertEqual(res["kernels"][1]["total_votes"], 5)
        mock_client.list_kernels.assert_called_once_with(user=None, page_size=10, sort_by="hotness", competition="comp-x")

    def test_cli_watch_and_list_parsers(self):
        parser = build_parser()
        args_w = parser.parse_args(["kaggle", "watch", "user/title", "--poll", "10", "--timeout", "500"])
        self.assertEqual(args_w.args, ["watch", "user/title", "--poll", "10", "--timeout", "500"])

        args_l = parser.parse_args(["kaggle", "list", "--competition", "my-comp", "--sort-by", "votes"])
        self.assertEqual(args_l.args, ["list", "--competition", "my-comp", "--sort-by", "votes"])


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
