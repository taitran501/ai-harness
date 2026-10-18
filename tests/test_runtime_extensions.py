"""Explicit factory context, shell-independent JSON, and live redirected progress."""
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import yaml

from harness.runner import UniversalRunner
from tests.test_research_continuity import task_document
from tools.agent_bridge_cli import _parse_json_arg, main


class FactoryContracts(unittest.TestCase):
    def setUp(self):
        self.runner = UniversalRunner.__new__(UniversalRunner)
        self.runner.task_spec = object()
        self.runner.plugin_context = object()
        self.runner.cfg = {"model": {"factory": "factory_fixture.build"}}

    def invoke(self, fn, **params):
        module = ModuleType("factory_fixture")
        module.build = fn
        with patch.dict(sys.modules, {"factory_fixture": module}):
            return self.runner._get_model_factory()("custom", **params)

    def test_explicit_task_plugin_and_keyword_context(self):
        def task_factory(task_spec, name, **params):
            return task_spec, name, params
        def plugin_factory(plugin_context, name, **params):
            return plugin_context, name, params
        def keyword_factory(name, *, task_spec, **params):
            return task_spec, name, params
        self.assertEqual(self.invoke(task_factory, C=2), (self.runner.task_spec, "custom", {"C": 2}))
        self.assertEqual(self.invoke(plugin_factory)[0], self.runner.plugin_context)
        self.assertEqual(self.invoke(keyword_factory)[0], self.runner.task_spec)

    def test_legacy_and_configured_context_name(self):
        def legacy(name, **params):
            return name, params
        self.assertEqual(self.invoke(legacy, C=3), ("custom", {"C": 3}))
        def named_context(ctx, name, **params):
            return ctx, name, params
        self.runner.cfg["model"]["factory_context"] = "plugin_context"
        self.assertEqual(self.invoke(named_context)[0], self.runner.plugin_context)
        self.runner.cfg["model"]["factory_context"] = "none"
        self.assertEqual(self.invoke(legacy), ("custom", {}))

    def test_factory_internal_typeerror_is_not_retried_and_context_cannot_be_overridden(self):
        calls = []
        def broken(task_spec, name, **params):
            calls.append(task_spec)
            raise TypeError("factory implementation error")
        with self.assertRaisesRegex(TypeError, "implementation error"):
            self.invoke(broken)
        self.assertEqual(len(calls), 1)
        with self.assertRaisesRegex(ValueError, "override injected"):
            self.invoke(broken, task_spec="wrong task")
        self.assertEqual(len(calls), 1)


class RuntimeIOContracts(unittest.TestCase):
    def test_param_file_preserves_quotes_unicode_and_bom(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "configs").mkdir()
            (root / "configs/task.yaml").write_text(yaml.safe_dump(task_document()))
            params = {"summary": 'Bài học: "class weighting" còn chưa rõ.',
                      "next_action": 'Kiểm tra "minority" slices', "open_questions": ["Label noise?"]}
            (root / "checkpoint.json").write_text(json.dumps(params, ensure_ascii=False), encoding="utf-8-sig")
            output = io.StringIO()
            with patch.dict(os.environ), contextlib.redirect_stdout(output):
                self.assertEqual(main(["--project-root", str(root), "act", "--action", "record_research_checkpoint",
                                       "--param-file", "checkpoint.json"]), 0)
            payload = json.loads(output.getvalue())
            self.assertEqual(payload["observation"]["status"], "SUCCESS")
            self.assertEqual(payload["observation"]["data"]["conclusion"], params["summary"])

    def test_invalid_json_object_and_mutually_exclusive_parameters(self):
        with contextlib.redirect_stderr(io.StringIO()):
            for raw in ("[]", "42", "not json"):
                with self.assertRaises(SystemExit):
                    _parse_json_arg(raw)
            with self.assertRaises(SystemExit):
                main(["act", "--action", "search_research_memory", "--params", "{}", "--param-file", "p.json"])

    def test_progress_is_visible_in_redirected_file_before_process_finishes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stdout_path = root / "stdout.log"
            script = "\n".join([
                "from pathlib import Path", "from harness.runner import UniversalRunner", "import sys",
                "runner = UniversalRunner.__new__(UniversalRunner)", "runner.exp_id = 'progress'",
                "runner.exp_dir = Path(sys.argv[1])", "runner._progress('Fold 0 started')",
                "sys.stdin.readline()",
            ])
            env = dict(os.environ)
            env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent)
            with stdout_path.open("w") as log:
                process = subprocess.Popen([sys.executable, "-c", script, str(root)],
                    stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT, text=True, env=env)
                try:
                    deadline = time.monotonic() + 12
                    while "Fold 0 started" not in stdout_path.read_text() and time.monotonic() < deadline:
                        if process.poll() is not None:
                            break
                        time.sleep(.02)
                    self.assertIsNone(process.poll())
                    self.assertIn("Fold 0 started", stdout_path.read_text())
                    self.assertIn("Fold 0 started", (root / "run.log").read_text())
                finally:
                    process.communicate("\n", timeout=10)


if __name__ == "__main__":
    unittest.main()
