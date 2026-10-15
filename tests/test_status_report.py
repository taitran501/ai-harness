"""Tests for the single-command `ai-harness status` report."""

import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from harness.status_report import collect_status, render_markdown
from harness.cli import build_parser, main


def _run_main(argv):
    buf = StringIO()
    with redirect_stdout(buf):
        code = main(argv)
    return code, buf.getvalue()


class TestStatusCommand(unittest.TestCase):
    def test_parser_registers_status(self):
        parser = build_parser()
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
            # install section must succeed (harness is importable in this env)
            self.assertTrue(status["install"].get("found"), status["install"])
            self.assertEqual(status["install"]["version"], "0.2.0")
            # artifact section must not error on empty dir
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
            code, out = _run_main(["status", "--project-root", tmp, "--no-cloud"])
            self.assertIn(code, (0, 2))
            payload = json.loads(out)
            self.assertIn("install", payload)

    def test_main_markdown_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, out = _run_main(["status", "--project-root", tmp, "--no-cloud", "--markdown"])
            self.assertIn("ai-harness status", out)

    def test_exit_code_2_when_not_attached(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, _ = _run_main(["status", "--project-root", tmp, "--no-cloud", "--markdown"])
            self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
