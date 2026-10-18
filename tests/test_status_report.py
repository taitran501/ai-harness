"""Tests for the single-command `ai-harness status` report."""

import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO

from harness.cli import build_parser, main
from harness.status_report import collect_status, render_markdown
from tests.test_repair import _drift_metrics, _make_project, _seal_into


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


class TestArtifactDriftSurfacing(unittest.TestCase):
    """status must tell an agent which broken runs are repairable (drift) vs not."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="aih-status-")
        self.root, self.spec, self.artifacts = _make_project(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_all_verified_no_drift(self):
        _seal_into(self.artifacts / "exp001_ok", self.spec, 0.8)
        status = collect_status(project_root=str(self.root), include_cloud=False)
        art = status["artifacts"]
        self.assertEqual(art["verified"], 1)
        self.assertEqual(art["drifted_repairable"], 0)
        self.assertIsNone(art["repair_hint"])

    def test_drifted_run_is_repairable_with_hint(self):
        folder = _seal_into(self.artifacts / "exp002_ok", self.spec, 0.8)
        _drift_metrics(folder, self.spec, 0.9)  # change bytes after sealing
        status = collect_status(project_root=str(self.root), include_cloud=False)
        art = status["artifacts"]
        self.assertEqual(art["verified"], 0)
        self.assertEqual(art["drifted_repairable"], 1)
        self.assertIsNotNone(art["repair_hint"])
        self.assertIn("repair --check", art["repair_hint"])
        # markdown surfaces it too
        md = render_markdown(status)
        self.assertIn("1 drift-repairable", md)
        self.assertIn("repair --check", md)

    def test_mixed_verified_and_drifted(self):
        _seal_into(self.artifacts / "exp001_ok", self.spec, 0.8)
        bad = _seal_into(self.artifacts / "exp002_bad", self.spec, 0.8)
        _drift_metrics(bad, self.spec, 0.9)
        status = collect_status(project_root=str(self.root), include_cloud=False)
        art = status["artifacts"]
        self.assertEqual(art["verified"], 1)
        self.assertEqual(art["drifted_repairable"], 1)
        self.assertEqual(art["runs_total"], 2)

    def test_unverifiable_not_counted_as_repairable(self):
        # A run whose identity is broken (not a hash drift) must NOT be repairable.
        folder = _seal_into(self.artifacts / "exp003_bad", self.spec, 0.8)
        # corrupt an identity field so validate_run fails for non-drift reasons
        manifest = json.loads((folder / "artifact_manifest.json").read_text(encoding="utf-8"))
        manifest["exp_id"] = "tampered-not-a-real-hash"
        (folder / "artifact_manifest.json").write_text(json.dumps(manifest))
        status = collect_status(project_root=str(self.root), include_cloud=False)
        art = status["artifacts"]
        self.assertEqual(art["drifted_repairable"], 0)
        self.assertGreaterEqual(art["unverifiable"], 1)
        self.assertIsNone(art["repair_hint"])


if __name__ == "__main__":
    unittest.main()
