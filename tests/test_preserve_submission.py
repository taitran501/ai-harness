"""Unit tests for submission preservation, bridge action, and CLI parameter parsing."""

import json
import tempfile
import unittest
from pathlib import Path

from harness.agent_bridge import AgentBridge, get_action_schema
from harness.artifacts import sha256
from harness.submission_manager import SubmissionManager
from tests.fixtures import make_workspace
from tools.agent_bridge_cli import (
    _coerce_value,
    _format_action_schema,
    _parse_dynamic_kv_args,
)


class PreserveSubmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.p = Path(self.temp_dir.name)
        self.sub_dir = self.p / "submissions"
        self.mgr = SubmissionManager(submissions_dir=str(self.sub_dir), project_root=str(self.p))

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_preserve_file_submission_success(self):
        sample_file = self.p / "custom_submission.zip"
        sample_file.write_bytes(b"PK\x03\x04mock_zip_binary_payload")
        expected_hash = sha256(sample_file)

        entry = self.mgr.preserve_file_submission(
            file_path=sample_file,
            notes="Trained on external cluster from debate consensus",
            reported_score=0.895,
            reported_partition="public",
        )

        self.assertTrue(entry.id.endswith("_preserved"))
        self.assertEqual(entry.status, "PRESERVED_UNVERIFIED")
        self.assertEqual(entry.exp_id, "preserved")
        self.assertEqual(entry.payload_sha256, expected_hash)
        self.assertEqual(entry.official_scores, {"reported_public": 0.895})
        self.assertEqual(entry.official_score_source, "user_reported_unverified")

        # Verify receipt
        receipt_path = Path(entry.receipt_path)
        self.assertTrue(receipt_path.is_file())
        with open(receipt_path, "r", encoding="utf-8") as f:
            receipt = json.load(f)
        self.assertEqual(receipt["record_type"], "preservation_only")
        self.assertEqual(receipt["verification_status"], "unverified")
        self.assertFalse(receipt["dispatch_authorized"])
        self.assertEqual(receipt["payload_sha256"], expected_hash)
        self.assertEqual(receipt["byte_count"], sample_file.stat().st_size)
        self.assertEqual(receipt["reported_score"], 0.895)
        self.assertEqual(receipt["reported_partition"], "public")

        # Verify SUBMISSIONS.md rendering
        md_text = self.mgr.render_submissions_markdown()
        self.assertIn("📦 `PRESERVED_UNVERIFIED`", md_text)
        self.assertIn(f"| `{entry.id}` | `preserved` | 📦 `PRESERVED` |", md_text)
        self.assertNotIn(f"| `{entry.id}` | `preserved` | 🔒 `LOCAL_VERIFIED` |", md_text)

    def test_preserve_file_submission_validation_errors(self):
        # Non-existent file
        with self.assertRaises(FileNotFoundError):
            self.mgr.preserve_file_submission(file_path=self.p / "does_not_exist.csv")

        # Valid file for subsequent error tests
        valid_file = self.p / "valid_name.csv"
        valid_file.write_text("a,b\n1,2\n", encoding="utf-8")

        # Non-finite scores
        with self.assertRaises(ValueError):
            self.mgr.preserve_file_submission(file_path=valid_file, reported_score=float("nan"))
        with self.assertRaises(ValueError):
            self.mgr.preserve_file_submission(file_path=valid_file, reported_score=float("inf"))

    def test_bridge_preserve_submission_action(self):
        root, task_config_path = make_workspace(prefix="test_preserve_ws_")
        bridge = AgentBridge(project_root=root, task_spec_path=task_config_path)

        sub_file = root / "final_ensemble.zip"
        sub_file.write_bytes(b"PK\x03\x04test_data")
        expected_hash = sha256(sub_file)

        _sess, obs = bridge.act(
            action="preserve_submission",
            params={
                "file": str(sub_file),
                "notes": "Adjudicated debate winner",
                "score": 0.912,
                "partition": "private",
            },
        )

        self.assertEqual(obs.status, "SUCCESS")
        self.assertIn("Preserved submission artifact", obs.summary)
        self.assertEqual(obs.data["sha256"], expected_hash)
        self.assertFalse(obs.data["dispatch_authorized"])
        self.assertEqual(obs.data["status"], "PRESERVED_UNVERIFIED")

    def test_agent_bridge_cli_formatting_and_parsing(self):
        schema = get_action_schema("preserve_submission")
        self.assertIsNotNone(schema)
        formatted = _format_action_schema("preserve_submission", schema)
        self.assertIn("--file <value>", formatted)
        self.assertIn("--notes <value> (optional)", formatted)
        self.assertIn('ai-harness-agent act --action preserve_submission --file "VALUE"', formatted)
        self.assertIn("PowerShell:", formatted)

        # Dynamic args parsing
        parsed = _parse_dynamic_kv_args(["--file", "output.zip", "--score", "0.85", "--notes", "test"])
        self.assertEqual(parsed["file"], "output.zip")
        self.assertEqual(parsed["score"], 0.85)
        self.assertEqual(parsed["notes"], "test")

        # Positional args must raise ValueError
        with self.assertRaises(ValueError):
            _parse_dynamic_kv_args(["unexpected_positional_arg"])

        # Coerce values check
        self.assertEqual(_coerce_value("nan"), "nan")
        self.assertEqual(_coerce_value("inf"), "inf")
        self.assertEqual(_coerce_value("-inf"), "-inf")
        self.assertEqual(_coerce_value("123"), 123)
        self.assertEqual(_coerce_value("12.34"), 12.34)
        self.assertIsNone(_coerce_value("null"))
        self.assertTrue(_coerce_value("true"))
        self.assertFalse(_coerce_value("false"))

    def test_runner_custom_scorer_logging(self):
        import argparse
        import contextlib
        import io
        import yaml
        from harness.runner import UniversalRunner

        root, task_config_path = make_workspace(prefix="test_runner_notice_")

        scorer_py = root / "custom_eval.py"
        scorer_py.write_text("def score(y, p): return 1.0\n", encoding="utf-8")

        cfg = {
            "experiment": {"id": "exp001", "name": "notice_test"},
            "runtime": {"seed": 42},
            "paths": {"metrics_evaluator": "custom_eval:score"},
        }
        exp_yaml = root / "configs" / "exp001.yaml"
        exp_yaml.parent.mkdir(parents=True, exist_ok=True)
        exp_yaml.write_text(yaml.safe_dump(cfg), encoding="utf-8")

        args = argparse.Namespace(
            project_root=str(root),
            task_config=str(task_config_path),
            fold=0,
            debug=True,
            probe_samples=None,
            job_id=None,
        )

        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            UniversalRunner(cfg, str(exp_yaml), args)
        out = f.getvalue()
        self.assertIn("[ai-harness] INFO: Custom metric scorer active:", out)
        self.assertIn("custom_eval:score", out)


if __name__ == "__main__":
    unittest.main()
