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
    _safe_json_loads,
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
        # official_scores must be strictly empty for unverified entries
        self.assertEqual(entry.official_scores, {})
        self.assertEqual(entry.reported_scores, {"reported_public": 0.895})
        self.assertEqual(entry.official_score_source, "user_reported_unverified")

        # Invariant: verify payload archive was not overwritten by receipt or metadata writes
        self.assertEqual(sha256(Path(entry.archive_path)), expected_hash)

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
        # Official LB column should show '-' and reported score should be in notes
        self.assertIn("| - | - | [`zip`]", md_text)
        self.assertIn("Reported: reported_public:0.895", md_text)

    def test_preserve_file_submission_validation_errors(self):
        # Non-existent file
        with self.assertRaises(FileNotFoundError):
            self.mgr.preserve_file_submission(file_path=self.p / "does_not_exist.csv")

        # Source file outside project root (containment security check)
        with tempfile.TemporaryDirectory() as external_tmp:
            ext_file = Path(external_tmp) / "external.csv"
            ext_file.write_text("x,y\n1,2\n", encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                self.mgr.preserve_file_submission(file_path=ext_file)
            self.assertIn("project root", str(ctx.exception))

        # Reserved metadata filenames must be rejected
        for reserved in ("PRESERVATION_RECEIPT.json", "metadata.json", "registry.json"):
            res_file = self.p / reserved
            res_file.write_text("{}", encoding="utf-8")
            with self.assertRaises(ValueError) as ctx:
                self.mgr.preserve_file_submission(file_path=res_file)
            self.assertIn("reserved metadata filename", str(ctx.exception))

        # Valid file for score tests
        valid_file = self.p / "valid_name.csv"
        valid_file.write_text("a,b\n1,2\n", encoding="utf-8")

        # Non-finite scores and booleans
        with self.assertRaises(ValueError):
            self.mgr.preserve_file_submission(file_path=valid_file, reported_score=float("nan"))
        with self.assertRaises(ValueError):
            self.mgr.preserve_file_submission(file_path=valid_file, reported_score=float("inf"))
        with self.assertRaises(ValueError):
            self.mgr.preserve_file_submission(file_path=valid_file, reported_score=True)

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

        # Dynamic args parsing with schema types
        param_types = schema.get("parameter_types", {})
        parsed = _parse_dynamic_kv_args(
            ["--file", "123", "--notes", "false", "--score", "0.85"],
            parameter_types=param_types,
        )
        # Type preservation: file must remain string "123", notes must remain string "false"
        self.assertEqual(parsed["file"], "123")
        self.assertIsInstance(parsed["file"], str)
        self.assertEqual(parsed["notes"], "false")
        self.assertIsInstance(parsed["notes"], str)
        self.assertEqual(parsed["score"], 0.85)

        # Boolean passed to finite_float must raise ValueError
        with self.assertRaises(ValueError):
            _parse_dynamic_kv_args(["--score", "true"], parameter_types=param_types)

        # Missing value for non-boolean flag must raise ValueError
        with self.assertRaises(ValueError):
            _parse_dynamic_kv_args(["--file"], parameter_types=param_types)

        # Positional args must raise ValueError
        with self.assertRaises(ValueError):
            _parse_dynamic_kv_args(["unexpected_positional_arg"])

        # JSON non-finite parsing rejection
        with self.assertRaises(ValueError):
            _safe_json_loads('{"score": NaN}')
        with self.assertRaises(ValueError):
            _safe_json_loads('{"score": Infinity}')

        # Coerce values check
        self.assertEqual(_coerce_value("nan"), "nan")
        self.assertEqual(_coerce_value("inf"), "inf")
        self.assertEqual(_coerce_value("-inf"), "-inf")
        self.assertEqual(_coerce_value("123"), 123)
        self.assertEqual(_coerce_value("12.34"), 12.34)
        self.assertIsNone(_coerce_value("null"))
        self.assertTrue(_coerce_value("true"))
        self.assertFalse(_coerce_value("false"))

    def test_runner_notice_and_spec_conflict(self):
        import argparse
        import contextlib
        import io
        import yaml
        from harness.runner import UniversalRunner
        from tests.fixtures import task_document

        # Create task spec with intentional direction conflict (rmse with maximize)
        doc = task_document(primary_metric="rmse", direction="maximize")
        root, task_config_path = make_workspace(
            prefix="test_runner_notice_",
            task=doc,
        )

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
        # Custom metric scorer notice must be printed
        self.assertIn("[ai-harness] INFO: Custom metric scorer active:", out)
        self.assertIn("custom_eval:score", out)
        # Genuine spec conflict warning must NOT be suppressed
        self.assertIn("[ai-harness] WARNING: spec:", out)


if __name__ == "__main__":
    unittest.main()
