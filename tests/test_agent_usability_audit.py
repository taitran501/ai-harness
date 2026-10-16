"""Unit tests for agent usability audit enhancements (Phase 4).

Covers:
1. Package & module aliases (`ai_harness`, `ml_harness`, `harness.agent`, `Runner`, `SubmissionValidator`, `kaggle_manager`).
2. Label normalization in evaluation contracts when ground truth or predictions are float representations of integer classes.
3. Helpful diagnostic error when continuous probabilities are passed where discrete class labels are expected.
4. Runner argument parser extraction (`build_parser`) and `--force-import` flag.
5. Kaggle remote runner error log extraction (`get_kernel_error_log`), transient 403 retry in `wait_for_dataset`, and `force_import` tolerance.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd

from harness.evaluation import evaluate, validate_prediction_contract
from harness.task_spec import UniversalTaskSpec
from harness.templates.kaggle_template import KaggleKernelManager


class TestAgentUsabilityAudit(unittest.TestCase):
    """Test suite covering usability audit fixes."""

    def test_import_aliases(self):
        """Verify all package and module aliases resolve and function correctly."""
        import ai_harness
        import ml_harness
        from ai_harness.task_spec import UniversalTaskSpec as SpecAlias
        from ml_harness.runner import build_parser
        from harness.agent import AgentBridge
        from harness.endgame_submission import SubmissionValidator, SubmissionPreflightValidator
        from harness import kaggle_manager, KaggleKernelManager as KKM

        self.assertIsNotNone(ai_harness)
        self.assertIsNotNone(ml_harness)
        self.assertIs(SpecAlias, UniversalTaskSpec)
        self.assertTrue(callable(build_parser))
        self.assertIs(SubmissionValidator, SubmissionPreflightValidator)
        self.assertIs(kaggle_manager, KKM)
        self.assertIsNotNone(AgentBridge)

    def test_runner_build_parser_and_force_import(self):
        """Verify runner exposes build_parser() and accepts --force-import."""
        from harness.runner import build_parser

        parser = build_parser()
        args = parser.parse_args(["--config", "configs/test.yaml", "--force-import"])
        self.assertTrue(args.force_import)
        self.assertEqual(args.config, "configs/test.yaml")

    def test_evaluation_float_class_normalization(self):
        """Verify evaluation contract normalizes float representations (0.0, 1.0) of int classes."""
        spec = UniversalTaskSpec.from_dict({
            "task": {
                "id_column": "id",
                "primary_metric": "accuracy",
                "direction": "maximize",
                "subtasks": {
                    "target": {
                        "task_type": "binary_classification",
                        "primary_metric": "accuracy",
                        "direction": "maximize",
                        "weight": 1.0,
                        "classes": [0, 1],
                    }
                },
            }
        })

        # Ground truth has float 0.0, 1.0 (common when loaded with pandas if missing values existed or read as float)
        truth_df = pd.DataFrame({"id": [1, 2, 3, 4], "target": [0.0, 1.0, 0.0, 1.0]})
        # Model predictions are float 0.0, 1.0
        preds = {"target": np.array([0.0, 1.0, 1.0, 1.0])}

        # validate_prediction_contract should pass without 'Predictions outside target domain'
        batch = validate_prediction_contract(spec, truth_df, preds)
        self.assertIsNotNone(batch)

        # evaluate should pass without 'Ground truth outside declared classes'
        result = evaluate(spec, truth_df, preds)
        self.assertAlmostEqual(result["primary_score"], 0.75)

    def test_evaluation_continuous_probability_diagnostic(self):
        """Verify clear error message is raised when continuous probabilities are passed instead of discrete labels."""
        spec = UniversalTaskSpec.from_dict({
            "task": {
                "id_column": "id",
                "primary_metric": "accuracy",
                "direction": "maximize",
                "subtasks": {
                    "target": {
                        "task_type": "binary_classification",
                        "primary_metric": "accuracy",
                        "direction": "maximize",
                        "weight": 1.0,
                        "classes": [0, 1],
                    }
                },
            }
        })

        truth_df = pd.DataFrame({"id": [1, 2, 3], "target": [0, 1, 0]})
        preds = {"target": np.array([0.25, 0.85, 0.40])}  # continuous floats, not 0/1

        with self.assertRaises(ValueError) as ctx:
            validate_prediction_contract(spec, truth_df, preds)

        msg = str(ctx.exception)
        self.assertIn("contain continuous probabilities", msg)
        self.assertIn("proba_target", msg)

    def test_wait_for_dataset_transient_error_retry(self):
        """Verify wait_for_dataset retries on initial transient 403 Forbidden before succeeding."""
        with patch("subprocess.run") as mock_run:
            # Simulate: 1st call 403 Forbidden, 2nd call 403 Forbidden, 3rd call READY
            call1 = MagicMock(returncode=1, stdout="", stderr="403 Client Error: Forbidden for url")
            call2 = MagicMock(returncode=1, stdout="", stderr="403 Client Error: Forbidden for url")
            call3 = MagicMock(returncode=0, stdout='"READY"', stderr="")
            mock_run.side_effect = [call1, call2, call3]

            # Should succeed without raising RuntimeError
            KaggleKernelManager.wait_for_dataset(
                "owner/test-dataset", poll_interval=0.01, timeout_seconds=5.0
            )
            self.assertEqual(mock_run.call_count, 3)

    def test_get_kernel_error_log(self):
        """Verify get_kernel_error_log parses JSON stream and extracts stderr traceback."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            log_file = Path(tmp_dir) / "test_kernel.log"
            log_entries = [
                {"stream_name": "stdout", "data": "Starting kernel execution\n"},
                {"stream_name": "stderr", "data": "Traceback (most recent call last):\n"},
                {"stream_name": "stderr", "data": "  File 'runner.py', line 10, in <module>\n"},
                {"stream_name": "stderr", "data": "ValueError: Test intentional failure\n"},
            ]
            lines = [json.dumps(e) for e in log_entries]
            log_file.write_text("\n".join(lines), encoding="utf-8")

            with patch("subprocess.run") as mock_run:
                # When kaggle kernels output runs, copy our log_file to target dir
                def fake_run(cmd, capture_output=True, text=True, timeout=60):
                    dest_dir = cmd[cmd.index("-p") + 1]
                    import shutil
                    shutil.copy(log_file, Path(dest_dir) / "test_kernel.log")
                    return MagicMock(returncode=0, stdout="", stderr="")

                mock_run.side_effect = fake_run
                err_log = KaggleKernelManager.get_kernel_error_log("owner/test-kernel")
                self.assertIsNotNone(err_log)
                self.assertIn("ValueError: Test intentional failure", err_log)
                self.assertIn("Traceback", err_log)


if __name__ == "__main__":
    unittest.main()
