"""Tests for structured failure classification and actionable recovery guidance (P1-02 & P1-03)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from harness.failure_classifier import FailureCode, classify_failure  # noqa: E402


class TestFailureClassifier(unittest.TestCase):
    def test_missing_dependency_module_not_found(self):
        log = "Traceback:\nModuleNotFoundError: No module named 'lightgbm'"
        res = classify_failure(log, returncode=1)
        self.assertEqual(res["code"], FailureCode.DEPENDENCY_MISSING)
        self.assertEqual(res["category"], "environment")
        self.assertFalse(res["retryable"])
        self.assertIn("pip install lightgbm", res["recovery"]["suggested_command"])

    def test_missing_symbol_cannot_import(self):
        log = "ImportError: cannot import name 'FastModel' from 'models'"
        res = classify_failure(log, returncode=1)
        self.assertEqual(res["code"], FailureCode.IMPORT_SYMBOL_ERROR)
        self.assertIsNone(res["recovery"]["suggested_command"])
        self.assertIn("FastModel", res["recovery"]["suggested_action"])

    def test_cuda_oom(self):
        log = "torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 MiB"
        res = classify_failure(log, returncode=1)
        self.assertEqual(res["code"], FailureCode.RESOURCE_OOM)
        self.assertEqual(res["category"], "resource")
        self.assertIn("batch size", res["recovery"]["suggested_action"])

    def test_auth_required(self):
        log = "Kaggle authentication unavailable. Please set KAGGLE_USERNAME and KAGGLE_KEY."
        res = classify_failure(log, returncode=1)
        self.assertEqual(res["code"], FailureCode.AUTH_REQUIRED)
        self.assertEqual(res["category"], "provider")

    def test_quota_exceeded(self):
        log = "API error: 429 Too Many Requests. Quota exceeded for today."
        res = classify_failure(log, returncode=1)
        self.assertEqual(res["code"], FailureCode.QUOTA_EXCEEDED)
        self.assertTrue(res["retryable"])
        self.assertTrue(res["recovery"]["retryable"])

    def test_config_invalid(self):
        log = "yaml.parser.ParserError: while parsing a block mapping in configs/exp.yaml"
        res = classify_failure(log, returncode=1)
        self.assertEqual(res["code"], FailureCode.CONFIG_INVALID)
        self.assertEqual(res["recovery"]["suggested_command"], "ai-harness validate-task")

    def test_artifact_unverifiable(self):
        log = "Run exp001 has artifact hash drift; seal is broken."
        res = classify_failure(log, returncode=1)
        self.assertEqual(res["code"], FailureCode.ARTIFACT_UNVERIFIABLE)
        self.assertEqual(res["recovery"]["suggested_command"], "ai-harness repair --check")

    def test_data_contract_invalid(self):
        log = "ValueError: Task ground truth missing id_column 'sample_id'"
        res = classify_failure(log, returncode=1)
        self.assertEqual(res["code"], FailureCode.DATA_CONTRACT_INVALID)
        self.assertEqual(res["category"], "data")

    def test_unknown_fallback(self):
        log = "Fatal syntax or unhandled exception in line 42: division by zero"
        res = classify_failure(log, returncode=139)
        self.assertEqual(res["code"], FailureCode.UNKNOWN)
        self.assertFalse(res["retryable"])
        self.assertIn("139", res["detail"])


if __name__ == "__main__":
    unittest.main()
