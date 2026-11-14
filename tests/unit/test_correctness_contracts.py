"""Thin shim re-exporting EvaluationContracts and MetricReferenceTests for critical-invariants smoke gate."""
from __future__ import annotations

import unittest

from tests.unit.test_evaluation_contracts import (  # noqa: F401
    EvaluationContracts,
    MetricReferenceTests,
    task,
)

__all__ = ["EvaluationContracts", "MetricReferenceTests", "task"]


if __name__ == "__main__":
    unittest.main()
