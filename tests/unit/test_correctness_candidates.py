"""Thin shim re-exporting CandidateContracts for critical-invariants smoke gate."""
from __future__ import annotations

import unittest

from tests.unit.test_label_integrity import CandidateContracts  # noqa: F401

__all__ = ["CandidateContracts"]


if __name__ == "__main__":
    unittest.main()
