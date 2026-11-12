"""Thin shim re-exporting ScopeStabilityTests for critical-invariants smoke gate."""
import unittest

from tests.unit.test_research_memory import (  # noqa: F401
    ScopeStabilityTests,
    TestResearchMemoryScopeStability,
)

__all__ = ["ScopeStabilityTests", "TestResearchMemoryScopeStability"]


if __name__ == "__main__":
    unittest.main()
