"""Console-script wrappers for optional ML-Harness runtime features."""

from __future__ import annotations

import importlib
from typing import Any

from harness.optional_deps import require_extra


def _run(module_name: str, feature: str) -> int:
    if not require_extra("ml", feature):
        return 2
    module = importlib.import_module(module_name)
    result: Any = module.main()
    return int(result or 0)


def run_main() -> int:
    return _run("harness.runner", "Experiment execution")


def audit_main() -> int:
    return _run("tools.audit_gate", "Submission auditing")


def bootstrap_main() -> int:
    return _run("tools.bootstrap_task", "Dataset bootstrap")
