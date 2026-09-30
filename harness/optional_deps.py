"""Optional feature dependency checks for ML-Harness commands."""

from __future__ import annotations

import importlib.util
import sys
from typing import Dict, List, Tuple

_EXTRA_MODULES: Dict[str, Tuple[str, ...]] = {
    "ml": ("numpy", "pandas", "sklearn", "scipy"),
    "kaggle": ("kaggle",),
    "modal": ("modal",),
}
_MODULE_DISTRIBUTIONS = {
    "sklearn": "scikit-learn",
}


def missing_extra_modules(extra: str) -> List[str]:
    """Return importable modules required by an optional extra but not installed."""
    modules = _EXTRA_MODULES.get(extra)
    if modules is None:
        raise ValueError(f"Unknown ML-Harness extra: {extra}")
    missing = []
    for module in modules:
        try:
            present = importlib.util.find_spec(module) is not None
        except (ImportError, ValueError):
            present = False
        if not present:
            missing.append(_MODULE_DISTRIBUTIONS.get(module, module))
    return missing


def require_extra(extra: str, feature: str) -> bool:
    """Print installation instructions and return False when a feature extra is absent."""
    missing = missing_extra_modules(extra)
    if not missing:
        return True

    command = (
        "uv pip install "
        f"\"ml-harness[{extra}] @ git+https://github.com/taitran501/ml-harness.git"
        "@feature/ml-harness-installable-plugin\""
    )
    print(
        f"{feature} requires the optional '{extra}' dependencies: {', '.join(missing)}.\n"
        f"Install the extra in this environment, for example:\n  {command}",
        file=sys.stderr,
    )
    return False
