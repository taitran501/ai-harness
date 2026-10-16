"""Self-healing toolchain routing for the ai-harness CLI entry points.

The ``ai-harness*`` console scripts are ONE global (editable) install whose
target changes whenever a workspace syncs. Without pinning, a CLI launched
in workspace A can silently execute the framework code installed for
workspace B (observed: with the global target pointing at another clone,
``import harness`` in workspace A resolved to that other clone). ``sync``
pinned its attach step (PR #62); this module applies the same pin to the
EXECUTION entry points.

Every CLI ``main()`` calls :func:`ensure_pinned_toolchain` right after
project discovery. When the loaded ``harness`` package does not originate
from the workspace's own framework clone, the command re-executes itself
as ``python -m <module>`` with ``PYTHONPATH`` pinned to the clone (cwd and
argv preserved so relative paths keep their meaning). The re-exec is
announced on stderr so it is never invisible.

Invariants (same doctrine as the host hooks):
- fail-open: any error in this module lets the current process continue;
- escape hatch: ``AI_HARNESS_TOOLCHAIN_NO_PIN=1`` disables the re-exec.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, Sequence, Tuple

PIN_DISABLE_ENV = "AI_HARNESS_TOOLCHAIN_NO_PIN"
CLONE_DIR_NAMES = ("ai-harness", "ai_harness")


def find_workspace_clone(project_root: Optional[Path | str]) -> Optional[Path]:
    """Return the workspace's framework clone dir, or None when absent.

    A clone must contain a real ``harness`` package — a plain directory is
    not enough (the sync fixture lesson: a stub cannot stand in for code).
    """
    try:
        root = Path(project_root).expanduser()  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    for name in CLONE_DIR_NAMES:
        cand = root / name
        if (cand / "harness" / "__init__.py").is_file():
            return cand
    return None


def current_origin() -> Optional[Path]:
    """Path of the ``harness.__init__`` this process actually loaded."""
    try:
        import harness  # noqa: F401 - already imported by every caller

        return Path(harness.__file__).resolve()
    except Exception:
        return None


def _needs_pin(project_root: Optional[Path | str]) -> Tuple[bool, str, str]:
    """(drift?, loaded_origin, expected_origin). No clone -> no pin."""
    clone = find_workspace_clone(project_root)
    origin = current_origin()
    if clone is None or origin is None:
        return False, "", ""
    expected = (clone / "harness" / "__init__.py").resolve()
    return origin != expected, str(origin), str(expected)


def ensure_pinned_toolchain(
    project_root: Optional[Path | str],
    module: str,
    argv: Optional[Sequence[str]] = None,
    runner=subprocess.run,
) -> Optional[subprocess.CompletedProcess]:
    """Re-exec ``module`` under the workspace clone when the loaded code
    originates elsewhere.

    Returns ``None`` when the current process may continue (no drift, no
    clone, disabled, or re-exec failed — fail-open), otherwise the
    CompletedProcess of the pinned child; the caller must exit with that
    child's return code.
    """
    if os.environ.get(PIN_DISABLE_ENV) == "1":
        return None
    try:
        needs, origin, _expected = _needs_pin(project_root)
    except Exception:
        return None
    if not needs:
        return None
    clone = find_workspace_clone(project_root)
    if clone is None:
        return None
    cmd = [sys.executable, "-m", module] + list(argv if argv is not None else sys.argv[1:])
    env = dict(os.environ)
    env["PYTHONPATH"] = str(clone) + os.pathsep + env.get("PYTHONPATH", "")
    print(
        f"[ai-harness] toolchain pin: re-executing '{module}' under the workspace "
        f"clone ({clone}) - loaded code originated from {origin}",
        file=sys.stderr,
    )
    try:
        return runner(cmd, env=env, cwd=os.getcwd())
    except Exception as exc:  # noqa: BLE001 - fail-open by doctrine
        print(
            f"[ai-harness] toolchain pin: re-exec failed ({exc}); continuing with {origin}",
            file=sys.stderr,
        )
        return None
