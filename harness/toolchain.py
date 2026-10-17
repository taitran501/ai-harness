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
#: Re-exec generation counter. Any process launched with this set is ALREADY
#: a pinned child; if drift persists there, re-exec cannot converge (the
#: child's `-m` sys.path puts the cwd first, and a cwd that itself holds a
#: different ``harness`` package — e.g. the framework repo — will always
#: win over PYTHONPATH). Observed live: an unguarded chain spawned 35+
#: generations of ``python -m harness.cli`` and pegged the CPU. The guard
#: bounds the chain to one re-exec and fails open with a clear message.
PIN_DEPTH_ENV = "AI_HARNESS_PIN_DEPTH"
CLONE_DIR_NAMES = ("ai-harness", "ai_harness")
#: Environment variable pointing at the single canonical framework checkout
#: shared by every workspace (single-install architecture). When set and
#: valid it takes precedence over any per-workspace clone — this is what
#: removes cross-workspace drift at the source.
FRAMEWORK_DIR_ENV = "HARNESS_FRAMEWORK_DIR"


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


def find_framework_root(project_root: Optional[Path | str]) -> Optional[Path]:
    """Return the authoritative framework root for this invocation.

    Resolution order (single-install architecture):

    1. ``HARNESS_FRAMEWORK_DIR`` — an explicit canonical checkout shared by
       every workspace. When set and valid it is always preferred, which is
       exactly what removes cross-workspace drift.
    2. The per-workspace clone (``<workspace>/ai-harness``) — the legacy
       layout, kept as a fallback so not-yet-migrated workspaces keep
       working.

    Returns the directory that directly contains the ``harness`` package, or
    ``None`` when no framework checkout can be found.
    """
    env_dir = os.environ.get(FRAMEWORK_DIR_ENV)
    if env_dir:
        candidate = Path(env_dir).expanduser()
        if not candidate.is_absolute():
            base = Path(project_root).expanduser() if project_root is not None else Path.cwd()
            candidate = base / candidate
        candidate = candidate.resolve()
        if (candidate / "harness" / "__init__.py").is_file():
            return candidate
    return find_workspace_clone(project_root)


def current_origin() -> Optional[Path]:
    """Path of the ``harness.__init__`` this process actually loaded."""
    try:
        import harness  # noqa: F401 - already imported by every caller

        return Path(harness.__file__).resolve()
    except Exception:
        return None


def _needs_pin(project_root: Optional[Path | str]) -> Tuple[bool, str, str]:
    """(drift?, loaded_origin, expected_origin). No framework root -> no pin."""
    framework_root = find_framework_root(project_root)
    origin = current_origin()
    if framework_root is None or origin is None:
        return False, "", ""
    expected = (framework_root / "harness" / "__init__.py").resolve()
    return origin != expected, str(origin), str(expected)


def ensure_pinned_toolchain(
    project_root: Optional[Path | str],
    module: str,
    argv: Optional[Sequence[str]] = None,
    runner=subprocess.run,
) -> Optional[subprocess.CompletedProcess]:
    """Re-exec ``module`` under the framework root when the loaded code
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
    # Circuit breaker: we are already the pinned child and the clone still
    # does not win path resolution. Re-exec would loop forever (each child
    # re-detects the same drift); continue with the loaded code instead.
    try:
        depth = int(os.environ.get(PIN_DEPTH_ENV, "0") or "0")
    except ValueError:
        depth = 0
    if depth >= 1:
        cwd = os.getcwd()
        hint = (
            f" the cwd ({cwd}) appears to hold a different harness package, which "
            f"'-m' path resolution always prefers over the pinned clone. Run the "
            f"command from inside the workspace, or from outside it set "
            f"{PIN_DISABLE_ENV}=1."
        ) if (Path(cwd) / "harness" / "__init__.py").is_file() else (
            " check that the clone at the pinned dir is complete."
        )
        print(
            f"[ai-harness] toolchain pin: drift still present in the pinned child; "
            f"NOT re-executing again (would loop) — continuing with {origin}.{hint}",
            file=sys.stderr,
        )
        return None
    pinned_root = find_framework_root(project_root)
    if pinned_root is None:
        return None
    cmd = [sys.executable, "-m", module] + list(argv if argv is not None else sys.argv[1:])
    env = dict(os.environ)
    env["PYTHONPATH"] = str(pinned_root) + os.pathsep + env.get("PYTHONPATH", "")
    env[PIN_DEPTH_ENV] = str(depth + 1)
    print(
        f"[ai-harness] toolchain pin: re-executing '{module}' under the framework "
        f"root ({pinned_root}) - loaded code originated from {origin}",
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
