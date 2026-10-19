"""Parse and import ``paths.metrics_evaluator`` hook references.

The hook reference appears in task/experiment configs in two spellings:
``module:function`` (colon) or ``module.function`` (dot). Every consumer
(runner scoring, OOF re-validation, artifact integrity gate) must parse
it identically — this module is the single source of truth.

Deliberately a standalone module: it is NOT part of the scorer identity
hash scope (``scorer_sha256`` seals ``harness.evaluation`` +
``harness.task_spec`` sources), so hardening the parser can never
invalidate sealed runs.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Tuple


def resolve_metrics_hook(hook: str) -> Tuple[str, str]:
    """Split a metrics_evaluator reference into ``(module, function)``.

    ``"src.evaluator:evaluate_rhae"``  -> ``("src.evaluator", "evaluate_rhae")``
    ``"src.evaluator.evaluate_rhae"``  -> ``("src.evaluator", "evaluate_rhae")``

    The colon takes precedence when present; otherwise the *last* dot
    splits the (possibly dotted) module path from the function name.
    """
    text = str(hook).strip()
    if ":" in text:
        module, name = text.split(":", 1)
    else:
        module, name = text.rsplit(".", 1)
    module, name = module.strip(), name.strip()
    if not module or not name:
        raise ValueError(
            f"Invalid metrics_evaluator reference {hook!r}: "
            "expected 'module:function' or 'module.function'"
        )
    return module, name


def ensure_hook_project_roots_on_sys_path(project_root: str | os.PathLike | None = None) -> None:
    """Put the task workspace on ``sys.path`` before custom scorer imports.

    ``validate_run`` executes outside the runner on journal sync, repair
    and artifact-import paths, where the runner's own ``sys.path``
    insertion is absent. An explicit ``project_root`` (known by the
    caller, e.g. the dispatch/pull context) takes precedence and is
    inserted first; without it, remote kernels rely on
    ``HARNESS_PROJECT_ROOT`` and attached workspaces resolve through
    ``harness.project``.
    """
    candidates = [os.environ.get("HARNESS_PROJECT_ROOT") or ""]
    try:
        from harness.project import resolve_project_root

        candidates.append(str(resolve_project_root()))
    except Exception:
        pass
    if project_root is not None:
        candidates.append(str(Path(project_root).expanduser()))
    for root in candidates:
        if root and root not in sys.path:
            sys.path.insert(0, root)
