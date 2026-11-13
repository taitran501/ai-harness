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
import importlib
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Tuple

_HOOK_IMPORT_LOCK = threading.RLock()


@contextmanager
def isolated_metrics_hook(hook, project_root=None):
    """Import and call workspace hooks without retaining their module graph."""
    with _HOOK_IMPORT_LOCK:
        original_path = sys.path[:]
        before = dict(sys.modules)
        roots, prefixes = [], set()
        try:
            ensure_hook_project_roots_on_sys_path(project_root)
            if project_root is None:
                from harness.project import resolve_project_root

                project_root = os.environ.get("HARNESS_PROJECT_ROOT") or resolve_project_root()
            roots = [Path(project_root).resolve()]
            root_text = str(roots[0])
            sys.path[:] = [root_text, *[p for p in sys.path if p != root_text]]
            for root in roots:
                if root.is_dir():
                    prefixes.update(p.stem if p.is_file() else p.name for p in root.iterdir()
                                    if (p.suffix == ".py" or p.is_dir()) and p.stem.isidentifier())
            prefixes.discard("harness")
            if roots[0] == Path(__file__).resolve().parent.parent:
                prefixes.difference_update({"tests", "tools"})
            def local_name(name):
                return name.split(".", 1)[0] in prefixes
            for name in list(sys.modules):
                if local_name(name):
                    del sys.modules[name]
            importlib.invalidate_caches()
            module, name = resolve_metrics_hook(hook)
            yield getattr(importlib.import_module(module), name)
        finally:
            sys.path[:] = original_path
            for name in list(sys.modules):
                local = name.split(".", 1)[0] in prefixes
                if name in before and not local:
                    continue
                obj = sys.modules[name]
                file = getattr(obj, "__dict__", {}).get("__file__")
                from_root = file and any(Path(file).resolve().is_relative_to(root) for root in roots)
                if local or (name not in before and from_root):
                    sys.modules.pop(name, None)
            for name, obj in before.items():
                if name not in sys.modules:
                    sys.modules[name] = obj


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
