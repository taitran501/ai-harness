"""Atomic File I/O Utilities for ML-Harness.

Prevents file corruption and race conditions when multiple processes (CLI,
web dashboard, background runners) read and write shared JSON and text states.
Uses same-directory temporary files combined with atomic `os.replace()`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any


def atomic_write_text(
    path: Path | str,
    text: str,
    encoding: str = "utf-8",
) -> None:
    """Atomically write text to disk using a temporary file and os.replace()."""
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)

    # Temporary file must be created on the same filesystem/directory for atomic rename
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=str(target.parent),
        prefix=f".{target.name}.tmp_",
        text=True,
    )
    try:
        with open(tmp_fd, "w", encoding=encoding) as f:
            f.write(text)
        os.replace(tmp_path, str(target))
    except Exception:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise


def atomic_write_json(
    path: Path | str,
    data: Any,
    indent: int = 2,
    ensure_ascii: bool = False,
    encoding: str = "utf-8",
) -> None:
    """Atomically serialize data to a JSON file on disk."""
    dumped = json.dumps(data, indent=indent, ensure_ascii=ensure_ascii)
    atomic_write_text(path, dumped, encoding=encoding)


def get_project_root() -> Path:
    """Finds the root directory of the active project or competition workspace.

    Handles both standalone execution and subfolder plugin installation
    (e.g., when the harness is inside an 'ml-harness/' subfolder of a competition).
    """
    if "HARNESS_PROJECT_ROOT" in os.environ:
        return Path(os.environ["HARNESS_PROJECT_ROOT"]).resolve()

    cwd = Path.cwd().resolve()
    if (cwd / "configs" / "task.yaml").exists() or (cwd / "dataset").exists() or (cwd / "artifacts").exists():
        return cwd

    file_dir = Path(__file__).resolve().parent.parent
    if file_dir.name in ("ml-harness", "ml_harness"):
        parent = file_dir.parent
        if (parent / "configs" / "task.yaml").exists() or (parent / "dataset").exists():
            return parent
    return file_dir
