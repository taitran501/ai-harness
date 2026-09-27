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
