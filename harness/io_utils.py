"""Atomic File I/O Utilities for AI-Harness.

Prevents file corruption and race conditions when multiple processes (CLI,
web dashboard, background runners) read and write shared JSON and text states.
Uses same-directory temporary files combined with atomic `os.replace()`.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Optional


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


def json_default_serializer(obj: Any) -> Any:
    """Fallback serializer handling NumPy types, Path objects, and common non-JSON types."""
    import pathlib
    if isinstance(obj, (pathlib.Path, pathlib.PurePath)):
        return str(obj)

    try:
        import numpy as np
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:
        pass

    if hasattr(obj, "item") and callable(obj.item):
        try:
            return obj.item()
        except Exception:
            pass

    if hasattr(obj, "tolist") and callable(obj.tolist):
        try:
            return obj.tolist()
        except Exception:
            pass

    import datetime
    if isinstance(obj, (datetime.date, datetime.datetime)):
        return obj.isoformat()

    if isinstance(obj, (set, frozenset)):
        return list(obj)

    import dataclasses
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)

    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        try:
            return obj.to_dict()
        except Exception:
            pass

    import uuid
    if isinstance(obj, uuid.UUID):
        return str(obj)

    if isinstance(obj, (bytes, bytearray)):
        return obj.decode("utf-8", errors="replace")

    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def sanitize_for_json(obj: Any, seen: Optional[set[int]] = None) -> Any:
    """Recursively convert NumPy scalars/arrays, Paths, and dict keys to JSON-compatible types."""
    import pathlib

    if seen is None:
        seen = set()

    obj_id = id(obj)
    if isinstance(obj, (dict, list, tuple, set, frozenset)):
        if obj_id in seen:
            raise ValueError("Circular reference detected")
        seen.add(obj_id)

    try:
        if isinstance(obj, (pathlib.Path, pathlib.PurePath)):
            return str(obj)

        try:
            import numpy as np
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.bool_):
                return bool(obj)
            if isinstance(obj, np.ndarray):
                return [sanitize_for_json(x, seen) for x in obj.tolist()]
        except ImportError:
            pass

        if hasattr(obj, "item") and callable(obj.item):
            try:
                return sanitize_for_json(obj.item(), seen)
            except Exception:
                pass

        if hasattr(obj, "tolist") and callable(obj.tolist):
            try:
                return [sanitize_for_json(x, seen) for x in obj.tolist()]
            except Exception:
                pass

        if isinstance(obj, dict):
            clean_dict = {}
            for k, v in obj.items():
                if isinstance(k, (pathlib.Path, pathlib.PurePath)):
                    clean_k = str(k)
                elif hasattr(k, "item") and callable(k.item):
                    try:
                        clean_k = k.item()
                    except Exception:
                        clean_k = str(k)
                elif not isinstance(k, (str, int, float, bool, type(None))):
                    clean_k = str(k)
                else:
                    clean_k = k

                if not isinstance(clean_k, (str, int, float, bool, type(None))):
                    clean_k = str(clean_k)

                clean_dict[clean_k] = sanitize_for_json(v, seen)
            return clean_dict

        if isinstance(obj, (list, tuple)):
            return [sanitize_for_json(x, seen) for x in obj]

        if isinstance(obj, (set, frozenset)):
            return [sanitize_for_json(x, seen) for x in obj]

        import datetime
        if isinstance(obj, (datetime.date, datetime.datetime)):
            return obj.isoformat()

        import dataclasses
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return sanitize_for_json(dataclasses.asdict(obj), seen)

        if hasattr(obj, "to_dict") and callable(obj.to_dict):
            try:
                return sanitize_for_json(obj.to_dict(), seen)
            except Exception:
                pass

        import uuid
        if isinstance(obj, uuid.UUID):
            return str(obj)

        if isinstance(obj, (bytes, bytearray)):
            return obj.decode("utf-8", errors="replace")

        return obj
    finally:
        if isinstance(obj, (dict, list, tuple, set, frozenset)):
            seen.remove(obj_id)


def atomic_write_json(
    path: Path | str,
    data: Any,
    indent: int = 2,
    ensure_ascii: bool = False,
    encoding: str = "utf-8",
    default: Optional[Callable[[Any], Any]] = None,
) -> None:
    """Atomically serialize data to a JSON file on disk with fallback serialization."""
    def _fallback(o: Any) -> Any:
        if default is not None:
            try:
                return default(o)
            except Exception:
                pass
        return json_default_serializer(o)

    try:
        dumped = json.dumps(
            data,
            indent=indent,
            ensure_ascii=ensure_ascii,
            default=_fallback,
        )
    except TypeError:
        # Handles non-primitive dict keys (e.g. np.int64 or Path keys) that bypass default serializer
        clean_data = sanitize_for_json(data)
        dumped = json.dumps(
            clean_data,
            indent=indent,
            ensure_ascii=ensure_ascii,
            default=_fallback,
        )

    atomic_write_text(path, dumped, encoding=encoding)


def get_project_root() -> Path:
    """Finds the root directory of the active project or competition workspace.

    Shared resolver honors an explicit root or an attached workspace manifest.
    """
    from harness.project import resolve_project_root

    return resolve_project_root()


def set_seed(seed: int = 42) -> None:
    """Set global seed for Python, NumPy, and PyTorch (GPU runs only).

    PyTorch is heavy to import (~4-7s cold on a CUDA build). It is only needed for
    GPU determinism, so on a CPU-only / no-torch environment we skip it entirely.
    On a machine that has a GPU but is running CPU workloads (e.g. the unit /
    regression test suite, which never touches CUDA), set
    ``HARNESS_DISABLE_TORCH_SEED=1`` to skip the torch import without changing any
    seeded value on the CPU path. When a real GPU run needs torch/cuDNN seeding,
    leave it unset (the default) and torch is imported and seeded as before.
    """
    import random
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    if os.environ.get("HARNESS_DISABLE_TORCH_SEED"):
        return
    try:
        import torch
        if not torch.cuda.is_available():
            # CPU-only run: torch import was paid but nothing to seed on GPU.
            torch.manual_seed(seed)
            return
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def get_git_commit_hash(cwd: Optional[Path | str] = None) -> str:
    """Return the current git commit hash and dirty status, or 'unversioned'."""
    import subprocess
    work_dir = str(cwd) if cwd else None
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            cwd=work_dir,
        )
        if result.returncode == 0:
            commit_hash = result.stdout.strip()
            status_res = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                check=False,
                cwd=work_dir,
            )
            dirty = "-dirty" if status_res.stdout.strip() else ""
            return f"{commit_hash}{dirty}"
    except Exception:
        pass
    return "unversioned"


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load configuration from a YAML file."""
    import yaml
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config or {}


def save_json(path: str | Path, payload: Any) -> None:
    """Save payload as a formatted JSON file using atomic write."""
    atomic_write_json(path, payload, indent=2, ensure_ascii=False)


