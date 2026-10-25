"""Kaggle kernel wrap + dispatch CLI (``ai-harness dispatch-kernel``).

The harness-native, in-process alternative to the raw ``kaggle kernels push``
shell command (which the host hook blocks). It wraps a local notebook into a
Kaggle kernel (staging dir + ``kernel-metadata.json``) and pushes it via the
mockable :class:`harness.kaggle_compat.KaggleClient.kernels_push`, so the
dispatch is attributable and auditable.

This is the exact flow the RSNA endgame agents were hand-rolling with
``build_expXXX.py`` wrappers + 45s polling loops — now a single first-class
command.

Examples:
    ai-harness dispatch-kernel my_notebook.ipynb \
        --kernel-id trant/exp048 --title "Exp048: Ensemble" \
        --machine-shape KaggleGPU --dataset owner/dataset

    ai-harness dispatch-kernel train.py --kernel-id trant/exp048 \
        --title "Exp048" --machine-shape P100 --enable-internet
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

# Accelerator / machine-shape values recognised by Kaggle kernel push.
KNOWN_MACHINE_SHAPES = (
    "KaggleGPU",
    "KaggleGPU-P100",
    "KaggleTPUv2-8",
    "KaggleTPUv3-8",
    "P100",
    "T4",
    "V100",
    "A100",
)


def build_kernel_metadata(
    kernel_id: str,
    title: str,
    code_file: str = "code.py",
    machine_shape: Optional[str] = None,
    language: str = "python",
    kernel_type: str = "notebook",
    is_private: bool = True,
    enable_gpu: bool = False,
    enable_tpu: bool = False,
    enable_internet: bool = False,
    dataset_sources: Optional[List[str]] = None,
    competition_sources: Optional[List[str]] = None,
    kernel_sources: Optional[List[str]] = None,
    model_sources: Optional[List[str]] = None,
    keywords: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Build a ``kernel-metadata.json`` dict (the schema ``kernels push`` reads).

    Args:
        kernel_id: The kernel slug (``owner/kernel``).
        title: Human-readable kernel title.
        code_file: The code filename inside the kernel (default ``code.py``).
        machine_shape: Accelerator (``KaggleGPU``, ``P100``, ...). Omitted when
            None (runs CPU-only).
        language: Kernel language (default ``python``).
        kernel_type: ``notebook`` (default) or ``script``.
        is_private: Private (default True) vs public kernel.
        enable_gpu / enable_tpu: Legacy accelerator flags (machine_shape is the
            preferred, modern field; these are kept for back-compat).
        enable_internet: Whether the kernel may access the internet.
        dataset_sources / competition_sources / kernel_sources / model_sources:
            Source slugs (``owner/name`` lists).
        keywords: Free-form keyword tags.

    Returns:
        The metadata dict, ready to be written as ``kernel-metadata.json``.
    """
    meta: Dict[str, Any] = {
        "id": kernel_id,
        "title": title,
        "code_file": code_file,
        "language": language,
        "kernel_type": kernel_type,
        "is_private": bool(is_private),
        "enable_gpu": bool(enable_gpu),
        "enable_internet": bool(enable_internet),
        "enable_tpu": bool(enable_tpu),
        "keywords": list(keywords or []),
        "dataset_sources": list(dataset_sources or []),
        "competition_sources": list(competition_sources or []),
        "kernel_sources": list(kernel_sources or []),
        "model_sources": list(model_sources or []),
    }
    if machine_shape:
        meta["machine_shape"] = machine_shape
    return meta


def dispatch_kernel(
    notebook_path: str,
    kernel_id: str,
    title: str,
    code_file: Optional[str] = None,
    machine_shape: Optional[str] = None,
    dataset_sources: Optional[List[str]] = None,
    competition_sources: Optional[List[str]] = None,
    kernel_sources: Optional[List[str]] = None,
    model_sources: Optional[List[str]] = None,
    keywords: Optional[List[str]] = None,
    enable_internet: bool = False,
    timeout: Optional[int] = None,
    keep_staging: bool = False,
    client: Any = None,
    watch: bool = False,
    poll_interval: float = 15.0,
    watch_timeout: float = 3600.0,
) -> Dict[str, Any]:
    """Wrap a local notebook into a Kaggle kernel and push it.

    Creates a temp staging dir, copies the notebook in (as ``code_file``),
    writes ``kernel-metadata.json``, and pushes via the (mockable) client.

    Args:
        notebook_path: Path to the local ``.ipynb`` (or ``.py``) to push.
        kernel_id: The kernel slug (``owner/kernel``).
        title: Human-readable kernel title.
        code_file: Code filename inside the kernel (default: notebook name).
        machine_shape: Accelerator; also passed to ``kernels_push(acc=...)``.
        dataset_sources / competition_sources / kernel_sources / model_sources:
            Source slugs.
        keywords: Free-form keyword tags.
        enable_internet: Whether the kernel may access the internet.
        timeout: Kernel run timeout in seconds (optional).
        keep_staging: Keep the staging dir after the push (for debugging).
        client: A :class:`harness.kaggle_compat.KaggleClient` (or any object
            exposing ``kernels_push``). Created lazily when omitted.

    Returns:
        dict with ``ok``, ``ref``, ``url``, ``kernel_id``, ``version_number``,
        ``metadata`` and any ``invalid_*`` source warnings / ``error``.
    """
    nb = Path(notebook_path)
    if not nb.exists():
        raise FileNotFoundError(f"Notebook not found: {notebook_path}")
    if code_file is None:
        code_file = nb.name

    if client is None:
        from harness.kaggle_compat import KaggleClient

        client = KaggleClient()

    staging = Path(tempfile.mkdtemp(prefix="harness_kernel_"))
    try:
        from harness.cleaner import STAGING_MARKER

        (staging / STAGING_MARKER).write_text(str(os.getpid()), encoding="ascii")
    except Exception:
        pass
    try:
        shutil.copy2(nb, staging / code_file)
        meta = build_kernel_metadata(
            kernel_id=kernel_id,
            title=title,
            code_file=code_file,
            machine_shape=machine_shape,
            dataset_sources=dataset_sources,
            competition_sources=competition_sources,
            kernel_sources=kernel_sources,
            model_sources=model_sources,
            keywords=keywords,
            enable_internet=enable_internet,
        )
        (staging / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))

        resp = client.kernels_push(str(staging), timeout=timeout, acc=machine_shape)
        result: Dict[str, Any] = {
            "ok": True,
            "ref": getattr(resp, "ref", None),
            "url": getattr(resp, "url", None),
            "kernel_id": getattr(resp, "kernel_id", None),
            "version_number": getattr(resp, "version_number", None),
            "metadata": meta,
        }
        if keep_staging:
            result["staging_dir"] = str(staging)
        # Surface validation warnings / errors from the API response.
        for field in (
            "invalid_dataset_sources",
            "invalid_competition_sources",
            "invalid_kernel_sources",
            "invalid_model_sources",
            "error",
        ):
            val = getattr(resp, field, None)
            if val:
                result[field] = val
        # Dispatch governance (PR-C): append-only audit entry + advisory
        # duplicate warning (same notebook content re-pushed to the same
        # kernel id). Best-effort — audit failure never blocks the push.
        try:
            from harness.dispatch_audit import record_dispatch

            audit = record_dispatch(
                kind="kernel",
                kernel_id=kernel_id,
                notebook_path=str(nb),
                message=title,
                ref=result.get("ref"),
            )
        except Exception as exc:  # noqa: BLE001 - audit must never break dispatch
            import sys as _sys

            print(
                f"[dispatch_audit] warning: audit entry not recorded: {exc}",
                file=_sys.stderr,
            )
            audit = None
        if audit and audit.get("duplicate_of"):
            dup = audit["duplicate_of"]
            result["duplicate_of"] = dup
            result["duplicate_warning"] = (
                f"Duplicate kernel: identical notebook content was already pushed "
                f"to kernel '{kernel_id}' earlier; pushing again creates a new "
                "identical kernel version."
            )

        if watch and result.get("ok"):
            from harness.templates.kaggle_template import KaggleKernelManager

            target_slug = result.get("ref") or kernel_id
            watch_res = KaggleKernelManager.wait_for_kernel(
                exp_id="",
                kernel_slug=target_slug,
                poll_interval=poll_interval,
                timeout_seconds=watch_timeout,
            )
            result["watch"] = watch_res
            result["status"] = watch_res.get("status")
        return result
    finally:
        if not keep_staging:
            shutil.rmtree(staging, ignore_errors=True)


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point for ``ai-harness dispatch-kernel``."""
    parser = argparse.ArgumentParser(
        prog="ai-harness dispatch-kernel",
        description="Wrap a local notebook into a Kaggle kernel and push it.",
    )
    parser.add_argument("notebook", help="Local notebook/script path to push.")
    parser.add_argument("--kernel-id", "-i", required=True, help="Kernel slug (owner/kernel).")
    parser.add_argument("--title", "-t", required=True, help="Kernel title.")
    parser.add_argument("--code-file", default=None, help="Code filename in the kernel (default: notebook name).")
    parser.add_argument(
        "--machine-shape", default=None,
        help=f"Accelerator: {' / '.join(KNOWN_MACHINE_SHAPES)} (default: CPU).",
    )
    parser.add_argument("--dataset", action="append", default=[], help="Dataset source slug (repeatable).")
    parser.add_argument("--competition", action="append", default=[], help="Competition source slug (repeatable).")
    parser.add_argument("--kernel-source", action="append", default=[], help="Kernel source slug (repeatable).")
    parser.add_argument("--model", action="append", default=[], help="Model source slug (repeatable).")
    parser.add_argument("--keyword", action="append", default=[], help="Keyword tag (repeatable).")
    parser.add_argument("--enable-internet", action="store_true", help="Enable internet access in the kernel.")
    parser.add_argument("--timeout", type=int, default=None, help="Kernel run timeout seconds.")
    parser.add_argument("--keep-staging", action="store_true", help="Keep the staging dir after the push.")
    parser.add_argument("--watch", action="store_true", help="Poll until the dispatched kernel finishes execution.")
    parser.add_argument("--poll", type=float, default=15.0, help="Poll interval in seconds when --watch is used.")
    parser.add_argument("--watch-timeout", type=float, default=3600.0, help="Max wait time in seconds when --watch is used.")

    args = parser.parse_args(argv)
    try:
        result = dispatch_kernel(
            args.notebook,
            args.kernel_id,
            args.title,
            code_file=args.code_file,
            machine_shape=args.machine_shape,
            dataset_sources=args.dataset,
            competition_sources=args.competition,
            kernel_sources=args.kernel_source,
            model_sources=args.model,
            keywords=args.keyword,
            enable_internet=args.enable_internet,
            timeout=args.timeout,
            keep_staging=args.keep_staging,
            watch=args.watch,
            poll_interval=args.poll,
            watch_timeout=args.watch_timeout,
        )
    except Exception as exc:  # noqa: BLE001 - surface any dispatch failure as JSON.
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2, default=str))
        return 1
    print(json.dumps(result, indent=2, default=str))
    if result.get("duplicate_warning"):
        print(f"WARNING: {result['duplicate_warning']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
