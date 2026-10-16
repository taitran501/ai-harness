"""`ai-harness kaggle` — inspect Kaggle kernel state without re-dispatching.

Solves the #1 most expensive failure in the conversation audit: when a kernel
FAILS, the agent previously had to re-run the whole `ai-harness-run --backend
kaggle` dispatch (re-pushing a new kernel version, burning quota + waiting for
polling) just to see why it failed. These commands fetch status/log/output from
an *existing* kernel slug with no new push.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


def _resolve_exp_kernel_slug(exp_id: str) -> Optional[str]:
    """Return the kernel slug actually pushed for ``exp_id``, if a job state
    recorded one. The dispatch slug carries a job-hash suffix that slug
    derivation cannot reconstruct (``<owner>/<task>-<exp>-<hash8>``), so the
    derived default slug 404s on every real dispatch. Job state is the source
    of truth; derivation remains only as a fallback for legacy/unknown jobs.
    """
    if not exp_id:
        return None
    try:
        from harness.project import HarnessProject

        project = HarnessProject.discover(Path.cwd())
    except Exception:
        return None
    jobs_dir: Optional[Path] = None
    try:
        jobs_dir = project.artifacts_dir / "jobs"
    except Exception:
        return None
    if not jobs_dir.is_dir():
        return None
    best: Optional[tuple] = None
    for state_file in jobs_dir.glob("*.json"):
        if state_file.name.endswith(".worker.json"):
            continue
        try:
            state = json.loads(state_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if state.get("exp_id") != exp_id or not state.get("remote_id"):
            continue
        stamp = state.get("started_at") or 0
        if best is None or stamp >= best[0]:
            best = (stamp, str(state["remote_id"]))
    return best[1] if best else None


def _status_for(kernel_slug: Optional[str], exp_id: Optional[str]):
    from harness.templates.kaggle_template import KaggleKernelManager

    if not kernel_slug and exp_id:
        kernel_slug = _resolve_exp_kernel_slug(exp_id)
    return KaggleKernelManager.get_kernel_status(exp_id or "", kernel_slug)


def kernel_status(kernel_slug: Optional[str] = None, exp_id: Optional[str] = None) -> Dict[str, Any]:
    status = _status_for(kernel_slug, exp_id)
    if not kernel_slug and exp_id:
        status["kernel_slug"] = _resolve_exp_kernel_slug(exp_id) or status.get("kernel_slug")
    return status


def kernel_log(kernel_slug: Optional[str] = None, exp_id: Optional[str] = None, max_lines: int = 60) -> Dict[str, Any]:
    from harness.templates.kaggle_template import KaggleKernelManager

    status = _status_for(kernel_slug, exp_id)
    slug = status.get("kernel_slug") or (_resolve_exp_kernel_slug(exp_id) if exp_id else None) or kernel_slug or ""
    err_log = KaggleKernelManager.get_kernel_error_log(slug, max_lines=max_lines) if slug else None
    return {
        "kernel_slug": slug,
        "status": status.get("status"),
        "raw_output": status.get("raw_output", ""),
        "error_log": err_log,
    }


def kernel_output(kernel_slug: Optional[str] = None, exp_id: Optional[str] = None) -> Dict[str, Any]:
    """Download the kernel output to a temp dir and list files present (nothing is written to artifacts/)."""
    import subprocess
    import tempfile

    status = _status_for(kernel_slug, exp_id)
    slug = status.get("kernel_slug") or (_resolve_exp_kernel_slug(exp_id) if exp_id else None) or kernel_slug or ""
    if not slug:
        return {"kernel_slug": slug, "status": status.get("status"), "files": [], "error": "no kernel slug provided"}
    try:
        with tempfile.TemporaryDirectory(prefix="aih-inspect-") as temp_dir:
            result = subprocess.run(
                ["kaggle", "kernels", "output", slug, "-p", temp_dir],
                capture_output=True, text=True, timeout=120,
            )
            files = sorted(p.name for p in Path(temp_dir).rglob("*") if p.is_file())
            if result.returncode != 0:
                stderr_tail = (result.stderr or "").strip()[-500:]
                return {
                    "kernel_slug": slug,
                    "status": status.get("status"),
                    "files": files,
                    "error": (
                        f"kaggle CLI failed (rc={result.returncode}): {stderr_tail}"
                    ),
                }
            return {"kernel_slug": slug, "status": status.get("status"), "files": files}
    except Exception as exc:  # noqa: BLE001
        return {"kernel_slug": slug, "status": status.get("status"), "files": [], "error": str(exc)}


def main(argv=None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="ai-harness kaggle", description="Inspect Kaggle kernel state without re-dispatching.")
    sub = parser.add_subparsers(dest="action", required=True)

    p_status = sub.add_parser("status", help="Check a kernel's current status (no push).")
    p_status.add_argument("kernel", nargs="?", help="Kernel slug (e.g. user/title) or --exp to derive from task slug")
    p_status.add_argument("--exp", default=None, help="exp_id to derive the default kernel slug")
    p_status.add_argument("--json", action="store_true", default=True)

    p_log = sub.add_parser("log", help="Fetch a failed kernel's error log/stderr (no push).")
    p_log.add_argument("kernel", nargs="?", help="Kernel slug or --exp")
    p_log.add_argument("--exp", default=None, help="exp_id to derive the default kernel slug")
    p_log.add_argument("--max-lines", type=int, default=60)

    p_out = sub.add_parser("output", help="Download and list a kernel's output files (no push).")
    p_out.add_argument("kernel", nargs="?", help="Kernel slug or --exp")
    p_out.add_argument("--exp", default=None, help="exp_id to derive the default kernel slug")

    args = parser.parse_args(argv)
    try:
        if args.action == "status":
            result = kernel_status(args.kernel, exp_id=args.exp)
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 0 if result.get("status") in {"COMPLETED", "QUEUED", "RUNNING", "UNKNOWN"} else 2
        if args.action == "log":
            result = kernel_log(args.kernel, exp_id=args.exp, max_lines=args.max_lines)
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 0
        if args.action == "output":
            result = kernel_output(args.kernel, exp_id=args.exp)
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 0
    except Exception as exc:  # noqa: BLE001 - report, don't crash
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}, indent=2, ensure_ascii=False))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
