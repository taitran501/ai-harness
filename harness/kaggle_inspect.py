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


def kernel_watch(
    kernel_slug: Optional[str] = None,
    exp_id: Optional[str] = None,
    poll_interval: float = 15.0,
    timeout_seconds: float = 3600.0,
) -> Dict[str, Any]:
    """Wait/poll until a kernel completes (no re-dispatch)."""
    from harness.templates.kaggle_template import KaggleKernelManager

    if not kernel_slug and exp_id:
        kernel_slug = _resolve_exp_kernel_slug(exp_id)
    slug = kernel_slug or ""
    res = KaggleKernelManager.wait_for_kernel(
        exp_id=exp_id or "",
        kernel_slug=slug,
        poll_interval=poll_interval,
        timeout_seconds=timeout_seconds,
    )
    if not slug and res.get("kernel_slug"):
        slug = res["kernel_slug"]
    return {
        "kernel_slug": slug,
        "status": res.get("status"),
        "success": res.get("success", False),
        "timed_out": res.get("timed_out", False),
        "raw_output": res.get("raw_output", ""),
    }


def kernel_list(
    competition: Optional[str] = None,
    user: Optional[str] = None,
    page_size: int = 20,
    sort_by: str = "hot",
    search: Optional[str] = None,
) -> Dict[str, Any]:
    """List Kaggle kernels by competition or user."""
    from harness.kaggle_compat import KaggleClient

    client = KaggleClient()
    sort_map = {
        "hot": "hotness",
        "hotness": "hotness",
        "votes": "voteCount",
        "votecount": "voteCount",
        "voteCount": "voteCount",
        "updated": "dateRun",
        "daterun": "dateRun",
        "dateRun": "dateRun",
        "created": "dateCreated",
        "datecreated": "dateCreated",
        "dateCreated": "dateCreated",
        "relevance": "relevance",
    }
    canonical_sort = sort_map.get(sort_by.lower(), sort_by)
    kwargs: Dict[str, Any] = {"page_size": page_size, "sort_by": canonical_sort}
    if competition:
        kwargs["competition"] = competition
    if search:
        kwargs["search"] = search
    kernels = client.list_kernels(user=user, **kwargs)
    items = []
    for k in kernels:
        items.append({
            "ref": getattr(k, "ref", None),
            "title": getattr(k, "title", None),
            "author": getattr(k, "author", None),
            "total_votes": getattr(k, "total_votes", None) or getattr(k, "votes", None),
            "last_run_time": str(getattr(k, "last_run_time", "")) if getattr(k, "last_run_time", None) else None,
            "url": getattr(k, "url", None),
        })
    return {
        "action": "list",
        "competition": competition,
        "user": user,
        "count": len(items),
        "kernels": items,
    }


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

    p_watch = sub.add_parser("watch", help="Wait/poll until a kernel completes (no push).")
    p_watch.add_argument("kernel", nargs="?", help="Kernel slug (e.g. user/title) or --exp to derive from task slug")
    p_watch.add_argument("--exp", default=None, help="exp_id to derive the default kernel slug")
    p_watch.add_argument("--poll", type=float, default=15.0, help="Poll interval in seconds (default: 15)")
    p_watch.add_argument("--timeout", type=float, default=3600.0, help="Max wait time in seconds (default: 3600)")

    p_list = sub.add_parser("list", help="List kernels by competition or user.")
    p_list.add_argument("--competition", "-c", default=None, help="Filter by competition slug")
    p_list.add_argument("--user", "-u", default=None, help="Filter by username")
    p_list.add_argument("--sort-by", default="hot", help="Sort order: hot/hotness, votes/voteCount, updated/dateRun, created/dateCreated (default: hot)")
    p_list.add_argument("--page-size", "-n", type=int, default=20, help="Number of kernels (default: 20)")
    p_list.add_argument("--search", "-s", default=None, help="Search terms")

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
        if args.action == "watch":
            result = kernel_watch(args.kernel, exp_id=args.exp, poll_interval=args.poll, timeout_seconds=args.timeout)
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 0 if result.get("status") in {"COMPLETED", "QUEUED", "RUNNING"} else 2
        if args.action == "list":
            result = kernel_list(
                competition=args.competition,
                user=args.user,
                page_size=args.page_size,
                sort_by=args.sort_by,
                search=args.search,
            )
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 0
    except Exception as exc:  # noqa: BLE001 - report, don't crash
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}, indent=2, ensure_ascii=False))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
