"""Kaggle submission dispatch + score-polling CLI (``ai-harness submission``).

The harness-native, in-process alternative to the raw ``kaggle competitions
submit`` shell command (which the host hook blocks). The tracker drives the
Kaggle API internally via :class:`harness.kaggle_compat.KaggleClient`, so
every dispatch is attributable and auditable.

Actions:
    dispatch  Submit a file (or code-comp kernel output) and print the ref.
    status    One-shot status/score check for a submission ref.
    watch     Poll until the score lands or the timeout elapses.
    latest    Show the best scored submission for a competition.

Examples:
    ai-harness submission dispatch rsna-knee-abnormality-detection submission.csv \
        -m "Exp 22 ensemble"
    ai-harness submission status rsna-knee-abnormality-detection 57314333
    ai-harness submission watch rsna-knee-abnormality-detection 57314333 \
        --timeout 1800 --poll 15
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional

from harness.endgame_submission import KaggleLeaderboardTracker


def _print_payload(payload: Dict[str, Any]) -> None:
    """Print a payload as pretty JSON (JSON is machine-readable for the agent)."""
    print(json.dumps(payload, indent=2, default=str))


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point for ``ai-harness submission``."""
    parser = argparse.ArgumentParser(
        prog="ai-harness submission",
        description="Kaggle submission dispatch + score polling (harness-native).",
    )
    sub = parser.add_subparsers(dest="action", required=True)

    p_dispatch = sub.add_parser(
        "dispatch",
        help="Submit and return the submission ref.",
        description=(
            "Submit to a competition. For standard competitions pass the local "
            "submission file; for code competitions pass the kernel output "
            "filename plus --kernel (owner/kernel) and --kernel-version."
        ),
    )
    p_dispatch.add_argument("competition", help="Kaggle competition slug.")
    p_dispatch.add_argument("file", help="Submission file (or kernel output filename).")
    p_dispatch.add_argument("--message", "-m", default="", help="Leaderboard label.")
    p_dispatch.add_argument(
        "--kernel", "-k", default=None,
        help="Code competition: public kernel slug (owner/kernel).",
    )
    p_dispatch.add_argument(
        "--kernel-version", default=None,
        help="Code competition: kernel version number.",
    )

    p_status = sub.add_parser("status", help="One-shot status/score check.")
    p_status.add_argument("competition", help="Kaggle competition slug.")
    p_status.add_argument("ref", help="Submission ref.")

    p_watch = sub.add_parser("watch", help="Poll until the score lands or timeout.")
    p_watch.add_argument("competition", help="Kaggle competition slug.")
    p_watch.add_argument("ref", help="Submission ref.")
    p_watch.add_argument(
        "--timeout", type=float, default=1800.0,
        help="Max seconds to wait (default 1800).",
    )
    p_watch.add_argument(
        "--poll", type=float, default=15.0,
        help="Initial poll interval seconds (default 15, grows 1.5x per round).",
    )
    p_watch.add_argument(
        "--max-poll", type=float, default=60.0,
        help="Max poll interval seconds (default 60).",
    )

    p_latest = sub.add_parser("latest", help="Best scored submission for a competition.")
    p_latest.add_argument("competition", help="Kaggle competition slug.")

    p_list = sub.add_parser("list", help="List recent submissions for a competition.")
    p_list.add_argument("competition", help="Kaggle competition slug.")
    p_list.add_argument("--limit", "-n", type=int, default=20, help="Max submissions to list.")

    args = parser.parse_args(argv)
    tracker = KaggleLeaderboardTracker()

    try:
        if args.action == "dispatch":
            payload = tracker.dispatch(
                args.competition,
                args.file,
                args.message,
                kernel=args.kernel,
                kernel_version=args.kernel_version,
            )
            _print_payload({"ok": True, "action": "dispatch", **payload})
            if payload.get("duplicate_warning"):
                print(f"WARNING: {payload['duplicate_warning']}", file=sys.stderr)
            return 0

        if args.action == "status":
            _print_payload({"action": "status", **tracker.get_submission(args.competition, args.ref)})
            return 0

        if args.action == "watch":
            payload = tracker.wait_for_score(
                args.competition,
                args.ref,
                timeout_s=args.timeout,
                poll_interval_s=args.poll,
                max_poll_interval_s=args.max_poll,
            )
            _print_payload({"action": "watch", **payload})
            return 0

        if args.action == "latest":
            _print_payload({"action": "latest", **tracker.latest(args.competition)})
            return 0

        if args.action == "list":
            subs = tracker.list_submissions(args.competition, limit=args.limit)
            _print_payload({"action": "list", "competition": args.competition, "count": len(subs), "submissions": subs})
            return 0
    except Exception as exc:  # noqa: BLE001 - surface any dispatch/poll failure as JSON.
        _print_payload({"ok": False, "action": args.action, "error": str(exc)})
        return 1

    parser.error(f"unknown action: {args.action}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
