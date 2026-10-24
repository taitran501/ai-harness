"""`ai-harness competition` — inspect Kaggle competitions, metadata, deadlines and leaderboards."""
from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List, Optional

from harness.kaggle_compat import KaggleClient


def _print_payload(payload: Dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, default=str))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ai-harness competition",
        description="Inspect Kaggle competition metadata, deadlines, and active status.",
    )
    sub = parser.add_subparsers(dest="action", required=True)

    p_info = sub.add_parser("info", help="View competition metadata (deadline, metric, rules, user rank).")
    p_info.add_argument("competition", help="Kaggle competition slug.")

    p_list = sub.add_parser("list", help="List competitions.")
    p_list.add_argument("--search", "-s", default=None, help="Search keyword.")
    p_list.add_argument("--category", default=None, help="Category filter (e.g. Featured, Research, Playground).")
    p_list.add_argument("--sort-by", default=None, help="Sort order (e.g. latestDeadline, hotness).")
    p_list.add_argument("--page-size", "-n", type=int, default=20, help="Number of competitions (default: 20).")

    p_lb = sub.add_parser("leaderboard", aliases=["lb"], help="View public leaderboard.")
    p_lb.add_argument("competition", help="Kaggle competition slug.")
    p_lb.add_argument("--top", "-n", type=int, default=20, help="Top N entries (default: 20).")

    args = parser.parse_args(argv)
    client = KaggleClient()

    try:
        if args.action == "info":
            info = client.get_competition(args.competition)
            if not info:
                _print_payload({"ok": False, "error": f"Competition '{args.competition}' not found."})
                return 1
            _print_payload({"action": "info", "competition": args.competition, "info": info})
            return 0

        if args.action == "list":
            comps = client.list_competitions(
                search=args.search,
                category=args.category,
                sort_by=args.sort_by,
                page_size=args.page_size,
            )
            items = []
            for c in comps:
                items.append({
                    "id": getattr(c, "id", None),
                    "ref": getattr(c, "ref", None),
                    "title": getattr(c, "title", None),
                    "url": getattr(c, "url", None),
                    "deadline": str(getattr(c, "deadline", "")),
                    "category": getattr(c, "category", None),
                    "reward": getattr(c, "reward", None),
                    "team_count": getattr(c, "team_count", None) or getattr(c, "teamCount", None),
                    "evaluation_metric": getattr(c, "evaluation_metric", None) or getattr(c, "evaluationMetric", None),
                    "is_code_competition": bool(getattr(c, "is_kernels_submissions_only", False) or getattr(c, "isKernelsSubmissionsOnly", False)),
                })
            _print_payload({"action": "list", "count": len(items), "competitions": items})
            return 0

        if args.action in ("leaderboard", "lb"):
            entries = client.get_leaderboard(args.competition, page_size=args.top)
            results = []
            for rank, entry in enumerate(entries, 1):
                results.append({
                    "rank": rank,
                    "team_id": getattr(entry, "team_id", None),
                    "team_name": getattr(entry, "team_name", None),
                    "score": str(getattr(entry, "score", "")),
                    "submission_date": str(getattr(entry, "submission_date", "")),
                })
            _print_payload({
                "action": "leaderboard",
                "competition": args.competition,
                "count": len(results),
                "leaderboard": results,
            })
            return 0
    except Exception as exc:  # noqa: BLE001
        _print_payload({"ok": False, "error": str(exc)})
        return 1

    parser.error(f"unknown action: {args.action}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
