"""Kaggle competition discussion topics and messages CLI (``ai-harness discussion``).

Enables agents to query discussions and retrieve insights safely without raw
SDK imports or missing positional argument friction.

Actions:
    list  List discussion topics for a competition.
    view  Read messages/comments for a specific discussion topic.

Examples:
    ai-harness discussion list arc-prize-2026-arc-agi-3
    ai-harness discussion list playground-series-s6e10 --sort-by hotness
    ai-harness discussion view 713634 --competition arc-prize-2026-arc-agi-3
"""
from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List, Optional

from harness.kaggle_compat import KaggleClient


def _print_payload(payload: Dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, default=str))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ai-harness discussion",
        description="Inspect Kaggle competition discussions and read messages (harness-native).",
    )
    sub = parser.add_subparsers(dest="action", required=True)

    p_list = sub.add_parser("list", help="List discussion topics for a competition.")
    p_list.add_argument("competition", help="Kaggle competition slug.")
    p_list.add_argument(
        "--sort-by", "-s", default="hotness",
        choices=["hotness", "recent", "top", "active"],
        help="Sort order (default: hotness).",
    )
    p_list.add_argument("--page", "-p", type=int, default=1, help="Page number (default: 1).")
    p_list.add_argument("--page-size", type=int, default=20, help="Page size (default: 20).")

    p_view = sub.add_parser("view", help="Read messages/comments for a discussion topic.")
    p_view.add_argument("topic_id", type=int, help="Numeric topic ID.")
    p_view.add_argument(
        "--competition", "-c", default=None,
        help="Kaggle competition slug (optional but recommended).",
    )
    p_view.add_argument("--page-size", type=int, default=50, help="Page size (default: 50).")

    args = parser.parse_args(argv)
    client = KaggleClient()

    try:
        if args.action == "list":
            kwargs: Dict[str, Any] = {}
            if args.page > 1:
                kwargs["page"] = args.page
            if args.page_size != 20:
                kwargs["page_size"] = args.page_size
            if args.sort_by:
                kwargs["sort_by"] = args.sort_by
            topics_raw = client.competition_topics(args.competition, **kwargs)
            topics = []
            for t in topics_raw:
                topics.append(
                    {
                        "id": getattr(t, "id", None),
                        "title": getattr(t, "title", ""),
                        "author": getattr(t, "author", "") or getattr(t, "author_name", ""),
                        "votes": getattr(t, "votes", 0) or getattr(t, "total_votes", 0) or 0,
                        "comments": getattr(t, "comments", 0) or getattr(t, "comment_count", 0) or 0,
                        "date": str(getattr(t, "date", "") or getattr(t, "post_date", "") or ""),
                        "url": getattr(t, "topic_url", "") or "",
                    }
                )
            _print_payload(
                {
                    "action": "list",
                    "competition": args.competition,
                    "count": len(topics),
                    "topics": topics,
                }
            )
            return 0

        if args.action == "view":
            kwargs = {}
            if args.page_size != 50:
                kwargs["page_size"] = args.page_size
            msgs_raw = client.competition_topic_messages(
                args.topic_id, competition=args.competition, **kwargs
            )
            messages = []
            for m in msgs_raw:
                messages.append(
                    {
                        "id": getattr(m, "id", None),
                        "author": getattr(m, "author", "") or getattr(m, "author_name", ""),
                        "date": str(getattr(m, "date", "") or getattr(m, "post_date", "") or ""),
                        "votes": getattr(m, "votes", 0) or getattr(m, "total_votes", 0) or 0,
                        "content": getattr(m, "content", "") or getattr(m, "message", "") or "",
                    }
                )
            _print_payload(
                {
                    "action": "view",
                    "topic_id": args.topic_id,
                    "competition": args.competition,
                    "count": len(messages),
                    "messages": messages,
                }
            )
            return 0
    except Exception as exc:  # noqa: BLE001
        _print_payload({"ok": False, "action": args.action, "error": str(exc)})
        return 1

    parser.error(f"unknown action: {args.action}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
