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
from pathlib import Path
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

    p_view = sub.add_parser("view", help="Read topic details and all comments.")
    p_view.add_argument("topic_id", type=int, help="Numeric topic ID.")
    p_view.add_argument(
        "--competition", "-c", default=None,
        help="Kaggle competition slug (optional).",
    )
    p_view.add_argument("--raw", action="store_true", help="Include raw HTML content.")
    p_view.add_argument("--save", default=None, help="Save the complete fetched payload as JSON before display filtering.")
    p_view.add_argument("--register", action="store_true", help="Register the complete payload in .ai-harness/evidence.")
    p_view.add_argument("--project-root", default=".", help="Workspace root used by --register.")
    p_view.add_argument("--snapshot-version", default=None, help="Explicit label when registering changed versions of a topic.")

    p_register = sub.add_parser("register", help="Register one explicitly selected topic JSON file as citation evidence.")
    p_register.add_argument("topic_id", type=int, help="Numeric topic ID to bind to the payload.")
    p_register.add_argument("--file", required=True, help="Selected JSON evidence file; no workspace search is performed.")
    p_register.add_argument("--adapter", required=True,
                            choices=["normalized_topic_v1", "data_wrapper_v1", "string_list_v1"],
                            help="Explicit supported source schema adapter.")
    p_register.add_argument("--project-root", default=".", help="Workspace root used for evidence registry storage.")
    p_register.add_argument("--competition", default=None, help="Competition slug if known.")
    p_register.add_argument("--canonical-url", default=None, help="Canonical source URL if known.")
    p_register.add_argument("--snapshot-version", default=None, help="Explicit label for a conflicting source version.")
    p_register.add_argument("--incomplete", action="store_true", help="Mark retrieval incomplete, such as failed comment retrieval.")

    args = parser.parse_args(argv)

    try:
        if args.action == "register":
            from harness.citation_sources import SourceRegistry

            record = SourceRegistry(args.project_root).register_topic_source(
                args.topic_id, None, adapter=args.adapter, source_path=args.file,
                competition=args.competition, canonical_url=args.canonical_url,
                snapshot_version=args.snapshot_version, complete=not args.incomplete,
                retrieval_provenance={"method": "explicit local file registration", "source_path": Path(args.file).name},
            )
            _print_payload({"action": "register", "source_id": record["source_id"],
                            "topic_id": record["topic_id"], "raw_sha256": record["raw_sha256"],
                            "complete": record["completeness"]["complete"]})
            return 0

        client = KaggleClient()
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
            topic_data = client.get_topic(args.topic_id, competition=args.competition)
            if args.save:
                from harness.io_utils import atomic_write_json

                atomic_write_json(args.save, topic_data, indent=2, ensure_ascii=False)
            registration = None
            if args.register:
                from harness.citation_sources import SourceRegistry

                registration = SourceRegistry(args.project_root).register_topic_source(
                    args.topic_id, topic_data, adapter="normalized_topic_v1",
                    competition=args.competition, canonical_url=topic_data.get("url"),
                    complete=topic_data.get("comments_retrieved", True),
                    snapshot_version=args.snapshot_version,
                    retrieval_provenance={"method": "ai-harness discussion view", "comments_retrieved": topic_data.get("comments_retrieved", True),
                                          "retrieval_errors": topic_data.get("retrieval_errors", [])},
                )
            if not args.raw:
                topic_data.pop("content_raw", None)
                for c in topic_data.get("comments", []):
                    c.pop("content_raw", None)
            _print_payload(
                {
                    "action": "view",
                    "competition": args.competition,
                    **topic_data,
                    **({"registration": {"source_id": registration["source_id"], "raw_sha256": registration["raw_sha256"],
                                          "complete": registration["completeness"]["complete"]}} if registration else {}),
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
