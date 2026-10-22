"""Codabench integration CLI (``ai-harness codabench``).

Subcommands:
    status    Check status and scores for a submission.
    download  Download logs and evaluation artifacts for a submission.

Examples:
    ai-harness codabench status 1007196 --session 288nmtzcsxx26hnrl8xa7wrqpdamcgta
    ai-harness codabench download 1007196 --output scratch/sub_1007196
"""
from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List, Optional

from harness.codabench_compat import CodabenchClient


def _print_payload(payload: Dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, default=str))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ai-harness codabench",
        description="Codabench platform integration (status, logs, artifact download).",
    )
    sub = parser.add_subparsers(dest="action", required=True)

    p_status = sub.add_parser("status", help="Check status and scores for a submission.")
    p_status.add_argument("sub_id", type=int, help="Numeric Codabench submission ID.")
    p_status.add_argument("--session", default=None, help="Codabench sessionid cookie value.")
    p_status.add_argument("--base-url", default="https://www.codabench.org", help="Codabench base URL.")

    p_download = sub.add_parser("download", help="Download logs and evaluation artifacts.")
    p_download.add_argument("sub_id", type=int, help="Numeric Codabench submission ID.")
    p_download.add_argument("--output", "-o", default=None, help="Destination directory (default: scratch/audit_<sub_id>).")
    p_download.add_argument("--session", default=None, help="Codabench sessionid cookie value.")
    p_download.add_argument("--base-url", default="https://www.codabench.org", help="Codabench base URL.")

    p_auth = sub.add_parser("auth", help="Save session token to .codabench_session.")
    p_auth.add_argument("session", help="Codabench sessionid cookie value.")

    args = parser.parse_args(argv)

    if args.action == "auth":
        token = str(args.session).strip()
        from pathlib import Path
        Path(".codabench_session").write_text(token, encoding="utf-8")
        _print_payload({"action": "auth", "saved": True, "file": ".codabench_session", "token_length": len(token)})
        return 0

    base_url = getattr(args, "base_url", "https://www.codabench.org")
    client = CodabenchClient(session_id=args.session, base_url=base_url)

    try:

        if args.action == "status":
            info = client.get_submission(args.sub_id)
            _print_payload({"action": "status", **info})
            return 0

        if args.action == "download":
            out_dir = args.output or f"scratch/audit_{args.sub_id}"
            res = client.download_submission(args.sub_id, output_dir=out_dir)
            _print_payload({"action": "download", **res})
            return 0
    except Exception as exc:  # noqa: BLE001
        _print_payload({"ok": False, "action": args.action, "error": str(exc)})
        return 1

    parser.error(f"unknown action: {args.action}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
