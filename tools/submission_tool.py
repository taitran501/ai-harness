#!/usr/bin/env python3
"""CLI tool for managing StereoQueerEval competition submissions and score tracking.

Enables agents and human engineers to:
1. Register verified submissions through the statistical audit gate into submissions/.
2. List all submissions and view Local CV vs Remote Leaderboard deltas.
3. Record official evaluation scores returned by competition organizers.
4. Track submission statuses (LOCAL_VERIFIED, SUBMITTED_PENDING, SCORED).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.submission_manager import SubmissionManager


def main() -> None:
    parser = argparse.ArgumentParser(
        description="StereoQueerEval Submission Lifecycle & Score Tracking CLI"
    )
    subparsers = parser.add_subparsers(dest="action", help="Action to perform")

    # 1. Register Action
    reg_parser = subparsers.add_parser("register", help="Register and audit a new submission")
    reg_parser.add_argument("--exp-id", required=True, help="Experiment ID that generated the predictions")
    reg_parser.add_argument("--candidate", "-c", required=True, help="Path to candidate prediction CSV")
    reg_parser.add_argument("--baseline", "-b", default=None, help="Path to parent baseline prediction CSV")
    reg_parser.add_argument(
        "--ensemble-components",
        nargs="*",
        default=None,
        help="Constituent prediction CSVs if candidate is an ensemble",
    )
    reg_parser.add_argument("--notes", default="", help="Descriptive notes or hypothesis context")
    reg_parser.add_argument(
        "--min-diff-pct",
        type=float,
        default=2.0,
        help="Minimum required prediction change percentage (MMES, default: 2.0%%)",
    )

    # 2. List Action
    list_parser = subparsers.add_parser("list", help="List all registered submissions and score comparison")

    # 3. Record Score Action
    score_parser = subparsers.add_parser("record-score", help="Record official scores from competition leaderboard")
    score_parser.add_argument("--sub-id", required=True, help="Submission ID to update (e.g., sub_001_exp001)")
    score_parser.add_argument("--macro-a", type=float, default=None, help="Official Subtask A Macro-F1")
    score_parser.add_argument("--macro-b", type=float, default=None, help="Official Subtask B Macro-F1")
    score_parser.add_argument("--macro-c-type", type=float, default=None, help="Official Subtask C Target Type Macro-F1")
    score_parser.add_argument("--macro-c-id", type=float, default=None, help="Official Subtask C Identity Micro/Macro-F1")
    score_parser.add_argument("--split-type", choices=["public", "private"], default="public", help="Leaderboard split")
    score_parser.add_argument("--notes", default="", help="Additional feedback or notes from competition portal")

    # 4. Mark Submitted Action
    sub_parser = subparsers.add_parser("mark-submitted", help="Mark submission as uploaded to competition portal")
    sub_parser.add_argument("--sub-id", required=True, help="Submission ID that was uploaded")
    sub_parser.add_argument("--ref", default="", help="Competition portal reference or submission ID")

    args = parser.parse_args()
    mgr = SubmissionManager()

    if args.action == "register":
        print(f"===========================================================")
        print(f"📦 REGISTERING SUBMISSION FOR EXPERIMENT: {args.exp_id}")
        print(f"===========================================================")
        try:
            entry = mgr.register_submission(
                exp_id=args.exp_id,
                candidate_csv=args.candidate,
                baseline_csv=args.baseline,
                ensemble_components=args.ensemble_components,
                notes=args.notes,
                min_diff_pct=args.min_diff_pct,
            )
            print(f"✔ Submission ID     : {entry.id}")
            print(f"✔ Status            : {entry.status}")
            print(f"✔ Archive Package   : {entry.archive_path}")
            print(f"✔ Audit Receipt     : {entry.receipt_path}")
            print(f"✔ Local CV Scores   : {entry.local_cv_scores}")
            print(f"✔ Registered in     : submissions/SUBMISSIONS.md")
            print(f"===========================================================")
            print(f"🎉 SUCCESS: Submission package verified and ready for upload!")
        except Exception as e:
            print(f"❌ REGISTRATION FAILED: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.action == "record-score":
        try:
            entry = mgr.record_official_score(
                sub_id=args.sub_id,
                macro_a=args.macro_a,
                macro_b=args.macro_b,
                macro_c_type=args.macro_c_type,
                macro_c_id=args.macro_c_id,
                split_type=args.split_type,
                notes=args.notes,
            )
            print(f"===========================================================")
            print(f"🎯 RECORDED OFFICIAL SCORES FOR: {entry.id}")
            print(f"===========================================================")
            print(f"Status              : {entry.status}")
            print(f"Official Scores     : {entry.official_scores}")
            print(f"Score Deltas (LB-CV): {entry.score_deltas}")
            if entry.divergence_warning:
                print(f"🚨 WARNING: CV vs LB Divergence detected! LB score dropped > 0.05 below Local CV.")
            print(f"Updated in          : submissions/SUBMISSIONS.md")
            print(f"===========================================================")
        except Exception as e:
            print(f"❌ FAILED TO RECORD SCORE: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.action == "mark-submitted":
        try:
            entry = mgr.mark_submitted(sub_id=args.sub_id, portal_reference=args.ref)
            print(f"✔ Submission '{entry.id}' marked as SUBMITTED_PENDING.")
            print(f"Updated in submissions/SUBMISSIONS.md")
        except Exception as e:
            print(f"❌ ERROR: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.action == "list" or not args.action:
        # Default action: display markdown table
        md_table = mgr.render_submissions_markdown()
        print(md_table)


if __name__ == "__main__":
    main()
