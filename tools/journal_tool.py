#!/usr/bin/env python3
"""CLI interface for TreeJournal (AIDE-style Solution Tree Explorer).

Allows Antigravity Brain or the Human engineer to quickly query past experiments,
inspect branches, view Pareto trade-offs, and get UCB recommendations without
overloading the LLM context window.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HARNESS_DIR = Path(__file__).resolve().parent.parent
if str(HARNESS_DIR) not in sys.path:
    sys.path.insert(0, str(HARNESS_DIR))

from harness.io_utils import get_project_root

PROJECT_ROOT = get_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from harness.tree_journal import TreeJournal


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TreeJournal Query & Branch Recommendation CLI"
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print compact high-density summary of the solution tree",
    )
    parser.add_argument(
        "--recommend",
        action="store_true",
        help="Recommend the next atomic intervention using AIDE tree search",
    )
    parser.add_argument(
        "--max-params",
        type=int,
        default=150_000_000,
        help="Maximum parameter ceiling (default: 150M)",
    )
    parser.add_argument(
        "--node",
        type=str,
        default=None,
        help="Show detailed view of a specific node ID",
    )
    parser.add_argument(
        "--best",
        action="store_true",
        help="Show current top-performing model under parameter constraints",
    )
    parser.add_argument(
        "--inject-skill",
        action="store_true",
        help="Dynamically generate/inject task-specific agent skill into .agents/skills/",
    )
    parser.add_argument(
        "--journal-path",
        type=str,
        default="artifacts/journal.json",
        help="Path to journal storage JSON file",
    )

    args = parser.parse_args()

    if args.inject_skill:
        from harness.skill_injector import inject_skill_file
        dest = inject_skill_file()
        print(f"✔ Successfully injected dynamic agent skill into: {dest}")
        return

    j_path = Path(args.journal_path)
    if not j_path.is_absolute():
        j_path = PROJECT_ROOT / j_path
    journal = TreeJournal(storage_path=str(j_path))

    if args.node:
        node = journal.get_node(args.node)
        if not node:
            print(f"Node '{args.node}' not found in journal.", file=sys.stderr)
            sys.exit(1)
        print(json.dumps(node.__dict__, indent=2, ensure_ascii=False))
        return

    if args.recommend:
        rec = journal.recommend_next_branch(max_params=args.max_params)
        print("===========================================================")
        print("🧠 AIDE TREE SEARCH: NEXT EXPERIMENT RECOMMENDATION")
        print("===========================================================")
        print(f"Parent Node   : {rec.get('parent_id') or 'ROOT'}")
        print(f"Branch Type   : {rec.get('recommended_branch')}")
        print(f"Parent Score  : {rec.get('parent_score')}")
        print(f"Parent Params : {rec.get('parent_params') or 'N/A'}")
        print(f"UCB Potential : {rec.get('ucb_score')}")
        if rec.get("bottlenecks"):
            print("-----------------------------------------------------------")
            print("🚨 Identified Bottlenecks:")
            for b in rec["bottlenecks"]:
                print(f"   - {b}")
        if rec.get("diagnostic_cue"):
            print(f"💡 Diagnostic Cue: {rec['diagnostic_cue']}")
        print("-----------------------------------------------------------")
        print(f"Guideline:\n{rec.get('guideline')}")
        print("===========================================================")
        return

    if args.best:
        best = journal.get_best_node(max_params=args.max_params)
        if not best:
            print("No valid nodes found.", file=sys.stderr)
            sys.exit(1)
        print(f"Best Node: {best.id} (Score: {best.get_score():.4f}, Params: {best.param_count or 'N/A'})")
        return

    # Default action: summary
    print(journal.generate_compact_summary())


if __name__ == "__main__":
    main()
