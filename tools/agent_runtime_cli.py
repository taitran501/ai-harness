#!/usr/bin/env python3
"""Headless CLI Controller for the Autonomous Agent Runtime (Level 5 Harness).

Allows running, inspecting, and approving multi-step AgentRuns (`run_0001`, ...)
without requiring the Web UI to be open:

  ml-harness-agent start-session --goal "Improve the primary metric" --max-experiments 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from harness.agent_runtime import AgentRuntime


def main() -> None:
    parser = argparse.ArgumentParser(description="Autonomous Agent Runtime CLI (Goal -> Plan -> Act -> Observe -> Stop)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 1. run
    run_p = subparsers.add_parser("run", help="Launch an autonomous AgentRun with a specific goal and budget")
    run_p.add_argument("--goal", required=True, help="Scientific / competition goal for this AgentRun")
    run_p.add_argument("--max-experiments", type=int, default=3, help="Maximum experiments budget (default: 3)")
    run_p.add_argument("--max-wall-seconds", type=float, default=1800.0, help="Maximum wall-clock time in seconds")
    run_p.add_argument("--auto-approve", action="store_true", help="Deprecated compatibility flag; starting a session never freezes candidates")

    # 2. list
    subparsers.add_parser("list", help="List recent AgentRuns and their statuses")

    # 3. inspect
    insp_p = subparsers.add_parser("inspect", help="Inspect detailed actions, observations, and decisions of an AgentRun")
    insp_p.add_argument("--run-id", required=True, help="AgentRun ID (e.g. run_0001)")

    # 4. approve
    appr_p = subparsers.add_parser("approve", help="Approve a pending USER_APPROVAL action and resume the AgentRun")
    appr_p.add_argument("--run-id", required=True, help="AgentRun ID (e.g. run_0001)")

    args = parser.parse_args()
    rt = AgentRuntime(PROJECT_ROOT)

    if args.command == "run":
        run = rt.start_run(
            goal=args.goal,
            max_experiments=args.max_experiments,
            max_wall_seconds=args.max_wall_seconds,
            auto_approve_all=args.auto_approve,
        )
        print(f"============================================================")
        print(f"🤖 AGENT RUN: {run.id} | STATUS: {run.status}")
        print(f"🎯 GOAL     : {run.goal}")
        print(f"🛑 STOP/GATE: {run.stop_reason}")
        print(f"============================================================")
        for i, obs in enumerate(run.observations, 1):
            print(f"  {i}. [{obs['status']}] {obs['action']} -> {obs['summary']}")
        if run.pending_approval_action:
            act = run.pending_approval_action
            print(f"\n⚠️  PENDING USER APPROVAL: [{act['action']}] ({act['rationale']})")
            print(f"   Run `python tools/agent_runtime_cli.py approve --run-id {run.id}` to approve & resume.")

    elif args.command == "list":
        runs = rt.list_runs()
        if not runs:
            print("No AgentRuns recorded yet.")
            return
        print(f"{'Run ID':<12} | {'Status':<22} | {'Exp Budget':<10} | {'Goal'}")
        print("-" * 85)
        for r in runs:
            b = r.get("budget", {})
            b_str = f"{b.get('experiments_used', 0)}/{b.get('max_experiments', 3)}"
            print(f"{r['id']:<12} | {r['status']:<22} | {b_str:<10} | {r['goal']}")

    elif args.command == "inspect":
        run = rt.load_run(args.run_id)
        print(json.dumps(run.__dict__, indent=2, ensure_ascii=False))

    elif args.command == "approve":
        run = rt.approve_pending_action(args.run_id)
        print(f"✔ Approved and resumed {run.id}. Final Status: {run.status} ({run.stop_reason})")


if __name__ == "__main__":
    main()
