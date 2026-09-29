#!/usr/bin/env python3
"""Bidirectional CLI Bridge between Antigravity (IDE Agent Runtime) and the ML Harness.

Provides structured Context Packets and Typed Harness Action execution with stateful
BridgeSession tracking, optional budget enforcement, and USER_APPROVAL governance:

  # 1. Read compressed Context Packet (Competition State + UCB + Diagnostics + Session):
  ml-harness-agent context
  ml-harness-agent context --format json

  # 2. Start or configure a BridgeSession with optional budget limits:
  ml-harness-agent start-session --goal "Improve the primary metric" --max-experiments 3

  # 3. Invoke a Typed Harness Action (returns structured ActionObservation JSON):
  ml-harness-agent act --action inspect_diagnostics --params '{"exp_id": "exp001_baseline"}'
  ml-harness-agent act --action create_experiment --params '{"branch": "feat", "descriptor": "feature_v1"}'
  ml-harness-agent act --action run_smoke_test --params '{"config_path": "configs/exp002_feat_feature_v1.yaml"}'
  ml-harness-agent act --action run_full_validation --params '{"config_path": "configs/exp002_feat_feature_v1.yaml", "backend": "local"}'
  ml-harness-agent act --action synthesize_candidate --params '{"name": "ensemble_v1", "mode": "ensemble_vote"}'

  # 4. Approve a paused USER_APPROVAL action (submit, freeze, unfreeze, finalize):
  ml-harness-agent approve --session-id session_0001
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

HARNESS_DIR = Path(__file__).resolve().parent.parent
if str(HARNESS_DIR) not in sys.path:
    sys.path.insert(0, str(HARNESS_DIR))

from harness.agent_bridge import AgentBridge
from harness.project import HarnessProject


def _parse_json_arg(raw: str | None) -> Dict[str, Any]:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except Exception as e:
        print(f"Error parsing --params JSON: {e}", file=sys.stderr)
        sys.exit(1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Antigravity <-> ML Harness Bidirectional Context & Action Bridge CLI"
    )
    parser.add_argument("--project-root", default=None, help="Competition workspace root")
    parser.add_argument("--task-config", default=None, help="Task spec path relative to workspace root")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 1. context
    ctx_p = subparsers.add_parser("context", help="Generate compressed Context Packet for Antigravity in IDE")
    ctx_p.add_argument("--session-id", default=None, help="Optional session ID to inspect")
    ctx_p.add_argument(
        "--format",
        choices=["markdown", "json"],
        default="markdown",
        help="Output format (default: markdown)",
    )

    # 2. start-session
    start_p = subparsers.add_parser("start-session", help="Start a stateful BridgeSession with optional budgets")
    start_p.add_argument("--goal", required=True, help="Session goal description")
    start_p.add_argument("--session-id", default=None, help="Optional custom session ID")
    start_p.add_argument("--max-experiments", type=int, default=None, help="Optional ceiling on full CV / synthesis runs")
    start_p.add_argument("--max-gpu-hours", type=float, default=None, help="Optional ceiling on cloud GPU hours")
    start_p.add_argument("--max-cloud-runs", type=int, default=None, help="Optional ceiling on Modal/Kaggle dispatches")
    start_p.add_argument(
        "--max-consecutive-rejections",
        type=int,
        default=3,
        help="Max non-improving full runs before circuit breaker pauses (default: 3)",
    )

    # 3. act
    act_p = subparsers.add_parser("act", help="Execute a single Typed Harness Action and return ActionObservation")
    act_p.add_argument("--action", required=True, help="Typed action name (e.g. create_experiment, run_smoke_test)")
    act_p.add_argument("--session-id", default=None, help="Target session ID (uses active session if omitted)")
    act_p.add_argument("--goal", default=None, help="Optional goal if creating a new session")
    act_p.add_argument("--rationale", default="", help="Why Antigravity selected this action")
    act_p.add_argument("--params", default="{}", help="JSON object of action parameters")
    act_p.add_argument(
        "--backend",
        choices=["local", "modal", "kaggle"],
        default=None,
        help="Execution backend override for run_full_validation",
    )
    act_p.add_argument(
        "--user-approved",
        action="store_true",
        help="Confirm explicit human approval for USER_APPROVAL tier actions (submit, freeze, unfreeze, finalize)",
    )

    # 4. approve
    appr_p = subparsers.add_parser("approve", help="Approve and resume a pending USER_APPROVAL action")
    appr_p.add_argument("--session-id", required=True, help="Session ID to approve (e.g. session_0001)")

    # 5. status
    stat_p = subparsers.add_parser("status", help="Inspect detailed state of a BridgeSession")
    stat_p.add_argument("--session-id", default=None, help="Session ID (defaults to latest session)")

    # 6. list
    subparsers.add_parser("list", help="List recent BridgeSessions")

    args = parser.parse_args(argv)
    project = HarnessProject.discover(args.project_root, args.task_config)
    os.environ["HARNESS_PROJECT_ROOT"] = str(project.root)
    os.environ["HARNESS_TASK_CONFIG"] = str(project.task_config_path)
    if str(project.root) not in sys.path:
        sys.path.insert(0, str(project.root))
    bridge = AgentBridge(project.root, task_spec_path=project.task_config_path)

    if args.command == "context":
        if args.format == "json":
            pkt = bridge.get_context_packet(session_id=args.session_id)
            print(json.dumps(pkt, indent=2, ensure_ascii=False))
        else:
            print(bridge.render_context_markdown(session_id=args.session_id))

    elif args.command == "start-session":
        budget = {
            "max_experiments": args.max_experiments,
            "max_gpu_hours": args.max_gpu_hours,
            "max_cloud_runs": args.max_cloud_runs,
            "max_consecutive_rejections": args.max_consecutive_rejections,
        }
        sess = bridge.start_session(goal=args.goal, budget=budget, session_id=args.session_id)
        print(json.dumps(asdict(sess), indent=2, ensure_ascii=False))

    elif args.command == "act":
        params = _parse_json_arg(args.params)
        if args.backend:
            params["backend"] = args.backend

        sess, obs = bridge.execute_action(
            action=args.action,
            params=params,
            rationale=args.rationale,
            session_id=args.session_id,
            goal=args.goal,
            user_approved=args.user_approved,
        )
        output = {
            "session_id": sess.session_id,
            "session_status": sess.status,
            "compute_usage": sess.compute_usage,
            "consecutive_rejections": sess.consecutive_rejections,
            "observation": asdict(obs),
        }
        print(json.dumps(output, indent=2, ensure_ascii=False))

    elif args.command == "approve":
        sess, obs = bridge.approve_pending_action(args.session_id)
        output = {
            "session_id": sess.session_id,
            "session_status": sess.status,
            "observation": asdict(obs) if obs else None,
        }
        print(json.dumps(output, indent=2, ensure_ascii=False))

    elif args.command == "status":
        if args.session_id:
            sess = bridge.load_session(args.session_id)
            print(json.dumps(asdict(sess), indent=2, ensure_ascii=False))
        else:
            sessions = bridge.list_sessions(limit=1)
            if not sessions:
                print("No BridgeSessions recorded yet.")
            else:
                print(json.dumps(sessions[0], indent=2, ensure_ascii=False))

    elif args.command == "list":
        sessions = bridge.list_sessions()
        if not sessions:
            print("No BridgeSessions recorded yet.")
            return 0
        print(f"{'Session ID':<14} | {'Status':<22} | {'Exp Used':<10} | {'Goal'}")
        print("-" * 88)
        for s in sessions:
            cu = s.get("compute_usage", {})
            bg = s.get("budget", {})
            max_e = bg.get("max_experiments")
            exp_str = f"{cu.get('experiments_used', 0)}/{max_e if max_e is not None else '∞'}"
            print(f"{s['session_id']:<14} | {s['status']:<22} | {exp_str:<10} | {s.get('goal', '')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
