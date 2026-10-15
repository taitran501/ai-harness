#!/usr/bin/env python3
"""Bidirectional CLI Bridge between Antigravity (IDE Agent Runtime) and the AI Harness.

Provides structured Context Packets and Typed Harness Action execution with stateful
BridgeSession tracking, optional budget enforcement, and USER_APPROVAL governance:

  # 1. Read compressed Context Packet (Competition State + UCB + Diagnostics + Session):
  ai-harness-agent context
  ai-harness-agent context --format json

  # 2. Start or configure a BridgeSession with optional budget limits:
  ai-harness-agent start-session --goal "Improve the primary metric" --max-experiments 3

  # 3. Invoke a Typed Harness Action (returns structured ActionObservation JSON):
  ai-harness-agent act --action inspect_diagnostics --params '{"exp_id": "exp001_baseline"}'
  ai-harness-agent act --action create_experiment --params '{"branch": "feat", "descriptor": "feature_v1"}'
  ai-harness-agent act --action run_smoke_test --params '{"config_path": "configs/exp002_feat_feature_v1.yaml"}'
  ai-harness-agent act --action run_full_validation --params '{"config_path": "configs/exp002_feat_feature_v1.yaml", "backend": "local"}'
  ai-harness-agent act --action synthesize_candidate --params '{"name": "ensemble_v1", "mode": "ensemble_vote"}'

  # 4. Approve a paused USER_APPROVAL action (submit, freeze, unfreeze, finalize):
  ai-harness-agent approve --session-id session_0001
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


def _coerce_value(val: str) -> Any:
    """Coerce string into int, float, bool, or json dict/list if applicable."""
    val_lower = val.lower()
    if val_lower == "true":
        return True
    if val_lower == "false":
        return False
    if val_lower in {"none", "null"}:
        return None
    try:
        if val.isdigit() or (val.startswith("-") and val[1:].isdigit()):
            return int(val)
        return float(val)
    except ValueError:
        pass
    if (val.startswith("{") and val.endswith("}")) or (val.startswith("[") and val.endswith("]")):
        try:
            return json.loads(val)
        except Exception:
            pass
    return val


def _parse_dynamic_kv_args(args_list: list[str]) -> Dict[str, Any]:
    """Parse dynamic CLI flags (--exp-id value, --weight 0.5, --flag) into a dict.

    Converts kebab-case keys (--config-path) to snake_case (config_path) and parses
    ints, floats, booleans, and JSON-like strings automatically.
    """
    params: Dict[str, Any] = {}
    i = 0
    while i < len(args_list):
        arg = args_list[i]
        if arg.startswith("--"):
            # Handle --key=val format
            if "=" in arg:
                key_part, val_part = arg[2:].split("=", 1)
                key = key_part.replace("-", "_")
                params[key] = _coerce_value(val_part)
                i += 1
                continue

            key = arg[2:].replace("-", "_")
            # Peek next token
            if i + 1 < len(args_list) and not args_list[i + 1].startswith("--"):
                val_part = args_list[i + 1]
                params[key] = _coerce_value(val_part)
                i += 2
            else:
                params[key] = True
                i += 1
        else:
            i += 1
    return params


def _parse_json_arg(raw: str | None) -> Dict[str, Any]:
    if not raw:
        return {}
    # 1. Standard JSON parse
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    # 2. YAML safe_load (handles single quotes, unquoted keys often generated on PowerShell)
    try:
        import yaml
        parsed = yaml.safe_load(raw)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    # 3. Python AST literal eval
    try:
        import ast
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    print(f"Error parsing --params JSON: {raw}", file=sys.stderr)
    sys.exit(1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Antigravity <-> AI Harness Bidirectional Context & Action Bridge CLI"
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
        help="Comparable non-improving full runs before reassessment advisory (default: 3; no automatic freeze)",
    )

    # 3. act
    act_p = subparsers.add_parser("act", help="Execute a single Typed Harness Action and return ActionObservation")
    act_p.add_argument("--action", required=True, help="Typed action name (e.g. create_experiment, run_smoke_test)")
    act_p.add_argument("--session-id", default=None, help="Target session ID (uses active session if omitted)")
    act_p.add_argument("--goal", default=None, help="Optional goal if creating a new session")
    act_p.add_argument("--rationale", default="", help="Why Antigravity selected this action")
    parameters = act_p.add_mutually_exclusive_group()
    parameters.add_argument("--params", default=None, help="JSON object of action parameters")
    parameters.add_argument("--param-file", "--params-file", dest="param_file", default=None,
                            help="UTF-8 JSON object file, absolute or relative to project root (PowerShell friendly)")
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

    args, extra_argv = parser.parse_known_args(argv)
    if args.command != "act" and extra_argv:
        parser.error(f"Unrecognized arguments: {' '.join(extra_argv)}")

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
        raw = args.params
        if args.param_file:
            path = Path(args.param_file)
            if not path.is_absolute():
                path = project.root / path
            try:
                raw = path.read_text(encoding="utf-8-sig")
            except OSError as error:
                parser.error(f"Cannot read --param-file: {error}")
        params = _parse_json_arg(raw)
        dynamic_params = _parse_dynamic_kv_args(extra_argv)
        params.update(dynamic_params)
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
