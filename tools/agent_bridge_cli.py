#!/usr/bin/env python3
"""Bidirectional CLI Bridge between Antigravity (IDE Agent Runtime) and the AI Harness.

Provides structured Context Packets and Typed Harness Action execution with stateful
BridgeSession tracking, optional budget enforcement, and USER_APPROVAL governance:

  # 1. Read compressed Context Packet (Competition State + UCB + Diagnostics + Session):
  ai-harness-agent context
  ai-harness-agent context --format json

  # 2. Start or configure a BridgeSession with optional budget limits:
  ai-harness-agent start-session --goal "Improve the primary metric" --max-experiments 3

  # 3. Invoke a Typed Harness Action (returns structured ActionObservation JSON).
  # Prefer flat flags; for complex params on PowerShell use --param-file with a
  # UTF-8 JSON file (inline --params JSON breaks under PowerShell quoting):
  ai-harness-agent act --action inspect_diagnostics --exp-id exp001_baseline
  ai-harness-agent act --action create_experiment --branch feat --descriptor feature_v1
  ai-harness-agent act --action run_smoke_test --config-path configs/exp002_feat_feature_v1.yaml
  ai-harness-agent act --action run_full_validation --config-path configs/exp002_feat_feature_v1.yaml --backend local
  ai-harness-agent act --action synthesize_candidate --param-file synthesize.json
  ai-harness-agent act --action get_macro_hypothesis_map
  ai-harness-agent act --action check_retrospective_status
  ai-harness-agent act --action record_macro_review --param-file retrospective.json
  ai-harness-agent act --action list_model_assets --limit 20
  ai-harness-agent act --action inspect_model_asset --model-ref <model-id-or-digest>
  ai-harness-agent act --action register_model_asset --file configs/models/backbone.yaml
  ai-harness-agent act --action validate_model_plan --config-path configs/exp012.yaml

  # 4. Approve a paused USER_APPROVAL action (submit, freeze, unfreeze, finalize):
  ai-harness-agent approve --session-id session_0001
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

HARNESS_DIR = Path(__file__).resolve().parent.parent
if str(HARNESS_DIR) not in sys.path:
    sys.path.insert(0, str(HARNESS_DIR))

from harness.agent_bridge import (
    AgentBridge,
    TYPED_ACTION_SCHEMAS,
    _TYPED_ACTION_NAMES,
    get_action_schema,
    resolve_action_name,
)
from harness.project import HarnessProject


def _format_action_schema(name: str, schema: Dict[str, Any]) -> str:
    """Format action parameter schema for human or agent inspection."""
    lines = [
        f"Action: {name}",
        f"Summary: {schema.get('summary', '')}",
        f"Description: {schema.get('description', '')}",
    ]
    req = schema.get("required", [])
    if req:
        lines.append("Required Parameters:")
        for p in req:
            lines.append(f"  --{p.replace('_', '-')} <value>")
    else:
        lines.append("Required Parameters: (none)")
    opt = schema.get("optional", {})
    if opt:
        lines.append("Optional Parameters:")
        for p, default_val in opt.items():
            def_str = f" [default: {default_val!r}]" if default_val is not None else ""
            lines.append(f"  --{p.replace('_', '-')} <value> (optional){def_str}")
    flags = " ".join(f'--{p.replace("_", "-")} "VALUE"' for p in req)
    lines.append("Example Invocation:")
    lines.append(
        f"  ai-harness-agent act --action {name}"
        + (f" {flags}" if flags else "")
    )
    lines.append(
        "PowerShell: quote values containing spaces. Use --param-file for nested objects or lists."
    )
    return "\n".join(lines)


def _json_serial(obj: Any) -> Any:
    """Safe fallback serializer for datetime, Path, dataclass, numpy, and custom objects."""
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return str(obj)


def _safe_json_loads(val: str) -> Any:
    """Safe JSON deserialization that rejects non-finite constants (NaN, Infinity)."""
    def _reject_constant(c: str) -> Any:
        raise ValueError(f"Non-finite numeric constant '{c}' not allowed in JSON input.")
    return json.loads(val, parse_constant=_reject_constant)


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
        f = float(val)
        if not math.isnan(f) and not math.isinf(f):
            return f
    except ValueError:
        pass
    if (val.startswith("{") and val.endswith("}")) or (val.startswith("[") and val.endswith("]")):
        try:
            return _safe_json_loads(val)
        except Exception:
            pass
    return val


def _coerce_typed_value(key: str, val: Any, declared_type: str | None) -> Any:
    """Coerce dynamic CLI flag values according to action schema parameter_types."""
    flag_name = key.replace("_", "-")
    if declared_type in ("string", "path"):
        if val is True:
            raise ValueError(f"Flag '--{flag_name}' requires a {declared_type} value.")
        return str(val)

    if declared_type == "finite_float":
        if val is True or isinstance(val, bool):
            raise ValueError(f"Flag '--{flag_name}' requires a finite number, got boolean.")
        try:
            f = float(val)
            if math.isnan(f) or math.isinf(f):
                raise ValueError()
            return f
        except (ValueError, TypeError):
            raise ValueError(f"Flag '--{flag_name}' requires a finite number, got {val!r}.")

    if declared_type == "int":
        if val is True or isinstance(val, bool):
            raise ValueError(f"Flag '--{flag_name}' requires an integer, got boolean.")
        try:
            return int(val)
        except (ValueError, TypeError):
            raise ValueError(f"Flag '--{flag_name}' requires an integer, got {val!r}.")

    if declared_type == "bool":
        if isinstance(val, bool):
            return val
        s = str(val).lower().strip()
        if s in ("true", "1", "yes"):
            return True
        if s in ("false", "0", "no"):
            return False
        raise ValueError(f"Flag '--{flag_name}' requires a boolean value, got {val!r}.")

    # Fallback to general coercion
    if isinstance(val, str):
        return _coerce_value(val)
    return val


def _parse_dynamic_kv_args(
    args_list: list[str],
    parameter_types: Dict[str, str] | None = None,
) -> Dict[str, Any]:
    """Parse dynamic CLI flags (--exp-id value, --weight 0.5, --flag) into a dict.

    Converts kebab-case keys (--config-path) to snake_case (config_path) and applies
    schema-declared parameter_types when available.
    """
    params: Dict[str, Any] = {}
    types = parameter_types or {}
    i = 0
    while i < len(args_list):
        arg = args_list[i]
        if arg.startswith("--"):
            # Handle --key=val format
            if "=" in arg:
                key_part, val_part = arg[2:].split("=", 1)
                key = key_part.replace("-", "_")
                decl_type = types.get(key)
                params[key] = _coerce_typed_value(key, val_part, decl_type)
                i += 1
                continue

            key = arg[2:].replace("-", "_")
            decl_type = types.get(key)
            # Peek next token
            if i + 1 < len(args_list) and not args_list[i + 1].startswith("--"):
                val_part = args_list[i + 1]
                params[key] = _coerce_typed_value(key, val_part, decl_type)
                i += 2
            else:
                if decl_type and decl_type not in ("bool",):
                    raise ValueError(f"Flag '--{key.replace('_', '-')}' requires a {decl_type} value.")
                params[key] = _coerce_typed_value(key, True, decl_type)
                i += 1
        else:
            raise ValueError(
                f"Unrecognized positional argument '{arg}'. All action parameters must be specified as --flag <value>."
            )
    return params


def _parse_json_arg(raw: str | None) -> Dict[str, Any]:
    if not raw:
        return {}
    # 1. Standard JSON parse
    try:
        parsed = _safe_json_loads(raw)
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
        description="Antigravity <-> AI Harness Bidirectional Context & Action Bridge CLI",
        allow_abbrev=False,
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
    act_p = subparsers.add_parser(
        "act",
        help="Execute a single Typed Harness Action and return ActionObservation",
        allow_abbrev=False,
    )
    act_p.add_argument("--action", required=True, help="Typed action name (e.g. create_experiment, list_model_assets, run_smoke_test)")
    act_p.add_argument("--session-id", default=None, help="Target session ID (uses active session if omitted)")
    act_p.add_argument("--goal", default=None, help="Optional goal if creating a new session")
    act_p.add_argument("--rationale", default="", help="Why Antigravity selected this action")
    act_p.add_argument("--rule-id", default=None)
    act_p.add_argument("--candidate-evidence-id", default=None)
    act_p.add_argument("--evidence-kind", choices=("full_run", "official_submission"), default=None)
    act_p.add_argument("--evidence-partition", choices=("oof", "holdout", "confirmation", "public", "private"), default=None)
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

    # 7. schema
    schema_p = subparsers.add_parser("schema", help="Inspect parameter schema for typed harness actions")
    schema_p.add_argument("--action", default=None, help="Specific action to inspect (lists all if omitted)")
    schema_p.add_argument("--format", choices=["text", "json"], default="text", help="Output format (default: text)")

    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if "act" in raw_argv and ("--help" in raw_argv or "-h" in raw_argv) and "--action" in raw_argv:
        try:
            act_idx = raw_argv.index("--action")
            if act_idx + 1 < len(raw_argv):
                act_name = raw_argv[act_idx + 1]
                schema = get_action_schema(act_name)
                if schema:
                    print(_format_action_schema(act_name, schema))
                    return 0
        except (IndexError, ValueError):
            pass

    args, extra_argv = parser.parse_known_args(argv)
    if args.command != "act" and extra_argv:
        parser.error(f"Unrecognized arguments: {' '.join(extra_argv)}")

    if args.command == "schema":
        if args.action:
            schema = get_action_schema(args.action)
            if not schema:
                print(
                    f"Unknown action: '{args.action}'. Valid actions:\n  "
                    + "\n  ".join(sorted(_TYPED_ACTION_NAMES)),
                    file=sys.stderr,
                )
                return 1
            if args.format == "json":
                print(json.dumps({args.action: schema}, indent=2))
            else:
                print(_format_action_schema(args.action, schema))
        else:
            if args.format == "json":
                print(json.dumps(TYPED_ACTION_SCHEMAS, indent=2))
            else:
                print(f"{'Action Name':<32} | {'Summary'}")
                print("-" * 88)
                for name, s in sorted(TYPED_ACTION_SCHEMAS.items()):
                    print(f"{name:<32} | {s.get('summary', '')}")
                print("\nRun 'ai-harness-agent schema --action <name>' or 'ai-harness-agent act --action <name> --help' for details.")
        return 0

    project = HarnessProject.discover(args.project_root, args.task_config)
    from harness.toolchain import ensure_pinned_toolchain

    _pinned = ensure_pinned_toolchain(project.root, "tools.agent_bridge_cli", sys.argv[1:])
    if _pinned is not None:
        return int(_pinned.returncode)
    os.environ["HARNESS_PROJECT_ROOT"] = str(project.root)
    os.environ["HARNESS_TASK_CONFIG"] = str(project.task_config_path)
    if str(project.root) not in sys.path:
        sys.path.insert(0, str(project.root))
    bridge = AgentBridge(project.root, task_spec_path=project.task_config_path)

    if args.command == "context":
        if args.format == "json":
            pkt = bridge.get_context_packet(session_id=args.session_id)
            print(json.dumps(pkt, indent=2, ensure_ascii=False, default=_json_serial))
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
        payload = asdict(sess)
        # T3: the adoption gate belongs at the session ENTRY point, not only
        # in the separate `context` call - an agent that goes straight from
        # start-session to act would otherwise never see it.
        from harness.agent_bridge import adoption_gate_lines
        from harness.hygiene import scan_workspace_hygiene

        packet = bridge.get_context_packet(session_id=sess.session_id) or {}
        if packet.get("competition_research_directive"):
            payload["competition_research_directive"] = packet["competition_research_directive"]
        memory = packet.get("research_memory") or {}
        if memory.get("macro_hypothesis_map"):
            payload["macro_hypothesis_map"] = memory["macro_hypothesis_map"]
            payload["retrospective_status"] = memory["retrospective_status"]
        gate = adoption_gate_lines(
            scan_workspace_hygiene(project.root),
            int(
                (packet.get("competition_hud") or {}).get("total_experiments") or 0
            ),
        )
        if gate:
            payload["adoption_gate"] = [""] + gate
        print(json.dumps(payload, indent=2, ensure_ascii=False, default=_json_serial))

    elif args.command == "act":
        act_name = resolve_action_name(args.action)
        if act_name not in _TYPED_ACTION_NAMES:
            print(
                f"Unknown action: '{args.action}'. Run 'ai-harness-agent schema' to list all valid actions.",
                file=sys.stderr,
            )
            return 1
        raw = args.params
        if args.param_file:
            path = Path(args.param_file)
            if not path.is_absolute():
                path = project.root / path
            try:
                raw = path.read_text(encoding="utf-8-sig")
            except OSError as error:
                parser.error(f"Cannot read --param-file: {error}")
        schema = get_action_schema(act_name) or {}
        param_types = schema.get("parameter_types") or {}
        params = _parse_json_arg(raw)
        try:
            dynamic_params = _parse_dynamic_kv_args(extra_argv, parameter_types=param_types)
        except ValueError as err:
            parser.error(str(err))
        params.update(dynamic_params)
        for field in ("rule_id", "candidate_evidence_id", "evidence_kind", "evidence_partition"):
            if getattr(args, field, None) is not None:
                params[field] = getattr(args, field)
        if args.backend:
            params["backend"] = args.backend

        sess, obs = bridge.execute_action(
            action=act_name,
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
        print(json.dumps(output, indent=2, ensure_ascii=False, default=_json_serial))

    elif args.command == "approve":
        sess, obs = bridge.approve_pending_action(args.session_id)
        output = {
            "session_id": sess.session_id,
            "session_status": sess.status,
            "observation": asdict(obs) if obs else None,
        }
        print(json.dumps(output, indent=2, ensure_ascii=False, default=_json_serial))

    elif args.command == "status":
        if args.session_id:
            sess = bridge.load_session(args.session_id)
            print(json.dumps(asdict(sess), indent=2, ensure_ascii=False, default=_json_serial))
        else:
            sessions = bridge.list_sessions(limit=1)
            if not sessions:
                print("No BridgeSessions recorded yet.")
            else:
                print(json.dumps(sessions[0], indent=2, ensure_ascii=False, default=_json_serial))

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
