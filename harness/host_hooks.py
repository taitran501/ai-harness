"""Host-agnostic guardrail hooks for attached competition workspaces.

Adoption audit (rsna-knee): an agent bypassed the harness 100% for cloud
runs — 22/22 manual ``kaggle kernels push`` dirs, zero harness experiments,
zero warnings. Layer-1 (AGENTS.md) guidance is advisory only; every
external guardrail project we reviewed (block-no-verify, airlock, noslop,
agentic-experiments) converges on the same lesson: *markdown rules are
suggestions, not enforcement*. The reliable layers are:

1. **Session-start gates** in a tool the agent must use (T3, in
   ``agent_bridge``) — host-agnostic.
2. **Per-host tool hooks** (this module, T4): Claude Code
   ``.claude/settings.json`` PreToolUse and Cursor
   ``.cursor/hooks.json`` beforeShellExecution. Both receive JSON on
   stdin, and exit code 2 blocks the command with the stderr/stdout
   reason fed back to the agent so it adjusts its approach.

Design invariants:
- **Fail-open.** The hook script allows (exit 0) on ANY parse/IO error.
  A guardrail that breaks a host's tooling is worse than no guardrail.
- **Never destroy user config.** Existing ``settings.json`` /
  ``hooks.json`` are loaded and merged; only our own managed entry
  (tagged ``managed_by: ai-harness``) is replaced. Unknown formats are
  left untouched.
- **Human override.** ``AI_HARNESS_ALLOW_MANUAL=1`` disables the block —
  manual pushes can be legitimate (the operator's own notebooks). This
  mirrors block-no-verify's "AI must fix it properly, humans keep full
  autonomy".

Codex and other hosts without a documented hook surface rely on T3
(session-start gate) and T5 (adoption health in ``ai-harness status``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

MANAGED_TAG = "managed_by"
MANAGED_BY = "ai-harness"
HOOK_SCRIPT_REL = Path(".ai-harness") / "hooks" / "block_manual_kaggle.py"
HUMAN_OVERRIDE_ENV = "AI_HARNESS_ALLOW_MANUAL"

_HOOK_SCRIPT = '''#!/usr/bin/env python3
"""ai-harness managed hook (managed_by: ai-harness) — do not edit in place.

Blocks manual kaggle submissions in an attached workspace so experiment
runs keep their evidence trail. Regenerated on `ai-harness attach` /
`ai-harness sync`; hand edits will be overwritten.

Protocol (Claude Code PreToolUse and Cursor beforeShellExecution agree):
  - JSON payload on stdin (tool input).
  - exit 0  -> allow.
  - exit 2  -> block; stderr reason is fed back to the agent.

Fail-open: any unexpected condition allows the command.
"""
import json
import os
import re
import sys

HUMAN_OVERRIDE_ENV = "AI_HARNESS_ALLOW_MANUAL"

# Manual cloud push/submit — the exact bypass observed in the rsna-knee
# workspace (22/22 kaggle_sub_* dirs created outside the harness).
BLOCK_RE = re.compile(r"\\bkaggle\\s+(kernels\\s+push|competitions\\s+submit)\\b", re.I)


def _extract_command(payload):
    """Lenient command extraction across host payload shapes (fail-open)."""
    if not isinstance(payload, dict):
        return ""
    tool_input = payload.get("tool_input")
    if isinstance(tool_input, dict) and isinstance(tool_input.get("command"), str):
        return tool_input["command"]
    for key in ("shell_command", "command"):
        value = payload.get(key)
        if isinstance(value, str):
            return value
    return ""


def main():
    try:
        raw = sys.stdin.read()
        payload = json.loads(raw) if raw.strip() else {}
    except Exception:
        return 0
    if os.environ.get(HUMAN_OVERRIDE_ENV):
        return 0
    command = _extract_command(payload)
    if command and BLOCK_RE.search(command):
        sys.stderr.write(
            "ai-harness: manual kaggle push is blocked in this attached workspace "
            "(no evidence trail, invisible to the framework). "
            "Run experiments through the harness instead: "
            "`ai-harness-agent act --action create_experiment` -> `run_smoke_test` -> "
            "`run_full_validation` (contracts in .agents/skills/ai-harness/SKILL.md). "
            "If harness dispatch fails, REPORT the failure - do not silently fall back. "
            "Human override: " + HUMAN_OVERRIDE_ENV + "=1 <command>\\n"
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''


def _hook_command() -> str:
    """Command line hosts execute; relative to the workspace root."""
    return "python " + HOOK_SCRIPT_REL.as_posix()


def claude_settings_payload() -> Dict[str, Any]:
    """Full Claude Code ``.claude/settings.json`` shape for our hook."""
    return {
        "hooks": {
            "PreToolUse": [
                {
                    "matcher": "Bash",
                    "hooks": [
                        {
                            "type": "command",
                            "command": _hook_command(),
                            MANAGED_TAG: MANAGED_BY,
                        }
                    ],
                }
            ]
        }
    }


def cursor_hooks_payload() -> Dict[str, Any]:
    """Full Cursor ``.cursor/hooks.json`` shape for our hook."""
    return {
        "version": 1,
        "hooks": {
            "beforeShellExecution": [
                {
                    "command": _hook_command(),
                    MANAGED_TAG: MANAGED_BY,
                }
            ]
        },
    }


def _replace_managed_entry(existing: Any, new_entry: Dict[str, Any]) -> Dict[str, Any]:
    """Return ``existing`` with our managed entry removed, in place."""
    if not isinstance(existing, list):
        return {}
    kept = [
        entry
        for entry in existing
        if not (isinstance(entry, dict) and entry.get(MANAGED_TAG) == MANAGED_BY)
    ]
    # The entry may be nested one level deeper (Claude: group["hooks"]).
    for entry in kept:
        if isinstance(entry, dict) and isinstance(entry.get("hooks"), list):
            entry["hooks"] = [
                h
                for h in entry["hooks"]
                if not (isinstance(h, dict) and h.get(MANAGED_TAG) == MANAGED_BY)
            ]
    return {"hooks": kept}


def _merge_config(path: Path, fresh: Dict[str, Any]) -> Dict[str, str]:
    """Write/refresh our managed hook entry in a host config file.

    Returns ``{"path": ..., "status": created|merged|refreshed|skipped_*}``.
    Never raises: unreadable or unknown formats are skipped untouched.
    """
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(fresh, indent=2) + "\n", encoding="utf-8")
        return {"path": str(path), "status": "created"}
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"path": str(path), "status": "skipped_unreadable"}
    if not isinstance(existing, dict):
        return {"path": str(path), "status": "skipped_unknown_format"}
    existing.setdefault("hooks", {})
    hooks = existing["hooks"]
    event = next(iter(fresh.get("hooks", {})))
    if "version" in fresh:
        existing.setdefault("version", fresh["version"])
    # Remove our previous managed entry, keep everything else.
    kept = _replace_managed_entry(hooks.get(event), None)
    new_event = fresh["hooks"][event]
    hooks[event] = kept.get("hooks", []) + new_event
    path.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
    return {"path": str(path), "status": "refreshed"}


def write_host_hooks(root: Path) -> Dict[str, Any]:
    """Install the managed hook script + host hook configs in a workspace.

    Called from ``attach_project``. Idempotent; never touches user files
    it does not own.
    """
    root = Path(root)
    script_path = root / HOOK_SCRIPT_REL
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(_HOOK_SCRIPT, encoding="utf-8")
    claude = _merge_config(root / ".claude" / "settings.json", claude_settings_payload())
    cursor = _merge_config(root / ".cursor" / "hooks.json", cursor_hooks_payload())
    return {
        "script": str(script_path),
        "hosts": [claude, cursor],
        "override_env": HUMAN_OVERRIDE_ENV,
    }
