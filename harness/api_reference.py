"""Verified API surface for the agent (anti-hallucination reference).

Agents repeatedly fail with ``AttributeError`` / ``ModuleNotFoundError`` /
``KeyError`` by *guessing* the Kaggle SDK surface (``publicScore``,
``totalVotes`` …) or the harness's own module/method names (``harness.agents``,
``_ACTION_REGISTRY``, ``task_config`` …). This module holds the **verified**
surface as data so that:

1. ``skill_injector`` renders it into the generated ``SKILL.md`` — the agent
   looks up real names instead of inventing them.
2. ``tests/test_api_reference.py`` validates that every entry actually
   resolves, keeping the reference honest (a stale or invented name fails the
   suite instead of silently rotting).

The data is curated (not auto-discovered) because one-line descriptions have
value; ``validate_reference()`` — not the renderer — is the source of truth
for correctness.
"""

from __future__ import annotations

import argparse
import importlib
from typing import Any, List, Set, Tuple


# ---------------------------------------------------------------------------
# Curated surface data. Each entry is (name, description). Every target is
# checked by :func:`validate_reference` so the reference cannot drift.
# ---------------------------------------------------------------------------

#: Top-level ``ai-harness`` subcommands (see ``harness.cli.build_parser``).
CLI_COMMANDS: List[Tuple[str, str]] = [
    ("attach", "Attach the installed harness to a task workspace."),
    ("status", "Single-command install + workspace health report (default JSON)."),
    ("sync", "Update the shared framework, reinstall, and re-attach."),
    ("kaggle", "Inspect a Kaggle kernel's status/log/output without re-dispatching."),
    ("submission", "Kaggle submission dispatch + score polling (harness-native)."),
    ("dispatch-kernel", "Wrap a local notebook into a Kaggle kernel and push it."),
    ("repair", "Re-seal artifact manifest file hashes that drifted after a run."),
    ("import-artifacts", "Import verified local run outputs into the artifact tree."),
    ("adoption-scan", "Cross-workspace adoption overview + unimported verified runs."),
    ("validate-task", "Static task.yaml validation (schema, data files, column checks)."),
    ("plugins", "List installed task plugins."),
    ("phase", "Manage the 8-phase prompt workflow (init/validate/status/render/record)."),
    ("eda", "General-purpose exploratory data analysis (eda run)."),
    ("cloud", "Cloud readiness + task-scoped upload consent (preflight/consent/status/pull)."),
    ("export-kaggle-notebook", "Export a standalone offline Kaggle submission notebook."),
]

#: Console entry points (``pyproject.toml`` ``[project.scripts]``).
ENTRY_POINTS: List[Tuple[str, str]] = [
    ("ai-harness", "Framework CLI — the commands listed above."),
    ("ai-harness-agent", "Session lifecycle, Context HUD, and typed harness actions."),
    ("ai-harness-run", "Run an experiment config (smoke/full, local or remote)."),
    ("ai-harness-audit", "Deterministic SWE-agent audit gate."),
    ("ai-harness-bootstrap", "Bootstrap a new task workspace."),
]

#: Methods on the stable Kaggle facade (``harness.kaggle_compat.KaggleClient``).
#: Call these instead of the raw SDK — the facade normalises attribute names so
#: the legacy camelCase and the new snake_case surface read identically.
FACADE_METHODS: List[Tuple[str, str]] = [
    ("list_submissions", "List competition submissions (normalised: .public_score, .total_votes, …)."),
    ("submit_submission", "Submit a file or a code-comp kernel output; returns a normalised ref."),
    ("kernels_push", "Push a kernel folder (kernel-metadata.json + code) to Kaggle."),
    ("list_kernels", "List kernels (normalised)."),
    ("kernel_output", "Download a kernel's output."),
    ("kernels_status", "Query a kernel's status."),
    ("kernels_logs", "Fetch a kernel's logs."),
]

#: Normalised fields on the objects returned by ``list_submissions()`` (and on
#: the submission reference from ``submit_submission()``). The proxy
#: (``kaggle_compat._NormalisedObject``) maps the legacy camelCase names to these
#: canonical snake_case names, so callers use ONLY these spellings. This closes
#: the field-name guessing gap (agents used to invent ``.date`` / ``.submittedDate``).
SUBMISSION_FIELDS: List[Tuple[str, str]] = [
    ("ref", "Submission id — pass to status / wait_for_score tracking."),
    ("status", "Submission status string (e.g. IN_PROGRESS, COMPLETE, ERROR)."),
    ("description", "The message/description attached to the submission."),
    ("public_score", "Public (leaderboard) score; None until scored. (from publicScore)"),
    ("private_score", "Private score, if present. (from privateScore)"),
    ("total_votes", "Community votes, if present. (from totalVotes)"),
    ("created_at", "Submission timestamp, if present. (from createdAt)"),
]

#: Python entry points the agent may reach for. Dotted path = module[.attr[.attr…]].
PYTHON_SYMBOLS: List[Tuple[str, str]] = [
    ("harness.kaggle_compat.KaggleClient", "Unified Kaggle SDK facade (use this, not the raw SDK)."),
    ("harness.kaggle_compat.detect_sdk", "Which SDK is present: kagglesdk / kaggle / none."),
    ("harness.endgame_submission.KaggleLeaderboardTracker", "In-process submission tracker (dispatch / get_submission / wait_for_score / latest)."),
    ("harness.kernel_dispatch.build_kernel_metadata", "Build kernel-metadata.json content."),
    ("harness.kernel_dispatch.dispatch_kernel", "Stage + push a notebook as a Kaggle kernel."),
    ("harness.submission_dispatch.main", "The ``ai-harness submission`` CLI (dispatch/status/watch/latest)."),
    ("harness.grounding_gate.GroundingGate", "Anti-hallucination gate for experiment configs."),
]

#: Harness modules with their CORRECT names. The names agents keep inventing
#: (``harness.agents``, ``harness.task_config``) are deliberately absent.
KEY_MODULES: List[Tuple[str, str]] = [
    ("harness.agent_bridge", "Session lifecycle + Context HUD (module is ``agent_bridge``, NOT ``harness.agents``)."),
    ("harness.task_spec", "task.yaml schema + spec (module is ``task_spec``, NOT ``harness.task_config``)."),
    ("harness.runner", "Execution runtime behind ``ai-harness-run``."),
    ("harness.competition_state", "8-phase competition state machine."),
    ("harness.evaluation", "Metric computation + audit."),
    ("harness.research_memory", "Research continuity (insights, probes, decisions)."),
    ("harness.grounding_gate", "Anti-hallucination grounding gate."),
    ("harness.kaggle_compat", "Kaggle SDK compatibility facade."),
]


# ---------------------------------------------------------------------------
# Resolution + validation
# ---------------------------------------------------------------------------


def _resolve(dotted: str) -> Any:
    """Resolve a dotted path to its target.

    Imports the longest importable module prefix, then walks the remaining
    dotted attributes. ``harness.kaggle_compat.KaggleClient.list_submissions``
    imports ``harness.kaggle_compat`` then walks ``KaggleClient ->
    list_submissions``.
    """
    parts = dotted.split(".")
    for i in range(len(parts), 0, -1):
        try:
            obj: Any = importlib.import_module(".".join(parts[:i]))
        except ImportError:
            continue
        for attr in parts[i:]:
            obj = getattr(obj, attr)
        return obj
    raise ImportError("cannot import dotted path: %s" % dotted)


def _ai_harness_commands() -> Set[str]:
    """Return the set of registered top-level ``ai-harness`` subcommands."""
    from harness.cli import build_parser

    parser = build_parser()
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction) and action.dest == "command":
            return set(action.choices.keys())
    return set()


def _submission_fixture():
    """A stand-in raw submission (legacy camelCase + direct names).

    Used by :func:`validate_reference` to prove each canonical field in
    :data:`SUBMISSION_FIELDS` resolves through the normalising proxy — so a typo
    or a missing alias in the reference fails the suite instead of rotting.
    """

    class _Raw:
        ref = "57313652"
        status = "COMPLETE"
        description = "Exp021 Verified SOTA"
        publicScore = 0.958
        privateScore = 0.957
        totalVotes = 3
        createdAt = "2026-10-17T21:14:27Z"

    return _Raw()


def validate_reference() -> List[str]:
    """Return a list of problems (empty == the reference is fully valid).

    Checks that every listed CLI command is registered, every Python symbol
    resolves, every key module imports, and every facade method exists on
    ``KaggleClient``.
    """
    problems: List[str] = []

    commands = _ai_harness_commands()
    for name, _ in CLI_COMMANDS:
        if name not in commands:
            problems.append("cli: %r is not a registered ai-harness command" % name)

    for dotted, _ in PYTHON_SYMBOLS:
        try:
            _resolve(dotted)
        except Exception as exc:  # noqa: BLE001 - report, do not raise
            problems.append("python: %r does not resolve (%s)" % (dotted, exc))

    for dotted, _ in KEY_MODULES:
        try:
            importlib.import_module(dotted)
        except Exception as exc:  # noqa: BLE001
            problems.append("module: %r cannot be imported (%s)" % (dotted, exc))

    try:
        client_cls = _resolve("harness.kaggle_compat.KaggleClient")
        for method, _ in FACADE_METHODS:
            if not hasattr(client_cls, method):
                problems.append("facade: KaggleClient.%r is missing" % method)
    except Exception as exc:  # noqa: BLE001
        problems.append("facade: cannot load KaggleClient (%s)" % exc)

    try:
        from harness.kaggle_compat import _NormalisedObject

        sub = _NormalisedObject(_submission_fixture(), "kaggle")
        for name, _ in SUBMISSION_FIELDS:
            try:
                getattr(sub, name)
            except AttributeError:
                problems.append(
                    "submission-field: %r does not resolve on the normalised "
                    "Submission object" % name
                )
    except Exception as exc:  # noqa: BLE001 - report, do not raise
        problems.append("submission-field: cannot build fixture (%s)" % exc)

    return problems


# ---------------------------------------------------------------------------
# Rendering (used by skill_injector to build SKILL.md §7)
# ---------------------------------------------------------------------------


def _table(rows: List[Tuple[str, str]]) -> List[str]:
    out = ["| name | what |", "|---|---|"]
    for name, desc in rows:
        out.append("| `%s` | %s |" % (name, desc))
    return out


def render_reference() -> str:
    """Render the API Reference section (Markdown) for the generated SKILL.md."""
    example = (
        "python -c \"import harness.kaggle_compat as k; "
        "print([n for n in dir(k.KaggleClient) if not n.startswith('_')])\""
    )
    lines: List[str] = [
        "## 7. Verified API surface — look names up here, do NOT invent them",
        "",
        (
            "> Agents repeatedly fail with `AttributeError` / `ModuleNotFoundError` / `KeyError` by guessing the Kaggle SDK or harness surface. The names below are the **verified, current** surface (validated by `tests/test_api_reference.py`). If what you need is not listed, inspect it (e.g. `%s`) or ask — **never guess a module or method name**."
            % example
        ),
        "",
        "### CLI (`ai-harness`)",
        "",
    ]
    lines += _table(CLI_COMMANDS)
    lines += ["", "### Entry points", ""]
    for name, desc in ENTRY_POINTS:
        lines.append("- `%s` — %s" % (name, desc))
    lines += [
        "",
        "### Kaggle facade — call this, not the raw SDK (`harness.kaggle_compat`)",
        "",
        "`client = KaggleClient()` (check `detect_sdk()` first if unsure), then:",
        "",
    ]
    lines += _table(FACADE_METHODS)
    lines += [
        "",
        "### Submission object — normalised fields (`list_submissions()[i]`) — "
        "use these names, not camelCase",
    ]
    lines += _table(SUBMISSION_FIELDS)
    lines += ["", "### Dispatch (harness-native, in-process)", ""]
    lines += _table(PYTHON_SYMBOLS)
    lines += ["", "### Key modules (correct names)", ""]
    lines += _table(KEY_MODULES)
    lines.append("")
    return "\n".join(lines)
