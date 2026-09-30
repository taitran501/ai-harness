"""ML-Harness public API, loaded lazily to keep workflow-only installs small."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "Node": (".tree_journal", "Node"),
    "TreeJournal": (".tree_journal", "TreeJournal"),
    "ExperimentValidator": (".validator", "ExperimentValidator"),
    "CompetitionStateMachine": (".competition_state", "CompetitionStateMachine"),
    "CompetitionState": (".competition_state", "CompetitionState"),
    "AgentBridge": (".agent_bridge", "AgentBridge"),
    "EnsembleGuard": (".ensemble_guard", "EnsembleGuard"),
    "SubmissionManager": (".submission_manager", "SubmissionManager"),
    "atomic_write_json": (".io_utils", "atomic_write_json"),
    "atomic_write_text": (".io_utils", "atomic_write_text"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    """Load public runtime APIs only when callers request them."""
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = target
    value = getattr(import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
