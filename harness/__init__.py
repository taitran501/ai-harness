"""ML Experiment Harness - Autonomous Agentic ML/DL Experimentation Platform.

Combines AIDE ML (Tree Search & Memory Compression) with SWE-agent (Deterministic Audit Gate).
"""

from .tree_journal import Node, TreeJournal
from .validator import ExperimentValidator
from .competition_state import CompetitionStateMachine, CompetitionState
from .agent_bridge import AgentBridge
from .ensemble_guard import EnsembleGuard
from .submission_manager import SubmissionManager
from .io_utils import (
    atomic_write_json,
    atomic_write_text,
    json_default_serializer,
    sanitize_for_json,
)

__all__ = [
    "Node",
    "TreeJournal",
    "ExperimentValidator",
    "CompetitionStateMachine",
    "CompetitionState",
    "AgentBridge",
    "EnsembleGuard",
    "SubmissionManager",
    "atomic_write_json",
    "atomic_write_text",
    "json_default_serializer",
    "sanitize_for_json",
]
