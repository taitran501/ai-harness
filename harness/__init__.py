"""ML Experiment Harness for StereoQueerEval (SemEval-2027 Task 8).

Combines AIDE ML (Tree Search & Memory Compression) with SWE-agent (Deterministic Audit Gate).
"""

from .tree_journal import Node, TreeJournal
from .validator import ExperimentValidator

__all__ = ["Node", "TreeJournal", "ExperimentValidator"]
