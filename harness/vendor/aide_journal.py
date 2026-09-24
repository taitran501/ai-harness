"""Direct port of WecoAI/aideml/aide/journal.py.

Original Author: Weco AI (https://github.com/WecoAI/aideml)
License: MIT License
Paper: AIDE: Human-Level Performance on Data Science Competitions (arXiv:2502.13138)

The journal is the core data structure in AIDE that contains:
- generated code / intervention samples
- tree structure linking parent and children nodes
- evaluation information and metrics
- compact summary generator for LLM agents
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from dataclasses_json import DataClassJsonMixin

logger = logging.getLogger("aide")


def trim_long_string(string: str, threshold: int = 5100, k: int = 2500) -> str:
    """Trim a string if it exceeds the threshold length (from aide.utils.response)."""
    if len(string) > threshold:
        first_k = string[:k]
        last_k = string[-k:]
        truncated_len = len(string) - 2 * k
        return f"{first_k}\n ... [{truncated_len} characters truncated] ... \n{last_k}"
    return string


@dataclass(eq=False)
class AIDEOriginalNode(DataClassJsonMixin):
    """Direct port of Node from WecoAI/aideml/aide/journal.py."""

    # ---- code & plan ----
    code: str = ""
    plan: str = field(default="", kw_only=True)

    # ---- general attrs ----
    step: int = field(default=0, kw_only=True)
    id: str = field(default_factory=lambda: uuid.uuid4().hex, kw_only=True)
    ctime: float = field(default_factory=time.time, kw_only=True)
    parent_id: Optional[str] = field(default=None, kw_only=True)
    branch: str = field(default="baseline", kw_only=True)

    # ---- execution info ----
    _term_out: list[str] = field(default_factory=list, kw_only=True)
    exec_time: float = field(default=0.0, kw_only=True)
    exc_type: str | None = field(default=None, kw_only=True)
    exc_info: dict | None = field(default=None, kw_only=True)

    # ---- evaluation & metrics ----
    analysis: str = field(default="", kw_only=True)
    metric_value: float = field(default=0.0, kw_only=True)
    metrics_dict: Dict[str, Any] = field(default_factory=dict, kw_only=True)
    is_buggy: bool = field(default=False, kw_only=True)

    # ---- domain scientific constraints (SemEval adaptation) ----
    param_count: Optional[int] = field(default=None, kw_only=True)
    atomic_hypothesis: str = field(default="", kw_only=True)
    intervention: str = field(default="", kw_only=True)
    commit_hash: Optional[str] = field(default=None, kw_only=True)
    artifact_dir: Optional[str] = field(default=None, kw_only=True)
    status: str = field(default="COMPLETED", kw_only=True)

    @property
    def stage_name(self) -> Literal["draft", "debug", "improve"]:
        """Return the stage of the node: draft, debug, or improve."""
        if self.parent_id is None:
            return "draft"
        return "debug" if self.is_buggy else "improve"

    @property
    def term_out(self) -> str:
        """Get the terminal output of the code execution after truncating."""
        return trim_long_string("".join(self._term_out))

    def __eq__(self, other):
        return isinstance(other, AIDEOriginalNode) and self.id == other.id

    def __hash__(self):
        return hash(self.id)


@dataclass
class AIDEOriginalJournal(DataClassJsonMixin):
    """Direct port of Journal from WecoAI/aideml/aide/journal.py."""

    nodes: List[AIDEOriginalNode] = field(default_factory=list)
    metric_maximize: bool = True

    def __len__(self) -> int:
        return len(self.nodes)

    def __getitem__(self, idx: int) -> AIDEOriginalNode:
        return self.nodes[idx]

    def append(self, node: AIDEOriginalNode) -> None:
        """Append a new node to the journal."""
        node.step = len(self.nodes)
        self.nodes.append(node)

    @property
    def draft_nodes(self) -> List[AIDEOriginalNode]:
        """Return a list of nodes representing initial coding drafts."""
        return [n for n in self.nodes if n.parent_id is None]

    @property
    def buggy_nodes(self) -> List[AIDEOriginalNode]:
        """Return a list of nodes that are considered buggy."""
        return [n for n in self.nodes if n.is_buggy]

    @property
    def good_nodes(self) -> List[AIDEOriginalNode]:
        """Return a list of nodes that are not considered buggy."""
        return [n for n in self.nodes if not n.is_buggy]

    def get_best_node(self, only_good: bool = True) -> Optional[AIDEOriginalNode]:
        """Return the best solution found so far (node with highest validation metric)."""
        nodes = self.good_nodes if only_good else self.nodes
        if not nodes:
            return None
        return max(nodes, key=lambda n: n.metric_value if self.metric_maximize else -n.metric_value)

    def generate_summary(self, include_code: bool = False, max_nodes: int = 15) -> str:
        """Direct port of AIDE's generate_summary().

        Extracts Plan, Analysis, and Metric for each good node to provide constant-size
        memory for LLM agents without token explosion.
        """
        summary = []
        best_node = self.get_best_node()

        for n in self.good_nodes[-max_nodes:]:
            summary_part = f"Node ID: {n.id} (Parent: {n.parent_id or 'ROOT'})\n"
            summary_part += f"Branch: {n.branch} | Params: {f'{n.param_count/1e6:.1f}M' if n.param_count else 'N/A'}\n"
            summary_part += f"Design/Plan: {n.plan or n.atomic_hypothesis}\n"
            if n.intervention:
                summary_part += f"Intervention: {n.intervention}\n"
            if include_code and n.code:
                summary_part += f"Code: {n.code[:200]}...\n"
            summary_part += f"Results: {n.analysis or 'Executed successfully.'}\n"
            summary_part += f"Validation Metric: {n.metric_value:.4f}"
            if best_node and n.id == best_node.id:
                summary_part += " (⭐ CURRENT BEST)"
            summary.append(summary_part)

        return "\n-------------------------------\n".join(summary)
