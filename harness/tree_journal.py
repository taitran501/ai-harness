"""AIDE-inspired Solution Tree and Compressed Experiment Journal.

Manages experiment lineage, prevents context saturation, enforces parameter
constraints, and recommends next scientific interventions via tree search and
diagnostic failure analysis.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class Node:
    """A single experiment node in the solution tree."""

    id: str
    parent_id: Optional[str] = None
    branch: str = "baseline"  # baseline, feat, model, loss, mtl, ens
    approach_type: str = "ml"  # "ml" (classical tabular/n-gram), "dl" (deep learning/transformers), "hybrid"
    model_family: str = "linear"  # linear, tree, transformer, cnn, rnn, ensemble, other
    primary_metric_name: str = "overall_composite"
    direction: str = "maximize"  # maximize or minimize
    atomic_hypothesis: str = ""
    intervention: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)
    param_count: Optional[int] = None
    param_size: Optional[str] = None
    llm_spec: Optional[Dict[str, Any]] = None
    commit_hash: Optional[str] = None
    status: str = "COMPLETED"  # COMPLETED, BUGGY, PRUNED, RUNNING
    verified: bool = False
    eligible: bool = False
    is_buggy: bool = False
    artifact_dir: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    notes: str = ""
    diagnostics_summary: Dict[str, Any] = field(default_factory=dict)

    def get_score(self, key: Optional[str] = None) -> float:
        """Extract a scalar comparison score from metrics dict in a task-agnostic manner."""
        if not self.metrics or self.is_buggy:
            return 0.0

        if key is None and "primary_score" in self.metrics:
            return float(self.metrics["primary_score"])
        target_key = key or self.primary_metric_name
        val = self.metrics.get(target_key)
        if isinstance(val, (int, float)):
            return float(val)

        # Check common standard metric keys
        for candidate in ["macro_f1", "f1", "accuracy", "roc_auc", "auc", "map", "score", "rmse", "loss"]:
            if candidate in self.metrics and isinstance(self.metrics[candidate], (int, float)):
                return float(self.metrics[candidate])

        # Dynamically check any nested subtask/target dictionaries
        scores = []
        for task, task_val in self.metrics.items():
            if isinstance(task_val, dict):
                for m_cand in ["macro_f1", "f1", "accuracy", "roc_auc", "auc", "type_macro_f1", "score", "rmse"]:
                    if m_cand in task_val and isinstance(task_val[m_cand], (int, float)):
                        scores.append(float(task_val[m_cand]))
                        break
        return sum(scores) / len(scores) if scores else 0.0

    def get_utility(self):
        from harness.evaluation import utility
        return utility(self.get_score(), self.direction)


class TreeJournal:
    """The central registry and tree search manager for all experiments."""

    def __init__(
        self,
        storage_path: Optional[str | Path] = None,
        auto_sync: bool = False,
        task_spec: Optional[Any] = None,
        journal_path: Optional[str | Path] = None,
        artifacts_dir: Optional[str | Path] = None,
        task_plugin=None,
    ):
        from harness.project import HarnessProject

        project = HarnessProject.discover()
        path = journal_path or storage_path or (project.artifacts_dir / "journal.json")
        self.storage_path = Path(path)
        self.artifacts_dir = (
            Path(artifacts_dir)
            if artifacts_dir
            else (self.storage_path.parent if journal_path or storage_path else project.artifacts_dir)
        )
        if task_spec is None:
            from harness.task_spec import load_active_task_spec
            from harness.plugins import TaskPluginContext,load_task_plugin
            task_spec=load_active_task_spec(project.task_config_path)
            task_plugin=load_task_plugin(project.plugin_id,TaskPluginContext(project.root,project.task_config_path,task_spec,project.plugin_id or "default"))
        self.task_spec = task_spec
        self.task_plugin=task_plugin
        self.nodes: Dict[str, Node] = {}
        self.load()
        from harness.artifacts import validate_run
        for node in self.nodes.values():
            if node.artifact_dir:
                try: validate_run(node.artifact_dir,task_spec=self.task_spec,plugin=self.task_plugin)
                except (ValueError,OSError,KeyError):
                    node.verified=node.eligible=False
                    node.status="LEGACY_UNVERIFIED"
        self._snapshot=[asdict(n) for n in self.nodes.values()]
        if auto_sync:
            self.sync_from_artifacts(self.artifacts_dir)

    def add_node(
        self,
        node_id: str,
        parent_id: Optional[str] = None,
        branch: str = "baseline",
        hypothesis: str = "",
        score: float = 0.0,
        metrics: Optional[Dict[str, Any]] = None,
        model_family: str = "linear",
        approach_type: str = "ml",
    ) -> Node:
        """Helper to programmatically create, register, and save a node."""
        m = metrics or {}
        if "overall_composite" not in m:
            m["overall_composite"] = score
        node = Node(
            id=node_id,
            parent_id=parent_id,
            branch=branch,
            approach_type=approach_type,
            model_family=model_family,
            atomic_hypothesis=hypothesis,
            metrics=m,
            status="COMPLETED",
        )
        self.append(node)
        return node

    def load(self) -> None:
        """Load the journal from disk if it exists."""
        if not self.storage_path.exists():
            return
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            node_fields = Node.__dataclass_fields__.keys()
            for item in data.get("nodes", []):
                filtered = {k: v for k, v in item.items() if k in node_fields}
                node = Node(**filtered)
                self.nodes[node.id] = node
        except Exception as e:
            print(f"[TreeJournal] Warning: Failed to load {self.storage_path}: {e}")

    def sync_from_artifacts(self, artifacts_dir: Optional[str | Path] = None) -> int:
        """Scan artifacts/ directory and auto-discover completed experiment runs."""
        from harness.artifacts import validate_run
        import yaml
        art_path = Path(artifacts_dir) if artifacts_dir is not None else self.artifacts_dir
        count = 0
        if not art_path.exists(): return 0
        for directory in art_path.iterdir():
            if not directory.is_dir() or not (directory/"metrics.json").is_file(): continue
            metrics = json.loads((directory/"metrics.json").read_text(encoding="utf-8"))
            cfg = yaml.safe_load((directory/"run_config.yaml").read_text(encoding="utf-8")) if (directory/"run_config.yaml").is_file() else {}
            try:
                manifest = validate_run(directory, task_spec=self.task_spec,plugin=self.task_plugin)
                verified = True
            except (ValueError, KeyError, OSError):
                verified = False
            exp = cfg.get("experiment", {})
            model_cfg = cfg.get("model", {})
            task_cfg = cfg.get("task", {})
            m_name = str(model_cfg.get("name", "") or metrics.get("model", "")).lower()
            m_fam_raw = (
                task_cfg.get("model_family")
                or model_cfg.get("family")
                or metrics.get("model_family")
            )
            if any(k in m_name for k in ["lightgbm", "lgbm", "xgboost", "xgb", "catboost"]):
                m_fam = "gbdt"
            elif any(k in m_name for k in ["roberta", "deberta", "bert"]):
                m_fam = "transformer"
            elif any(k in m_name for k in ["llm", "llama", "qwen"]):
                m_fam = "llm"
            elif m_fam_raw:
                m_fam = m_fam_raw
            else:
                m_fam = "linear"
            diagnostics = {}
            if verified and "diagnostics.json" in manifest["files"]:
                diagnostics = json.loads((directory / "diagnostics.json").read_text(encoding="utf-8"))
            node = Node(id=directory.name,parent_id=exp.get("parent_id"),branch=exp.get("branch","baseline"),
                model_family=m_fam,
                metrics=metrics if isinstance(metrics,dict) else {"legacy_folds":metrics},
                primary_metric_name=metrics.get("primary_metric","unknown") if isinstance(metrics,dict) else "unknown",
                direction=metrics.get("direction","maximize") if isinstance(metrics,dict) else "maximize",
                verified=verified,eligible=verified,status="COMPLETED" if verified else "LEGACY_UNVERIFIED",
                artifact_dir=str(directory),atomic_hypothesis=exp.get("hypothesis",""),
                intervention=exp.get("intervention", ""), diagnostics_summary=diagnostics,
                timestamp=(directory / "artifact_manifest.json").stat().st_mtime
                    if (directory / "artifact_manifest.json").exists() else directory.stat().st_mtime)
            self.nodes[node.id]=node; count+=1
        if count: self.save()
        return count

    def save(self) -> None:
        """Save the journal to disk."""
        from harness.state_store import save_registry
        records=save_registry(self.storage_path,"nodes",[asdict(n) for n in self.nodes.values()],getattr(self,"_snapshot",[]))
        self.nodes={r["id"]:Node(**r) for r in records}
        self._snapshot=records

    def append(self, node: Node) -> None:
        """Register or update a node in the journal."""
        self.nodes[node.id] = node
        self.save()

    def get_node(self, node_id: str) -> Optional[Node]:
        return self.nodes.get(node_id)

    def get_children(self, parent_id: str) -> List[Node]:
        return [n for n in self.nodes.values() if n.parent_id == parent_id]

    def get_graph_data(self) -> Dict[str, Any]:
        """Generate structured DAG nodes and directed edges with delta scores for frontend visualizers."""
        nodes_list = []
        edges_list = []
        best = self.get_best_node()
        best_id = best.id if best else None

        for n in self.nodes.values():
            delta = None
            if n.parent_id and n.parent_id in self.nodes:
                parent_node = self.nodes[n.parent_id]
                delta = round(n.get_utility() - parent_node.get_utility(), 4)
                edges_list.append({
                    "from": n.parent_id,
                    "to": n.id,
                    "delta": delta,
                    "improved": delta > 0,
                })

            nodes_list.append({
                "id": n.id,
                "label": n.id,
                "parent_id": n.parent_id,
                "branch": n.branch,
                "approach_type": getattr(n, "approach_type", "ml"),
                "model_family": getattr(n, "model_family", "linear"),
                "score": round(n.get_score(), 4),
                "parent_delta": delta,
                "param_count": n.param_count,
                "param_size": getattr(n, "param_size", None),
                "llm_spec": getattr(n, "llm_spec", None),
                "status": n.status,
                "verified":n.verified,"eligible":n.eligible,"direction":n.direction,"primary_metric":n.primary_metric_name,
                "is_best": (n.id == best_id),
                "is_buggy": n.is_buggy,
                "atomic_hypothesis": n.atomic_hypothesis,
                "intervention": n.intervention,
                "timestamp": n.timestamp,
                "metrics": n.metrics,
                "diagnostics_summary": n.diagnostics_summary,
            })

        return {"nodes": nodes_list, "edges": edges_list, "best_id": best_id}

    def get_best_node(
        self, metric_key: Optional[str] = None, max_params: Optional[int] = None
    ) -> Optional[Node]:
        """Find the top-performing valid node adhering to parameter constraints."""
        candidates = [n for n in self.nodes.values() if not n.is_buggy and n.eligible and n.verified]
        if max_params is not None:
            candidates = [
                n for n in candidates if n.param_count is None or n.param_count <= max_params
            ]
        if not candidates:
            return None
        first = candidates[0]
        direction = getattr(first, "direction", "maximize")
        if direction == "minimize":
            return min(candidates, key=lambda n: n.get_score(metric_key))
        return max(candidates, key=lambda n: n.get_score(metric_key))

    def count_distinct_model_families(self) -> int:
        """Count distinct architecture families among completed nodes to prevent pseudo-ensembling."""
        families = set()
        for n in self.nodes.values():
            if n.is_buggy:
                continue
            mf = str(getattr(n, "model_family", "") or "").lower()
            model_name = str(n.metrics.get("model", "")).lower()
            node_id = n.id.lower()

            # 1. GBDT Family
            if mf in ("tree", "gbdt") or any(
                k in node_id or k in model_name
                for k in ["lightgbm", "lgbm", "xgboost", "xgb", "catboost", "gbdt"]
            ):
                families.add("gbdt")
            # 2. Vision Family
            elif any(
                k in node_id or k in model_name
                for k in ["convnext", "swin", "vit", "resnet", "efficientnet", "vision", "timm"]
            ):
                families.add("vision")
            # 3. Audio Family
            elif any(
                k in node_id or k in model_name
                for k in ["whisper", "wav2vec", "ast", "audio"]
            ):
                families.add("audio")
            # 4. Transformer / Deep Learning / LLM Families
            elif "xlmr" in node_id or "xlm-roberta" in model_name:
                families.add("xlm_roberta")
            elif "deberta" in node_id or "deberta" in model_name:
                families.add("deberta")
            elif "roberta" in node_id or "roberta" in model_name:
                families.add("roberta")
            elif "bert" in node_id or "bert" in model_name:
                families.add("bert")
            elif mf in ("transformer",):
                families.add("transformer")
            elif mf in ("dl", "deep_learning"):
                families.add("deep_learning")
            elif mf in ("llm", "llm_peft") or "llm" in node_id or "llama" in model_name:
                families.add("llm_generative")
            # 3. Linear / Baseline / TF-IDF Family
            elif (
                mf == "linear"
                or any(k in model_name for k in ["ridge", "logistic", "logreg"])
            ):
                families.add("linear")
            elif "tfidf" in node_id or "tfidf" in model_name:
                families.add("tfidf")
            elif "baseline" in node_id:
                families.add("baseline")
            elif mf and mf not in ("other",):
                families.add(mf)
            else:
                families.add(f"custom_{n.id}")
        return len(families)

    def generate_compact_summary(self, max_nodes: int = 15) -> str:
        """Compress the entire history into a token-efficient summary (~300-800 tokens)."""
        if not self.nodes:
            return "No experiments recorded yet in TreeJournal."

        best_node = self.get_best_node()
        lines = [
            f"# TreeJournal Summary (Total Runs: {len(self.nodes)})",
            f"Best Run: {best_node.id if best_node else 'None'} "
            f"(Score: {best_node.get_score() if best_node else 'N/A'})\n",
            "| ID | Type | Branch | Parent | Score | Params | Hypothesis & Delta | Status |",
            "|---|---|---|---|---:|---:|---|---|",
        ]

        sorted_nodes = sorted(
            self.nodes.values(),
            key=lambda n: (n.id == (best_node.id if best_node else ""), n.timestamp),
            reverse=True,
        )

        for n in sorted_nodes[:max_nodes]:
            parent = n.parent_id or "ROOT"
            score_str = f"{n.get_score():.4f}" if not n.is_buggy else "ERR"
            params_str = f"{n.param_count / 1e6:.1f}M" if n.param_count else "-"
            hypo_snippet = (
                (n.atomic_hypothesis[:50] + "...")
                if len(n.atomic_hypothesis) > 50
                else (n.atomic_hypothesis or n.intervention or "-")
            )
            status_badge = "✅" if not n.is_buggy else "❌"
            app_type = getattr(n, "approach_type", "ml").upper()
            lines.append(
                f"| `{n.id}` | `{app_type}` | `{n.branch}` | `{parent}` | **{score_str}** | {params_str} | {hypo_snippet} | {status_badge} {n.status} |"
            )

        if len(self.nodes) > max_nodes:
            lines.append(f"\n*(Truncated {len(self.nodes) - max_nodes} older runs to preserve agent context. Query specific nodes via `python tools/journal_tool.py --node <id>`)*")

        return "\n".join(lines)

    def recommend_next_branch(
        self,
        max_params: Optional[int] = None,
        exploration_weight: float = 1.414,
    ) -> Dict[str, Any]:
        """AIDE-style UCB-guided branch recommendation enhanced by diagnostic signals.

        Prevents agents from arbitrary pivoting, quitting, or performing pseudo-ensembling.
        """
        valid_nodes = [n for n in self.nodes.values() if not n.is_buggy and n.eligible and n.verified]
        if not valid_nodes:
            return {
                "recommendation": "INITIAL_BASELINE",
                "parent_id": None,
                "branch": "baseline",
                "guideline": "Establish strong initial baseline with TF-IDF or default backbone.",
            }

        # Count visits per parent
        parent_visits: Dict[str, int] = {}
        for n in self.nodes.values():
            p = n.parent_id or "ROOT"
            parent_visits[p] = parent_visits.get(p, 0) + 1

        total_runs = len(self.nodes)

        # Compute scores and range for UCB normalization across candidates
        raw_scores = [n.get_utility() for n in valid_nodes]
        min_s = min(raw_scores) if raw_scores else 0.0
        max_s = max(raw_scores) if raw_scores else 1.0
        range_s = (max_s - min_s) if (max_s - min_s) > 1e-6 else 1.0

        # Compute UCB scores for each valid candidate node as parent
        scored_candidates = []
        for n in valid_nodes:
            if max_params and n.param_count and n.param_count > max_params:
                continue

            exploit = (n.get_utility()-min_s)/range_s if len(valid_nodes)>1 else 1.0

            visits = parent_visits.get(n.id, 0)
            explore = exploration_weight * math.sqrt(math.log(total_runs + 1) / (visits + 1))
            ucb_score = exploit + explore

            scored_candidates.append((ucb_score, n))

        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        if not scored_candidates: return {"recommendation":"INITIAL_BASELINE","parent_id":None,"branch":"baseline"}
        top_ucb, top_node = scored_candidates[0]

        # Check diagnostic bottlenecks in top_node
        bottlenecks = top_node.diagnostics_summary.get("critical_bottlenecks", [])
        diagnostic_cue = ""

        # Check existing child branches
        children = self.get_children(top_node.id)
        existing_branches = {c.branch for c in children}

        # Anti-Pseudo-Ensemble Rule: Check distinct model families
        num_families = self.count_distinct_model_families()
        all_branches = ["feat", "model", "loss", "mtl", "ens"]

        # If bottlenecks exist, prioritize relevant branches dynamically
        b_text = " ".join(bottlenecks).lower()
        if any(k in b_text for k in ["imbalance", "rare", "zero recall", "low recall", "focal"]):
            prioritized = ["loss", "feat", "model", "mtl", "ens"]
            diagnostic_cue = "Parent diagnostics indicate class imbalance or low recall on minority slices. Prioritize class re-weighting, threshold tuning, or Focal Loss."
        elif any(k in b_text for k in ["representation", "implicit", "context", "underfitting", "high bias"]):
            prioritized = ["model", "feat", "loss", "mtl", "ens"]
            diagnostic_cue = "Parent diagnostics indicate complex semantic or contextual underfitting. Prioritize richer pretrained model backbones or augmented feature representations."
        elif any(k in b_text for k in ["overfitting", "high variance", "generalization"]):
            prioritized = ["feat", "ens", "loss", "model", "mtl"]
            diagnostic_cue = "Parent diagnostics indicate overfitting or variance across folds. Prioritize regularization, feature pruning, or ensembling."
        else:
            prioritized = all_branches

        candidate_branches = [b for b in prioritized if b not in existing_branches]
        if not candidate_branches:
            candidate_branches = prioritized

        recommended_branch = candidate_branches[0]

        # Enforce Ensemble Guard: Disallow 'ens' if fewer than 2 distinct model families
        if recommended_branch == "ens" and num_families < 2:
            recommended_branch = "model"
            diagnostic_cue += f" (Note: 'ens' locked because only {num_families} model family exists. Explore a new architecture backbone like XLM-R before ensembling)."

        return {
            "recommendation": f"Branch from {top_node.id} on {recommended_branch}",
            "parent_id": top_node.id,
            "parent_score": top_node.get_score(),
            "parent_params": top_node.param_count,
            "branch": recommended_branch,
            "recommended_branch": recommended_branch,
            "ucb_score": round(top_ucb, 4),
            "bottlenecks": bottlenecks,
            "diagnostic_cue": diagnostic_cue,
            "guideline": (
                f"Branch from `{top_node.id}` (Score: {top_node.get_score():.4f}). "
                f"Recommended Branch: `{recommended_branch}`. "
                + (f"Diagnostic finding: {diagnostic_cue} " if diagnostic_cue else "")
                + f"Formulate a single atomic hypothesis. Parameters must remain <= {max_params or 'unlimited'}."
            ),
        }
