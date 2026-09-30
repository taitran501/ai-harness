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
    is_buggy: bool = False
    artifact_dir: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    notes: str = ""
    diagnostics_summary: Dict[str, Any] = field(default_factory=dict)

    def get_score(self, key: Optional[str] = None) -> float:
        """Extract a scalar comparison score from metrics dict in a task-agnostic manner."""
        if not self.metrics or self.is_buggy:
            return 0.0

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


class TreeJournal:
    """The central registry and tree search manager for all experiments."""

    def __init__(
        self,
        storage_path: Optional[str | Path] = None,
        auto_sync: bool = False,
        task_spec: Optional[Any] = None,
        journal_path: Optional[str | Path] = None,
        artifacts_dir: Optional[str | Path] = None,
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
        self.task_spec = task_spec
        self.nodes: Dict[str, Node] = {}
        self.load()
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
        art_path = Path(artifacts_dir) if artifacts_dir is not None else self.artifacts_dir
        if not art_path.exists():
            return 0

        synced_count = 0
        for exp_dir in art_path.iterdir():
            if not exp_dir.is_dir():
                continue

            exp_id = exp_dir.name
            if "template" in exp_id.lower():
                continue

            metrics_file = exp_dir / "metrics.json"
            cfg_file = exp_dir / "run_config.yaml"
            summary_csv = exp_dir / "summary.csv"

            if not metrics_file.exists() and not summary_csv.exists():
                continue

            if exp_id in self.nodes and self.nodes[exp_id].status == "COMPLETED" and self.nodes[exp_id].get_score() > 0:
                continue

            cfg_data = {}
            if cfg_file.exists():
                try:
                    import yaml
                    with open(cfg_file, "r", encoding="utf-8") as f:
                        cfg_data = yaml.safe_load(f) or {}
                except Exception:
                    pass

            exp_info = cfg_data.get("experiment", {})
            task_info = cfg_data.get("task", {})
            model_info = cfg_data.get("model", {})

            parent_id = exp_info.get("parent_id")
            if parent_id in ("none", "null", "ROOT", ""):
                parent_id = None

            branch = exp_info.get("branch", "baseline")
            hypothesis = exp_info.get("hypothesis", "")
            intervention = exp_info.get("intervention", "")
            approach_type = task_info.get("approach_type")
            if not approach_type:
                lower_id = exp_id.lower()
                if any(k in lower_id for k in ["llm", "gemini", "gpt", "claude", "deepseek", "qwen", "llama", "qlora", "peft", "fewshot", "cot"]):
                    approach_type = "llm"
                elif "deberta" in lower_id or "xlm" in lower_id or "roberta" in lower_id or "bert" in lower_id:
                    approach_type = "dl"
                else:
                    approach_type = "ml"

            model_family = task_info.get("model_family")
            if not model_family:
                if approach_type == "llm":
                    model_family = "llm_peft" if "qlora" in exp_id.lower() or "peft" in exp_id.lower() else "llm_api"
                elif approach_type == "dl":
                    model_family = "transformer"
                else:
                    model_family = "linear"
            primary_metric_name = task_info.get("primary_metric", "macro_f1")
            direction = task_info.get("direction", "maximize")

            metrics_dict: Dict[str, Any] = {"model": model_info.get("name", exp_id)}
            calculated_score = 0.0

            if summary_csv.exists():
                try:
                    import pandas as pd
                    df_sum = pd.read_csv(summary_csv)
                    num_cols = df_sum.select_dtypes(include=["number"]).columns
                    for col in num_cols:
                        if col not in ("dev_fold", "train_rows", "dev_rows", "fold"):
                            metrics_dict[col] = round(float(df_sum[col].mean()), 4)

                    for candidate in [primary_metric_name, "score", "macro_f1", "overall_composite", "val_score", "accuracy", "auc_roc", "f1"]:
                        if candidate in df_sum.columns:
                            calculated_score = round(float(df_sum[candidate].mean()), 4)
                            metrics_dict["overall_composite"] = calculated_score
                            metrics_dict["macro_f1"] = calculated_score
                            break

                    if calculated_score == 0.0:
                        try:
                            from harness.task_spec import load_active_task_spec
                            task_spec = load_active_task_spec()
                            st_scores = []
                            total_weight = 0.0
                            for st_id, st in task_spec.subtasks.items():
                                for cand_col in [f"{st_id}_macro_f1", f"{st_id}_f1", f"{st.column}_f1", st_id]:
                                    if cand_col in df_sum.columns:
                                        sc = float(df_sum[cand_col].mean())
                                        st_scores.append(sc * st.weight)
                                        total_weight += st.weight
                                        metrics_dict[st_id] = round(sc, 4)
                                        break
                            if st_scores and total_weight > 0:
                                calculated_score = round(sum(st_scores) / total_weight, 4)
                                metrics_dict["overall_composite"] = calculated_score
                                metrics_dict["macro_f1"] = calculated_score
                        except Exception:
                            pass
                except Exception:
                    pass

            if calculated_score == 0.0 and metrics_file.exists():
                try:
                    with open(metrics_file, "r", encoding="utf-8") as f:
                        m_data = json.load(f)
                    if isinstance(m_data, list) and len(m_data) > 0:
                        first_m = m_data[0]
                        for candidate in [primary_metric_name, "overall_composite", "macro_f1", "score", "accuracy"]:
                            if candidate in first_m:
                                calculated_score = round(float(first_m[candidate]), 4)
                                metrics_dict["overall_composite"] = calculated_score
                                break
                        if calculated_score == 0.0:
                            try:
                                from harness.task_spec import load_active_task_spec
                                task_spec = load_active_task_spec()
                                st_scores = []
                                total_weight = 0.0
                                for st_id, st in task_spec.subtasks.items():
                                    val = first_m.get(st_id)
                                    if isinstance(val, (int, float)):
                                        sc = float(val)
                                    elif isinstance(val, dict):
                                        sc = float(val.get("macro_f1") or val.get("score") or val.get("f1", 0.0))
                                    else:
                                        sc = None
                                    if sc is not None:
                                        st_scores.append(sc * st.weight)
                                        total_weight += st.weight
                                        metrics_dict[st_id] = round(sc, 4)
                                if st_scores and total_weight > 0:
                                    calculated_score = round(sum(st_scores) / total_weight, 4)
                                    metrics_dict["overall_composite"] = calculated_score
                                    metrics_dict["macro_f1"] = calculated_score
                            except Exception:
                                pass
                    elif isinstance(m_data, dict):
                        for candidate in [primary_metric_name, "overall_composite", "macro_f1", "score", "accuracy"]:
                            if candidate in m_data:
                                calculated_score = round(float(m_data[candidate]), 4)
                                metrics_dict["overall_composite"] = calculated_score
                                break
                        metrics_dict.update({k: v for k, v in m_data.items() if isinstance(v, (int, float, dict))})
                except Exception:
                    pass

            diag_summary = {}
            diag_file = exp_dir / "diagnostics.json"
            if diag_file.exists():
                try:
                    with open(diag_file, "r", encoding="utf-8") as f:
                        diag_summary = json.load(f)
                except Exception:
                    pass

            param_count = cfg_data.get("constraints", {}).get("max_params_limit")
            llm_spec = cfg_data.get("llm") or model_info.get("llm_spec")
            param_size = (
                (cfg_data.get("llm") or {}).get("param_size")
                or model_info.get("param_size")
            )
            if not param_size:
                if param_count:
                    if param_count >= 1_000_000_000:
                        param_size = f"{round(param_count / 1e9, 1)}B"
                    elif param_count >= 1_000_000:
                        param_size = f"{round(param_count / 1e6, 1)}M"
                else:
                    id_lower = exp_id.lower()
                    for sz in ["70b", "32b", "14b", "8b", "7b", "3b", "1.5b", "1b"]:
                        if sz in id_lower or (model_info.get("name") and sz in model_info["name"].lower()):
                            param_size = sz.upper()
                            break

            node = Node(
                id=exp_id,
                parent_id=parent_id,
                branch=branch,
                approach_type=approach_type,
                model_family=model_family,
                primary_metric_name=primary_metric_name,
                direction=direction,
                atomic_hypothesis=hypothesis,
                intervention=intervention,
                metrics=metrics_dict,
                param_count=param_count,
                param_size=param_size,
                llm_spec=llm_spec,
                status="COMPLETED",
                artifact_dir=str(exp_dir),
                timestamp=exp_dir.stat().st_mtime,
                notes=exp_info.get("description", ""),
                diagnostics_summary=diag_summary,
            )
            self.nodes[exp_id] = node
            synced_count += 1

        if synced_count > 0:
            self.save()
        return synced_count

    def save(self) -> None:
        """Save the journal to disk."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": "1.0",
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_experiments": len(self.nodes),
            "nodes": [asdict(n) for n in self.nodes.values()],
        }
        from harness.io_utils import atomic_write_json
        atomic_write_json(self.storage_path, data, indent=2, ensure_ascii=False)

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
                delta = round(n.get_score() - parent_node.get_score(), 4)
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
        candidates = [n for n in self.nodes.values() if not n.is_buggy]
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
            if getattr(n, "model_family", None) and n.model_family not in ("other", "linear"):
                families.add(n.model_family)
            else:
                model_name = n.metrics.get("model", "")
                if "tfidf" in n.id.lower() or "tfidf" in model_name.lower():
                    families.add("sparse_tfidf")
                elif "baseline" in n.id.lower() and ("linear" in model_name.lower() or "logreg" in model_name.lower() or "tfidf" in model_name.lower()):
                    families.add("sparse_tfidf")
                elif "xlmr" in n.id.lower() or "xlm-roberta" in model_name.lower():
                    families.add("xlm_roberta")
                elif "deberta" in n.id.lower() or "deberta" in model_name.lower():
                    families.add("deberta")
                elif "llm" in n.id.lower() or "llama" in model_name.lower():
                    families.add("llm_generative")
                elif getattr(n, "model_family", None):
                    families.add(n.model_family)
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
            f"(Score: {best_node.get_score():.4f}, Params: {best_node.param_count or 'N/A'})\n",
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
        valid_nodes = [n for n in self.nodes.values() if not n.is_buggy]
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
        raw_scores = [n.get_score() for n in valid_nodes]
        min_s = min(raw_scores) if raw_scores else 0.0
        max_s = max(raw_scores) if raw_scores else 1.0
        range_s = (max_s - min_s) if (max_s - min_s) > 1e-6 else 1.0

        # Compute UCB scores for each valid candidate node as parent
        scored_candidates = []
        for n in valid_nodes:
            if max_params and n.param_count and n.param_count > max_params:
                continue

            raw_score = n.get_score()
            direction = getattr(n, "direction", "maximize")
            if direction == "minimize":
                # Lower raw_score is better -> map to higher exploit [0, 1]
                exploit = (max_s - raw_score) / range_s if len(valid_nodes) > 1 else (1.0 / (1.0 + max(0.0, raw_score)))
            else:
                exploit = raw_score

            visits = parent_visits.get(n.id, 0)
            explore = exploration_weight * math.sqrt(math.log(total_runs + 1) / (visits + 1))
            ucb_score = exploit + explore

            scored_candidates.append((ucb_score, n))

        scored_candidates.sort(key=lambda x: x[0], reverse=True)
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
