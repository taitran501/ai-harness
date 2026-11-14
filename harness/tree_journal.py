"""AIDE-inspired Solution Tree and Compressed Experiment Journal.

Manages experiment lineage, prevents context saturation, enforces parameter
constraints, and recommends next scientific interventions via tree search and
diagnostic failure analysis.
"""

from __future__ import annotations

import copy
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


class _RunParsingError(ValueError):
    def __init__(self, filename, reason):
        self.filename = filename
        super().__init__(str(reason))


def _read_run_mapping(path, *, yaml_format=False):
    try:
        text = path.read_text(encoding="utf-8")
        value = yaml.safe_load(text) if yaml_format else json.loads(text)
        if not isinstance(value, dict):
            raise ValueError("Expected a mapping")
        return value
    except (ValueError, OSError, UnicodeError, yaml.YAMLError) as exc:
        raise _RunParsingError(path.name, exc) from exc


@dataclass
class Node:
    """A single experiment node in the solution tree."""

    id: str
    parent_id: Optional[str] = None
    parent_ids: List[str] = field(default_factory=list)
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

    def __post_init__(self) -> None:
        """Harmonize parent_id and parent_ids for multi-parent DAG lineage."""
        if self.parent_id and not self.parent_ids:
            self.parent_ids = [self.parent_id]
        elif self.parent_ids and not self.parent_id:
            self.parent_id = self.parent_ids[0]

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
            from harness.plugins import TaskPluginContext, load_task_plugin
            from harness.task_spec import load_active_task_spec
            task_spec=load_active_task_spec(project.task_config_path)
            task_plugin=load_task_plugin(project.plugin_id,TaskPluginContext(project.root,project.task_config_path,task_spec,project.plugin_id or "default"))
        self.task_spec = task_spec
        self.task_plugin=task_plugin
        self.nodes: Dict[str, Node] = {}
        self.sync_diagnostics = []
        self.corrupt_run_count = 0
        if auto_sync:
            self._load_records(self.storage_path)
            self.sync_from_artifacts(self.artifacts_dir)
        else:
            self.load()

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
        parent_ids: Optional[List[str]] = None,
    ) -> Node:
        """Helper to programmatically create, register, and save a node."""
        m = metrics or {}
        if "overall_composite" not in m:
            m["overall_composite"] = score
        p_ids = parent_ids or ([parent_id] if parent_id else [])
        node = Node(
            id=node_id,
            parent_id=parent_id or (p_ids[0] if p_ids else None),
            parent_ids=p_ids,
            branch=branch,
            approach_type=approach_type,
            model_family=model_family,
            atomic_hypothesis=hypothesis,
            metrics=m,
            status="COMPLETED",
        )
        self.append(node)
        return node

    @classmethod
    def load_from_file(cls, path: str | Path, **kwargs) -> "TreeJournal":
        """Load a TreeJournal directly from an explicit file path."""
        return cls(storage_path=path, **kwargs)

    @staticmethod
    def _migrate_records(data):
        if isinstance(data, list):
            return {"nodes": data}
        if isinstance(data, dict):
            data = dict(data)
            if isinstance(data.get("nodes"), dict):
                for key, value in data["nodes"].items():
                    if not isinstance(value, dict) or value.get("id") != key:
                        raise ValueError("Journal node key/ID mismatch")
                data["nodes"] = list(data["nodes"].values())
        return data

    @staticmethod
    def _validate_records(data):
        rows = data.get("nodes")
        if not isinstance(rows, list):
            raise ValueError("Journal nodes must be a list or mapping")
        ids = set()
        for item in rows:
            if not isinstance(item, dict) or not isinstance(item.get("id"), str) or item["id"] in ids:
                raise ValueError("Journal nodes require unique string IDs")
            if not isinstance(item.get("metrics", {}), dict):
                raise ValueError("Node metrics must be a mapping")
            for flag in ("verified", "eligible"):
                if type(item.get(flag, False)) is not bool:
                    raise ValueError(f"Node {flag} must be boolean")
            ids.add(item["id"])

    def _load_records(self, path):
        from harness.state_store import read_json_state
        data = read_json_state(path, default_factory=lambda: {"nodes": []},
            migrate=self._migrate_records, validate=self._validate_records)
        fields = Node.__dataclass_fields__
        nodes = {item["id"]: Node(**{k: v for k, v in item.items() if k in fields})
                 for item in data["nodes"]}
        self.nodes = nodes
        # Baseline must represent persisted claims before validation demotes them.
        self._snapshot = copy.deepcopy(data["nodes"])
        self._decoded_snapshot = [asdict(n) for n in nodes.values()]

    def _validate_loaded_nodes(self, *, skip_ids=frozenset()):
        from harness.artifacts import validate_run
        for node in self.nodes.values():
            if node.id in skip_ids:
                continue
            try:
                if not node.artifact_dir:
                    raise ValueError("Missing artifact_dir")
                validate_run(node.artifact_dir, task_spec=self.task_spec, plugin=self.task_plugin,
                             expected={"exp_id": node.id})
            except (ValueError, OSError, KeyError, TypeError, UnicodeError, yaml.YAMLError):
                node.verified = node.eligible = False
                node.status = "LEGACY_UNVERIFIED"

    def load(self, path: Optional[str | Path] = None) -> "TreeJournal":
        """Read and validate a fresh journal; never persist demotions during loading."""
        if isinstance(self, (str, Path)):
            return TreeJournal(storage_path=self)
        target = Path(path) if path is not None else self.storage_path
        self._load_records(target)
        self._validate_loaded_nodes()
        return self

    def sync_from_artifacts(self, artifacts_dir: Optional[str | Path] = None) -> int:
        """Materialize readable runs, isolating corrupt siblings and persisting demotions."""
        art_path = Path(artifacts_dir) if artifacts_dir is not None else self.artifacts_dir
        # An action may have published journal records through another instance.
        # Rebase local changes on that registry before refreshing derived fields.
        from harness.state_store import merge_changed
        previous = {r["id"]: r for r in self._decoded_snapshot}
        pending = [asdict(n) for n in self.nodes.values() if asdict(n) != previous.get(n.id)]
        self._load_records(self.storage_path)
        for record in pending:
            node = self.nodes.get(record["id"])
            merged = merge_changed(asdict(node) if node else {}, previous.get(record["id"], {}), record)
            self.nodes[record["id"]] = Node(**merged)
        present = {d.name for d in art_path.glob("*") if d.is_dir() and (d / "metrics.json").is_file()}
        self._validate_loaded_nodes(skip_ids=present)
        self.sync_diagnostics = []
        self.corrupt_run_count = 0
        count = 0
        changed = any(asdict(n) not in self._snapshot for n in self.nodes.values())
        if not art_path.exists():
            if changed:
                self.save()
            return 0
        for directory in art_path.iterdir():
            if not directory.is_dir():
                continue
            if not (directory / "metrics.json").exists() and directory.name not in self.nodes:
                continue
            try:
                node = self._node_from_artifact_directory(directory)
            except (ValueError, OSError, KeyError, TypeError, UnicodeError, yaml.YAMLError) as exc:
                self.corrupt_run_count += 1
                if len(self.sync_diagnostics) < 20:
                    self.sync_diagnostics.append({"exp_id": directory.name,
                        "filename": str(getattr(exc, "filename", None) or "metrics.json/run_config.yaml/diagnostics.json"),
                        "reason": str(exc)[:300]})
                stale = self.nodes.get(directory.name)
                if stale:
                    stale.verified = stale.eligible = False
                    stale.status = "LEGACY_UNVERIFIED"
                    changed = True
                continue
            changed |= self.nodes.get(node.id) != node
            self.nodes[node.id] = node
            count += 1
        if changed:
            self.save()
        return count

    def _node_from_artifact_directory(self, directory: Path) -> Node:
        from harness.artifacts import validate_run
        metrics = _read_run_mapping(directory / "metrics.json")
        cfg_path = directory / "run_config.yaml"
        cfg = _read_run_mapping(cfg_path, yaml_format=True) if cfg_path.is_file() else {}
        try:
            manifest = validate_run(directory, task_spec=self.task_spec,plugin=self.task_plugin)
            verified = True
        except (ValueError, KeyError, OSError, TypeError, UnicodeError, yaml.YAMLError):
            verified = False
            manifest = {}
        for name in ("experiment", "model", "task"):
            if not isinstance(cfg.get(name, {}), dict):
                raise _RunParsingError("run_config.yaml", f"Section {name} must be a mapping")
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
        if (directory / "diagnostics.json").exists():
            diagnostics = _read_run_mapping(directory / "diagnostics.json")
        node = Node(id=directory.name,parent_id=exp.get("parent_id"),branch=exp.get("branch","baseline"),
            model_family=m_fam,
            metrics=metrics if isinstance(metrics,dict) else {"legacy_folds":metrics},
            primary_metric_name=metrics.get("primary_metric","unknown") if isinstance(metrics,dict) else "unknown",
            direction=metrics.get("direction","maximize") if isinstance(metrics,dict) else "maximize",
            verified=verified,eligible=bool(verified and manifest.get(
                "eligible",
                manifest.get("provenance", "local_run") in {"local_run", "reproduced_external_code"},
            )),status="COMPLETED" if verified else "LEGACY_UNVERIFIED",
            artifact_dir=str(directory),atomic_hypothesis=exp.get("hypothesis",""),
            intervention=exp.get("intervention", ""), diagnostics_summary=diagnostics,
            timestamp=(directory / "artifact_manifest.json").stat().st_mtime
                if (directory / "artifact_manifest.json").exists() else directory.stat().st_mtime)
        return node

    def save(self) -> None:
        """Save the journal to disk."""
        import dataclasses

        from harness.state_store import save_registry
        valid_fields = {f.name for f in dataclasses.fields(Node)}
        records=save_registry(self.storage_path,"nodes",[asdict(n) for n in self.nodes.values()],getattr(self,"_snapshot",[]),
                              validate=self._validate_records, migrate=self._migrate_records)
        self.nodes={r["id"]:Node(**{k: v for k, v in r.items() if k in valid_fields}) for r in records}
        self._snapshot=copy.deepcopy(records)
        self._decoded_snapshot=[asdict(n) for n in self.nodes.values()]

    def append(self, node: Node) -> None:
        """Register or update a node in the journal."""
        self.nodes[node.id] = node
        self.save()

    def get_node(self, node_id: str) -> Optional[Node]:
        return self.nodes.get(node_id)

    def get_children(self, parent_id: str) -> List[Node]:
        return [
            n for n in self.nodes.values()
            if n.parent_id == parent_id or parent_id in getattr(n, "parent_ids", [])
        ]

    def get_graph_data(self) -> Dict[str, Any]:
        """Generate structured DAG nodes and directed edges with delta scores for frontend visualizers."""
        nodes_list = []
        edges_list = []
        best = self.get_best_node()
        best_id = best.id if best else None

        for n in self.nodes.values():
            delta = None
            parents = n.parent_ids if getattr(n, "parent_ids", None) else ([n.parent_id] if n.parent_id else [])
            for p_id in parents:
                if p_id in self.nodes:
                    parent_node = self.nodes[p_id]
                    edge_delta = round(n.get_utility() - parent_node.get_utility(), 4)
                    if delta is None:
                        delta = edge_delta
                    edges_list.append({
                        "from": p_id,
                        "to": n.id,
                        "delta": edge_delta,
                        "improved": edge_delta > 0,
                    })

            nodes_list.append({
                "id": n.id,
                "label": n.id,
                "parent_id": n.parent_id,
                "parent_ids": getattr(n, "parent_ids", []),
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
            parent = (
                ",".join(n.parent_ids)
                if len(getattr(n, "parent_ids", [])) > 1
                else (n.parent_id or "ROOT")
            )
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
                "guideline": "Establish strong initial baseline with default model backbone.",
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
        if not scored_candidates:
            return {"recommendation": "INITIAL_BASELINE", "parent_id": None, "branch": "baseline"}
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
            diagnostic_cue += f" (Note: 'ens' locked because only {num_families} model family exists. Explore a diverse orthogonal architecture family before ensembling)."

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
