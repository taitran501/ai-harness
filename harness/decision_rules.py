"""Prospective, evidence-bound promotion rules and paired metric uncertainty.

State is kept beside sealed runs. Scores supplied by callers, free-text lessons,
and public leaderboard notes are never substituted for recomputed evidence.
"""
from __future__ import annotations

import copy
import json
import math
import os
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yaml

from harness.artifacts import digest, sha256, validate_run
from harness.evaluation import metric_name
from harness.io_utils import atomic_write_json
from harness.policy_schema import validate_document
from harness.state_store import transaction, workspace_lock

_EVALUATION_CACHE = {}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


class DecisionRuleNotSatisfied(ValueError):
    def __init__(self, reason_codes):
        self.reason_codes = list(reason_codes)
        super().__init__("Decision rule not satisfied: " + ", ".join(self.reason_codes))


class EvidenceNotComparable(DecisionRuleNotSatisfied):
    pass


@dataclass(frozen=True)
class MinimumEffect:
    value: float
    scale: str = "absolute"


@dataclass(frozen=True)
class MetricSelector:
    scope: str
    subtask_id: str | None
    name: str


@dataclass(frozen=True)
class EvidenceRef:
    kind: str
    id: str
    partition: str


@dataclass(frozen=True)
class UncertaintySpec:
    method: str
    unit: str
    unit_column: str | None
    stratify_column: str | None
    confidence_level: float
    resamples: int
    seed: int


@dataclass(frozen=True)
class DecisionRuleV1:
    schema_version: int
    kind: str
    metric: dict
    direction: str
    control: dict
    minimum_effect: dict
    comparator: str
    score_target: dict | None
    evidence_requirements: dict
    uncertainty: dict
    multiplicity: dict

    def __post_init__(self):
        validate_document("decision_rule", asdict(self))

    @classmethod
    def from_dict(cls, value):
        value = copy.deepcopy(value)
        value["metric"]["name"] = metric_name(value["metric"]["name"])
        return cls(**value)

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class ResearchPolicy:
    schema_version: int = 1
    policy_id: str = "resolved-default-v1"
    minimum_effect: dict | None = None
    confidence_level: float = .95
    max_comparisons: int = 20
    bootstrap_resamples: int = 10000
    seed: int = 42
    max_spearman: float = .995
    stagnation_patience: int = 3
    near_duplicate_policy: str = "require_independent_promotion"
    require_confirmation_for_adaptive_search: bool = True

    def __post_init__(self):
        validate_document("research_policy", asdict(self))

    def to_dict(self):
        return asdict(self)


class DecisionEvaluationV1(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key) from None

    def to_dict(self):
        return dict(self)


def read_decision_evaluation(path):
    """Sidecar integrity is checked before any projection or cache can use it."""
    value = json.loads(path.read_text(encoding="utf-8"))
    validate_document("decision_evaluation", value)
    expected_id = "evaluation-" + digest({k: v for k, v in value.items() if k != "evaluation_id"})
    if value["evaluation_id"] != expected_id or path.stem != expected_id:
        raise ValueError("DECISION_EVALUATION_INTEGRITY: receipt content changed")
    return value


def _metadata(spec):
    return spec.get("task", spec.get("metadata", spec)) if isinstance(spec, dict) else spec.metadata


def _project(project):
    from harness.project import HarnessProject
    return project if isinstance(project, HarnessProject) else HarnessProject.discover(project)


def resolve_research_policy(project, task_spec, session_budget=None):
    project = _project(project)
    metadata = _metadata(task_spec)
    sub = getattr(task_spec, "submission_spec", None) or metadata.get("submission_spec", {})
    sidecar = project.root / ".ai-harness/policies/research_policy.json"
    policy_data = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.exists() else sub.get("audit_policy", {})
    if sidecar.exists():
        validate_document("research_policy", policy_data)
    policy = ResearchPolicy(**policy_data)
    if policy.minimum_effect is None:
        explicit = None
        for source in (session_budget or {}, sub, metadata, metadata.get("evaluation_protocol", {})):
            for key in ("min_meaningful_improvement", "minimum_meaningful_improvement"):
                if source.get(key) is not None:
                    explicit = float(source[key])
                    if explicit == 0:
                        # Historical session defaults used zero for "unspecified".
                        # This is not a positive promotion rule and cannot verify support.
                        explicit = None
                        continue
                    break
            if explicit is not None:
                break
        normalized = {"accuracy", "balanced_accuracy", "macro_f1", "micro_f1", "weighted_f1", "roc_auc", "pr_auc", "exact_match"}
        origin = "explicit_legacy_alias" if explicit is not None else "legacy_practical_default"
        if explicit is None and metric_name(metadata.get("primary_metric", "")) in normalized:
            explicit = .002
        if explicit is not None:
            policy = ResearchPolicy(**{**asdict(policy), "policy_id": origin, "minimum_effect": {"value": explicit, "scale": "absolute"}})
    return policy


def policy_directory(project):
    return _project(project).artifacts_dir / "policy_evidence"


def _rule(value):
    return value if isinstance(value, DecisionRuleV1) else DecisionRuleV1.from_dict(value)


def _validate_rule_policy(rule, policy, task_spec):
    selected = task_spec if rule.metric["scope"] == "primary" else task_spec.subtasks[rule.metric["subtask_id"]]
    if metric_name(selected.primary_metric) != rule.metric["name"] or selected.direction != rule.direction:
        raise EvidenceNotComparable(["IDENTITY_MISMATCH"])
    if rule.control["partition"] != rule.evidence_requirements["partition"]:
        raise EvidenceNotComparable(["IDENTITY_MISMATCH"])
    if policy.minimum_effect is None:
        # A predeclared explicit rule supplies the missing unit-bearing effect.
        pass
    elif (rule.minimum_effect["scale"] != policy.minimum_effect["scale"] or
          rule.minimum_effect["value"] < policy.minimum_effect["value"]):
        raise ValueError("Decision rule weakens minimum effect policy")
    if (rule.uncertainty["confidence_level"] < policy.confidence_level or
        rule.uncertainty["resamples"] < policy.bootstrap_resamples or
        rule.multiplicity["max_comparisons"] < policy.max_comparisons):
        raise ValueError("Decision rule weakens uncertainty policy")


def resolve_evidence(reference, *, project, task_spec, plugin=None):
    """Resolve full sealed evidence. Unpaired official scores fail closed."""
    from harness.oof_revalidation import rederive_primary_score
    from harness.plugins import TaskPluginContext, load_task_plugin
    project = _project(project)
    reference = asdict(reference) if isinstance(reference, EvidenceRef) else reference
    if reference["kind"] != "full_run":
        raise EvidenceNotComparable(["UNSUPPORTED_METRIC_PROTOCOL"])
    directory = (project.artifacts_dir / reference["id"]).resolve()
    directory.relative_to(project.artifacts_dir.resolve())
    manifest = validate_run(directory, task_spec, plugin=plugin)
    context = TaskPluginContext(project.root, project.task_config_path, task_spec, project.plugin_id or "default")
    plugin = plugin or load_task_plugin(project.plugin_id, context)
    cfg = yaml.safe_load((directory / "run_config.yaml").read_text(encoding="utf-8"))
    partition = cfg.get("validation", {}).get("partition", "oof")
    if partition != reference["partition"]:
        raise EvidenceNotComparable(["IDENTITY_MISMATCH"])
    report = rederive_primary_score(directory, task_spec=task_spec, plugin=plugin, plugin_context=context)
    if not report["ok"]:
        raise EvidenceNotComparable(["EVIDENCE_MISSING"])
    truth = pd.read_csv(directory / "evaluation_truth.csv")
    frame = pd.read_csv(directory / "oof_predictions.csv")
    variant = report.get("selection_variant")
    if "input_variant" in frame:
        if variant is None and frame.input_variant.nunique() != 1:
            raise EvidenceNotComparable(["UNSUPPORTED_METRIC_PROTOCOL"])
        if variant is not None:
            frame = frame[frame.input_variant == variant].reset_index(drop=True)
    from harness.audit_gate import align_prediction_frames
    truth, _ = align_prediction_frames(truth, frame.drop(columns=[c for c in ("fold", "input_variant", "split") if c in frame and c not in truth]), [task_spec.id_column])
    # Keep original fold/variant for comparison identity; align only its row order.
    positions = {v: i for i, v in enumerate(frame[task_spec.id_column])}
    frame = frame.iloc[[positions[v] for v in truth[task_spec.id_column]]].reset_index(drop=True)
    identity = {k: manifest.get(k) for k in ("task_sha256", "scorer_sha256", "custom_scorer_sha256", "data_sha256", "split_sha256")}
    identity.update(partition=partition, coverage=digest(list(truth[task_spec.id_column])),
                    scope=digest(frame[[c for c in (task_spec.id_column, "fold", "input_variant", "split") if c in frame]].to_dict("list")),
                    truth_sha256=sha256(directory / "evaluation_truth.csv"),
                    class_order={c: frame[c].tolist() for c in frame if c.startswith("classes_")})
    return dict(reference=reference, directory=directory, manifest=manifest, cfg=cfg, truth=truth, frame=frame,
                plugin=plugin, context=context, score=report["recomputed"], identity=identity,
                evidence_sha256=digest(manifest), config_sha256=digest(cfg))


def register_decision_rule(rule, *, project, task_spec, plugin=None):
    project, rule = _project(project), _rule(rule)
    policy = resolve_research_policy(project, task_spec)
    _validate_rule_policy(rule, policy, task_spec)
    control = resolve_evidence(rule.control, project=project, task_spec=task_spec, plugin=plugin)
    registration = dict(rule_id="rule-" + uuid.uuid4().hex, rule=rule.to_dict(), rule_sha256=digest(rule.to_dict()),
                        registered_at=utc_now(), policy_sha256=digest(policy.to_dict()),
                        policy=policy.to_dict(),
                        control_evidence_sha256=control["evidence_sha256"], comparison_identity_sha256=digest(control["identity"]))
    directory = policy_directory(project)
    with workspace_lock(directory):
        with transaction(directory / "comparison_families.json", {"families": {}}) as state:
            family = state["families"].setdefault(rule.multiplicity["family_id"], {
                "max_comparisons": rule.multiplicity["max_comparisons"], "control_sha256": control["evidence_sha256"], "slots": {}})
            if family["max_comparisons"] != rule.multiplicity["max_comparisons"] or family["control_sha256"] != control["evidence_sha256"]:
                raise ValueError("Comparison family cannot change control or horizon")
        atomic_write_json(directory / (registration["rule_id"] + ".json"), registration)
    return registration


def load_registration(project, rule_id):
    if not isinstance(rule_id, str) or not rule_id.startswith("rule-") or not rule_id[5:].isalnum():
        raise ValueError("Invalid rule_id")
    value = json.loads((policy_directory(project) / (rule_id + ".json")).read_text(encoding="utf-8"))
    if value["rule_id"] != rule_id or digest(value["rule"]) != value["rule_sha256"]:
        raise ValueError("Corrupt rule registration")
    return value


def reserve_comparison_slot(project, task_spec, cfg, *, plugin=None):
    """Called by the runner before training, never by post-hoc decision notes."""
    research = cfg.get("research", {})
    rule_id = research.get("decision_rule_id")
    if not rule_id:
        return None
    project = _project(project)
    registration = load_registration(project, rule_id)
    if registration.get("probe_plan_id") and research.get("probe_plan_id") != registration["probe_plan_id"]:
        raise ValueError("research.probe_plan_id must match the registered rule's plan")
    rule = _rule(registration["rule"])
    policy = resolve_research_policy(project, task_spec)
    _validate_rule_policy(rule, policy, task_spec)
    if registration["policy_sha256"] != digest(policy.to_dict()):
        raise ValueError("Policy changed since preregistration; register a new probe")
    candidate_id = cfg["experiment"]["id"]
    manifest = project.artifacts_dir / candidate_id / "artifact_manifest.json"
    if manifest.exists() and json.loads(manifest.read_text())["status"] == "COMPLETED":
        raise ValueError("Cannot preregister a completed result")
    with transaction(policy_directory(project) / "comparison_families.json", {"families": {}}) as state:
        family = state["families"][rule.multiplicity["family_id"]]
        key = rule_id + ":" + candidate_id
        reservation = family["slots"].get(key)
        if reservation:
            if reservation["config_sha256"] != digest(cfg):
                raise ValueError("Reserved candidate configuration changed")
            return reservation
        if len(family["slots"]) >= family["max_comparisons"]:
            raise DecisionRuleNotSatisfied(["COMPARISON_BUDGET_EXHAUSTED"])
        reservation = dict(rule_id=rule_id, candidate_id=candidate_id, index=len(family["slots"]) + 1,
                           config_sha256=digest(cfg), reserved_at=utc_now(), probe_plan_id=research.get("probe_plan_id"))
        family["slots"][key] = reservation
        return reservation


def prepare_remote_research(project, config_path):
    """Reserve on the originating workspace before cloud dispatch.

    A small capsule travels with code/config, never the control's training data
    or sealed predictions. The canonical workspace evaluates imported results.
    """
    from harness.stagnation import StagnationController
    from harness.task_spec import load_active_task_spec
    project = _project(project)
    config_path = project.root / config_path
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    research = cfg.get("research", {})
    candidate_id = cfg["experiment"]["id"]
    from harness.macro_research import enforce_retrospective
    enforce_retrospective(project, cfg=cfg, recovery_probe=bool(research.get("recovery_probe_id")))
    blocker = StagnationController(project).check_action("run_full_validation",
        recovery_probe_id=research.get("recovery_probe_id"), candidate_id=candidate_id)
    if blocker:
        raise DecisionRuleNotSatisfied(["STAGNATION_PAUSE"])
    if not research.get("decision_rule_id"):
        return None
    spec = load_active_task_spec(project.task_config_path)
    reserve_comparison_slot(project, spec, cfg)
    registration = load_registration(project, research["decision_rule_id"])
    rule = _rule(registration["rule"])
    with transaction(policy_directory(project) / "comparison_families.json", {"families": {}}) as state:
        slot = state["families"][rule.multiplicity["family_id"]]["slots"][registration["rule_id"] + ":" + candidate_id]
        slot.update(remote=True, source_config_sha256=sha256(config_path), captured_research=research)
        capsule = dict(exp_id=candidate_id, source_config_sha256=slot["source_config_sha256"],
                       registration=registration, reservation=dict(slot))
    path = project.root / ".ai-harness/research_execution" / (candidate_id + ".json")
    atomic_write_json(path, capsule)
    return path


def validate_remote_research_capsule(project, config_path, cfg):
    """Validate the dispatched source config before remote training starts."""
    project = _project(project)
    if os.environ.get("AI_HARNESS_REMOTE_KERNEL") != "1" or not cfg.get("research", {}).get("decision_rule_id"):
        return False
    candidate_id = cfg["experiment"]["id"]
    path = project.root / ".ai-harness/research_execution" / (candidate_id + ".json")
    capsule = json.loads(path.read_text(encoding="utf-8"))
    registration = capsule["registration"]
    if (capsule["exp_id"] != candidate_id or capsule["source_config_sha256"] != sha256(config_path) or
        registration["rule_id"] != cfg["research"]["decision_rule_id"] or
        digest(registration["rule"]) != registration["rule_sha256"] or
        capsule["reservation"]["captured_research"] != cfg["research"]):
        raise ValueError("Remote research capsule/config binding mismatch")
    return True


def paired_metric_bootstrap(truth, candidate, control, scorer, *, direction="maximize", confidence_level=.95,
                            max_comparisons=1, resamples=10000, seed=42, groups=None, strata=None):
    """One-sided paired metric bootstrap, bounded memory, deterministic RNG.

    scorer(truth, predictions, sampled_indices) recomputes the actual metric.
    Whole groups are sampled together; stratification applies only to IID rows.
    """
    if (resamples < 1000 or max_comparisons < 1 or not .5 < confidence_level < 1 or
        direction not in {"maximize", "minimize"} or len(truth) != len(candidate) or len(truth) != len(control)):
        raise ValueError("Invalid paired bootstrap protocol")
    if groups is not None and strata is not None:
        raise ValueError("Row stratification cannot override dependent units")
    n = len(truth)
    if n < 2:
        raise ValueError("Insufficient comparison units")
    if groups is not None:
        labels = pd.Series(groups)
        if len(labels) != n or labels.isna().any():
            raise ValueError("Invalid group membership")
        units = [np.flatnonzero(labels.to_numpy() == value) for value in pd.unique(labels)]
    else:
        units = None
    if strata is not None:
        labels = pd.Series(strata)
        if len(labels) != n or labels.isna().any():
            raise ValueError("Invalid strata")
        strata_indices = [np.flatnonzero(labels.to_numpy() == value) for value in pd.unique(labels)]
    else:
        strata_indices = None
    rng, gains, invalid = np.random.default_rng(seed), [], 0
    for _ in range(resamples):
        if units is not None:
            indices = np.concatenate([units[i] for i in rng.integers(0, len(units), size=len(units))])
        elif strata_indices is not None:
            indices = np.concatenate([rng.choice(s, len(s), replace=True) for s in strata_indices])
        else:
            indices = rng.integers(0, n, n)
        try:
            a, b = scorer(truth, candidate, indices), scorer(truth, control, indices)
            gain = float(a - b if direction == "maximize" else b - a)
            if not math.isfinite(gain):
                raise ValueError("Undefined bootstrap metric")
            gains.append(gain)
        except (ValueError, KeyError, IndexError, ZeroDivisionError):
            invalid += 1
    alpha = (1 - confidence_level) / max_comparisons
    return dict(method="paired_bootstrap", unit="group" if groups is not None else "row", units=len(units) if units is not None else n,
                resamples=resamples, seed=seed, alpha=alpha, lower_bound=float(np.quantile(gains, alpha)) if gains else None,
                upper_bound=None, invalid_resamples=invalid)


def evaluate_decision_rule(rule, candidate, *, project, task_spec, plugin=None, registration=None):
    from harness.oof_revalidation import build_variant_predictions, score_predictions
    project, rule = _project(project), _rule(rule)
    candidate = asdict(candidate) if isinstance(candidate, EvidenceRef) else candidate
    policy = resolve_research_policy(project, task_spec)
    _validate_rule_policy(rule, policy, task_spec)
    receipt = DecisionEvaluationV1(schema_version=1, evaluation_id="evaluation-" + uuid.uuid4().hex,
        rule_id=registration["rule_id"] if registration else "retrospective", rule_sha256=digest(rule.to_dict()),
        policy_sha256=digest(policy.to_dict()), comparison_identity_sha256=None, candidate=candidate, control=rule.control,
        evidence_sha256=dict(candidate=None, control=None, truth=None), metric=rule.metric, direction=rule.direction,
        candidate_score=None, control_score=None, gain=None, minimum_effect=rule.minimum_effect,
        threshold_origin=policy.policy_id, uncertainty=None, comparison_slot=None, preregistered=False,
        independent_confirmation=False, checks=dict(identity="inconclusive", metric_gain="inconclusive", score_target="not_required" if rule.score_target is None else "inconclusive", uncertainty="inconclusive"),
        outcome="inconclusive", reason_codes=[], evaluated_at=utc_now())
    reasons = receipt["reason_codes"]
    try:
        a = resolve_evidence(candidate, project=project, task_spec=task_spec, plugin=plugin)
        b = resolve_evidence(rule.control, project=project, task_spec=task_spec, plugin=plugin)
        receipt["evidence_sha256"] = dict(candidate=a["evidence_sha256"], control=b["evidence_sha256"], truth=sha256(a["directory"] / "evaluation_truth.csv"))
        if a["identity"] != b["identity"] or candidate["partition"] != rule.control["partition"]:
            raise EvidenceNotComparable(["IDENTITY_MISMATCH"])
        receipt["comparison_identity_sha256"] = digest(a["identity"])
        receipt["checks"]["identity"] = "pass"
        verified_registration = load_registration(project, registration["rule_id"]) if registration else None
        if verified_registration and verified_registration != registration:
            raise EvidenceNotComparable(["NOT_PREREGISTERED"])
        if registration:
            if (registration["rule_sha256"] != receipt["rule_sha256"] or registration["policy_sha256"] != receipt["policy_sha256"] or
                registration["control_evidence_sha256"] != b["evidence_sha256"]):
                raise EvidenceNotComparable(["IDENTITY_MISMATCH"])
            families_path = policy_directory(project) / "comparison_families.json"
            families = json.loads(families_path.read_text(encoding="utf-8"))["families"]
            slot = families[rule.multiplicity["family_id"]]["slots"].get(registration["rule_id"] + ":" + candidate["id"])
            remote_matches = (slot and slot.get("remote") and
                a["manifest"].get("config_sha256") == slot.get("source_config_sha256") and
                a["cfg"].get("research", {}) == slot.get("captured_research"))
            if slot and (slot["config_sha256"] == a["config_sha256"] or remote_matches) and a["cfg"].get("research", {}).get("decision_rule_id") == registration["rule_id"]:
                receipt["preregistered"] = True
                receipt["comparison_slot"] = dict(family_id=rule.multiplicity["family_id"], index=slot["index"], max_comparisons=rule.multiplicity["max_comparisons"])
        if not receipt["preregistered"]:
            reasons.append("NOT_PREREGISTERED")
        # Confirmation is an explicitly contracted independent holdout, not an agent label.
        contracted_partition = _metadata(task_spec).get("validation", {}).get("partition")
        receipt["independent_confirmation"] = candidate["partition"] == "confirmation" and contracted_partition == "confirmation"
        if policy.require_confirmation_for_adaptive_search and not receipt["independent_confirmation"]:
            reasons.append("CONFIRMATION_REQUIRED")
        def selected_score(result):
            if rule.metric["scope"] == "primary":
                return float(result["primary_score"])
            target = rule.metric["subtask_id"]
            detail = result.get("subtasks", {}).get(target, result.get(target))
            if not isinstance(detail, dict) or rule.metric["name"] not in detail:
                if isinstance(detail, dict) and metric_name(detail.get("metric", "")) == rule.metric["name"] and "score" in detail:
                    return float(detail["score"])
                raise EvidenceNotComparable(["UNSUPPORTED_METRIC_PROTOCOL"])
            return float(detail[rule.metric["name"]])
        if rule.metric["scope"] == "primary":
            cs, bs = float(a["score"]), float(b["score"])
        else:
            scores = []
            for item in (a, b):
                result = score_predictions(task_spec, item["plugin"], item["context"], item["cfg"], item["truth"], build_variant_predictions(item["frame"], task_spec))
                scores.append(selected_score(result))
            cs, bs = scores
        gain = cs - bs if rule.direction == "maximize" else bs - cs
        if rule.minimum_effect["scale"] == "relative":
            if bs == 0:
                raise EvidenceNotComparable(["UNSUPPORTED_METRIC_PROTOCOL"])
            gain /= abs(bs)
        receipt.update(candidate_score=cs, control_score=bs, gain=gain)
        clears = gain >= rule.minimum_effect["value"] if rule.comparator == "gte" else gain > rule.minimum_effect["value"]
        if gain <= 0:
            reasons.append("ZERO_GAIN" if gain == 0 else "BELOW_MINIMUM_EFFECT")
        elif not clears:
            reasons.append("BELOW_MINIMUM_EFFECT")
        receipt["checks"]["metric_gain"] = "pass" if gain > 0 and clears else "fail"
        if rule.score_target:
            target = rule.score_target
            operators = {"gte": lambda x, y: x >= y, "gt": lambda x, y: x > y, "lte": lambda x, y: x <= y, "lt": lambda x, y: x < y}
            passes = operators[target["operator"]](cs, target["value"])
            receipt["checks"]["score_target"] = "pass" if passes else "fail"
            if not passes:
                reasons.append("SCORE_TARGET_FAILED")
        if receipt["checks"]["metric_gain"] == "fail" or receipt["checks"]["score_target"] == "fail":
            receipt["outcome"] = "rejected"
            validate_document("decision_evaluation", receipt)
            write_decision_evaluation(project, receipt)
            return receipt
        # Revalidate live evidence and registration above before reusing expensive uncertainty.
        cache_key = digest({k: receipt[k] for k in ("rule_sha256", "policy_sha256", "evidence_sha256", "candidate", "preregistered")})
        if cache_key in _EVALUATION_CACHE:
            # Only machine-computed values from this process are reusable as
            # results. A loaded JSON outcome/interval is never a computation.
            cached = copy.deepcopy(_EVALUATION_CACHE[cache_key])
            read_decision_evaluation(policy_directory(project) / (cached["evaluation_id"] + ".json"))
            return DecisionEvaluationV1(cached)
        uncertainty = rule.uncertainty
        if uncertainty["method"] != "paired_bootstrap":
            raise EvidenceNotComparable(["UNSUPPORTED_METRIC_PROTOCOL"])
        unit_column = uncertainty["unit_column"]
        stratify = uncertainty["stratify_column"]
        if unit_column and unit_column not in a["truth"] or stratify and stratify not in a["truth"]:
            raise EvidenceNotComparable(["UNSUPPORTED_METRIC_PROTOCOL"])
        if rule.metric["name"] == "roc_auc" and uncertainty["unit"] == "row" and not stratify:
            stratify = task_spec.primary_target
        def scorer(truth, frame, indices):
            sampled_truth, sampled_frame = truth.iloc[indices].copy(), frame.iloc[indices].copy()
            sampled_truth[task_spec.id_column] = np.arange(len(indices))
            sampled_frame[task_spec.id_column] = np.arange(len(indices))
            result = score_predictions(task_spec, a["plugin"], a["context"], a["cfg"], sampled_truth,
                                       build_variant_predictions(sampled_frame, task_spec))
            return selected_score(result)
        # Exact additive metrics avoid rebuilding DataFrames and unrelated metric
        # diagnostics on every resample. Custom scorers and routing keep the full
        # plugin adapter. Verify the fast adapter against both recomputed scores.
        from harness.plugins import TaskPlugin
        active = [s for s in task_spec.subtasks.values() if s.weight > 0]
        if (type(a["plugin"]).evaluate_predictions is TaskPlugin.evaluate_predictions and
            not a["cfg"].get("paths", {}).get("metrics_evaluator") and len(active) == 1 and
            not active[0].column_prefix and rule.metric["scope"] == "primary" and rule.metric["name"] in {"accuracy", "rmse", "mae"}):
            sub = active[0]
            mask = task_spec.compute_evaluation_masks(a["truth"])[sub.id]
            if np.asarray(mask).all():
                y = a["truth"][sub.column or sub.id].to_numpy()
                prepared = {}
                from harness.evaluation import _normalize_class_array
                for item in (a, b):
                    mapping = build_variant_predictions(item["frame"], task_spec)
                    p = np.asarray(mapping[f"pred_{sub.id}"])
                    if rule.metric["name"] == "accuracy":
                        losses = (_normalize_class_array(y) == _normalize_class_array(p)).astype(float)
                    else:
                        losses = (y.astype(float) - p.astype(float)) ** 2 if rule.metric["name"] == "rmse" else np.abs(y.astype(float) - p.astype(float))
                    prepared[id(item["frame"])] = losses
                    check_score = float(np.mean(losses))
                    if rule.metric["name"] == "rmse":
                        check_score = math.sqrt(check_score)
                    if not math.isclose(check_score, item["score"], rel_tol=1e-9, abs_tol=1e-12):
                        prepared = {}
                        break
                if prepared:
                    def scorer(truth, frame, indices):
                        value = float(np.mean(prepared[id(frame)][indices]))
                        return math.sqrt(value) if rule.metric["name"] == "rmse" else value
        interval = paired_metric_bootstrap(a["truth"], a["frame"], b["frame"], scorer, direction=rule.direction,
            confidence_level=uncertainty["confidence_level"], max_comparisons=rule.multiplicity["max_comparisons"],
            resamples=uncertainty["resamples"], seed=uncertainty["seed"],
            groups=a["truth"][unit_column] if unit_column else None,
            strata=a["truth"][stratify] if stratify and not unit_column else None)
        interval["unit"] = uncertainty["unit"]
        receipt["uncertainty"] = interval
        if interval["lower_bound"] is not None and interval["lower_bound"] > 0 and not interval["invalid_resamples"]:
            receipt["checks"]["uncertainty"] = "pass"
        else:
            reasons.append("UNCERTAINTY_NOT_CLEARED")
            receipt["checks"]["uncertainty"] = "fail"
        if receipt["checks"]["metric_gain"] == "fail" or receipt["checks"]["score_target"] == "fail":
            receipt["outcome"] = "rejected"
        elif not reasons:
            receipt["outcome"] = "supported"
    except EvidenceNotComparable as exc:
        reasons.extend(exc.reason_codes)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        reasons.append("EVIDENCE_MISSING")
        # Preserve a useful diagnostic outside the schema-bound receipt.
        if isinstance(exc, DecisionRuleNotSatisfied):
            reasons.extend(exc.reason_codes)
    receipt["reason_codes"] = list(dict.fromkeys(reasons))
    validate_document("decision_evaluation", receipt)
    write_decision_evaluation(project, receipt)
    return receipt


def write_decision_evaluation(project, receipt):
    validate_document("decision_evaluation", receipt)
    directory = policy_directory(project)
    key = digest({k: receipt[k] for k in ("rule_sha256", "policy_sha256", "evidence_sha256", "candidate", "preregistered")})
    with workspace_lock(directory):
        index_path = directory / "evaluations.json"
        with transaction(index_path, {"evaluations": {}}) as state:
            previous = state["evaluations"].get(key)
            if previous:
                old = read_decision_evaluation(directory / (previous + ".json"))

                def scientific(value):
                    return {k: v for k, v in value.items() if k not in {"evaluation_id", "evaluated_at"}}

                if scientific(old) != scientific(receipt):
                    raise ValueError("DECISION_EVALUATION_INTEGRITY: receipt does not match live computation")
                receipt.clear()
                receipt.update(old)
            else:
                receipt["evaluation_id"] = "evaluation-" + digest({k: v for k, v in receipt.items() if k != "evaluation_id"})
                atomic_write_json(directory / (receipt["evaluation_id"] + ".json"), dict(receipt))
                state["evaluations"][key] = receipt["evaluation_id"]
            _EVALUATION_CACHE[key] = copy.deepcopy(dict(receipt))
    from harness.stagnation import StagnationController
    states = StagnationController(project).reconcile_results([receipt])
    for state in states.values():
        if state["state"] != "paused_for_stagnation":
            continue
        key = state["comparison_identity_sha256"] + ":" + str(state["generation"])
        with transaction(directory / "pause_checkpoints.json", {"recorded": []}) as checkpoints:
            if key in checkpoints["recorded"]:
                continue
            checkpoints["recorded"].append(key)
        from harness.research_memory import ResearchMemory
        memory = ResearchMemory(project_root=_project(project).root)
        memory.checkpoint(summary="STAGNATION_PAUSE: " + str(state["consecutive_failures"]) + " comparable candidates missed the promotion rule.",
            next_action="Diagnose the failures and register one bounded recovery probe; submissions remain paused.")


def effective_outcome(entry):
    if entry.get("outcome") == "supported" and entry.get("verification_status") != "machine_evaluated":
        return "inconclusive"
    return entry.get("effective_outcome", entry.get("outcome", "inconclusive"))
