"""Epistemic Grounding & Assumption Register Validator (Anti-Outdated & Anti-Hallucination Gate).

Inspired by:
1. Microsoft RD-Agent: Separating Research Grounding from Code Execution.
2. Edwards & Schuster (2026): 'Ask or Assume? Uncertainty-aware clarification-seeking in coding agents'.
3. OpenAI PaperBench & MLE-bench: Objective verification rubrics against empirical research.

Enforces physical code assertions so the agent cannot rely on stale parametric memory,
recycled classical defaults (e.g. pseudo-ensembling, arbitrary tuning), or unverified assumptions.
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class EpistemicGroundingError(Exception):
    """Raised when an experiment fails epistemic grounding or assumption verification."""
    pass


class GroundingGate:
    """Deterministic circuit breaker that stops experiments lacking grounded evidence."""

    VALID_SOURCE_TYPES = {
        "huggingface_hub",
        "arxiv",
        "paperswithcode",
        "empirical_journal",
        "verified_baseline",
    }

    VALID_UNCERTAINTY_LEVELS = {"low", "medium", "high"}

    # Banned / Lazy intervention patterns that indicate trial-and-error or hallucinated certainty
    LAZY_INTERVENTION_PATTERNS = [
        r"^(tune|tuning|adjust|adjusting|tweak|tweaking)\s+(parameters?|hyperparameters?)$",
        r"^(try|test)\s+different\s+(things|models|parameters)$",
        r"^(simple|basic|naive)\s+(ensemble|blending|averaging)$",
        r"^random\s+search$",
        r"^(tweak|modify)\s+prompt\s+slightly$",
    ]

    @classmethod
    def check_lazy_intervention(cls, intervention: str) -> Optional[str]:
        """Check if intervention description matches known lazy or ungrounded trial-and-error patterns."""
        normalized = intervention.strip().lower()
        for pattern in cls.LAZY_INTERVENTION_PATTERNS:
            if re.match(pattern, normalized):
                return (
                    f"Intervention '{intervention}' matches lazy trial-and-error pattern '{pattern}'. "
                    f"Agent must declare a specific, hypothesis-driven, atomic intervention."
                )
        return None

    @classmethod
    def verify_config(
        cls,
        config: Dict[str, Any],
        config_path: Optional[str | Path] = None,
        raise_on_error: bool = True,
    ) -> Tuple[bool, List[str], Dict[str, Any]]:
        """Validate config for empirical grounding and explicit assumptions.

        Args:
            config: Full experiment configuration dictionary.
            config_path: Optional path to YAML file for diagnostic messages.
            raise_on_error: Whether to raise EpistemicGroundingError if checks fail.

        Returns:
            Tuple of (passed: bool, errors: list[str], receipt_data: dict).
        """
        errors: List[str] = []
        cfg_name = str(config_path) if config_path else "in-memory config"

        exp_meta = config.get("experiment", {})
        exp_id = exp_meta.get("id", "unnamed_experiment")
        branch = exp_meta.get("branch", "unknown")
        parent_id = str(exp_meta.get("parent_id", "")).strip().lower()
        is_root_baseline = branch == "baseline" and (parent_id in ["none", "", "null"])

        # 1. Inspect Experiment Metadata
        problem = exp_meta.get("problem_addressed", "")
        if not problem or len(str(problem).strip()) < 15:
            errors.append(
                f"[EXPERIMENT SPECIFICATION ERROR] 'problem_addressed' must be substantive (>=15 chars), "
                f"directly referencing a documented bottleneck from DIAGNOSTICS.md or past runs."
            )

        hypothesis = exp_meta.get("hypothesis", "")
        if not hypothesis or len(str(hypothesis).strip()) < 20:
            errors.append(
                f"[EXPERIMENT SPECIFICATION ERROR] 'hypothesis' must clearly state expected mechanism "
                f"and quantitative target delta (>=20 chars)."
            )

        intervention = exp_meta.get("intervention", "")
        if not intervention or len(str(intervention).strip()) < 10:
            errors.append(
                f"[EXPERIMENT SPECIFICATION ERROR] 'intervention' must clearly define Control vs Treatment."
            )
        else:
            lazy_err = cls.check_lazy_intervention(str(intervention))
            if lazy_err:
                errors.append(f"[LAZY INTERVENTION REJECTED] {lazy_err}")

        # 2. Inspect Grounding Section (Anti-Outdated Knowledge Guard)
        grounding = config.get("grounding")
        if not grounding:
            if is_root_baseline:
                try:
                    from harness.task_spec import load_active_task_spec
                    spec = load_active_task_spec()
                    task_name = spec.name
                except Exception:
                    task_name = "competition"
                # Provide standard fallback for root baseline if not declared
                grounding = {
                    "source_type": "verified_baseline",
                    "source_id": "sklearn_tfidf_logreg_canonical",
                    "retrieval_summary": f"Initial canonical baseline for {task_name} invariant control.",
                }
            else:
                errors.append(
                    f"[EPISTEMIC GROUNDING ERROR] Missing required 'grounding' section in config {cfg_name}.\n"
                    f"Agent is physically forbidden from proposing models/techniques without empirical grounding.\n"
                    f"You MUST verify modern literature (Hugging Face Hub, ArXiv, docs/KNOWLEDGE_BASE.md) and specify:\n"
                    f"  grounding:\n"
                    f"    source_type: 'huggingface_hub' | 'arxiv' | 'paperswithcode' | 'empirical_journal'\n"
                    f"    source_id: '<model_id_or_paper_id>'\n"
                    f"    retrieval_summary: '<explanation of why this architecture solves the diagnosed bottleneck>'\n"
                    f"    verified_via: 'search_web' | 'read_url_content' | 'mcp_gemini-docs' | 'local_registry'"
                )

        if grounding:
            stype = grounding.get("source_type")
            if stype not in cls.VALID_SOURCE_TYPES:
                errors.append(
                    f"[GROUNDING TYPE ERROR] Invalid source_type '{stype}'. "
                    f"Must be one of {sorted(cls.VALID_SOURCE_TYPES)}."
                )

            sid = grounding.get("source_id", "")
            if not sid or len(str(sid).strip()) < 3:
                errors.append(
                    f"[GROUNDING SOURCE ERROR] 'source_id' must specify a valid model ID, paper ID, or benchmark key."
                )

            summary = grounding.get("retrieval_summary", "")
            if not summary or len(str(summary).strip()) < 15:
                errors.append(
                    f"[GROUNDING RATIONALE ERROR] 'retrieval_summary' must explain empirical justification (>=15 chars)."
                )

        # 3. Inspect Assumption Register (According to 'Ask or Assume?', Edwards & Schuster 2026)
        assumption_reg = config.get("assumption_register") or exp_meta.get("assumption_register")
        if not assumption_reg:
            if is_root_baseline:
                try:
                    from harness.task_spec import load_active_task_spec
                    spec = load_active_task_spec()
                    task_name = spec.name
                except Exception:
                    task_name = "competition"
                assumption_reg = {
                    "known_facts": [f"{task_name} dataset validated through active CompetitionSpec."],
                    "assumptions": ["Standard baseline provides an empirical anchor for downstream iterations."],
                    "uncertainty_level": "low",
                }
            else:
                errors.append(
                    f"[ASSUMPTION REGISTER MISSING] Config must define 'assumption_register' to decouple known facts "
                    f"from speculative hypotheses:\n"
                    f"  assumption_register:\n"
                    f"    known_facts:\n"
                    f"      - '<fact 1 with metrics from parent_id>'\n"
                    f"    assumptions:\n"
                    f"      - '<explicit assumption 1>'\n"
                    f"    uncertainty_level: 'low' | 'medium' | 'high'\n"
                    f"    verification_probe: '<how assumption will be verified before heavy full-scale training>'"
                )

        if assumption_reg:
            known_facts = assumption_reg.get("known_facts", [])
            if not isinstance(known_facts, list) or len(known_facts) == 0:
                errors.append("[ASSUMPTION REGISTER ERROR] 'known_facts' must contain at least 1 verified fact.")

            assumptions = assumption_reg.get("assumptions", [])
            if not isinstance(assumptions, list) or len(assumptions) == 0:
                errors.append("[ASSUMPTION REGISTER ERROR] 'assumptions' must contain at least 1 explicit assumption.")

            uncertainty = str(assumption_reg.get("uncertainty_level", "medium")).lower()
            if uncertainty not in cls.VALID_UNCERTAINTY_LEVELS:
                errors.append(
                    f"[ASSUMPTION REGISTER ERROR] 'uncertainty_level' must be in {sorted(cls.VALID_UNCERTAINTY_LEVELS)}."
                )
            elif uncertainty == "high":
                probe = assumption_reg.get("verification_probe")
                if not probe or len(str(probe).strip()) < 10:
                    errors.append(
                        f"[HIGH UNCERTAINTY FLAGGED] Experiments with uncertainty_level='high' MUST declare a "
                        f"'verification_probe' (e.g. '--debug --fold 0' smoke test or mini-probe)."
                    )

        # 4. Anti-Pseudo-Ensemble Guard for 'ens' branch
        if branch == "ens":
            ens_components = config.get("ensemble", {}).get("components", [])
            if len(ens_components) < 2:
                errors.append(
                    f"[ENSEMBLE POLICY VIOLATION] Branch 'ens' requires at least 2 distinct constituent model paths "
                    f"under 'ensemble.components'."
                )

        # Construct Audit Receipt
        receipt = {
            "receipt_id": f"GROUNDING_{exp_id}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "experiment_id": exp_id,
            "branch": branch,
            "parent_id": parent_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "status": "PASSED" if not errors else "FAILED",
            "grounding": grounding,
            "assumption_register": assumption_reg,
            "error_count": len(errors),
            "errors": errors,
        }

        if errors and raise_on_error:
            formatted_errors = "\n  - " + "\n  - ".join(errors)
            raise EpistemicGroundingError(
                f"\n{'='*70}\n"
                f"[CIRCUIT BREAKER: EPISTEMIC GROUNDING GATE TRIGGERED]\n"
                f"Experiment '{exp_id}' failed pre-flight grounding validation:\n"
                f"{formatted_errors}\n\n"
                f"ACTION REQUIRED:\n"
                f"1. Do not guess or recycle outdated classical tricks.\n"
                f"2. Inspect docs/KNOWLEDGE_BASE.md or query external tools (search_web / read_url_content).\n"
                f"3. Populate 'grounding' and 'assumption_register' in {cfg_name}.\n"
                f"{'='*70}"
            )

        return (len(errors) == 0, errors, receipt)

    @classmethod
    def write_receipt(cls, receipt: Dict[str, Any], output_path: str | Path) -> None:
        """Write JSON grounding receipt to disk atomically."""
        from harness.io_utils import atomic_write_json
        atomic_write_json(output_path, receipt, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Epistemic Grounding & Assumption Register CLI Gate")
    parser.add_argument("--config", required=True, help="Path to experiment YAML configuration")
    parser.add_argument("--output-receipt", help="Optional path to output GROUNDING_RECEIPT.json")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file '{config_path}' not found.", file=sys.stderr)
        sys.exit(1)

    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    try:
        passed, errors, receipt = GroundingGate.verify_config(cfg, config_path=config_path, raise_on_error=False)
        if passed:
            print(f"SUCCESS: Epistemic Grounding Gate PASSED for '{receipt['experiment_id']}'.")
            print(f"  Source Type: {receipt['grounding'].get('source_type')}")
            print(f"  Source ID:   {receipt['grounding'].get('source_id')}")
            print(f"  Uncertainty: {receipt['assumption_register'].get('uncertainty_level')}")
            if args.output_receipt:
                GroundingGate.write_receipt(receipt, args.output_receipt)
                print(f"  Receipt written to: {args.output_receipt}")
            sys.exit(0)
        else:
            print(f"FAILED: Grounding Gate rejected config with {len(errors)} error(s):", file=sys.stderr)
            for err in errors:
                print(f"  - {err}", file=sys.stderr)
            if args.output_receipt:
                GroundingGate.write_receipt(receipt, args.output_receipt)
            sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
