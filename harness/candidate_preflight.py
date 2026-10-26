"""Preflight verification of candidate predictions before costly runs or submissions.

Enforces epistemic grounding:
- Computes Spearman rho and Pearson r against incumbent or baseline.
- Measures rank swaps and distribution of absolute probability/score deltas.
- Flags pseudo-orthogonal micro-tinkering (rho >= 0.995).
- Emits structured PreflightReceipt validated against JSON schema.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats

from harness.artifacts import sha256
from harness.audit_gate import semantic_prediction_hash
from harness.io_utils import atomic_write_json
from harness.policy_schema import validate_document
from harness.project import HarnessProject


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def compute_prediction_similarity(
    cand_preds: np.ndarray,
    inc_preds: np.ndarray,
) -> Dict[str, Any]:
    """Calculate rank correlation, linear correlation, and delta distribution."""
    cand = np.asarray(cand_preds, dtype=np.float64)
    inc = np.asarray(inc_preds, dtype=np.float64)

    if cand.shape != inc.shape or len(cand) == 0:
        raise ValueError(f"Shape mismatch or empty arrays: cand={cand.shape}, inc={inc.shape}")

    # Spearman rank correlation
    s_corr, _ = stats.spearmanr(cand, inc)
    if np.isnan(s_corr):
        # Constant array
        s_corr = 1.0 if np.allclose(cand, inc) else 0.0

    # Pearson correlation
    p_corr, _ = stats.pearsonr(cand, inc)
    if np.isnan(p_corr):
        p_corr = 1.0 if np.allclose(cand, inc) else 0.0

    diffs = np.abs(cand - inc)
    mean_delta = float(np.mean(diffs))
    max_delta = float(np.max(diffs))
    p95_delta = float(np.percentile(diffs, 95))

    # Fast estimate of rank swaps (discordant pairs)
    # Using sample or full rank comparison
    n = len(cand)
    rank_cand = stats.rankdata(cand)
    rank_inc = stats.rankdata(inc)
    # For performance on large arrays (e.g. 300k), estimate rank swap ratio via Kendall tau
    # Kendall tau = (concordant - discordant) / (n*(n-1)/2)
    # rank_swap_ratio = (1 - tau) / 2
    if n > 5000:
        # Sample for speed on very large arrays
        sub_idx = np.random.RandomState(42).choice(n, size=5000, replace=False)
        tau, _ = stats.kendalltau(rank_cand[sub_idx], rank_inc[sub_idx])
    else:
        tau, _ = stats.kendalltau(rank_cand, rank_inc)

    if np.isnan(tau):
        tau = 1.0 if np.allclose(cand, inc) else 0.0

    swap_ratio = float(np.clip((1.0 - tau) / 2.0, 0.0, 1.0))
    # Estimated total discordant pairs
    total_pairs = (n * (n - 1)) // 2
    swap_count = int(round(swap_ratio * total_pairs))

    return {
        "spearman_rho": float(s_corr),
        "pearson_r": float(p_corr),
        "mean_abs_delta": mean_delta,
        "max_abs_delta": max_delta,
        "p95_abs_delta": p95_delta,
        "rank_swap_count": swap_count,
        "rank_swap_ratio": swap_ratio,
    }


def resolve_incumbent_prediction(project: HarnessProject) -> Optional[Path]:
    """Auto-resolve the incumbent prediction file from control OOF or best submission."""
    # 1. Check control OOF
    ctrl_oof = project.artifacts_dir / "control" / "oof_predictions.csv"
    if ctrl_oof.is_file():
        return ctrl_oof

    # 2. Check submissions registry
    reg_path = project.artifacts_dir / "submissions" / "registry.json"
    if reg_path.is_file():
        try:
            records = json.loads(reg_path.read_text(encoding="utf-8")).get("submissions", [])
            scored = [r for r in records if r.get("status") in {"SCORED", "SUBMITTED"} and r.get("candidate_path")]
            if scored:
                # Return the highest scored or latest
                return Path(scored[-1]["candidate_path"])
        except Exception:
            pass

    return None


def run_candidate_preflight(
    project_root: str | Path,
    candidate_path: str | Path,
    incumbent_path: Optional[str | Path] = None,
    id_col: Optional[str] = None,
    pred_col: Optional[str] = None,
) -> Dict[str, Any]:
    """Perform preflight analysis comparing candidate predictions to an incumbent baseline."""
    if not candidate_path:
        raise ValueError("candidate_path is required for candidate preflight verification.")
    project = HarnessProject.discover(Path(project_root).resolve())
    cand_p = Path(candidate_path)
    if not cand_p.is_absolute():
        cand_p = (project.root / cand_p).resolve()

    if not cand_p.is_file():
        raise FileNotFoundError(f"Candidate prediction file not found: {cand_p}")

    inc_p = Path(incumbent_path) if incumbent_path else resolve_incumbent_prediction(project)
    if inc_p and not inc_p.is_absolute():
        inc_p = (project.root / inc_p).resolve()

    if not inc_p or not inc_p.is_file():
        raise FileNotFoundError(
            f"Incumbent prediction file could not be resolved (target: {inc_p}). "
            "Please provide an explicit incumbent_path."
        )

    # Read CSVs
    df_cand = pd.read_csv(cand_p)
    df_inc = pd.read_csv(inc_p)

    # Determine ID column
    resolved_id_col = id_col
    if not resolved_id_col:
        for col in ("id", "ID", "Id", "row_id", "sample_id"):
            if col in df_cand.columns and col in df_inc.columns:
                resolved_id_col = col
                break

    if resolved_id_col:
        df_cand = df_cand.sort_values(resolved_id_col).reset_index(drop=True)
        df_inc = df_inc.sort_values(resolved_id_col).reset_index(drop=True)
        if not (df_cand[resolved_id_col].values == df_inc[resolved_id_col].values).all():
            # Align by ID
            common_ids = set(df_cand[resolved_id_col]) & set(df_inc[resolved_id_col])
            if not common_ids:
                raise ValueError("No common IDs found between candidate and incumbent!")
            df_cand = df_cand[df_cand[resolved_id_col].isin(common_ids)].sort_values(resolved_id_col).reset_index(drop=True)
            df_inc = df_inc[df_inc[resolved_id_col].isin(common_ids)].sort_values(resolved_id_col).reset_index(drop=True)

    if len(df_cand) != len(df_inc):
        raise ValueError(f"Row count mismatch after alignment: cand={len(df_cand)}, inc={len(df_inc)}")

    # Determine Prediction column
    resolved_pred_col = pred_col
    if not resolved_pred_col:
        cand_cols = [c for c in df_cand.columns if c != resolved_id_col]
        inc_cols = [c for c in df_inc.columns if c != resolved_id_col]
        if len(cand_cols) == 1 and len(inc_cols) == 1:
            resolved_pred_col = cand_cols[0]
            inc_pred_col = inc_cols[0]
        else:
            for c in ("satisfaction", "target", "label", "pred", "prediction", "probability"):
                if c in cand_cols and c in inc_cols:
                    resolved_pred_col = c
                    inc_pred_col = c
                    break
            if not resolved_pred_col:
                resolved_pred_col = cand_cols[0]
                inc_pred_col = inc_cols[0]
    else:
        inc_pred_col = resolved_pred_col

    y_cand = df_cand[resolved_pred_col].to_numpy(dtype=np.float64)
    y_inc = df_inc[inc_pred_col].to_numpy(dtype=np.float64)

    similarity = compute_prediction_similarity(y_cand, y_inc)
    rho = similarity["spearman_rho"]

    # Epistemic classification
    if rho >= 0.999:
        classification = "TRIVIAL_PERTURBATION"
        verdict = "FLAGGED_NEAR_DUPLICATE"
        delusion_warning = (
            f"CRITICAL DELUSION WARNING: Spearman rho = {rho:.6f} (>= 0.999). "
            "This candidate is mathematically nearly identical to the incumbent. "
            "Score variations on public leaderboard are pure sampling noise. "
            "Submitting this wastes daily quota without testing a distinct scientific mechanism."
        )
        recommendation = (
            "REJECT / HALT: Do not submit this candidate. Explore an untouched research family "
            "(e.g. Target Transformation, Calibration, or Neural Architecture) instead of micro-tinkering."
        )
    elif rho >= 0.995:
        classification = "MINOR_REWEIGHTING"
        verdict = "FLAGGED_NEAR_DUPLICATE"
        delusion_warning = (
            f"WARNING: Spearman rho = {rho:.6f} (>= 0.995). "
            "Candidate exhibits very high rank correlation with incumbent. "
            "Only submit if an independent, predeclared paired metric decision rule establishes verified gain."
        )
        recommendation = (
            "CAUTION: Preflight flagged near-duplicate. Requires verified paired OOF gain before submission."
        )
    else:
        classification = "DIVERSE_CANDIDATE"
        verdict = "PASS"
        delusion_warning = None
        recommendation = (
            f"PASS: Candidate exhibits healthy behavioral diversity (Spearman rho = {rho:.4f}, "
            f"rank swap ratio = {similarity['rank_swap_ratio']:.2%})."
        )

    try:
        from harness.task_spec import load_active_task_spec
        spec = load_active_task_spec(project.task_config_path)
        sem_cand = semantic_prediction_hash(df_cand, spec)
        sem_inc = semantic_prediction_hash(df_inc, spec)
    except Exception:
        import hashlib
        sem_cand = hashlib.sha256(df_cand.to_csv(index=False).encode("utf-8")).hexdigest()
        sem_inc = hashlib.sha256(df_inc.to_csv(index=False).encode("utf-8")).hexdigest()

    receipt = {
        "schema_version": 1,
        "preflight_id": "preflight-" + uuid.uuid4().hex,
        "candidate": {
            "path": str(cand_p),
            "sha256": sha256(cand_p),
            "semantic_hash": sem_cand,
            "row_count": len(df_cand),
        },
        "incumbent": {
            "path": str(inc_p),
            "sha256": sha256(inc_p),
            "semantic_hash": sem_inc,
            "row_count": len(df_inc),
        },
        "similarity": similarity,
        "classification": classification,
        "delusion_warning": delusion_warning,
        "recommendation": recommendation,
        "verdict": verdict,
        "created_at": _utc_now_iso(),
    }

    validate_document("candidate_preflight_receipt", receipt)

    # Persist receipt
    out_dir = project.root / ".ai-harness" / "preflights"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{receipt['preflight_id']}.json"
    atomic_write_json(out_path, receipt)

    return receipt
