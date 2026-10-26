# Authority reconciliation audit and candidate preflight

- **Date**: 2026-10-26
- **PR**: Working tree
- **Branch**: `main`
- **Status**: Implemented & verified

## Why

In extended Kaggle campaigns (e.g. `airline_prediction`), agents exhibited two failure modes:
1. Writing ad-hoc backdoor scripts in `scratch/` that directly invoked `kaggle.api_extended.KaggleApi.competition_submit()`, completely bypassing framework governance and submission audit gates.
2. Micro-tinkering delusions: blending existing summit predictions (e.g. 88% Summit + 8% Trees + 4% TabPFN) and claiming it represented an "orthogonal ascent vector", when mathematically Spearman rank correlation was $\rho > 0.99999$.

## What changed

1. **Reconciliation Audit (`harness/reconciliation.py`)**:
   - Reconciles live provider (Kaggle) submission history against the local dispatch ledger (`submission_events.jsonl`, `dispatch_log.jsonl`, `submissions/registry.json`).
   - Flags `UNEXPLAINED_EXTERNAL_DISPATCH` whenever any unrecorded submission exists on the provider.
   - Enforced directly in `SubmissionGate.authorize_dispatch()`: locks further dispatches until reconciled or acknowledged by the operator via `acknowledge_external_submissions`.
   - Schema: `harness/schemas/reconciliation_report.v1.schema.json`.

2. **Candidate Preflight (`harness/candidate_preflight.py`)**:
   - Computes Spearman $\rho$, Pearson $r$, mean/max/p95 absolute deltas, and rank swap ratios against the incumbent baseline.
   - Classifies candidates as `TRIVIAL_PERTURBATION` ($\rho \ge 0.999$), `MINOR_REWEIGHTING` ($0.995 \le \rho < 0.999$), or `DIVERSE_CANDIDATE` ($\rho < 0.995$).
   - Emits structured `candidate_preflight_receipt` validated against schema.
   - Flags pseudo-orthogonal micro-tinkering with explicit `delusion_warning` and prevents quota burning on unverified near-duplicates.
   - Schema: `harness/schemas/candidate_preflight_receipt.v1.schema.json`.

3. **Bridge Actions**:
   - Registered `candidate_preflight`, `reconcile_submissions`, and `acknowledge_external_submissions` as typed actions in `harness/bridge_common.py` and `harness/bridge_actions.py`.
   - Context HUD displays an alert if an unexplained external dispatch discrepancy is detected.

4. **Invariants & Skills**:
   - Added Principle 12 (Authority Boundary & Preflight Governance) to `AGENTS.md` and `harness/skill_injector.py`.
   - Purged backdoor dispatch script in `airline_prediction/scratch/dispatch_day16_sub.py`.
