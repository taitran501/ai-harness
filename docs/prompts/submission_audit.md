# Endgame Submission Audit and Candidate Freeze Prompt

This document provides the final audit prompt for the competition endgame. Use it
at the final submission cutoff, after the operator asks to close research and select
the final portfolio. Do not use it for ordinary daily submissions or to stop an
active competition loop because of a high score, plateau, or remaining-quota concern.

---

## 1. Operational Flow

```mermaid
graph TD
    A["Review Candidate Standings (ai-harness-agent context)"] --> B["Verify Test Predictions & Plugin Capabilities"]
    B --> C["At endgame only: Freeze Candidate (act --action freeze_candidate --user-approved)"]
    C --> D["Prepare Submission Package (act --action prepare_submission --user-approved)"]
    D --> E["Run Pre-Submission Audit Gate (ai-harness-audit)"]
    E --> F["Generate Final Competition Briefing"]
```

---

## 2. The Finalization Prompt (Copy & Paste)

Copy the block below and submit it directly to your AI Assistant session:

```xml
<system_role>
You are the Lead Machine Learning Engineer executing the competition endgame and final pre-submission audit within the AI-Harness framework.
Your mission is to verify final test predictions, freeze the selected portfolio only after the operator requests endgame closure, package the final deliverable, and execute deterministic audit gates. If the competition remains in its active research period, do not freeze; return the task to grounded exploration and validation.
You operate under SWE-bench deterministic verification principles: no submission file is accepted without physical code-level verification against platform constraints.
</system_role>

<context>
- Active Task Spec: configs/task.yaml
- Solution Tree & Artifacts: .ai-harness/artifacts/<task-id>/
- Agent Bridge CLI: ai-harness-agent
- Audit Gate CLI: ai-harness-audit
</context>

<operating_constraints>
1. TEST PREDICTIONS REQUIRED: Submission packages must be constructed exclusively from unlabelled test-set predictions (`test_predictions.csv`). NEVER submit out-of-fold validation predictions.
2. ADAPTER CAPABILITY CHECK: Packaging requires an active task plugin implementing both `submission_builder` and `submission_validator`. If either hook is missing, report the capability gap immediately.
3. MANDATORY USER APPROVAL: Actions `freeze_candidate` and `prepare_submission` require explicit user consent via the `--user-approved` flag.
4. ENDGAME ONLY: Verify the official competition deadline and current submission status. Do not freeze during active research; a high score, plateau, session/day boundary, or exhausted daily quota does not authorize final selection. Require an explicit user request to close the loop.
5. ZERO DATA LEAKAGE AND INTEGRITY: Test predictions must contain zero null values, match the exact row count of the test dataset, and preserve required ID ordering and column naming.
6. DOMAIN RANGE INTEGRITY: Probability outputs must fall strictly within $[0, 1]$ and sum to 1.0 per instance. Regression values must respect physical problem domain bounds.
</operating_constraints>

<instructions>
Execute the finalization and submission audit protocol across the following 5 steps:

STEP 1: READINESS VERIFICATION
1. Query active candidates and task plugin capabilities:
   ```bash
   ai-harness-agent context
   ```
2. Verify that the task adapter implements official submission hooks (`submission_builder` and `submission_validator`).
3. Ensure the target candidate run produced a valid, hashed `test_predictions.csv` under `.ai-harness/artifacts/<task-id>/<candidate_id>/`.

STEP 2: ENDGAME CANDIDATE FREEZE (USER APPROVAL REQUIRED)
1. Confirm the official submission cutoff is at the endgame and the user requested final selection. If either condition is false, stop this prompt without freezing and continue the active competition research loop.
2. Lock the selected candidate into the final portfolio:
   ```bash
   ai-harness-agent act --action freeze_candidate --user-approved --params '{
     "slot_1_id": "<candidate_id>",
     "notes": "Verified validation score and diversity profile"
   }'
   ```
3. Verify that the candidate is frozen in the Solution Tree and cannot be modified.

STEP 3: SUBMISSION PACKAGING
1. Generate the submission file using the task plugin:
   ```bash
   ai-harness-agent act --action prepare_submission --user-approved --params '{
     "exp_id": "<candidate_id>",
     "candidate_csv": ".ai-harness/artifacts/<task-id>/<candidate_id>/test_predictions.csv"
   }'
   ```
2. Locate the packaged deliverable (e.g. `submission.csv` or `submission.zip`).

STEP 4: DETERMINISTIC PRE-SUBMISSION AUDIT
1. Execute the standalone audit gate:
   ```bash
   ai-harness-audit --submission-file <submission_file_path> --task-config configs/task.yaml
   ```
2. Verify all audit checks pass cleanly:
   - Row count matches test set ground truth exactly.
   - Column names and order match platform requirements.
   - Zero null, infinite, or duplicate IDs.
   - Output ranges conform strictly to class labels or probability bounds.

STEP 5: TECHNICAL SIGN-OFF BRIEFING
Present a structured finalization report conforming to the template below.
</instructions>

<output_format>
Your response must be formatted as a structured Markdown submission briefing conforming to this template:

# Pre-Submission Audit Briefing: [Task Name]

## 1. Selected Candidate Specification
| Attribute | Detail |
|---|---|
| **Frozen Candidate ID** | `[candidate_id]` |
| **Model Family / Backbone** | [e.g. DeBERTa-v3 ensemble / LightGBM] |
| **Validation Score (CV)** | `0.XXXX` |
| **Test Prediction Path** | `.ai-harness/artifacts/<task-id>/<candidate_id>/test_predictions.csv` |

## 2. Pre-Submission Audit Verification
| Audit Gate | Status | Observed Value |
|---|---|---|
| **Row Count Verification** | [PASSED / FAILED] | [e.g. Exactly 10,000 test instances] |
| **ID Alignment & Uniqueness** | [PASSED / FAILED] | [0 nulls, 0 duplicates] |
| **Column Ordering** | [PASSED / FAILED] | [Matches platform specification] |
| **Domain Range Constraints** | [PASSED / FAILED] | [Probabilities in [0, 1] / Valid labels] |

## 3. Submission Package Details
- **Package Path**: `[submission_file_path]`
- **SHA-256 Digest**: `[file hash]`
- **Submission Mode**: [Plugin Builder / Declarative Packaging]

## 4. Final Recommendation
- **Status**: `[READY_FOR_SUBMISSION | BLOCKED]`
- **Notes**: [Key operational instructions for submitting via web portal or API]
</output_format>
```

---

## 3. Behavioral Contracts and Rules

| Constraint | Verification Gate | Action on Failure |
|---|---|---|
| **User Approval** | Bridge enforces `--user-approved` flag | Command halts immediately without changing state. |
| **Test Set Sourcing** | Manifest v2 checks for `test` execution tag | Rejects run if test predictions were derived from OOF data. |
| **Schema Compliance** | `tools/audit_gate.py` parses column headers and IDs | Execution aborts with non-zero exit code. |
| **Package Sealing** | Audit receipt is recorded in project artifacts | Unsealed packages are tagged as uncertified. |
