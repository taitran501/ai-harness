# Submission Audit and Candidate Freeze Prompt

Use the task's official submission contract and the selected plugin's declared capabilities. Local validation predictions are not test-set submission predictions.

## Operating rules

1. Read `ai-harness-agent --project-root . context` and confirm the active task, plugin, and submission capabilities.
2. For a task plugin, stop if either `submission_builder` or `submission_validator` is missing. Do not replace missing hooks with the default audit path.
3. Prepare submission data from test-set predictions in the task-declared format. Never submit out-of-fold validation predictions as test predictions.
4. Follow only validation, format, and governance checks declared by the task or plugin. Do not assume duplicate, significance, divergence, or label-space thresholds.
5. `freeze_candidate` and `prepare_submission` require explicit user approval through the Agent Bridge.

## Workflow

```powershell
ai-harness-agent --project-root . context
ai-harness-agent --project-root . act --action freeze_candidate --user-approved --params '{"slot_1_id": "<candidate-id>", "notes": "Reviewed final candidate"}'
ai-harness-agent --project-root . act --action prepare_submission --user-approved --params '{"exp_id": "<candidate-id>", "candidate_csv": ".ai-harness/artifacts/<task-id>/<candidate-id>/test_predictions.csv"}'
```

If the task or adapter is not submission-ready, stop before packaging and report the missing contract or hook. Record the final artifact path and validation receipt only when the task plugin actually produced them.

## Report

Summarize the selected candidate, source of test predictions, task/plugin submission capabilities, checks actually performed, generated package path, and any blocked capability. Keep local validation results separate from external competition scores and rank.
