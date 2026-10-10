# Model Synthesis and Ensemble Prompt

Use this prompt only when ensembling is supported by the task runtime and its validation contract.

## Operating rules

1. Read `ai-harness-agent --project-root . context` and inspect the task's metric, direction, validation scheme, and plugin capabilities.
2. Compare predictions generated for the same validation rows. Use out-of-fold predictions for any learned weights or stacking model.
3. Apply ensemble diversity checks only when they are declared by the task or selected plugin. Do not invent thresholds.
4. Promote an ensemble only when its measured task metric improves over the best eligible constituent under the declared validation protocol.

## Workflow

```powershell
ai-harness-agent --project-root . context
ai-harness-agent --project-root . act --action inspect_diagnostics --params '{"exp_id": "<parent-id>"}'
ai-harness-agent --project-root . act --action build_ensemble --params '{"name": "<ensemble-id>", "candidates": ["<candidate-1>", "<candidate-2>"]}'
```

Inspect the resulting metrics under `.ai-harness/artifacts/<task-id>/<ensemble-id>/` before promotion. If the task plugin does not support a requested ensemble or output type, report that capability gap instead of substituting another contract.

## Report

Include the candidate IDs, model families, validation metric and direction, validation protocol, synthesis method, measured result, and promotion decision. Separate local validation results from any external competition score.
