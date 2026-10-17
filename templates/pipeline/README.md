# Pipeline Stage Templates

Copy-paste starting points for `execution.mode: pipeline` stages. The harness
runs each stage as a **separate subprocess** sharing the experiment dir, so
each template is a standalone `.py` that reads the harness env vars and writes
artifacts into `--exp-dir`.

## Files

| File | Purpose |
|---|---|
| `train_template.py` | train one model, write OOF predictions |
| `blend_template.py` | combine the train stages' OOF into the final `oof_predictions.csv` |

## How to use

1. Copy the template(s) into your workspace's `src/`:
   ```bash
   cp <framework_root>/templates/pipeline/train_template.py src/train_a.py
   cp <framework_root>/templates/pipeline/blend_template.py  src/blend.py
   ```
   (`<framework_root>` is `find_framework_root()` / `HARNESS_FRAMEWORK_DIR`.)
2. Replace every `[CUSTOMIZE]` marker with real values.
3. Reference the scripts in your experiment config (see
   `configs/pipeline_template.yaml`):
   ```yaml
   experiment:
     approach: custom_script        # REQUIRED for pipeline mode
   execution:
     mode: pipeline
     pipeline:
       - name: train_a
         entrypoint: src/train_a.py
       - name: train_b
         entrypoint: src/train_b.py
       - name: blend
         entrypoint: src/blend.py
   ```
4. Smoke probe before the full run:
   ```bash
   ai-harness-run --project-root . --config configs/<exp>.yaml --debug --fold 0
   ```

## Contract (what the harness expects)

- Each stage is invoked as:
  `python <stage> --exp-dir <artifact_dir> --config <run_config.yaml>` (cwd =
  the project root).
- Env vars per stage: `HARNESS_EXP_ID`, `HARNESS_EXP_DIR`,
  `HARNESS_PROJECT_ROOT`, `HARNESS_SEED`, `HARNESS_STAGE_NAME`,
  `HARNESS_STAGE_INDEX`, `HARNESS_STAGE_TOTAL`, `HARNESS_STAGE_DIR`.
- The **last** stage must write `oof_predictions.csv` (columns
  `pred_<subtask_id>` per the task contract); optionally `test_predictions.csv`
  and `metrics.json`.
- A non-zero exit from any stage **halts the pipeline** (fail-fast); per-stage
  logs land in `stages/<name>.log` and `stage_manifest.json` records exit
  codes + durations.

## Notes

- If you only need a single train stage (no blend), keep `train_template.py`
  and set `OUT_NAME = "oof_predictions.csv"`.
- The harness is execution + audit only — it does not infer your stages,
  reorder them, or inject data. You choose the scripts, order, and content.
