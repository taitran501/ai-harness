# Change Log

Per-change documentation for **framework-repo** changes (code + docs). Each
change/PR gets one file here, written from real context — **never** reconstructed
from a PR description alone (that risks fabricating the "thought" and "approach").
Newer at the top.

## Conventions

- **File name**: `YYYY-MM-DD-<slug>.md` (slug = short kebab-case description).
- **Template**: copy [`_TEMPLATE.md`](_TEMPLATE.md) and fill every section.
- **When**: on every framework-repo change/PR — create the file **before**
  committing, then add a row to the index below.
- **Detail bar**: Why (motivation + thought) → What (concrete changes) → Approach
  (decisions + rejected alternatives) → Evidence → Impact & follow-ups.
- **Reading again**: open this index → click the change file you need. No need to
  scroll the historical `WORK_LOG.md`.

## Changes

| Date | Change | PR | File |
|------|--------|----|------|
| 2026-10-17 | Harness-native Kaggle submission + kernel dispatch | [#84](https://github.com/taitran501/ai-harness/pull/84) | [2026-10-17-kaggle-dispatch-polling](2026-10-17-kaggle-dispatch-polling.md) |
| 2026-10-17 | Skill source fix: pipeline section in generated SKILL.md | [#83](https://github.com/taitran501/ai-harness/pull/83) | [2026-10-17-skill-source-fix](2026-10-17-skill-source-fix.md) |
| 2026-10-17 | Pipeline adoption stack (audit gate, config hardening, discoverability, scaffolds) | [#82](https://github.com/taitran501/ai-harness/pull/82) | [2026-10-17-pipeline-adoption-stack](2026-10-17-pipeline-adoption-stack.md) |

## Older changes (archived)

Changes prior to this change log are recorded in **[`docs/WORK_LOG.md`](../WORK_LOG.md)**
(now a read-only archive). Those entries pre-date the per-file change-doc convention
and are kept as-is; they are not individually split out (reconstructing them from
thin descriptions would risk inaccuracy).
