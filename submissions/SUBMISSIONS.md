# StereoQueerEval Submissions & Leaderboard Registry

This registry tracks all verified competition packages, submission statuses,
and compares Local Cross-Validation against official Leaderboard scores.

> **Status Legend**:
> - 🔒 `LOCAL_VERIFIED`: Passed Deterministic & Statistical Audit Gate.
> - ⏳ `SUBMITTED_PENDING`: Uploaded to evaluation portal, awaiting leaderboard score.
> - 🎯 `SCORED`: Official scores recorded with CV vs LB delta analysis.
> - ⚠️ `DIVERGENCE`: Local CV significantly over-estimated performance (LB - CV < -0.05).

| Sub ID | Exp ID | Status | Local CV (A/B/C) | Official LB (A/B/C) | Delta (LB - CV) | Package | Notes |
|---|---|---|---|---|---|---|---|
| `sub_001_exp000_template` | `exp000_template` | 🎯 `SCORED` | A:0.6563 / B:0.2593 / C:0.2556 | A:0.665 / B:0.261 / C:0.258 | A:+0.0087 / B:+0.0017 | [`zip`](file:///submissions/sub_001_exp000_template/submission.zip) | Initial baseline debug package | Codabench Public LB |
