# Evidence-derived research decisions and pre-dispatch authorization

- **Date**: 2026-10-25
- **PR**: None (working-tree implementation)
- **Branch**: `main` (managed permissions prohibit writing `.git`)
- **Status**: implemented, verification recorded below; uncommitted
- **Commits**: None

## Why

Research decisions previously accepted an agent-selected `supported` outcome.
Submission dispatch called the provider before its best-effort audit, while
row-difference thresholds could mistake score jitter for scientific improvement.
Session-local rejection advisories did not pause repeated unsuccessful sends.

The implementation follows `scratch/codex_implementation_plan.md`: metric gain,
uncertainty, prediction diversity, quota and workflow state are separate checks.
Remaining quota does not justify dispatch.

## What changed

- Added seven strict Draft 2020-12 schemas under `harness/schemas/`, included in
  wheel package data. `policy_schema.py` validates their runtime subset without
  introducing a production `jsonschema` dependency; independent schema validation
  is included in the test extra.
- Added `decision_rules.py`: validated policy/rule types, policy resolution,
  fixed-control registration, prospective comparison reservations, sealed evidence
  resolution, recomputation through the existing scorer, full-precision effect
  checks, paired row/group/query/block bootstrap, fixed-horizon alpha allocation,
  and idempotent evaluation sidecars. Exact additive metrics have a verified fast
  adapter; custom scoring keeps the existing plugin path.
- Updated research memory and bridge contracts for structured rules and typed
  candidate evidence. Unsupported `supported` requests raise before appending a
  false decision. Legacy text plans and historical raw outcomes remain readable;
  effective legacy support is inconclusive. New decisions retain the claimed
  outcome, machine outcome, rule/evaluation IDs and reasons separately.
- Extracted audit helpers into `harness/audit_gate.py`, kept the tools module as a
  thin compatibility wrapper, and moved argument/packaging behavior to
  `harness/audit_cli.py`. Validation receipts explicitly do not authorize a send.
  Failed validation replaces a requested stale success receipt, and output paths
  inside completed run bundles are rejected.
- Added aligned portfolio diversity: typed composite IDs, symmetric scope and
  coverage, declared output semantics, per-column signed Spearman (default .995),
  discrete/text disagreement, constant-array handling, semantic hashes and strict
  missing-history errors. Numeric class IDs are not ranked as continuous scores.
- Added `SubmissionGate`, nonserializable single-use expiring permits, exact
  payload/plugin/evidence binding, operator-attested payload approval, quota
  reservation, history snapshots and durable dispatch events. CLI, tracker and
  advertised Kaggle facade enforce the gate before submission. Unknown sends keep
  their reservation and block retries. Provider-observed intent markers permit
  deliberate reconciliation and terminal score observations.
- Preparation no longer consumes submission quota. Registry entries retain
  payload/prediction hashes and dispatch/approval/observation references. Declared
  external scores are distinguishable from API-observed results.
- Added task-scoped stagnation replay and projections. Three comparable missed
  promotion rules latch the pause across sessions; corrupt/missing scientific
  evidence does not count as a negative result. The incumbent remains fixed until
  supported promotion. Recovery requires a registered bounded probe and supported
  evidence; editing a projection or increasing future patience cannot clear the
  current latch. The HUD and research checkpoint expose the recovery requirement.
- Local runner completions evaluate registered rules outside sealed folders.
  Cloud dispatch reserves comparisons on the host and transports a small
  source-config-bound capsule; canonical import performs the evaluation. Kaggle
  bundle caching includes the capsule. Ordinary training pushes remain distinct
  from scored competition submissions.
- Replaced the atomic-variable/quota mandate in `AGENTS.md` and the skill
  generator with attributable mechanisms and the stagnation pause protocol.
  Updated CLI flags, generated examples and the verified API reference.
- Added sealed disposable fixtures and research/diversity/dispatch/recovery/schema
  tests. Updated old expectations that explicitly allowed unaudited dispatch or
  self-awarded support. Unrelated polling, kernel logging and SDK routing tests
  remain in place.

## Approach

The new services depend on existing task/plugin scoring and artifact validation,
never on the tools CLI or AgentBridge. All policy evidence, reservations, receipts
and projections live outside immutable completed runs. `evaluation.py` and
`task_spec.py` were deliberately left unchanged because their source participates
in scorer identity. A permit is a live capability, not a trusted JSON assertion.
Network calls run outside the short workspace lock; durable intent precedes the
single provider call. A timeout is not treated as a definite rejection.

## Evidence

- Requested audit/research/dispatch test selection: passed (61 tests at the first
  green checkpoint; final expanded counts are in the verification report).
- Expanded audit/research/dispatch/governance/facade/preparation set: 117 passed.
- Audit/research/dispatch plus Kaggle runtime/cloud-job set: 103 passed.
- Smoke tier: 47 passed at the first verification checkpoint.
- Ruff: `ruff check harness/ tools/ tests/` passed.
- Wheel: built without dependencies/build isolation; contains all seven schemas.
- Upgrade probe: an untouched `git archive HEAD` snapshot produced a sealed v2
  bundle; the changed runtime validated it and preserved its manifest hash.
- Fast tier: 1,078 tests, two errors and one skip. Both errors reproduce in the
  untouched HEAD snapshot's `test_ergonomics_and_resilience`: nested dataset-mount
  fixture mismatch, and a facade surface test that attempts network authentication
  in this restricted environment. They are not marked skipped or silently hidden.
- Final commands, counts and complete file inventory are recorded in
  `scratch/systemic_verification_report.md`; raw logs are in `scratch/systemic-*.log`.

## Impact & follow-ups

- New outbound sends require attached workspace/evidence, exact-payload approval,
  complete history, quota and a live permit. Old passing audit JSON and
  `--allow-initial` cannot substitute for those checks.
- Unverified code-competition kernel outputs fail `UNVERIFIED_KERNEL_OUTPUT`.
  A same-named local CSV cannot establish an immutable provider version/output
  binding. Ordinary kernel training pushes still work. No live competition send
  or live cloud training was performed during this implementation.
- An unimplemented plugin-specific paired protocol and unpaired public leaderboard
  scores fail closed for verified support. Row/group bootstrap is available through
  the existing metric contract; unsupported representations remain explicit.
- The two generated `.agents/skills/.../SKILL.md` files are read-only under this
  session's managed permissions. The updated generated outputs are staged under
  `scratch/generated-skills/`; their canonical generator is updated. Re-injection
  into the protected destinations remains an operator/environment step.
- Feature-branch creation was also denied by managed `.git` permissions. No commit,
  push, merge, publication or real competition submission was attempted.
