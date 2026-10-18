# Change: pytest-only tests converted to unittest (17 tests now in the official gate)

- **Date**: 2026-10-18
- **PR**: [#103](https://github.com/taitran501/ai-harness/pull/103) (open)
- **Status**: open

## Why

Audit finding (2026-10-18, "path D"): `tests/test_recursive_sealing.py` (8 tests) and
`tests/test_repair_integrity.py` (9 tests) were written with **module-level pytest-style
functions** instead of `unittest.TestCase` methods. The official test runner
(`tools/run_tests.py`, unittest-based, documented in AGENTS.md as the pre-push gate:
"full suite — all modules must pass 100%") collected **0 tests** from both modules:

```
python tools/run_tests.py tests.test_recursive_sealing → 0 tests in 0.2s — OK
python tools/run_tests.py tests.test_repair_integrity  → 0 tests in 1.5s — OK
```

Only `python -m pytest` (925 items vs the official runner's 908) executed them.
Consequence: a regression in recursive sealing (PR #96 finding M2) or the repair
evidence-integrity gate (PR #96 finding C1) would pass the documented full-suite gate
silently.

Overlap analysis vs the running suites (`test_repair.py`, `test_local_import.py`,
`test_kaggle_runtime.py`): ~3-4 of the 17 are duplicate, ~4 partial duplicates, but
**~10 verify unique behaviors the official gate never ran**, including:

| Behavior | Why it matters |
|---|---|
| Nested tamper detection (`stages/*.log`) | Regression test of the actual #96 M2 bug |
| Traversal / absolute key rejection (seal + import) | Path-traversal protection in evidence manifests |
| Tampered metrics blocked by default | Doctored `metrics.json` must not be re-sealed |
| `--verify` re-derivation (pass + block paths) | Score re-computed through the runner's own scoring path |
| `--force` audited | Operator override is never silent |
| Repair log outside sealed bundle | Audit trail independent of evidence integrity |
| Nested export materialization + old-manifest backward compat | Kaggle export/import round-trip |

## What changed

Mechanical conversion to `unittest.TestCase` — **no test logic, fixture data, or
assertion semantics changed**:

- `tests/test_recursive_sealing.py`: 8 module-level functions → 3 `TestCase` classes
  (`RecursiveSealingTests`, `ImportContentVerifiedTests`, `ExportNestedFilesTests`);
  `tmp_path` fixture → `tempfile.TemporaryDirectory` in `setUp`/`tearDown`;
  the manifest dict repeated 4× extracted into `_full_manifest()` (identical content).
- `tests/test_repair_integrity.py`: 9 module-level functions →
  `RepairIntegrityGateTests`; `workspace` pytest fixture → `setUp` (same
  `tests.fixtures.make_workspace(task=_regression_task_doc())` call);
  `capsys` → `contextlib.redirect_stdout` + `io.StringIO` (verified `run_repair`
  prints to stdout); `pytest.raises` → `assertRaises`; `pytest.approx` →
  `assertAlmostEqual(places=6)`.

## Evidence

| Check | Before | After |
|---|---|---|
| `run_tests.py tests.test_recursive_sealing` | 0 tests OK | **8 tests OK** (16.3s) |
| `run_tests.py tests.test_repair_integrity` | 0 tests OK | **9 tests OK** (4.0s) |
| `pytest` on both files | 17 passed | 17 passed |
| Official full suite | 908 tests OK | **925 tests OK** (262.3s) |
| pytest full collection | 925 | 925 — **runners now agree** |
| ruff | clean | clean |

Both modules sit in the FAST tier, so the default (no-flag) `run_tests.py` invocation
now also covers them.

## Impact

- `tools/run_tests.py` and `python -m pytest` now collect the identical 925 tests —
  the documented gate no longer silently skips any module.
- The ~10 unique security/audit behaviors (tamper detection, traversal rejection,
  re-derivation gate, force-audit, log persistence) are protected by the official
  suite for the first time.
- No production code changed; no public API changed.
