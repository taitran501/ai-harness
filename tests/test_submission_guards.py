"""Regression tests for the anti-bypass submission guards.

The agent drives `prepare_submission` through the bridge; the standalone audit gate
CLI is not on that path. These tests verify the three deterministic guards that now
run on the bridge path before a submission is registered:

1. Duplicate-submission dedup (SHA-256) — re-submitting the same bytes is blocked.
2. Minimum meaningful effect size (MMES) — a candidate that differs from the
   baseline anchor on fewer rows than the floor is not a genuine new experiment.
3. Significance — when ground truth is available, a non-significant (p >= 0.05)
   candidate is rejected.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from harness.agent_bridge import AgentBridge

SOURCE_ROOT = Path(__file__).resolve().parent.parent


def _write_oof(folder: Path, key_col: str, pred_col: str, values) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({key_col: [f"r{i}" for i in range(len(values))], pred_col: values})
    df.to_csv(folder / "oof_predictions.csv", index=False)


class TestSubmissionGuards(unittest.TestCase):
    def setUp(self) -> None:
        self._ws = tempfile.TemporaryDirectory(prefix="test_submg_")
        self.root = Path(self._ws.name)
        shutil.copytree(SOURCE_ROOT / "configs", self.root / "configs")
        self.bridge = AgentBridge(self.root)
        self.art = self.bridge.artifacts_dir
        self.baseline_exp = "baseline"
        self.bridge.comp_state.state["gates"]["baseline_exp_id"] = self.baseline_exp

    def tearDown(self) -> None:
        self._ws.cleanup()

    def _cand_folder(self, exp_id="exp010", changed_frac=0.5, seed=7):
        rng = np.random.default_rng(seed)
        n = 200
        base = rng.integers(0, 2, size=n)
        cand = base.copy()
        flip = rng.random(n) < changed_frac
        cand[flip] = 1 - cand[flip]
        _write_oof(self.art / self.baseline_exp, "id", "pred_label", base)
        _write_oof(self.art / exp_id, "id", "pred_label", cand)
        # Candidate test prediction file (provenance is satisfied in the unit test
        # by calling the guard directly, bypassing validate_test_prediction).
        (self.art / exp_id / "test_predictions.csv").write_text("id,pred_label\n")
        return self.art / exp_id / "test_predictions.csv"

    def test_dedup_blocks_identical_bytes(self) -> None:
        cand = self._cand_folder(exp_id="exp010", changed_frac=0.5)
        sub_mgr = self.bridge._new_submission_manager()
        # Register a submission pointing at the same candidate bytes.
        from harness.submission_manager import SubmissionEntry
        sub_mgr.submissions["sub_existing"] = SubmissionEntry(
            id="sub_existing", exp_id="exp010", created_at="t", status="LOCAL_VERIFIED",
            candidate_path=str(cand),
        )
        with self.assertRaises(ValueError) as ctx:
            self.bridge._guard_submission(cand, "exp010", sub_mgr)
        self.assertIn("dedup", str(ctx.exception).lower())

    def test_mmes_blocks_below_floor(self) -> None:
        # changed_frac=0.01 -> ~2% of 200 = 2 rows; required = max(1, ceil(200*2/100)=4) = 4
        cand = self._cand_folder(exp_id="exp011", changed_frac=0.005)
        sub_mgr = self.bridge._new_submission_manager()
        with self.assertRaises(ValueError) as ctx:
            self.bridge._guard_submission(cand, "exp011", sub_mgr)
        self.assertIn("mmes", str(ctx.exception).lower())

    def test_mmes_passes_above_floor(self) -> None:
        cand = self._cand_folder(exp_id="exp012", changed_frac=0.5)
        sub_mgr = self.bridge._new_submission_manager()
        evidence = self.bridge._guard_submission(cand, "exp012", sub_mgr)
        self.assertTrue(evidence["checks"]["mmes"].startswith("PASS"))

    def test_significance_enforced_when_truth_present(self) -> None:
        # Build a case where the candidate is *worse* than baseline on ground
        # truth, so the paired test is not significant -> guard must fail.
        rng = np.random.default_rng(0)
        n = 400
        truth = rng.integers(0, 2, size=n)
        base = truth.copy()  # baseline is perfect
        # candidate is worse on 30% of rows
        cand = base.copy()
        worse = rng.random(n) < 0.3
        cand[worse] = 1 - cand[worse]
        _write_oof(self.art / self.baseline_exp, "id", "pred_label", base)
        cand_folder = self.art / "exp013"
        cand_folder.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"id": [f"r{i}" for i in range(n)], "pred_label": cand}).to_csv(
            cand_folder / "oof_predictions.csv", index=False
        )
        pd.DataFrame({"id": [f"r{i}" for i in range(n)], "pred_label": truth}).to_csv(
            cand_folder / "evaluation_truth.csv", index=False
        )
        cand_csv = cand_folder / "test_predictions.csv"
        cand_csv.write_text("id,pred_label\n")
        sub_mgr = self.bridge._new_submission_manager()
        evidence = self.bridge._guard_submission(cand_csv, "exp013", sub_mgr)
        self.assertTrue(
            evidence["checks"]["significance"].startswith("FAIL"),
            f"Expected a FAIL significance result, got {evidence['checks']['significance']}",
        )

    def test_significance_passes_when_candidate_better(self) -> None:
        rng = np.random.default_rng(1)
        n = 400
        truth = rng.integers(0, 2, size=n)
        cand = truth.copy()  # candidate perfect
        base = cand.copy()
        worse = rng.random(n) < 0.3
        base[worse] = 1 - base[worse]  # baseline worse on 30%
        _write_oof(self.art / self.baseline_exp, "id", "pred_label", base)
        cand_folder = self.art / "exp014"
        cand_folder.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"id": [f"r{i}" for i in range(n)], "pred_label": cand}).to_csv(
            cand_folder / "oof_predictions.csv", index=False
        )
        pd.DataFrame({"id": [f"r{i}" for i in range(n)], "pred_label": truth}).to_csv(
            cand_folder / "evaluation_truth.csv", index=False
        )
        cand_csv = cand_folder / "test_predictions.csv"
        cand_csv.write_text("id,pred_label\n")
        sub_mgr = self.bridge._new_submission_manager()
        evidence = self.bridge._guard_submission(cand_csv, "exp014", sub_mgr)
        self.assertTrue(
            evidence["checks"]["significance"].startswith("PASS"),
            f"Expected PASS, got {evidence['checks']['significance']}",
        )


if __name__ == "__main__":
    unittest.main()
