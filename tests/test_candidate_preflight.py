import shutil
import unittest

import numpy as np
import pandas as pd

from harness.candidate_preflight import compute_prediction_similarity, run_candidate_preflight
from harness.project import attach_project
from tests.fixtures import make_workspace


class TestCandidatePreflight(unittest.TestCase):
    def setUp(self):
        self.root, self.spec_path = make_workspace(prefix="aih-test-preflight-")
        self.addCleanup(shutil.rmtree, self.root)
        self.project = attach_project(self.root, task_config=self.spec_path)

    def test_compute_prediction_similarity_identical_and_reversed(self):
        a = np.linspace(0, 1, 100)
        sim_same = compute_prediction_similarity(a, a)
        self.assertAlmostEqual(sim_same["spearman_rho"], 1.0)
        self.assertAlmostEqual(sim_same["pearson_r"], 1.0)
        self.assertEqual(sim_same["rank_swap_count"], 0)
        self.assertAlmostEqual(sim_same["mean_abs_delta"], 0.0)

        sim_rev = compute_prediction_similarity(a, -a)
        self.assertAlmostEqual(sim_rev["spearman_rho"], -1.0)
        self.assertAlmostEqual(sim_rev["rank_swap_ratio"], 1.0)

    def test_preflight_flags_near_duplicate_micro_tinkering(self):
        # 1000 samples
        n = 1000
        ids = np.arange(n)
        inc_preds = np.random.RandomState(42).uniform(0.1, 0.9, size=n)

        # Micro-perturbation: 98% incumbent + 2% other
        cand_preds = 0.98 * inc_preds + 0.02 * np.random.RandomState(99).uniform(0.1, 0.9, size=n)

        cand_df = pd.DataFrame({"id": ids, "satisfaction": cand_preds})
        inc_df = pd.DataFrame({"id": ids, "satisfaction": inc_preds})

        cand_file = self.root / "candidate.csv"
        inc_file = self.root / "incumbent.csv"
        cand_df.to_csv(cand_file, index=False)
        inc_df.to_csv(inc_file, index=False)

        receipt = run_candidate_preflight(self.root, cand_file, inc_file)
        self.assertIn("preflight_id", receipt)
        self.assertEqual(receipt["verdict"], "FLAGGED_NEAR_DUPLICATE")
        self.assertIn(receipt["classification"], {"TRIVIAL_PERTURBATION", "MINOR_REWEIGHTING"})
        self.assertIsNotNone(receipt["delusion_warning"])
        self.assertIn("CRITICAL", receipt["delusion_warning"])

    def test_preflight_passes_diverse_candidate(self):
        n = 1000
        ids = np.arange(n)
        inc_preds = np.random.RandomState(42).uniform(0.1, 0.9, size=n)
        diverse_preds = np.random.RandomState(1234).uniform(0.1, 0.9, size=n)

        cand_df = pd.DataFrame({"id": ids, "satisfaction": diverse_preds})
        inc_df = pd.DataFrame({"id": ids, "satisfaction": inc_preds})

        cand_file = self.root / "diverse.csv"
        inc_file = self.root / "incumbent.csv"
        cand_df.to_csv(cand_file, index=False)
        inc_df.to_csv(inc_file, index=False)

        receipt = run_candidate_preflight(self.root, cand_file, inc_file)
        self.assertEqual(receipt["verdict"], "PASS")
        self.assertEqual(receipt["classification"], "DIVERSE_CANDIDATE")
        self.assertIsNone(receipt["delusion_warning"])
        self.assertIn("PASS", receipt["recommendation"])


if __name__ == "__main__":
    unittest.main()
