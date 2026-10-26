import shutil
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from harness.agent_bridge import AgentBridge
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

    def test_preflight_missing_candidate_raises(self):
        with self.assertRaises(ValueError):
            run_candidate_preflight(self.root, candidate_path=None)

    def test_bridge_action_candidate_preflight(self):
        n = 100
        ids = np.arange(n)
        preds = np.random.RandomState(42).uniform(0, 1, size=n)
        cand_df = pd.DataFrame({"id": ids, "satisfaction": preds})
        inc_df = pd.DataFrame({"id": ids, "satisfaction": preds * 0.999})
        cand_file = self.root / "cand.csv"
        inc_file = self.root / "inc.csv"
        cand_df.to_csv(cand_file, index=False)
        inc_df.to_csv(inc_file, index=False)

        bridge = AgentBridge(self.root, task_spec_path=self.spec_path)
        sess = bridge.start_session(goal="Test preflight action")
        _, obs = bridge.execute_action(
            action="candidate_preflight",
            params={"candidate": str(cand_file), "incumbent": str(inc_file)},
            session_id=sess.session_id,
        )
        self.assertEqual(obs.status, "SUCCESS")
        self.assertEqual(obs.data["verdict"], "FLAGGED_NEAR_DUPLICATE")

    def test_cli_allow_abbrev_preserves_candidate_flag(self):
        from tools.agent_bridge_cli import main

        n = 50
        ids = np.arange(n)
        preds = np.random.RandomState(42).uniform(0, 1, size=n)
        cand_file = self.root / "c.csv"
        inc_file = self.root / "i.csv"
        pd.DataFrame({"id": ids, "satisfaction": preds}).to_csv(cand_file, index=False)
        pd.DataFrame({"id": ids, "satisfaction": preds}).to_csv(inc_file, index=False)

        argv = [
            "--project-root",
            str(self.root),
            "act",
            "--action",
            "candidate_preflight",
            "--candidate",
            str(cand_file),
            "--incumbent",
            str(inc_file),
        ]
        with patch.dict("os.environ", {"AI_HARNESS_TOOLCHAIN_NO_PIN": "1"}):
            code = main(argv)
            self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
