import json
import shutil
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from harness.agent_bridge import AgentBridge
from harness.candidate_preflight import (
    compute_prediction_similarity,
    load_preflight_receipt,
    run_candidate_preflight,
    validate_candidate_pitch,
)
from harness.project import attach_project
from tests.fixtures import make_submission_evidence, make_workspace, workspace_with_verified_pair


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

        loaded = load_preflight_receipt(self.root, receipt["preflight_id"])
        self.assertEqual(loaded, receipt)
        pitch = validate_candidate_pitch(
            self.root,
            preflight_id=receipt["preflight_id"],
            candidate_path=cand_file,
            claim_kind="submission_candidate",
        )
        self.assertTrue(pitch["citation_valid"])
        self.assertTrue(pitch["allowed"])
        self.assertEqual(pitch["citation_bundle"]["preflight_id"], receipt["preflight_id"])

        bridge = AgentBridge(self.root, task_spec_path=self.spec_path)
        session = bridge.start_session(goal="Validate cited candidate")
        _, observation = bridge.execute_action(
            action="validate_candidate_pitch",
            params={"preflight_id": receipt["preflight_id"], "candidate_path": str(cand_file)},
            session_id=session.session_id,
        )
        self.assertEqual(observation.status, "SUCCESS")
        self.assertTrue(observation.data["allowed"])

    def test_preflight_citation_rejects_missing_malformed_and_stale_receipts(self):
        candidate = self.root / "candidate.csv"
        incumbent = self.root / "incumbent.csv"
        pd.DataFrame({"id": range(20), "satisfaction": np.linspace(0.1, 0.9, 20)}).to_csv(candidate, index=False)
        pd.DataFrame({"id": range(20), "satisfaction": np.random.RandomState(5).uniform(0, 1, 20)}).to_csv(incumbent, index=False)
        receipt = run_candidate_preflight(self.root, candidate, incumbent)

        receipt_path = self.root / ".ai-harness" / "artifacts" / "preflight" / f"{receipt['preflight_id']}.json"
        original_receipt = receipt_path.read_text(encoding="utf-8")
        mismatched = json.loads(original_receipt)
        mismatched["preflight_id"] = "preflight-" + "0" * 32
        receipt_path.write_text(json.dumps(mismatched), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "does not match"):
            load_preflight_receipt(self.root, receipt["preflight_id"])
        receipt_path.write_text(original_receipt, encoding="utf-8")

        for preflight_id in ("bad/id", "preflight-" + "0" * 32):
            result = validate_candidate_pitch(self.root, preflight_id, candidate_path=candidate)
            self.assertFalse(result["citation_valid"])
            self.assertFalse(result["allowed"])

        candidate.write_text("id,satisfaction\n0,0.25\n", encoding="utf-8")
        stale = validate_candidate_pitch(
            self.root,
            receipt["preflight_id"],
            candidate_path=candidate,
        )
        self.assertFalse(stale["citation_valid"])
        self.assertTrue(any("hash differs" in error for error in stale["errors"]))

    def test_near_duplicate_pitch_requires_verified_paired_oof_gain(self):
        root, config = workspace_with_verified_pair(prefix="aih-test-pitch-paired-")
        self.addCleanup(shutil.rmtree, root)
        from harness.decision_rules import evaluate_decision_rule, register_decision_rule, reserve_comparison_slot
        from harness.task_spec import load_active_task_spec
        from tests.test_research_memory import rule_document

        attach_project(root, task_config=config)
        from harness.project import HarnessProject
        project = HarnessProject.discover(root)
        spec = load_active_task_spec(config)
        candidate_exp_id = "pitch-candidate"
        cfg = {
            "experiment": {"id": candidate_exp_id},
            "validation": {"partition": "oof"},
            "research": {"probe_plan_id": "probe-fixture"},
        }
        registration = register_decision_rule(rule_document(), project=project, task_spec=spec)
        cfg["research"]["decision_rule_id"] = registration["rule_id"]
        reserve_comparison_slot(project, spec, cfg)
        make_submission_evidence(root, candidate_exp_id, cfg=cfg)
        evaluation = evaluate_decision_rule(
            registration["rule"],
            {"kind": "full_run", "id": candidate_exp_id, "partition": "oof"},
            project=project,
            task_spec=spec,
            registration=registration,
        )
        self.assertEqual(evaluation["outcome"], "supported")

        n = 100
        incumbent_predictions = np.random.RandomState(42).uniform(0.1, 0.9, size=n)
        candidate_predictions = 0.98 * incumbent_predictions + 0.02 * np.random.RandomState(99).uniform(0.1, 0.9, size=n)
        candidate = root / "near-candidate.csv"
        incumbent = root / "near-incumbent.csv"
        pd.DataFrame({"id": range(n), "satisfaction": candidate_predictions}).to_csv(candidate, index=False)
        pd.DataFrame({"id": range(n), "satisfaction": incumbent_predictions}).to_csv(incumbent, index=False)
        preflight = run_candidate_preflight(root, candidate, incumbent)

        blocked = validate_candidate_pitch(root, preflight["preflight_id"], candidate_path=candidate)
        self.assertFalse(blocked["allowed"])
        self.assertTrue(any("paired OOF" in warning for warning in blocked["warnings"]))

        wrong_pair = validate_candidate_pitch(
            root,
            preflight["preflight_id"],
            candidate_path=candidate,
            candidate_exp_id="another-candidate",
            incumbent_exp_id="control",
            paired_oof_evaluation_id=evaluation["evaluation_id"],
        )
        self.assertFalse(wrong_pair["allowed"])

        allowed = validate_candidate_pitch(
            root,
            preflight["preflight_id"],
            candidate_path=candidate,
            candidate_exp_id=candidate_exp_id,
            incumbent_exp_id="control",
            paired_oof_evaluation_id=evaluation["evaluation_id"],
        )
        self.assertTrue(allowed["allowed"])
        self.assertEqual(allowed["citation_bundle"]["paired_oof_evaluation_id"], evaluation["evaluation_id"])

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
