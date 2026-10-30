import tempfile
import os
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

from harness.citation_audit import enforce_citation_admission
from harness.grounding_gate import GroundingGate
from harness.research_memory import ResearchMemory


class CitationAdmissionTests(unittest.TestCase):
    def test_baseline_is_explicitly_not_applicable_and_omitted_discussion_receipt_blocks(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            baseline = enforce_citation_admission(root, {"experiment": {"branch": "baseline"}})
            self.assertEqual(baseline["admission"], "not_applicable")
            discussion = enforce_citation_admission(root, {"grounding": {"source_type": "kaggle_discussion"}})
            self.assertEqual(discussion["admission"], "blocked")
            self.assertEqual(discussion["reason"], "DISCUSSION_CITATION_RECEIPT_REQUIRED")

    def test_unreviewed_receipt_reference_is_revalidated_and_blocked(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            result = enforce_citation_admission(
                root, {"research": {"citation_dependencies": [".ai-harness/citation_audits/missing.json"]}})
            self.assertEqual(result["admission"], "blocked")

    def test_discussion_grounding_cannot_pass_on_verified_via_alone(self):
        config = {
            "experiment": {"id": "exp1", "branch": "feature", "problem_addressed": "A documented bottleneck in validation",
                           "hypothesis": "The change will improve the score because of a causal mechanism.",
                           "intervention": "Control is fixed; treatment changes one mechanism."},
            "grounding": {"source_type": "kaggle_discussion", "source_id": "topic-750885",
                          "retrieval_summary": "The discussion motivates a bounded mechanism probe.", "verified_via": "kaggle_discussion"},
            "assumption_register": {"known_facts": ["The task contract is loaded."],
                                    "assumptions": ["The mechanism transfers to this split."],
                                    "uncertainty_level": "medium"},
        }
        passed, errors, receipt = GroundingGate.verify_config(config, raise_on_error=False, project_root=".")
        self.assertFalse(passed)
        self.assertTrue(any("CITATION ADMISSION BLOCKED" in error for error in errors))
        self.assertEqual(receipt["citation_admission"]["admission"], "blocked")

    def test_cloud_dispatch_blocks_before_provider_preflight_when_discussion_receipt_is_missing(self):
        from harness.cloud_runner import _prepare_remote_run
        from tests.fixtures import make_workspace

        root, task_path = make_workspace(prefix="aih-citation-cloud-")
        try:
            config = {"experiment": {"id": "exp-citation"},
                      "grounding": {"source_type": "kaggle_discussion"}}
            with patch.dict(os.environ, {}), \
                 patch("harness.cloud_runner.CloudPreflightInspector.run_full_preflight_check",
                       return_value=(True, [], config)), \
                 patch("harness.cloud_policy.provider_preflight") as provider:
                with self.assertRaisesRegex(ValueError, "CITATION_ADMISSION_BLOCKED"):
                    _prepare_remote_run(str(task_path), "kaggle", root)
                provider.assert_not_called()
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_unreviewed_external_note_is_kept_as_hypothesis_and_can_have_a_bounded_probe(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            artifacts = root / ".ai-harness" / "artifacts" / "task"
            memory = ResearchMemory(root, artifacts_dir=artifacts, task_spec={})
            insight = memory.record_insight(
                title="Possible scheduling effect", observation="A discussion asks about fairness.",
                mechanism="A scheduler policy may change which examples are processed first.",
                next_action="Compare one policy against a fixed control.", source_refs=["Topic #750885"],
                empirical_status="external_report")
            self.assertEqual(insight["empirical_status"], "hypothesis")
            self.assertEqual(insight["citation_admission"]["admission"], "blocked")
            plan = memory.plan_probe(
                insight_id=insight["id"], control="fixed scheduler baseline",
                intervention="change only queue scheduling policy",
                expected_observation="processing order changes under the treatment",
                alternative_explanation="the observed difference comes from sample ordering",
                decision_rule="compare paired outcomes under a preregistered rule",
                budget="one smoke fold", stop_rule="stop after the smoke fold shows no mechanism effect")
            self.assertEqual(plan["empirical_status"], "hypothesis")


if __name__ == "__main__":
    unittest.main()
