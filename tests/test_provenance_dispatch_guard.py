import tempfile
import unittest
from pathlib import Path

from harness.audit_gate import enforce_dispatch_script_guard, scan_dispatch_scripts, scan_markdown_citations
from harness.citation_sources import SourceRegistry
from harness.provenance import (
    PredictionProvenance,
    derive_submission_eligibility,
    require_allowed_provenance,
    validate_provenance_record,
)


class ProvenanceDispatchGuardTests(unittest.TestCase):
    def test_provenance_contract_and_eligibility_derivation(self):
        self.assertEqual(PredictionProvenance.LOCAL_RUN.value, "local_run")
        with self.assertRaisesRegex(ValueError, "PROVENANCE_MISSING_OR_INVALID"):
            validate_provenance_record({})
        with self.assertRaisesRegex(ValueError, "EXTERNAL_PREDICTION_NOT_SUBMITTABLE"):
            require_allowed_provenance([{"provenance": "external_prediction"}], purpose="candidate")
        self.assertFalse(derive_submission_eligibility(
            provenance_valid=True, validation_valid=True,
            test_predictions_valid=True, promotion_valid=False,
        )["submission_eligible"])

    def test_scanner_finds_root_and_scratch_dispatch_patterns(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "dispatch.py").write_text("kaggle competitions submit -f x.csv\n")
            (root / "scratch").mkdir()
            (root / "scratch" / "kernel.py").write_text("client.kernels_push_directory(path)\n")
            report = scan_dispatch_scripts(root)
            self.assertEqual(report["status"], "findings")
            self.assertEqual({entry["pattern"] for entry in report["findings"]},
                             {"kaggle competitions submit", "kernels_push_directory"})
            self.assertEqual(enforce_dispatch_script_guard(root)["blocker"], "DISPATCH_BACKDOOR_DETECTED")

    def test_acknowledged_script_is_reported_but_not_blocking(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "raw.py").write_text("competitions_submissions_create()\n")
            ack = root / ".ai-harness"
            ack.mkdir()
            (ack / "acknowledged_dispatch_scripts.json").write_text('{"acknowledged": ["raw.py"]}')
            self.assertEqual(len(scan_dispatch_scripts(root)["findings"]), 1)
            self.assertEqual(enforce_dispatch_script_guard(root)["status"], "clean")



    def test_scan_markdown_citations_finds_missing_and_ungrounded_topics(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            disc_dir = root / "discussions"
            disc_dir.mkdir()
            # Explicitly register a normalized source; audits never crawl arbitrary topic files.
            SourceRegistry(root).register_topic_source(
                "750885", {"id": 750885, "title": "Negative results",
                           "content": "14 streams vs 10 streams on RTX Pro 6000 and 110k token cutoff."},
                adapter="normalized_topic_v1",
            )
            # Markdown citing existing topic with fabricated terms and missing topic #999999
            md_path = root / "summary.md"
            md_path.write_text(
                "# Analysis\n"
                "- Proven scheduler D-prime by shiiin9 prevents high-level starvation (Topic #750885).\n"
                "- Untracked breakthrough reported in (Topic #999999).\n",
                encoding="utf-8"
            )
            report = scan_markdown_citations(root, file_path=md_path)
            self.assertEqual(report["status"], "findings")
            self.assertEqual(report["total_citations"], 2)
            issues = {f["issue"] for f in report["findings"]}
            self.assertIn("MISSING_TOPIC_FILE", issues)
            self.assertIn("UNGROUNDED_TOPIC_CITATION", issues)
            ungrounded = [f for f in report["findings"] if f["issue"] == "UNGROUNDED_TOPIC_CITATION"][0]
            self.assertEqual(ungrounded["topic_id"], "750885")
            self.assertIn("shiiin9", ungrounded["missing_terms"])


if __name__ == "__main__":
    unittest.main()
