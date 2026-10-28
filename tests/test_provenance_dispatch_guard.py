import tempfile
import unittest
from pathlib import Path

from harness.audit_gate import enforce_dispatch_script_guard, scan_dispatch_scripts
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


if __name__ == "__main__":
    unittest.main()
