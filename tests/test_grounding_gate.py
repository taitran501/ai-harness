"""Unit tests for the Epistemic Grounding Gate and Assumption Register."""

import json
import tempfile
import unittest
from pathlib import Path

from harness.grounding_gate import EpistemicGroundingError, GroundingGate


class GroundingGateTests(unittest.TestCase):
    def setUp(self):
        self.valid_config = {
            "experiment": {
                "id": "exp010_model_mdeberta",
                "branch": "model",
                "parent_id": "exp001_baseline_tfidf",
                "problem_addressed": "Extreme class imbalance and poor semantic transfer on rare Subtask C identities.",
                "hypothesis": "Pretrained multilingual representations in mDeBERTa-v3 will boost rare identity Macro-F1 by +0.08.",
                "intervention": "Switch backbone from TF-IDF to microsoft/mdeberta-v3-base with disentangled attention.",
            },
            "grounding": {
                "source_type": "huggingface_hub",
                "source_id": "microsoft/mdeberta-v3-base",
                "retrieval_summary": "Disentangled attention model with multilingual coverage for EN, IT, and NL; superior semantic representation.",
                "verified_via": "huggingface_hub",
            },
            "assumption_register": {
                "known_facts": [
                    "exp001 achieved 0.62 Subtask A Macro-F1 but 0.00 on rare identity labels."
                ],
                "assumptions": [
                    "Disentangled attention captures nuanced implicit hate better than sparse n-grams."
                ],
                "uncertainty_level": "medium",
            },
        }

    def test_valid_grounding_passes(self):
        """A properly grounded config should pass validation without errors."""
        passed, errors, receipt = GroundingGate.verify_config(self.valid_config, raise_on_error=False)
        self.assertTrue(passed)
        self.assertEqual(len(errors), 0)
        self.assertEqual(receipt["status"], "PASSED")
        self.assertEqual(receipt["experiment_id"], "exp010_model_mdeberta")

    def test_missing_grounding_rejected(self):
        """Non-baseline config missing 'grounding' section must be rejected."""
        cfg = dict(self.valid_config)
        del cfg["grounding"]
        passed, errors, receipt = GroundingGate.verify_config(cfg, raise_on_error=False)
        self.assertFalse(passed)
        self.assertTrue(any("EPISTEMIC GROUNDING ERROR" in err for err in errors))

    def test_invalid_source_type_rejected(self):
        """Unsupported or hallucinated source_type must be rejected."""
        cfg = dict(self.valid_config)
        cfg["grounding"] = {
            "source_type": "my_personal_blog",
            "source_id": "some_random_id",
            "retrieval_summary": "Tested some models on a blog post.",
        }
        passed, errors, receipt = GroundingGate.verify_config(cfg, raise_on_error=False)
        self.assertFalse(passed)
        self.assertTrue(any("GROUNDING TYPE ERROR" in err for err in errors))

    def test_lazy_intervention_rejected(self):
        """Lazy trial-and-error interventions like 'tune parameters' must be rejected."""
        lazy_phrases = [
            "tune parameters",
            "tuning hyperparameters",
            "try different things",
            "simple ensemble",
            "random search",
        ]
        for phrase in lazy_phrases:
            cfg = dict(self.valid_config)
            cfg["experiment"] = dict(self.valid_config["experiment"])
            cfg["experiment"]["intervention"] = phrase
            passed, errors, _ = GroundingGate.verify_config(cfg, raise_on_error=False)
            self.assertFalse(passed, f"Should have rejected lazy phrase: '{phrase}'")
            self.assertTrue(any("LAZY INTERVENTION REJECTED" in err for err in errors))

    def test_missing_assumption_register_rejected(self):
        """Configs lacking an explicit assumption register must be rejected."""
        cfg = dict(self.valid_config)
        del cfg["assumption_register"]
        passed, errors, _ = GroundingGate.verify_config(cfg, raise_on_error=False)
        self.assertFalse(passed)
        self.assertTrue(any("ASSUMPTION REGISTER MISSING" in err for err in errors))

    def test_high_uncertainty_requires_verification_probe(self):
        """High uncertainty experiments must declare a verification probe."""
        cfg = dict(self.valid_config)
        cfg["assumption_register"] = {
            "known_facts": ["exp001 completed."],
            "assumptions": ["A radically new architecture might generalize."],
            "uncertainty_level": "high",
            # missing verification_probe
        }
        passed, errors, _ = GroundingGate.verify_config(cfg, raise_on_error=False)
        self.assertFalse(passed)
        self.assertTrue(any("HIGH UNCERTAINTY FLAGGED" in err for err in errors))

        # Adding probe should pass
        cfg["assumption_register"]["verification_probe"] = "Run --debug --fold 0 first"
        passed_with_probe, errors_with_probe, _ = GroundingGate.verify_config(cfg, raise_on_error=False)
        self.assertTrue(passed_with_probe)

    def test_root_baseline_default_exemption(self):
        """Canonical root baseline with branch=baseline and parent_id=none gets default baseline grounding."""
        root_cfg = {
            "experiment": {
                "id": "exp001_baseline_tfidf",
                "branch": "baseline",
                "parent_id": "none",
                "problem_addressed": "Establish initial competitive control baseline across all subtasks.",
                "hypothesis": "Sparse n-gram TF-IDF provides a solid lower bound for Subtask A and B.",
                "intervention": "Canonical sparse baseline implementation across comment, title, and context.",
            }
        }
        passed, errors, receipt = GroundingGate.verify_config(root_cfg, raise_on_error=False)
        self.assertTrue(passed, f"Unexpected errors for root baseline: {errors}")
        self.assertEqual(receipt["grounding"]["source_type"], "verified_baseline")

    def test_raise_on_error_behavior(self):
        """When raise_on_error=True, invalid configs must raise EpistemicGroundingError."""
        bad_cfg = {"experiment": {"id": "bad_exp"}}
        with self.assertRaises(EpistemicGroundingError):
            GroundingGate.verify_config(bad_cfg, raise_on_error=True)

    def test_write_receipt(self):
        """GroundingGate.write_receipt should correctly serialize receipt to disk."""
        passed, _, receipt = GroundingGate.verify_config(self.valid_config, raise_on_error=False)
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = Path(tmpdir) / "GROUNDING_RECEIPT.json"
            GroundingGate.write_receipt(receipt, out_file)
            self.assertTrue(out_file.exists())
            with open(out_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.assertEqual(data["status"], "PASSED")
            self.assertEqual(data["experiment_id"], "exp010_model_mdeberta")


if __name__ == "__main__":
    unittest.main()
