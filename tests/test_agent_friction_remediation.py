"""Tests for agent friction remediation (Kaggle facade, grounding whitelist, action schemas).

Addresses the real friction points audited from Antigravity conversations:
- NormalisedObject dict-like interface (.get(), obj[], in)
- kernels_status() returning NormalisedObject & handling transient 404
- kernels_list alias on KaggleClient
- GroundingGate accepting Kaggle competitive evidence sources
- Action schema discovery on CLI and programmatic API
"""
from __future__ import annotations

import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from harness.bridge_common import _TYPED_ACTION_NAMES, get_action_schema
from harness.grounding_gate import GroundingGate
from harness.kaggle_compat import KaggleClient, _NormalisedObject
from tools.agent_bridge_cli import _format_action_schema, main as cli_main


class _FakeEnumStatus:
    def __init__(self, name: str):
        self.name = name


class NormalisedObjectDictInterfaceTests(unittest.TestCase):
    def test_get_existing_and_missing(self):
        raw = SimpleNamespace(status="COMPLETE", failureMessage=None)
        obj = _NormalisedObject(raw, "kagglesdk")
        self.assertEqual(obj.get("status"), "COMPLETE")
        self.assertEqual(obj.get("failure_message"), None)
        self.assertIsNone(obj.get("missing_key"))
        self.assertEqual(obj.get("missing_key", "default_val"), "default_val")

    def test_getitem_and_contains(self):
        raw = SimpleNamespace(status="RUNNING")
        obj = _NormalisedObject(raw, "kagglesdk")
        self.assertEqual(obj["status"], "RUNNING")
        self.assertIn("status", obj)
        self.assertNotIn("non_existent", obj)
        with self.assertRaises(KeyError):
            _ = obj["non_existent"]

    def test_dict_raw_object_support(self):
        raw_dict = {"status": "QUEUED", "failureMessage": None}
        obj = _NormalisedObject(raw_dict, "kagglesdk")
        self.assertEqual(obj.status, "QUEUED")
        self.assertEqual(obj.get("status"), "QUEUED")
        self.assertEqual(obj["status"], "QUEUED")
        self.assertEqual(obj.failure_message, None)

    def test_enum_status_normalization(self):
        raw = SimpleNamespace(status=_FakeEnumStatus("COMPLETE"))
        obj = _NormalisedObject(raw, "kagglesdk")
        self.assertEqual(obj.status, "COMPLETE")
        self.assertEqual(obj.get("status"), "COMPLETE")
        self.assertEqual(obj["status"], "COMPLETE")


class KaggleClientFacadeEnhancementTests(unittest.TestCase):
    def test_kernels_list_alias_exists(self):
        client = KaggleClient()
        self.assertTrue(hasattr(client, "kernels_list"))
        self.assertEqual(client.kernels_list, client.list_kernels)

    def test_kernels_status_returns_normalised_object(self):
        client = KaggleClient()
        mock_raw = SimpleNamespace(status="COMPLETE", failureMessage=None)
        with patch.object(client, "_legacy_api_instance") as mock_api:
            mock_api.return_value.kernels_status.return_value = mock_raw
            st = client.kernels_status("user/test-kernel")
            self.assertIsInstance(st, _NormalisedObject)
            self.assertEqual(st.status, "COMPLETE")
            self.assertEqual(st.get("status"), "COMPLETE")
            self.assertEqual(st["status"], "COMPLETE")

    def test_kernels_status_handles_transient_404(self):
        client = KaggleClient()
        mock_404 = Exception("404 Client Error: Not Found for url: ...")
        with patch.object(client, "_legacy_api_instance") as mock_api:
            mock_api.return_value.kernels_status.side_effect = mock_404
            st = client.kernels_status("user/newly-pushed-kernel")
            self.assertIsInstance(st, _NormalisedObject)
            self.assertEqual(st.status, "QUEUED")
            self.assertEqual(st.get("status"), "QUEUED")
            self.assertEqual(st["status"], "QUEUED")


class GroundingGateKaggleWhitelistTests(unittest.TestCase):
    def test_kaggle_public_lb_source_accepted(self):
        cfg = {
            "experiment": {
                "id": "exp001_kaggle_treatment",
                "branch": "feat",
                "parent_id": "exp000_baseline",
                "problem_addressed": "Severe spatial inequality in target variable distributions across folds.",
                "hypothesis": "Adding calibrated geo-spatial coordinates reduces RMSE by 0.05 on local CV.",
                "intervention": "Include longitude and latitude orthogonal projection features.",
            },
            "grounding": {
                "source_type": "kaggle_public_lb",
                "source_id": "rsna_knee_public_lb_benchmark",
                "retrieval_summary": "Empirical top rank 10 submissions use 3D spatial pooling.",
                "verified_via": "kaggle_public_lb",
            },
            "assumption_register": {
                "known_facts": ["Public LB score correlates with local out-of-fold validation."],
                "assumptions": ["Orthogonal projection improves convergence across folds."],
                "uncertainty_level": "low",
            },
        }
        passed, errors, _ = GroundingGate.verify_config(cfg, raise_on_error=False)
        self.assertTrue(passed, f"Expected pass, got errors: {errors}")

    def test_kaggle_discussion_and_notebook_accepted(self):
        for src in ("kaggle_discussion", "kaggle_notebook", "kaggle_api"):
            with self.subTest(source=src):
                cfg = {
                    "experiment": {
                        "id": f"exp_{src}",
                        "branch": "feat",
                        "parent_id": "exp000_baseline",
                        "problem_addressed": "Significant false positive rate on rare class predictions.",
                        "hypothesis": "Focal loss formulation down-weights well-classified examples by gamma=2.0.",
                        "intervention": "Replace cross-entropy loss with focal loss gamma=2.0.",
                    },
                    "grounding": {
                        "source_type": "kaggle_discussion" if src != "kaggle_notebook" else "kaggle_notebook",
                        "source_id": f"kaggle_ref_{src}",
                        "retrieval_summary": "Top discussion topic reveals high class imbalance impact.",
                        "verified_via": src,
                    },
                    "assumption_register": {
                        "known_facts": ["Dataset exhibits 90/10 class skew."],
                        "assumptions": ["Gamma=2.0 prevents gradient vanishing for hard examples."],
                        "uncertainty_level": "medium",
                    },
                }
                passed, errors, _ = GroundingGate.verify_config(cfg, raise_on_error=False)
                self.assertTrue(passed, f"Expected pass for {src}, got errors: {errors}")

    def test_unverified_sentinel_still_rejected(self):
        cfg = {
            "experiment": {
                "id": "exp_bad",
                "branch": "feat",
                "parent_id": "exp000_baseline",
                "problem_addressed": "Severe spatial inequality in target variable distributions across folds.",
                "hypothesis": "Adding calibrated geo-spatial coordinates reduces RMSE by 0.05 on local CV.",
                "intervention": "Include longitude and latitude orthogonal projection features.",
            },
            "grounding": {
                "source_type": "kaggle_public_lb",
                "source_id": "rsna_knee_public_lb_benchmark",
                "retrieval_summary": "Empirical top rank 10 submissions use 3D spatial pooling.",
                "verified_via": "not_verified",
            },
            "assumption_register": {
                "known_facts": ["Validation data loaded."],
                "assumptions": ["Projection works."],
                "uncertainty_level": "low",
            },
        }
        passed, errors, _ = GroundingGate.verify_config(cfg, raise_on_error=False)
        self.assertFalse(passed)
        self.assertTrue(any("not verified" in err for err in errors))


class ActionDiscoveryAndSchemaTests(unittest.TestCase):
    def test_get_action_schema_returns_registered_schema(self):
        schema = get_action_schema("record_research_insight")
        self.assertIsNotNone(schema)
        self.assertIn("title", schema["required"])
        self.assertIn("observation", schema["required"])
        self.assertIn("mechanism", schema["required"])
        self.assertIn("next_action", schema["required"])

    def test_all_typed_action_names_have_schema(self):
        for name in _TYPED_ACTION_NAMES:
            schema = get_action_schema(name)
            self.assertIsNotNone(schema, f"Missing schema for action '{name}'")
            self.assertIn("summary", schema)
            self.assertIn("description", schema)
            self.assertIn("required", schema)
            self.assertIn("optional", schema)

    def test_format_action_schema_output(self):
        schema = get_action_schema("record_research_insight")
        formatted = _format_action_schema("record_research_insight", schema)
        self.assertIn("Action: record_research_insight", formatted)
        self.assertIn("Required Parameters:", formatted)
        self.assertIn("--title", formatted)
        self.assertIn("Example Invocation:", formatted)

    def test_cli_schema_subcommand(self):
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            code = cli_main(["schema", "--action", "record_research_insight"])
        self.assertEqual(code, 0)
        output = buf.getvalue()
        self.assertIn("Action: record_research_insight", output)
        self.assertIn("--title", output)

    def test_cli_act_help_interception(self):
        buf = io.StringIO()
        with patch("sys.stdout", buf):
            code = cli_main(["act", "--action", "record_research_insight", "--help"])
        self.assertEqual(code, 0)
        output = buf.getvalue()
        self.assertIn("Action: record_research_insight", output)
        self.assertIn("--observation", output)


if __name__ == "__main__":
    unittest.main()
