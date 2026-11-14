"""Policy tests for total parameter accounting and competition rules."""

from __future__ import annotations

import unittest
from copy import deepcopy

from harness.model_assets import (
    MODEL_API_PROHIBITED,
    MODEL_LICENSE_UNRESOLVED,
    MODEL_PARAMETER_COUNT_UNKNOWN,
    MODEL_PARAMETER_LIMIT_EXCEEDED,
    MODEL_PRETRAINED_PROHIBITED,
    MODEL_REFERENCE_MISSING,
    MODEL_RELEASE_CUTOFF_VIOLATION,
    MODEL_RELEASE_EVIDENCE_MISSING,
    MODEL_REVISION_UNPINNED,
    ModelDefinition,
    evaluate_composition_policy,
    evaluate_model_policy,
    normalize_competition_rules,
)


def _model(
    *,
    model_id: str = "example-model",
    kind: str = "pretrained",
    family: str = "Example LLM",
    modality: str = "text",
    count: int | None = 1_000_000,
    trainable: int | None = None,
    release_date: str | None = "2026-01-01",
    license_expression: str | None = "Apache-2.0",
    provider: str = "huggingface",
    repository: str = "example/model",
    revision: str = "0123456789abcdef",
    source_uri: str = "hf://example/model@0123456789abcdef",
) -> ModelDefinition:
    return ModelDefinition.from_dict(
        {
            "model_id": model_id,
            "kind": kind,
            "family": family,
            "task_types": ["generation"],
            "modalities": [modality],
            "source": {
                "provider": provider,
                "repository": repository,
                "revision": revision,
                "source_uri": source_uri,
            },
            "artifacts": {"manifest_ref": None, "bundle_digest": None},
            "parameter_facts": {
                "exact_param_count": count,
                "count_basis": "logical_total",
                "trainable_param_count": trainable,
                "active_param_count": None,
                "evidence_refs": [],
            },
            "release": {"public_release_date": release_date, "evidence_refs": []},
            "license": {"expression": license_expression, "evidence_refs": []},
        }
    )


class TestModelGovernance(unittest.TestCase):
    def test_frozen_7b_backbone_plus_lora_counts_full_deployed_total(self):
        model = _model(count=7_008_000_000, trainable=8_000_000)
        assessment = evaluate_model_policy(
            model,
            normalize_competition_rules({"max_total_parameters": 6_000_000_000}),
        )

        self.assertFalse(assessment.eligible)
        self.assertEqual(assessment.status, "blocked")
        self.assertEqual(assessment.failure_code, MODEL_PARAMETER_LIMIT_EXCEEDED)
        self.assertEqual(assessment.facts_evaluated["exact_param_count"], 7_008_000_000)

    def test_unknown_parameter_count_is_not_treated_as_zero(self):
        assessment = evaluate_model_policy(
            _model(count=None),
            normalize_competition_rules({"max_total_parameters": 6_000_000_000}),
        )

        self.assertFalse(assessment.eligible)
        self.assertEqual(assessment.failure_code, MODEL_PARAMETER_COUNT_UNKNOWN)
        self.assertIn("unknown is not zero", assessment.reasons[0])

    def test_pretrained_release_before_cutoff_passes_and_after_fails(self):
        rules = normalize_competition_rules({"model_weight_release_cutoff": "2026-02-01"})

        before = evaluate_model_policy(_model(release_date="2026-01-31"), rules)
        after = evaluate_model_policy(_model(release_date="2026-02-02"), rules)

        self.assertTrue(before.eligible)
        self.assertFalse(after.eligible)
        self.assertEqual(after.failure_code, MODEL_RELEASE_CUTOFF_VIOLATION)

    def test_release_cutoff_inclusive_boundary_is_explicit(self):
        model = _model(release_date="2026-02-01")

        inclusive = evaluate_model_policy(
            model,
            normalize_competition_rules(
                {"model_weight_release_cutoff": "2026-02-01", "weight_cutoff_inclusive": True}
            ),
        )
        exclusive = evaluate_model_policy(
            model,
            normalize_competition_rules(
                {"model_weight_release_cutoff": "2026-02-01", "weight_cutoff_inclusive": False}
            ),
        )

        self.assertTrue(inclusive.eligible)
        self.assertEqual(exclusive.failure_code, MODEL_RELEASE_CUTOFF_VIOLATION)

    def test_pretrained_release_date_is_required_when_cutoff_applies(self):
        assessment = evaluate_model_policy(
            _model(release_date=None),
            normalize_competition_rules({"model_weight_release_cutoff": "2026-02-01"}),
        )
        self.assertEqual(assessment.failure_code, MODEL_RELEASE_EVIDENCE_MISSING)

    def test_allowed_license_comparison_is_case_insensitive(self):
        assessment = evaluate_model_policy(
            _model(license_expression="apache-2.0"),
            normalize_competition_rules({"allowed_licenses": ["Apache-2.0"]}),
        )
        self.assertTrue(assessment.eligible)

    def test_missing_or_disallowed_license_is_blocked(self):
        assessment = evaluate_model_policy(
            _model(license_expression="Proprietary"),
            normalize_competition_rules({"allowed_licenses": ["Apache-2.0"]}),
        )
        self.assertEqual(assessment.failure_code, MODEL_LICENSE_UNRESOLVED)

    def test_composition_sums_llm_vision_encoder_and_projector(self):
        components = [
            _model(model_id="llm", count=7_000_000_000, family="Example LLM"),
            _model(
                model_id="vision-encoder",
                count=1_000_000_000,
                family="Vision Transformer",
                modality="image",
            ),
            _model(
                model_id="projector",
                kind="checkpoint",
                count=10_000_000,
                family="Multimodal Projector",
                modality="image",
            ),
        ]

        assessment = evaluate_composition_policy(
            components,
            normalize_competition_rules({"max_total_parameters": 8_000_000_000}),
        )

        self.assertEqual(assessment.failure_code, MODEL_PARAMETER_LIMIT_EXCEEDED)
        self.assertEqual(assessment.facts_evaluated["total_parameters"], 8_010_000_000)
        self.assertEqual(assessment.facts_evaluated["llm_parameters"], 7_000_000_000)

    def test_composition_blocks_unknown_component_counts(self):
        assessment = evaluate_composition_policy(
            [_model(model_id="known", count=10), _model(model_id="unknown", count=None)],
            normalize_competition_rules({"max_total_parameters": 20}),
        )
        self.assertEqual(assessment.failure_code, MODEL_PARAMETER_COUNT_UNKNOWN)
        self.assertIsNone(assessment.facts_evaluated["total_parameters"])

    def test_llm_limit_only_applies_to_text_or_llm_family_models(self):
        image_encoder = _model(family="Vision Encoder", modality="image", count=None)
        assessment = evaluate_model_policy(
            image_encoder,
            normalize_competition_rules({"max_llm_parameters": 100}),
        )
        self.assertTrue(assessment.eligible)

    def test_competition_rule_normalization_copies_lists_and_preserves_input(self):
        raw = {
            "max_total_parameters": 100,
            "allowed_licenses": ["Apache-2.0"],
            "parameter_accounting": {"basis": "logical_total"},
        }
        original = deepcopy(raw)

        rules = normalize_competition_rules(raw)
        rules.allowed_licenses.append("MIT")

        self.assertEqual(raw, original)
        self.assertEqual(raw["allowed_licenses"], ["Apache-2.0"])

    def test_reference_revision_api_and_pretrained_restrictions(self):
        cases = [
            (
                _model(repository="", source_uri=""),
                normalize_competition_rules({}),
                MODEL_REFERENCE_MISSING,
            ),
            (
                _model(revision="main"),
                normalize_competition_rules({}),
                MODEL_REVISION_UNPINNED,
            ),
            (
                _model(kind="service", family="Hosted LLM"),
                normalize_competition_rules({"allow_api": False}),
                MODEL_API_PROHIBITED,
            ),
            (
                _model(),
                normalize_competition_rules({"allow_pretrained_models": False}),
                MODEL_PRETRAINED_PROHIBITED,
            ),
        ]
        for model, rules, failure_code in cases:
            with self.subTest(failure_code=failure_code):
                self.assertEqual(evaluate_model_policy(model, rules).failure_code, failure_code)

    def test_rule_normalization_rejects_malformed_values(self):
        cases = [
            {"max_total_parameters": True},
            {"max_llm_parameters": -1},
            {"model_weight_release_cutoff": "2026-2-1"},
            {"allow_api": "false"},
            {"allowed_licenses": "Apache-2.0"},
        ]
        for raw in cases:
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    normalize_competition_rules(raw)

    def test_task_spec_governance_accessors_are_non_mutating_and_safe(self):
        from harness.task_spec import UniversalTaskSpec

        original = {"competition_rules": {"max_total_parameters": 10}, "model_governance": {"mode": "required"}}
        spec = UniversalTaskSpec.from_dict(
            {"task": {"name": "task", "competition_rules": original["competition_rules"], "model_governance": original["model_governance"]}}
        )

        rules = spec.competition_rules
        governance = spec.model_governance_config
        rules["max_total_parameters"] = 5
        governance["mode"] = "optional"
        self.assertEqual(spec.competition_rules, {"max_total_parameters": 10})
        self.assertEqual(spec.model_governance_config, {"mode": "required"})

        malformed = UniversalTaskSpec.from_dict(
            {"task": {"name": "task", "competition_rules": [], "model_governance": "required"}}
        )
        self.assertEqual(malformed.competition_rules, {})
        self.assertEqual(malformed.model_governance_config, {})

    def test_additive_policy_accessors_do_not_change_legacy_scorer_identity_source(self):
        from harness.artifacts import _stable_task_spec_source

        legacy_source = (
            "class UniversalTaskSpec:\n"
            "    metadata = {}\n"
            "\n"
            "    @property\n"
            "    def platform(self):\n"
            "        return self.metadata.get('platform', '')\n"
        )
        extended_source = (
            "class UniversalTaskSpec:\n"
            "    metadata = {}\n"
            "\n"
            "    @property\n"
            "    def competition_rules(self):\n"
            "        raw = self.metadata.get('competition_rules', {})\n"
            "        return dict(raw) if isinstance(raw, dict) else {}\n"
            "\n"
            "    @property\n"
            "    def model_governance_config(self):\n"
            "        raw = self.metadata.get('model_governance', {})\n"
            "        return dict(raw) if isinstance(raw, dict) else {}\n"
            "\n"
            "    @property\n"
            "    def platform(self):\n"
            "        return self.metadata.get('platform', '')\n"
        )

        self.assertEqual(_stable_task_spec_source(extended_source), legacy_source)


if __name__ == "__main__":
    unittest.main()
