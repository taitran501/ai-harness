"""Bounded storage tests for model asset definitions and legacy model names."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from harness.model_assets import ModelDefinition, ModelStore
from harness.model_registry import builtin_model_names


def _definition() -> dict:
    return {
        "schema_version": 1,
        "model_id": "  test-model  ",
        "kind": "PRETRAINED",
        "family": "  Test LLM  ",
        "task_types": ["generation"],
        "modalities": ["text"],
        "source": {
            "provider": "huggingface",
            "repository": "example/test-model",
            "revision": "abcdef0123456789",
            "source_uri": "hf://example/test-model@abcdef0123456789",
        },
        "artifacts": {"manifest_ref": None, "bundle_digest": None},
        "parameter_facts": {
            "exact_param_count": 100,
            "count_basis": "LOGICAL_TOTAL",
            "active_param_count": None,
            "trainable_param_count": 100,
            "evidence_refs": ["count-report"],
        },
        "release": {"public_release_date": "2026-01-01", "evidence_refs": []},
        "license": {"expression": "Apache-2.0", "evidence_refs": []},
        "parents": [],
        "verification_refs": [],
    }


def test_load_valid_model_definition_yaml_under_size_limit(tmp_path: Path):
    path = tmp_path / "model.yaml"
    path.write_text(yaml.safe_dump(_definition(), sort_keys=False), encoding="utf-8")

    model = ModelStore().load_catalog_file(path)

    assert model.model_id == "test-model"
    assert model.kind == "pretrained"
    assert model.parameter_facts.count_basis == "logical_total"


def test_rejects_model_definition_larger_than_16_kib(tmp_path: Path):
    path = tmp_path / "oversized.yaml"
    path.write_bytes(b" " * (16 * 1024 + 1))

    with pytest.raises(ValueError, match="exceeds 16 KiB limit"):
        ModelStore().load_catalog_file(path)


def test_rejects_unknown_nested_fields_and_tensor_cards(tmp_path: Path):
    definition = _definition()
    definition["parameter_facts"]["tensor_card"] = {"weights": [1, 2, 3]}
    path = tmp_path / "extra-field.yaml"
    path.write_text(yaml.safe_dump(definition), encoding="utf-8")

    with pytest.raises(ValueError, match="Unknown field.*tensor_card"):
        ModelStore().load_catalog_file(path)


def test_rejects_yaml_aliases_and_duplicate_keys(tmp_path: Path):
    alias_path = tmp_path / "alias.yaml"
    alias_path.write_text("source: &source {provider: local}\ncopy: *source\n", encoding="utf-8")
    duplicate_path = tmp_path / "duplicate.yaml"
    duplicate_path.write_text("model_id: first\nmodel_id: second\n", encoding="utf-8")

    with pytest.raises(ValueError, match="aliases are not allowed"):
        ModelStore().load_catalog_file(alias_path)
    with pytest.raises(ValueError, match="Duplicate YAML key"):
        ModelStore().load_catalog_file(duplicate_path)


def test_json_and_yaml_definitions_parse_to_same_normalized_record(tmp_path: Path):
    definition = _definition()
    yaml_path = tmp_path / "model.yaml"
    json_path = tmp_path / "model.json"
    yaml_path.write_text(yaml.safe_dump(definition), encoding="utf-8")
    json_path.write_text(json.dumps(definition), encoding="utf-8")
    store = ModelStore()

    assert store.load_catalog_file(yaml_path).to_dict() == store.load_catalog_file(json_path).to_dict()


def test_compute_digest_uses_canonical_normalized_json():
    first = ModelDefinition.from_dict(_definition())
    reordered = dict(reversed(list(_definition().items())))
    second = ModelDefinition.from_dict(reordered)
    canonical = json.dumps(
        first.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")

    assert first.compute_digest() == second.compute_digest()
    assert first.compute_digest() == hashlib.sha256(canonical).hexdigest()
    assert len(first.compute_digest()) == 64


def test_register_definition_writes_atomically_at_content_addressed_path(tmp_path: Path):
    store = ModelStore()
    model = ModelDefinition.from_dict(_definition())

    digest = store.register_definition(model, tmp_path)
    object_path = tmp_path / ".ai-harness" / "models" / "objects" / f"{digest}.json"

    assert object_path.is_file()
    assert json.loads(object_path.read_text(encoding="utf-8")) == model.to_dict()
    assert list(object_path.parent.iterdir()) == [object_path]
    assert store.load_definition(digest, tmp_path).to_dict() == model.to_dict()


def test_load_definition_returns_none_for_absent_digest_and_rejects_bad_digest(tmp_path: Path):
    store = ModelStore()
    assert store.load_definition("0" * 64, tmp_path) is None
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        store.load_definition("../model", tmp_path)


def test_load_definition_detects_content_tampering(tmp_path: Path):
    store = ModelStore()
    digest = store.register_definition(ModelDefinition.from_dict(_definition()), tmp_path)
    path = tmp_path / ".ai-harness" / "models" / "objects" / f"{digest}.json"
    changed = json.loads(path.read_text(encoding="utf-8"))
    changed["family"] = "Tampered family"
    path.write_text(json.dumps(changed), encoding="utf-8")

    with pytest.raises(ValueError, match="digest mismatch"):
        store.load_definition(digest, tmp_path)


def test_existing_builtin_registry_still_exposes_sklearn_and_lightgbm_models():
    names = builtin_model_names()
    assert "logistic_regression" in names
    assert "lightgbm" in names

