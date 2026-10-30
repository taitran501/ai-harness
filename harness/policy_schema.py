"""Dependency-free validation of the bundled Draft 2020-12 policy schemas.

Only the keywords used by these schemas are interpreted. The JSON documents
remain consumable by a full Draft 2020-12 validator in integrations/tests.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path


def validate_document(name, value, *, version=1):
    """Validate a policy document using an explicitly selected schema version.

    Version 1 remains the compatibility default for existing callers. New
    citation-aware receipts must ask for version 2 explicitly.
    """
    if type(version) is not int or version < 1:
        raise ValueError("Schema version must be a positive integer")
    schema_path = Path(__file__).parent / "schemas" / f"{name}.v{version}.schema.json"
    if not schema_path.is_file():
        raise ValueError(f"Unsupported schema version: {name}.v{version}")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    def check(node, obj, path="$", root=schema):
        if "$ref" in node:
            ref = node["$ref"]
            target = root
            if not ref.startswith("#/"):
                raise ValueError("Unsupported external schema reference")
            for part in ref[2:].split("/"):
                target = target[part]
            check(target, obj, path)
        for mode in ("oneOf", "anyOf"):
            if mode in node:
                passed = 0
                for branch in node[mode]:
                    try:
                        check(branch, obj, path)
                        passed += 1
                    except ValueError:
                        pass
                if passed == 0 or (mode == "oneOf" and passed != 1):
                    raise ValueError(f"{path}: {mode} failed")
        for branch in node.get("allOf", []):
            check(branch, obj, path)
        if "if" in node:
            try:
                check(node["if"], obj, path)
                branch = "then"
            except ValueError:
                branch = "else"
            check(node.get(branch, {}), obj, path)
        if "const" in node and (obj != node["const"] or type(obj) is not type(node["const"])):
            raise ValueError(f"{path}: invalid constant")
        if "enum" in node and obj not in node["enum"]:
            raise ValueError(f"{path}: invalid enum")
        types = node.get("type", [])
        types = [types] if isinstance(types, str) else types
        matches = {"null": obj is None, "object": isinstance(obj, dict),
                   "array": isinstance(obj, list), "string": isinstance(obj, str),
                   "boolean": type(obj) is bool, "integer": type(obj) is int,
                   "number": type(obj) in (int, float) and math.isfinite(obj)}
        if types and not any(matches[t] for t in types):
            raise ValueError(f"{path}: expected {types}")
        if isinstance(obj, dict):
            missing = set(node.get("required", [])) - obj.keys()
            if missing:
                raise ValueError(f"{path}: missing {sorted(missing)}")
            props = node.get("properties", {})
            if node.get("additionalProperties") is False and obj.keys() - props.keys():
                raise ValueError(f"{path}: unknown fields {sorted(obj.keys() - props.keys())}")
            for key, val in obj.items():
                if key in props:
                    check(props[key], val, f"{path}.{key}")
                elif isinstance(node.get("additionalProperties"), dict):
                    check(node["additionalProperties"], val, f"{path}.{key}")
        if isinstance(obj, list):
            if len(obj) < node.get("minItems", 0):
                raise ValueError(f"{path}: too few items")
            if node.get("uniqueItems") and len({json.dumps(v, sort_keys=True) for v in obj}) != len(obj):
                raise ValueError(f"{path}: duplicate items")
            for item in obj:
                check(node.get("items", {}), item, path + "[]")
        if isinstance(obj, str):
            if len(obj) < node.get("minLength", 0) or ("pattern" in node and not re.search(node["pattern"], obj)):
                raise ValueError(f"{path}: invalid string")
        if type(obj) in (float, int):
            if not math.isfinite(obj):
                raise ValueError(f"{path}: non-finite number")
            for keyword, predicate in (("minimum", lambda a, b: a >= b), ("maximum", lambda a, b: a <= b),
                                       ("exclusiveMinimum", lambda a, b: a > b), ("exclusiveMaximum", lambda a, b: a < b)):
                if keyword in node and not predicate(obj, node[keyword]):
                    raise ValueError(f"{path}: violates {keyword}")

    check(schema, value)
    return value
