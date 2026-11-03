"""Task-scoped CLI and service helpers for native model assets.

The command surface is intentionally local-only. Registration stores a bounded
declaration; validation evaluates policy facts; verification is the only path
that reads model weights and computes their SHA-256 digests.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Dict, Optional, Sequence

from harness.io_utils import atomic_write_text
from harness.model_assets.policy import evaluate_model_policy, normalize_competition_rules
from harness.model_assets.resolution import (
    prepare_or_validate_model_plan,
    resolve_effective_config,
)
from harness.model_assets.resolver import ModelResolver
from harness.model_assets.store import ModelStore
from harness.model_assets.types import ModelDefinition
from harness.project import HarnessProject
from harness.task_spec import load_active_task_spec


_INDEX_NAME = "index.json"
_INDEX_VERSION = 1
_MAX_PAGE_SIZE = 100
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")


def _project(root: Path | str = ".", task_config: Optional[str | Path] = None) -> HarnessProject:
    return HarnessProject.discover(root, task_config)


def _task_id(project: HarnessProject, task_spec: Any) -> str:
    return project.task_id or str(getattr(task_spec, "name", "task")).strip().lower().replace(" ", "-")


def _registry_paths(project_root: Path) -> list[Path]:
    object_dir = ModelStore._objects_dir(project_root)
    if not object_dir.is_dir():
        return []
    return sorted(path for path in object_dir.glob("*.json") if path.is_file())


def _load_models(project_root: Path) -> tuple[list[tuple[str, ModelDefinition]], list[dict[str, str]]]:
    store = ModelStore()
    models: list[tuple[str, ModelDefinition]] = []
    invalid: list[dict[str, str]] = []
    for path in _registry_paths(project_root):
        digest = path.stem
        try:
            definition = store.load_definition(digest, project_root)
            if definition is None:
                invalid.append({"digest": digest, "error": "definition is missing"})
                continue
            models.append((digest, definition))
        except (OSError, TypeError, ValueError) as exc:
            invalid.append({"digest": digest, "error": str(exc)[:240]})
    return models, invalid


def _resolve_model_ref(project_root: Path, reference: str) -> tuple[str, ModelDefinition]:
    reference = str(reference).strip()
    store = ModelStore()
    if _DIGEST_RE.fullmatch(reference):
        definition = store.load_definition(reference, project_root)
        if definition is None:
            raise ValueError(f"No registered model definition for digest {reference}")
        return reference, definition

    matches = [(digest, model) for digest, model in _load_models(project_root)[0] if model.model_id == reference]
    if not matches:
        raise ValueError(f"No registered model found for model_id {reference!r}")
    if len(matches) != 1:
        digests = ", ".join(digest for digest, _ in matches[:5])
        raise ValueError(
            f"model_id {reference!r} has {len(matches)} immutable definitions; inspect by digest: {digests}"
        )
    return matches[0]


def _task_context(project: HarnessProject) -> tuple[Any, str]:
    if project.task_config_path is None or not project.task_config_path.is_file():
        raise FileNotFoundError(
            "Task config is missing for the selected workspace; pass --task-config or attach a task workspace."
        )
    task_spec = load_active_task_spec(project.task_config_path)
    return task_spec, _task_id(project, task_spec)


def _index_path(project_root: Path) -> Path:
    return project_root / ".ai-harness" / "models" / _INDEX_NAME


def _verification_path(project_root: Path, digest: str) -> Path:
    return project_root / ".ai-harness" / "models" / "verifications" / f"{digest}.json"


def _is_explicitly_verified(project_root: Path, digest: str) -> bool:
    try:
        receipt = json.loads(_verification_path(project_root, digest).read_text(encoding="utf-8"))
        return receipt.get("verified") is True and receipt.get("definition_digest") == digest
    except (OSError, json.JSONDecodeError, AttributeError):
        return False


def rebuild_model_index(
    project_root: Path | str,
    *,
    task_spec: Any = None,
    task_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Rebuild the small summary index from immutable definition objects.

    The object files remain authoritative. The index contains aggregate counts
    only, so the Context HUD can read it without loading model definitions.
    """
    root = Path(project_root).expanduser().resolve()
    models, invalid = _load_models(root)
    object_dir = ModelStore._objects_dir(root)
    try:
        object_mtime_ns = object_dir.stat().st_mtime_ns
    except OSError:
        object_mtime_ns = None

    rules = getattr(task_spec, "competition_rules", {}) if task_spec is not None else {}
    rules = dict(rules) if isinstance(rules, dict) else {}
    eligible_count = 0
    blocked_count = 0
    if task_spec is not None:
        try:
            normalized_rules = normalize_competition_rules(rules)
            for _, definition in models:
                assessment = evaluate_model_policy(definition, normalized_rules)
                if assessment.eligible is True:
                    eligible_count += 1
                elif assessment.status == "blocked":
                    blocked_count += 1
        except (TypeError, ValueError):
            # A malformed task policy cannot establish eligibility.
            eligible_count = 0
            blocked_count = 0

    verified_count = sum(_is_explicitly_verified(root, digest) for digest, _ in models)
    current_task_id = task_id or (
        _task_id(HarnessProject.discover(root), task_spec) if task_spec is not None else "unknown"
    )
    path = _index_path(root)
    prior_tasks: Dict[str, Any] = {}
    try:
        prior = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(prior.get("tasks"), dict):
            prior_tasks = dict(prior["tasks"])
    except (OSError, json.JSONDecodeError, AttributeError):
        pass
    if task_spec is not None:
        prior_tasks[current_task_id] = {
            "competition_rules": rules,
            "eligible_count": eligible_count,
            "blocked_count": blocked_count,
        }
    index = {
        "schema_version": _INDEX_VERSION,
        "object_count": len(_registry_paths(root)),
        "registered_count": len(models),
        "invalid_count": len(invalid),
        "verified_count": verified_count,
        "unverified_count": max(0, len(models) - verified_count),
        "objects_mtime_ns": object_mtime_ns,
        "tasks": prior_tasks,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_text(path, json.dumps(index, sort_keys=True, indent=2), encoding="utf-8")
    return index


def list_model_assets(
    project_root: Path | str = ".",
    *,
    limit: int = 20,
    offset: int = 0,
    task_config: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Return one bounded page of task-workspace model definitions."""
    project = _project(project_root, task_config)
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= _MAX_PAGE_SIZE:
        raise ValueError(f"limit must be between 1 and {_MAX_PAGE_SIZE}")
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise ValueError("offset must be a non-negative integer")
    models, invalid = _load_models(project.root)
    models.sort(key=lambda item: (item[1].model_id.casefold(), item[0]))
    page = models[offset : offset + limit]
    entries = [
        {
            "model_id": model.model_id,
            "definition_digest": digest,
            "kind": str(model.kind),
            "family": model.family,
            "release_date": model.release.public_release_date,
            "parameter_count": model.parameter_facts.exact_param_count,
            "license": model.license.expression,
        }
        for digest, model in page
    ]
    next_offset = offset + len(page) if offset + len(page) < len(models) else None
    task_spec, task_id = _task_context(project)
    del task_spec
    return {
        "status": "ok",
        "task_id": task_id,
        "operation": "list",
        "models": entries,
        "checks": {"definitions_validated": len(page), "invalid_objects": len(invalid)},
        "blockers": invalid[:5],
        "next_action": "Inspect a model by model_id or immutable definition digest." if entries else "Register a model definition with `ai-harness model register --file <path>`. ",
        "artifact_refs": [f".ai-harness/models/objects/{digest}.json" for digest, _ in page],
        "pagination": {
            "limit": limit,
            "offset": offset,
            "total": len(models),
            "next_offset": next_offset,
        },
    }


def _backend_mappings(task_spec: Any, model_id: str, digest: str) -> Dict[str, Any]:
    metadata = getattr(task_spec, "metadata", {}) or {}
    governance = getattr(task_spec, "model_governance_config", {}) or {}
    configured = (
        governance.get("model_locations")
        or governance.get("locations")
        or metadata.get("model_locations")
        or {}
    )
    result: Dict[str, Any] = {}
    if not isinstance(configured, dict):
        return result
    entry = configured.get(model_id, configured.get(digest, configured))
    if not isinstance(entry, dict):
        return result
    for backend in ("local", "kaggle", "modal"):
        value = entry.get(backend)
        if value is not None:
            result[backend] = value
    if not result and any(key in entry for key in ("path", "root", "directory", "dataset", "volume_name")):
        result["declared"] = entry
    return result


def inspect_model_asset(
    project_root: Path | str,
    model_ref: str,
    *,
    task_config: Optional[str | Path] = None,
) -> Dict[str, Any]:
    project = _project(project_root, task_config)
    task_spec, task_id = _task_context(project)
    digest, model = _resolve_model_ref(project.root, model_ref)
    return {
        "status": "ok",
        "task_id": task_id,
        "operation": "inspect",
        "model": model.to_dict(),
        "definition_digest": digest,
        "artifact_verification": "verified" if _is_explicitly_verified(project.root, digest) else "pending",
        "eligibility": "not assessed",
        "backend_mappings": _backend_mappings(task_spec, model.model_id, digest),
        "checks": {"definition_integrity": "passed", "weight_artifacts": "not checked"},
        "blockers": [],
        "next_action": f"Run `ai-harness model verify {digest}` to check local artifact files.",
        "artifact_refs": [f".ai-harness/models/objects/{digest}.json"],
    }


def register_model_asset(
    project_root: Path | str,
    file_path: str | Path,
    *,
    task_config: Optional[str | Path] = None,
) -> Dict[str, Any]:
    project = _project(project_root, task_config)
    source_path = Path(file_path).expanduser()
    if not source_path.is_absolute():
        source_path = project.root / source_path
    definition = ModelStore().load_catalog_file(source_path.resolve())
    digest = ModelStore().register_definition(definition, project.root)
    task_spec, task_id = _task_context(project)
    rebuild_model_index(project.root, task_spec=task_spec, task_id=task_id)
    return {
        "status": "registered",
        "task_id": task_id,
        "operation": "register",
        "model_id": definition.model_id,
        "definition_digest": digest,
        "checks": {"definition_schema": "passed", "definition_size_limit": "passed"},
        "blockers": [],
        "artifact_verification": "pending",
        "eligibility": "not assessed",
        "next_action": f"Run `ai-harness model verify {digest}` to check local artifact files.",
        "artifact_refs": [f".ai-harness/models/objects/{digest}.json"],
    }


def _model_plan(project: HarnessProject, config_path: str | Path) -> tuple[Any, Any, Dict[str, Any], Path]:
    task_spec, task_id = _task_context(project)
    config = Path(config_path).expanduser()
    if not config.is_absolute():
        config = project.root / config
    config = config.resolve()
    try:
        config.relative_to(project.root)
    except ValueError as exc:
        raise ValueError("Experiment config must be inside the selected task workspace") from exc
    effective = resolve_effective_config(project, config)
    plan = prepare_or_validate_model_plan(task_spec, effective, ModelStore(), project=project)
    return task_spec, plan, {"task_id": task_id, "effective_config": effective}, config


def validate_model_plan(
    project_root: Path | str,
    config_path: str | Path,
    *,
    task_config: Optional[str | Path] = None,
) -> Dict[str, Any]:
    project = _project(project_root, task_config)
    task_spec, plan, context, config = _model_plan(project, config_path)
    receipt = plan.to_receipt_dict()
    lock = plan.to_lock_dict()
    is_blocked = plan.status == "blocked"
    blockers = list(receipt.get("reasons", [])) if is_blocked else []
    return {
        "status": "blocked" if is_blocked else plan.status,
        "task_id": context["task_id"],
        "operation": "validate-config",
        "model_lock_digest": plan.lock_digest,
        "model_policy_receipt": receipt,
        "model_lock": lock,
        "checks": {
            "policy": plan.status,
            "eligible": plan.eligible,
            "selected_roots": len(lock.get("roots", [])),
            "competition_rules": "loaded",
        },
        "blockers": blockers,
        "next_action": (
            "Resolve the reported policy blockers before execution."
            if is_blocked
            else "Model plan is available for the configured execution path."
        ),
        "artifact_refs": [str(config.relative_to(project.root)).replace("\\", "/")],
    }


def resolve_model_assets(
    project_root: Path | str,
    config_path: str | Path,
    backend: str,
    *,
    task_config: Optional[str | Path] = None,
) -> Dict[str, Any]:
    project = _project(project_root, task_config)
    task_spec, plan, context, config = _model_plan(project, config_path)
    if plan.status == "blocked":
        result = validate_model_plan(project.root, config, task_config=project.task_config_path)
        result["operation"] = "resolve"
        result["dry_run"] = True
        return result
    scope = {
        "root": project.root,
        **(getattr(task_spec, "metadata", {}) or {}),
        **(getattr(task_spec, "model_governance_config", {}) or {}),
    }
    scope["root"] = project.root
    resource_plan = ModelResolver().plan_backend_resources(plan, backend, scope)
    return {
        "status": plan.status,
        "task_id": context["task_id"],
        "operation": "resolve",
        "backend": backend,
        "dry_run": True,
        "model_lock_digest": plan.lock_digest,
        "resource_plan": resource_plan.to_dict(),
        "checks": {"policy": plan.status, "resource_plan": "created", "provider_dispatch": "not performed"},
        "blockers": [
            error
            for source in resource_plan.required_sources
            for error in source.get("missing_prerequisites", [])
        ],
        "next_action": "Review the declared mount paths and prerequisites before a governed run.",
        "artifact_refs": [str(config.relative_to(project.root)).replace("\\", "/")],
    }


def verify_model_asset(
    project_root: Path | str,
    model_ref: str,
    *,
    task_config: Optional[str | Path] = None,
) -> Dict[str, Any]:
    project = _project(project_root, task_config)
    task_spec, task_id = _task_context(project)
    digest, model = _resolve_model_ref(project.root, model_ref)
    verification_config = {
        "model_assets": {
            "schema_version": 1,
            "roots": [{"slot": "model", "ref": digest, "phases": ["training", "inference"]}],
        }
    }
    plan = prepare_or_validate_model_plan(task_spec, verification_config, ModelStore(), project=project)
    scope = {
        "root": project.root,
        **(getattr(task_spec, "metadata", {}) or {}),
        **(getattr(task_spec, "model_governance_config", {}) or {}),
    }
    scope["root"] = project.root
    resource_plan = ModelResolver().plan_backend_resources(plan, "local", scope)
    resolver = ModelResolver()
    verified = resolver.verify_runtime_resources(resource_plan)
    receipt_path = _verification_path(project.root, digest)
    receipt = {
        "schema_version": 1,
        "definition_digest": digest,
        "verified": verified,
        "backend": "local",
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "files_checked": sum(len(source.get("files", [])) for source in resource_plan.required_sources),
        "errors": list(resolver.verification_errors),
    }
    atomic_write_text(receipt_path, json.dumps(receipt, sort_keys=True, indent=2), encoding="utf-8")
    rebuild_model_index(project.root, task_spec=task_spec, task_id=task_id)
    errors = list(resolver.verification_errors)
    return {
        "status": "verified" if verified else "failed",
        "task_id": task_id,
        "operation": "verify",
        "model_id": model.model_id,
        "definition_digest": digest,
        "checks": {
            "definition_integrity": "passed",
            "artifact_manifest": "passed" if resource_plan.required_sources and all(source.get("files") for source in resource_plan.required_sources) else "failed",
            "artifact_files": "passed" if verified else "failed",
            "files_checked": receipt["files_checked"],
        },
        "blockers": errors,
        "next_action": "Use the registered model in a config after policy validation." if verified else "Resolve the artifact verification blockers and rerun verify.",
        "artifact_refs": [
            f".ai-harness/models/objects/{digest}.json",
            str(receipt_path.relative_to(project.root)).replace("\\", "/"),
        ],
        "resource_plan": resource_plan.to_dict(),
    }


def _emit(result: Dict[str, Any], *, exit_code: int = 0) -> int:
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    return exit_code


def add_model_subparsers(parser: argparse.ArgumentParser) -> None:
    """Register the model command surface under the root ``ai-harness`` parser."""
    commands = parser.add_subparsers(dest="model_command", required=True)
    list_parser = commands.add_parser("list", help="List registered models in this task workspace")
    list_parser.add_argument("--project-root", default=".")
    list_parser.add_argument("--task-config", default=None)
    list_parser.add_argument("--limit", type=int, default=20)
    list_parser.add_argument("--offset", type=int, default=0)

    inspect_parser = commands.add_parser("inspect", help="Inspect a registered model definition")
    inspect_parser.add_argument("model_ref", help="Model ID or immutable SHA-256 definition digest")
    inspect_parser.add_argument("--project-root", default=".")
    inspect_parser.add_argument("--task-config", default=None)

    register_parser = commands.add_parser("register", help="Register a bounded model declaration")
    register_parser.add_argument("--file", required=True, help="YAML or JSON model definition")
    register_parser.add_argument("--project-root", default=".")
    register_parser.add_argument("--task-config", default=None)

    verify_parser = commands.add_parser("verify", help="Check local model artifacts and SHA-256 hashes")
    verify_parser.add_argument("model_ref", help="Model ID or immutable SHA-256 definition digest")
    verify_parser.add_argument("--project-root", default=".")
    verify_parser.add_argument("--task-config", default=None)

    validate_parser = commands.add_parser("validate-config", help="Check a config against task model policy")
    validate_parser.add_argument("--config", required=True, help="Experiment config path")
    validate_parser.add_argument("--project-root", default=".")
    validate_parser.add_argument("--task-config", default=None)

    resolve_parser = commands.add_parser("resolve", help="Plan backend model mount paths without dispatch")
    resolve_parser.add_argument("--config", required=True, help="Experiment config path")
    resolve_parser.add_argument("--backend", required=True, choices=("local", "kaggle", "modal"))
    resolve_parser.add_argument("--dry-run", action="store_true", help="Plan only; never dispatch or mutate provider state")
    resolve_parser.add_argument("--project-root", default=".")
    resolve_parser.add_argument("--task-config", default=None)

def run_model_command(args: argparse.Namespace) -> int:
    """Run a parsed command from either CLI entry point."""
    try:
        if args.model_command == "list":
            return _emit(
                list_model_assets(
                    args.project_root,
                    limit=args.limit,
                    offset=args.offset,
                    task_config=args.task_config,
                )
            )
        if args.model_command == "inspect":
            return _emit(
                inspect_model_asset(
                    args.project_root,
                    args.model_ref,
                    task_config=args.task_config,
                )
            )
        if args.model_command == "register":
            return _emit(register_model_asset(args.project_root, args.file, task_config=args.task_config))
        if args.model_command == "verify":
            result = verify_model_asset(args.project_root, args.model_ref, task_config=args.task_config)
            return _emit(result, exit_code=0 if result["status"] == "verified" else 1)
        if args.model_command == "validate-config":
            result = validate_model_plan(args.project_root, args.config, task_config=args.task_config)
            return _emit(result, exit_code=1 if result["status"] == "blocked" else 0)
        if args.model_command == "resolve":
            result = resolve_model_assets(
                args.project_root,
                args.config,
                args.backend,
                task_config=args.task_config,
            )
            result["dry_run"] = True
            return _emit(result, exit_code=1 if result["status"] == "blocked" else 0)
    except (OSError, TypeError, ValueError, KeyError) as exc:
        project_root = getattr(args, "project_root", ".")
        try:
            project = _project(project_root, getattr(args, "task_config", None))
            task_spec, task_id = _task_context(project)
            del task_spec
        except Exception:
            task_id = "unknown"
        return _emit(
            {
                "status": "error",
                "task_id": task_id,
                "operation": args.model_command,
                "checks": {},
                "blockers": [f"{type(exc).__name__}: {exc}"],
                "next_action": "Correct the reported input or workspace issue and retry.",
                "artifact_refs": [],
            },
            exit_code=2,
        )
    return 2


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="ai-harness model")
    add_model_subparsers(parser)
    return run_model_command(parser.parse_args(argv))


__all__ = [
    "inspect_model_asset",
    "list_model_assets",
    "main",
    "add_model_subparsers",
    "rebuild_model_index",
    "register_model_asset",
    "resolve_model_assets",
    "run_model_command",
    "validate_model_plan",
    "verify_model_asset",
]
