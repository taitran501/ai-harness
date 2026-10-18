"""`ai-harness repair` — re-seal artifact manifests that drifted after a run.

Failure mode it solves: a completed run's artifacts get modified after the fact
(IDE background process, manual edit, cloud re-import with a changed file, or a
split/consent file that moved). The manifest's per-file `sha256` hashes no longer
match, so `validate_run()` raises "Artifact hash mismatch" and the run becomes
invisible to `research_memory.sync()` and the dashboard.

Before this command the only remedy was a hand-written Python script that
recomputed hashes and rewrote the manifest. `repair` does that, and *only* that:
it recomputes the `files` map from the bytes on disk. It never touches run
identity fields (run_id, exp_id, task/scorer/data/split/config sha256) — those
express the provenance of the run and must not be forged by a repair tool.
If identity is genuinely wrong, that is a different (louder) problem, not a
hash drift.

Two modes:
  --check        report which runs are drifted, without writing anything
  default        re-seal the `files` map for each drifted run (requires --exp or --all)

Evidence integrity gate (the tamper defense):
  Drift in an *evidence* file (metrics.json, OOF/predictions CSVs,
  evaluation_truth.csv) is never re-sealed by default. Re-sealing it would
  launder a tampered score. Such runs are blocked (exit 3) unless:
    --verify   re-derives the primary score from on-disk OOF + truth via the
               runner's own scoring path (harness.oof_revalidation) and
               re-seals only if the recomputed score matches the recorded one;
               mismatches are blocked with both values reported.
    --force    re-seals without re-derivation; the operator override is
               recorded in the audit log.

Every repair action (including blocked attempts) is appended to
  <artifacts_dir>/repair_log.jsonl  (append-only, outside the sealed bundle)
and each re-sealed manifest gains `repaired: true` + `repaired_at` so a
re-sealed run is distinguishable from a never-repaired one.

Exit codes:
  0  success (nothing to repair, or repair applied)
  2  project not attached, or nothing matched
  1  an error occurred (bad path, unwritable, etc.)
  3  a run was blocked by the evidence integrity gate
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from harness.artifacts import seal_file_map, validate_run
from harness.oof_revalidation import is_evidence_file, rederive_primary_score

_MANIFEST_NAME = "artifact_manifest.json"
_REPAIR_LOG_NAME = "repair_log.jsonl"


class EvidenceDriftError(ValueError):
    """A run's evidence files drifted; re-sealing is refused without --force.

    This is the programmatic counterpart of the CLI evidence integrity gate:
    a doctored metrics.json / OOF / truth must never be laundered back into
    the sealed bundle by an unqualified caller. Pass ``force=True`` (or use
    the CLI ``--verify`` / ``--force``) for an explicit, audited override.
    """

    def __init__(self, exp_dir: str, evidence_files, detail: str = ""):
        self.evidence_files = list(evidence_files)
        super().__init__(
            f"Evidence drift in '{exp_dir}' ({', '.join(self.evidence_files)}): "
            "refusing to re-seal without explicit force.\n"
            + (detail + "\n" if detail else "")
            + "Use repair_run_dir(..., force=True) or the CLI --verify/--force."
        )

# Fields that are run provenance. repair.py must never modify these.
_IDENTITY_FIELDS = (
    "schema_version",
    "run_id",
    "run_kind",
    "exp_id",
    "status",
    "plugin_id",
    "task_sha256",
    "scorer_sha256",
    "data_sha256",
    "split_sha256",
    "resolved_config_sha256",
    "estimator",
)


def _find_artifacts_dir(project_root: str | Path) -> Optional[Path]:
    """Locate the workspace artifacts directory without importing HarnessProject
    (which can be heavy). Returns None if no attach manifest is present."""
    from harness.project import LEGACY_MANIFEST_RELATIVE_PATH, MANIFEST_RELATIVE_PATH

    root = Path(project_root).expanduser().resolve()
    if not ((root / MANIFEST_RELATIVE_PATH).is_file() or (root / LEGACY_MANIFEST_RELATIVE_PATH).is_file()):
        return None
    manifest_path = root / MANIFEST_RELATIVE_PATH
    if not manifest_path.is_file():
        manifest_path = root / LEGACY_MANIFEST_RELATIVE_PATH
    import yaml

    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    artifacts_value = manifest.get("artifacts_dir") or ".ai-harness/artifacts"
    art_dir = Path(artifacts_value).expanduser()
    if not art_dir.is_absolute():
        art_dir = (root / art_dir).resolve()
    return art_dir if art_dir.is_dir() else None


def _iter_run_dirs(artifacts_dir: Path) -> List[Path]:
    """Yield run folders that carry a manifest (the only ones repair can touch)."""
    return [
        p
        for p in sorted(artifacts_dir.iterdir())
        if p.is_dir() and (p / _MANIFEST_NAME).is_file()
    ]


def _load_manifest(run_dir: Path) -> Dict[str, Any]:
    return json.loads((run_dir / _MANIFEST_NAME).read_text(encoding="utf-8"))


def _file_map_from_disk(run_dir: Path) -> Dict[str, str]:
    """Recompute the per-file sha256 map from the bytes currently on disk.

    Recursive (via artifacts.seal_file_map): nested run files (e.g.
    `stages/*.log`) are sealed, matching what the runner seals at
    finalization time. Top-level files keep bare names; nested files use
    POSIX relative paths.
    """
    return seal_file_map(run_dir)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _operator_name() -> str:
    return os.environ.get("AI_HARNESS_OPERATOR") or getpass.getuser()


def _append_repair_log(artifacts_dir: Path, entry: Dict[str, Any]) -> None:
    """Append one audit entry to <artifacts_dir>/repair_log.jsonl.

    The log lives beside the run folders, never inside any run folder, so it
    is outside every sealed bundle and never affects validate_run. Best-effort:
    a log write failure must not undo the re-seal (the re-seal is the primary
    operation; the marker + manifest diff remain the authoritative trail).
    """
    try:
        path = artifacts_dir / _REPAIR_LOG_NAME
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError:
        pass


def classify_run(run_dir: Path) -> Dict[str, Any]:
    """Classify a single run folder as verified / drifted / unverifiable.

    `drifted` means the run is structurally complete (validate_run would pass on
    identity) but one or more artifact file bytes no longer match the manifest.
    That is the only case repair is meant to fix.
    """
    manifest = _load_manifest(run_dir)
    on_disk = _file_map_from_disk(run_dir)
    declared = manifest.get("files") or {}

    missing_on_disk = sorted(name for name in declared if not (run_dir / name).is_file())
    drifted = sorted(
        name
        for name, value in declared.items()
        if (run_dir / name).is_file() and on_disk.get(name) != value
    )
    extra = sorted(name for name in on_disk if name not in declared)

    if missing_on_disk or drifted:
        return {
            "exp_dir": run_dir.name,
            "state": "drifted",
            "repaired": bool(manifest.get("repaired")),
            "missing_on_disk": missing_on_disk,
            "drifted": drifted,
            "extra_undeclared": extra,
            "reason": (
                f"{len(drifted)} file(s) drifted, "
                f"{len(missing_on_disk)} declared file(s) missing on disk"
            ),
        }
    # No hash drift — but the run may still be unverifiable for other reasons
    # (identity fields, required files, custom scorer). Report that separately so
    # the agent does not assume repair will help.
    try:
        validate_run(run_dir)
        return {
            "exp_dir": run_dir.name,
            "state": "verified",
            "repaired": bool(manifest.get("repaired")),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "exp_dir": run_dir.name,
            "state": "unverifiable",
            "repaired": bool(manifest.get("repaired")),
            "reason": str(exc)[:200],
            "note": "not a file-hash drift; repair will not fix this",
        }


def _repair_one(
    run_dir: Path,
    drop_missing: bool = False,
    *,
    artifacts_dir: Optional[Path] = None,
    verify_report: Optional[Dict[str, Any]] = None,
    forced: bool = False,
) -> Dict[str, Any]:
    """Re-seal the `files` map for a single run. Identity fields are preserved.

    `artifacts_dir` (the workspace artifacts dir) enables the append-only
    audit trail; `verify_report` carries the re-derivation outcome when the
    evidence integrity gate was passed via --verify; `forced` marks an
    operator override via --force. Re-sealed manifests gain `repaired` /
    `repaired_at` markers so the run is distinguishable from a clean one.
    """
    manifest = _load_manifest(run_dir)
    on_disk = _file_map_from_disk(run_dir)
    declared = manifest.get("files") or {}

    if drop_missing:
        # drop_missing mode: keep only declared files that still exist on
        # disk. Undeclared (stray) files are never adopted into the evidence
        # bundle — that would violate the bundle-exclusivity invariant.
        new_files = {
            name: on_disk[name]
            for name in declared
            if name in on_disk and (run_dir / name).is_file()
        }
    else:
        # Default mode: keep the declared set verbatim so a missing file
        # stays a *visible* validation failure (stale hash). Refresh hashes
        # only for declared files that exist; never silently drop
        # declarations (that would let the run re-pass validate_run with a
        # silently shrunken bundle) and never add undeclared files.
        new_files = {}
        for name, value in declared.items():
            if (run_dir / name).is_file():
                new_files[name] = on_disk[name]
            else:
                new_files[name] = value
        # undeclared files on disk are surfaced to the caller, never adopted

    manifest["files"] = new_files
    # Re-seal provenance: an explicit, checkable marker — never a silent
    # rewrite. validate_run ignores unknown fields, so this is backward
    # compatible with every consumer.
    manifest["repaired"] = True
    manifest["repaired_at"] = _utc_now_iso()
    from harness.io_utils import atomic_write_json

    atomic_write_json(run_dir / _MANIFEST_NAME, manifest)
    # clear the per-file hash cache so the very next validate_run re-reads bytes
    from harness.artifacts import _HASH_CACHE

    _HASH_CACHE.clear()

    # verify the repair actually restored validation (identity fields untouched)
    try:
        validate_run(run_dir)
        now_verified = True
        verify_error = None
    except Exception as exc:  # noqa: BLE001
        now_verified = False
        verify_error = str(exc)[:200]

    result = {
        "exp_dir": run_dir.name,
        "resealed_files": sorted(new_files),
        "dropped_missing": sorted(set(declared) - set(new_files)),
        "changed": sorted(
            name
            for name, value in new_files.items()
            if name in declared and declared[name] != value
        ),
        "now_verified": now_verified,
        "verify_error": verify_error,
        "forced": forced,
        "metric_check": verify_report,
    }
    if artifacts_dir is not None:
        _append_repair_log(
            artifacts_dir,
            {
                "ts": _utc_now_iso(),
                "exp_dir": run_dir.name,
                "operator": _operator_name(),
                "mode": "resave",
                "drop_missing": drop_missing,
                "forced": forced,
                "metric_check": verify_report,
                "files_changed": {
                    name: {
                        "old": declared.get(name),
                        "new": new_files[name],
                    }
                    for name in result["changed"]
                },
                "missing": sorted(set(declared) - set(new_files)),
                "extra_undeclared": sorted(set(on_disk) - set(declared)),
                "now_verified": now_verified,
                "verify_error": verify_error,
            },
        )
    return result


def repair_run_dir(
    run_dir: str | Path,
    *,
    drop_missing: bool = False,
    artifacts_dir: Optional[Path] = None,
    force: bool = False,
) -> Dict[str, Any]:
    """Public entrypoint: repair a single run folder. Raises if folder/manifest bad.

    Enforces the evidence integrity gate: if the drift touches evidence files
    (metrics.json / OOF / predictions / truth), re-sealing is refused with
    :class:`EvidenceDriftError` unless ``force=True`` (operator override,
    recorded in the audit log when ``artifacts_dir`` is provided).
    """
    run_dir = Path(run_dir).expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run directory not found: {run_dir}")
    if not (run_dir / _MANIFEST_NAME).is_file():
        raise ValueError(f"no {_MANIFEST_NAME} in {run_dir}; nothing to repair")
    cls = classify_run(run_dir)
    evidence_touched = sorted(
        name
        for name in (
            list(cls.get("drifted", [])) + list(cls.get("missing_on_disk", []))
        )
        if is_evidence_file(name)
    )
    if evidence_touched and not force:
        raise EvidenceDriftError(run_dir.name, evidence_touched)
    return _repair_one(
        run_dir,
        drop_missing=drop_missing,
        artifacts_dir=artifacts_dir,
        forced=force and bool(evidence_touched),
    )


def scan_artifacts(artifacts_dir: Path) -> Dict[str, Any]:
    """Classify every run folder. Used by `repair --check` and by status."""
    verified: List[str] = []
    drifted: List[Dict[str, Any]] = []
    unverifiable: List[Dict[str, Any]] = []
    for run_dir in _iter_run_dirs(artifacts_dir):
        try:
            cls = classify_run(run_dir)
        except Exception as exc:  # noqa: BLE001
            unverifiable.append(
                {"exp_dir": run_dir.name, "state": "unverifiable", "reason": str(exc)[:200]}
            )
            continue
        if cls["state"] == "verified":
            verified.append(cls["exp_dir"])
        elif cls["state"] == "drifted":
            drifted.append(cls)
        else:
            unverifiable.append(cls)
    return {
        "runs_total": len(verified) + len(drifted) + len(unverifiable),
        "verified": verified,
        "drifted": drifted,
        "unverifiable": unverifiable,
        "repaired_runs": sorted(
            item["exp_dir"]
            for item in (*drifted, *unverifiable)
            if item.get("repaired")
        )
        + [name for name in verified if name in _repaired_verified(artifacts_dir)],
    }


def _repaired_verified(artifacts_dir: Path) -> set:
    """Verified runs that carry the `repaired` marker (transparency surface)."""
    names = set()
    for run_dir in _iter_run_dirs(artifacts_dir):
        try:
            manifest = _load_manifest(run_dir)
        except (OSError, json.JSONDecodeError):
            continue
        if manifest.get("repaired") and manifest.get("status") == "COMPLETED":
            names.add(run_dir.name)
    return names


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ai-harness repair",
        description="Re-seal artifact manifest file hashes that drifted after a run.",
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Attached task workspace root (default: current dir)",
    )
    parser.add_argument(
        "--exp",
        action="append",
        default=None,
        help="Experiment (run folder) name to repair. Repeat for several. Default: none",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Repair every drifted run in the workspace (implied when --exp is omitted)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only report drifted/unverifiable runs; do not write anything",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="For runs with evidence drift: re-derive the primary score from on-disk "
        "OOF + truth via the runner's scoring path and re-seal only if it matches "
        "the recorded metrics.json value (mismatch => blocked, exit 3)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-seal evidence-drifted runs WITHOUT re-derivation; the override is "
        "recorded in repair_log.jsonl (operator override; use with care)",
    )
    parser.add_argument(
        "--drop-missing",
        action="store_true",
        help="Also drop manifest file entries that are missing on disk (default: keep them, which leaves the run unverifiable)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable result (default: human summary)",
    )
    return parser


def _resolve_targets(
    artifacts_dir: Path,
    exp: Optional[List[str]],
    all_drifted: bool,
) -> List[Path]:
    """Resolve which run folders to operate on."""
    targets: List[Path] = []
    if exp:
        for name in exp:
            candidate = artifacts_dir / name
            if not candidate.is_dir():
                raise FileNotFoundError(f"experiment folder not found: {name}")
            targets.append(candidate)
    elif all_drifted:
        for item in scan_artifacts(artifacts_dir)["drifted"]:
            targets.append(artifacts_dir / item["exp_dir"])
    else:
        # no --exp and no --all: default to repairing all drifted (documented)
        for item in scan_artifacts(artifacts_dir)["drifted"]:
            targets.append(artifacts_dir / item["exp_dir"])
    return targets


def _emit_check(artifacts_dir: Path, scan: Dict[str, Any], as_json: bool) -> int:
    result = {
        "artifacts_dir": str(artifacts_dir),
        "runs_total": scan["runs_total"],
        "verified": len(scan["verified"]),
        "drifted": len(scan["drifted"]),
        "unverifiable": len(scan["unverifiable"]),
        "drifted_runs": scan["drifted"],
        "unverifiable_runs": scan["unverifiable"],
    }
    if as_json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f"Artifacts: {artifacts_dir}")
        print(f"  total runs     : {scan['runs_total']}")
        print(f"  verified       : {len(scan['verified'])}")
        print(f"  drifted        : {len(scan['drifted'])}")
        print(f"  unverifiable   : {len(scan['unverifiable'])}")
        if scan["drifted"]:
            print("\n  Drifted runs (repairable):")
            for item in scan["drifted"]:
                flag = " [repaired marker]" if item.get("repaired") else ""
                print(f"    - {item['exp_dir']}: {item['reason']}{flag}")
                for name in item["drifted"][:5]:
                    print(f"        drifted: {name}")
        if scan.get("repaired_runs"):
            print("\n  Runs carrying a repair marker:")
            for name in scan["repaired_runs"]:
                print(f"    - {name}")
        if scan["unverifiable"]:
            print("\n  Unverifiable runs (NOT a hash drift; repair will not help):")
            for item in scan["unverifiable"]:
                print(f"    - {item['exp_dir']}: {item.get('reason', '?')}")
    return 0


def _load_verify_bundle(project_root: Path):
    """Load (spec, plugin, plugin_context) for score re-derivation.

    Only paid when --verify is actually exercised; the default repair path
    keeps its lightweight no-project-import design.
    """
    from harness.plugins import TaskPluginContext, load_task_plugin
    from harness.project import HarnessProject
    from harness.task_spec import load_active_task_spec

    project = HarnessProject.discover(project_root)
    spec = load_active_task_spec(project.task_config_path)
    plugin_context = TaskPluginContext(
        project.root,
        project.task_config_path,
        spec,
        project.plugin_id or "default",
    )
    plugin = load_task_plugin(project.plugin_id, plugin_context)
    return spec, plugin, plugin_context


def _emit_repair(
    artifacts_dir: Path,
    project_root: Path,
    scan: Dict[str, Any],
    exp: Optional[List[str]],
    all_drifted: bool,
    drop_missing: bool,
    verify: bool,
    force: bool,
    as_json: bool,
) -> int:
    try:
        targets = _resolve_targets(artifacts_dir, exp, all_drifted)
    except FileNotFoundError as exc:
        if as_json:
            print(json.dumps({"error": str(exc)}))
        else:
            print(f"Error: {exc}")
        return 1

    if not targets:
        result = {
            "artifacts_dir": str(artifacts_dir),
            "repaired": 0,
            "message": "No drifted runs to repair.",
        }
        if as_json:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print(f"No drifted runs found under {artifacts_dir}. Nothing to repair.")
        return 0

    repaired: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    blocked: List[Dict[str, Any]] = []
    verify_bundle = None
    for target in targets:
        cls = classify_run(target)
        evidence_touched = sorted(
            name
            for name in (
                list(cls.get("drifted", [])) + list(cls.get("missing_on_disk", []))
            )
            if is_evidence_file(name)
        )
        verify_report: Optional[Dict[str, Any]] = None
        if evidence_touched and not (verify or force):
            # Default posture: never re-seal drifted evidence. This is the
            # tamper path — a doctored metrics.json (or OOF) must stay
            # visible as a validation failure until an operator re-derives
            # (or explicitly forces).
            blocked.append(
                {
                    "exp_dir": target.name,
                    "status": "blocked_evidence_drift",
                    "evidence_files": evidence_touched,
                    "hint": (
                        "run with --verify (re-derive the score from on-disk "
                        "OOF + truth) or --force (operator override, audited)"
                    ),
                }
            )
            _append_repair_log(
                artifacts_dir,
                {
                    "ts": _utc_now_iso(),
                    "exp_dir": target.name,
                    "operator": _operator_name(),
                    "mode": "blocked",
                    "status": "blocked_evidence_drift",
                    "evidence_files": evidence_touched,
                    "now_verified": False,
                },
            )
            continue
        if evidence_touched and verify:
            if verify_bundle is None:
                try:
                    verify_bundle = _load_verify_bundle(project_root)
                except Exception as exc:  # noqa: BLE001
                    errors.append(
                        {
                            "exp_dir": target.name,
                            "error": f"verify setup failed: {exc}",
                        }
                    )
                    continue
            spec, plugin, plugin_context = verify_bundle
            try:
                report = rederive_primary_score(
                    target, task_spec=spec, plugin=plugin, plugin_context=plugin_context
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(
                    {"exp_dir": target.name, "error": f"re-derivation error: {exc}"}
                )
                continue
            verify_report = report
            if not report["verifiable"]:
                blocked.append(
                    {
                        "exp_dir": target.name,
                        "status": "not_verifiable",
                        "evidence_files": evidence_touched,
                        "detail": report["detail"],
                    }
                )
            elif not report["ok"]:
                blocked.append(
                    {
                        "exp_dir": target.name,
                        "status": "metric_mismatch",
                        "evidence_files": evidence_touched,
                        "recorded": report["recorded"],
                        "recomputed": report["recomputed"],
                        "detail": report["detail"],
                    }
                )
            if blocked and blocked[-1]["exp_dir"] == target.name:
                _append_repair_log(
                    artifacts_dir,
                    {
                        "ts": _utc_now_iso(),
                        "exp_dir": target.name,
                        "operator": _operator_name(),
                        "mode": "blocked",
                        "status": blocked[-1]["status"],
                        "evidence_files": evidence_touched,
                        "metric_check": report,
                        "now_verified": False,
                    },
                )
                continue
        try:
            res = _repair_one(
                target,
                drop_missing=drop_missing,
                artifacts_dir=artifacts_dir,
                verify_report=verify_report,
                forced=force and bool(evidence_touched),
            )
            repaired.append(res)
        except Exception as exc:  # noqa: BLE001
            errors.append({"exp_dir": target.name, "error": str(exc)[:200]})

    ok = bool(repaired) and not errors and not blocked
    result = {
        "artifacts_dir": str(artifacts_dir),
        "repaired": len(repaired),
        "blocked": len(blocked),
        "errors": len(errors),
        "details": repaired,
        "blocked_details": blocked,
        "error_details": errors,
    }
    if as_json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f"Repaired {len(repaired)} run(s) under {artifacts_dir}:")
        for res in repaired:
            flag = "OK" if res["now_verified"] else f"STILL BROKEN ({res['verify_error']})"
            suffix = " [forced]" if res.get("forced") else ""
            print(f"  - {res['exp_dir']}: {flag}{suffix}")
            if res["metric_check"]:
                print(
                    "      metric check: "
                    f"recorded={res['metric_check'].get('recorded')} "
                    f"recomputed={res['metric_check'].get('recomputed')}"
                )
            if res["dropped_missing"]:
                print(f"      dropped missing: {', '.join(res['dropped_missing'])}")
        if blocked:
            print(f"\nBlocked by the evidence integrity gate ({len(blocked)}):")
            for item in blocked:
                print(
                    f"  - {item['exp_dir']}: {item['status']} "
                    f"({', '.join(item['evidence_files'])})"
                )
                if item.get("detail"):
                    print(f"      {item['detail']}")
        if errors:
            print(f"\nErrors ({len(errors)}):")
            for err in errors:
                print(f"  - {err['exp_dir']}: {err['error']}")
    if blocked:
        return 3
    return 0 if ok else (1 if errors and not repaired else 2)


def run_repair(
    *,
    project_root: str = ".",
    exp: Optional[List[str]] = None,
    all_drifted: bool = False,
    check: bool = False,
    drop_missing: bool = False,
    verify: bool = False,
    force: bool = False,
    as_json: bool = False,
) -> int:
    """Keyword-driven entrypoint used by the CLI. Returns a process exit code."""
    artifacts_dir = _find_artifacts_dir(project_root)
    if artifacts_dir is None:
        msg = (
            "No attached ai-harness project found at this root (no .ml-harness/ or "
            ".ai-harness/ manifest with an artifacts directory). Run `ai-harness attach` first."
        )
        if as_json:
            print(json.dumps({"error": msg, "attached": False}))
        else:
            print(msg)
        return 2

    scan = scan_artifacts(artifacts_dir)

    if check:
        return _emit_check(artifacts_dir, scan, as_json)

    return _emit_repair(
        artifacts_dir,
        Path(project_root).expanduser().resolve(),
        scan,
        exp,
        all_drifted,
        drop_missing,
        verify=verify,
        force=force,
        as_json=as_json,
    )


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return run_repair(
        project_root=args.project_root,
        exp=args.exp,
        all_drifted=args.all,
        check=args.check,
        drop_missing=args.drop_missing,
        verify=args.verify,
        force=args.force,
        as_json=args.json,
    )


if __name__ == "__main__":
    raise SystemExit(main())
