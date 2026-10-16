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

Exit codes:
  0  success (nothing to repair, or repair applied)
  2  project not attached, or nothing matched
  1  an error occurred (bad path, unwritable, etc.)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from harness.artifacts import sha256, validate_run

_MANIFEST_NAME = "artifact_manifest.json"

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
    from harness.project import MANIFEST_RELATIVE_PATH, LEGACY_MANIFEST_RELATIVE_PATH

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
    """Recompute the per-file sha256 map from the bytes currently on disk."""
    return {
        p.name: sha256(p)
        for p in sorted(run_dir.iterdir())
        if p.is_file() and p.name != _MANIFEST_NAME
    }


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
        return {"exp_dir": run_dir.name, "state": "verified"}
    except Exception as exc:  # noqa: BLE001
        return {
            "exp_dir": run_dir.name,
            "state": "unverifiable",
            "reason": str(exc)[:200],
            "note": "not a file-hash drift; repair will not fix this",
        }


def _repair_one(run_dir: Path, drop_missing: bool = False) -> Dict[str, Any]:
    """Re-seal the `files` map for a single run. Identity fields are preserved."""
    manifest = _load_manifest(run_dir)
    on_disk = _file_map_from_disk(run_dir)
    declared = manifest.get("files") or {}

    if drop_missing:
        # keep only files that exist on disk (drop declared-but-missing)
        new_files = on_disk
    else:
        # refresh hashes for declared files that exist; keep declared set stable
        # so we do not silently add undeclared files into the evidence bundle
        new_files = {}
        for name, value in declared.items():
            if (run_dir / name).is_file():
                new_files[name] = on_disk[name]
        # surface (but do not add) undeclared files so the caller can decide
        extra = sorted(name for name in on_disk if name not in declared)

    manifest["files"] = new_files
    manifest_path = run_dir / _MANIFEST_NAME
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    # clear the per-file hash cache so the very next validate_run re-reads bytes
    from harness.artifacts import _HASH_CACHE

    for name in new_files:
        _HASH_CACHE.clear()
        break

    # verify the repair actually restored validation (identity fields untouched)
    try:
        validate_run(run_dir)
        now_verified = True
        verify_error = None
    except Exception as exc:  # noqa: BLE001
        now_verified = False
        verify_error = str(exc)[:200]

    return {
        "exp_dir": run_dir.name,
        "resealed_files": sorted(new_files),
        "dropped_missing": sorted(set(declared) - set(new_files)),
        "now_verified": now_verified,
        "verify_error": verify_error,
    }


def repair_run_dir(run_dir: str | Path, *, drop_missing: bool = False) -> Dict[str, Any]:
    """Public entrypoint: repair a single run folder. Raises if folder/manifest bad."""
    run_dir = Path(run_dir).expanduser().resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"run directory not found: {run_dir}")
    if not (run_dir / _MANIFEST_NAME).is_file():
        raise ValueError(f"no {_MANIFEST_NAME} in {run_dir}; nothing to repair")
    return _repair_one(run_dir, drop_missing=drop_missing)


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
    }


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
                print(f"    - {item['exp_dir']}: {item['reason']}")
                for name in item["drifted"][:5]:
                    print(f"        drifted: {name}")
        if scan["unverifiable"]:
            print("\n  Unverifiable runs (NOT a hash drift; repair will not help):")
            for item in scan["unverifiable"]:
                print(f"    - {item['exp_dir']}: {item.get('reason', '?')}")
    return 0


def _emit_repair(
    artifacts_dir: Path,
    scan: Dict[str, Any],
    exp: Optional[List[str]],
    all_drifted: bool,
    drop_missing: bool,
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
    for target in targets:
        try:
            res = _repair_one(target, drop_missing=drop_missing)
            repaired.append(res)
        except Exception as exc:  # noqa: BLE001
            errors.append({"exp_dir": target.name, "error": str(exc)[:200]})

    ok = bool(repaired) and not errors
    result = {
        "artifacts_dir": str(artifacts_dir),
        "repaired": len(repaired),
        "errors": len(errors),
        "details": repaired,
        "error_details": errors,
    }
    if as_json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f"Repaired {len(repaired)} run(s) under {artifacts_dir}:")
        for res in repaired:
            flag = "OK" if res["now_verified"] else f"STILL BROKEN ({res['verify_error']})"
            print(f"  - {res['exp_dir']}: {flag}")
            if res["dropped_missing"]:
                print(f"      dropped missing: {', '.join(res['dropped_missing'])}")
        if errors:
            print(f"\nErrors ({len(errors)}):")
            for err in errors:
                print(f"  - {err['exp_dir']}: {err['error']}")
    return 0 if ok else (1 if errors and not repaired else 2)


def run_repair(
    *,
    project_root: str = ".",
    exp: Optional[List[str]] = None,
    all_drifted: bool = False,
    check: bool = False,
    drop_missing: bool = False,
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

    return _emit_repair(artifacts_dir, scan, exp, all_drifted, drop_missing, as_json)


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return run_repair(
        project_root=args.project_root,
        exp=args.exp,
        all_drifted=args.all,
        check=args.check,
        drop_missing=args.drop_missing,
        as_json=args.json,
    )


if __name__ == "__main__":
    raise SystemExit(main())
