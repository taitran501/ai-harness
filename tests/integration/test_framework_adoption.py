"""Consolidated integration test suite for Framework Adoption, Toolchain, and Dynamic Skill Injection.

Consolidates:
- tests/test_adoption_stack.py
- tests/test_bootstrap_task.py
- tests/test_local_import.py
- tests/test_single_install.py
- tests/test_skill_injector.py
- tests/test_toolchain_adoption.py
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import shutil
import socketserver
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.request
import zipfile
from pathlib import Path
from unittest import mock

import pandas as pd
import yaml

from harness.agent_bridge import AgentBridge
from harness.host_hooks import (
    HOOK_SCRIPT_REL,
    write_host_hooks,
)
from harness.import_local import import_verified_local_run, scan_importable_runs
from harness.project import (
    HarnessProject,
    attach_project,
    read_framework_pointer,
    write_framework_pointer,
)
from harness.skill_injector import (
    generate_compatibility_skill_content,
    generate_modular_skills,
    generate_skill_content,
    inject_skill_file,
    install_global_skills,
    install_workspace_skills,
)
from harness.status_report import _adoption_section, _framework_section, collect_status, render_markdown
from harness.sync import run_sync
from harness.task_spec import (
    UniversalTaskSpec,
    extract_archives_in_directory,
    load_active_task_spec,
    load_dataframe_file,
)
from harness.toolchain import (
    FRAMEWORK_DIR_ENV,
    FrameworkConfigurationError,
    PIN_DEPTH_ENV,
    PIN_DISABLE_ENV,
    _needs_pin,
    ensure_pinned_toolchain,
    find_framework_root,
    find_workspace_clone,
)
from tests.evidence_fixtures import seal_fixture
from tests.fixtures import (
    SOURCE_ROOT,
    make_workspace,
    text_task_document,
)
from tests.workspace_fixtures import (
    cleanup_dashboard_workspace,
    prepare_dashboard_workspace,
)
from tools.bootstrap_task import (
    bootstrap_competition_task,
    extract_context_from_markdown,
)
from tools.dashboard_server import DashboardHandler


# ---------------------------------------------------------------------------
# Section 1: Adoption Stack Hooks, Configs, Health & Gate
# (from tests/test_adoption_stack.py)
# ---------------------------------------------------------------------------


def _adoption_workspace() -> Path:
    root = Path(tempfile.mkdtemp(prefix="aih-adoption-"))
    (root / "configs").mkdir()
    (root / "configs/task.yaml").write_text(yaml.safe_dump(text_task_document()), encoding="utf-8")
    return root


class HookScriptTests(unittest.TestCase):
    """The generated hook script: block, allow, fail-open, human override."""

    def setUp(self):
        self.root = _adoption_workspace()
        info = write_host_hooks(self.root)
        self.script = Path(info["script"])
        self.assertTrue(self.script.is_file())

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def _run(self, stdin: str, env_extra=None):
        env = dict(os.environ)
        env.pop("AI_HARNESS_ALLOW_MANUAL", None)
        if env_extra:
            env.update(env_extra)
        return subprocess.run(
            [sys.executable, str(self.script)],
            input=stdin, capture_output=True, text=True, env=env, cwd=self.root,
            timeout=30,
        )

    def test_blocks_manual_kaggle_push_claude_payload(self):
        payload = json.dumps({"tool_name": "Bash", "tool_input": {"command": "kaggle kernels push -p kaggle_sub_exp001_manual"}})
        r = self._run(payload)
        self.assertEqual(r.returncode, 2, r.stderr)
        self.assertIn("ai-harness", r.stderr)
        self.assertIn("run_full_validation", r.stderr)

    def test_blocks_competitions_submit_cursor_payload(self):
        payload = json.dumps({"shell_command": "kaggle competitions submit -c x -f sub.csv"})
        r = self._run(payload)
        self.assertEqual(r.returncode, 2, r.stderr)

    def test_allows_benign_command(self):
        r = self._run(json.dumps({"tool_input": {"command": "python train.py"}}))
        self.assertEqual(r.returncode, 0)

    def test_allows_kaggle_non_push_commands(self):
        for cmd in ("kaggle kernels list", "kaggle datasets download x"):
            r = self._run(json.dumps({"tool_input": {"command": cmd}}))
            self.assertEqual(r.returncode, 0, cmd)

    def test_human_override_allows(self):
        payload = json.dumps({"tool_input": {"command": "kaggle kernels push -p x"}})
        r = self._run(payload, env_extra={"AI_HARNESS_ALLOW_MANUAL": "1"})
        self.assertEqual(r.returncode, 0)

    def test_fail_open_on_garbage_stdin(self):
        r = self._run("this is not json{{{")
        self.assertEqual(r.returncode, 0)

    def test_fail_open_on_empty_stdin(self):
        r = self._run("")
        self.assertEqual(r.returncode, 0)


class HostConfigTests(unittest.TestCase):
    """write_host_hooks: creates, merges, never destroys user config."""

    def test_creates_claude_and_cursor_configs(self):
        root = _adoption_workspace()
        try:
            write_host_hooks(root)
            claude = json.loads((root / ".claude" / "settings.json").read_text(encoding="utf-8"))
            cursor = json.loads((root / ".cursor" / "hooks.json").read_text(encoding="utf-8"))
            entries = claude["hooks"]["PreToolUse"][0]["hooks"]
            self.assertEqual(entries[0]["command"], "python " + HOOK_SCRIPT_REL.as_posix())
            self.assertEqual(entries[0]["managed_by"], "ai-harness")
            self.assertEqual(cursor["version"], 1)
            self.assertEqual(cursor["hooks"]["beforeShellExecution"][0]["managed_by"], "ai-harness")
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_merge_preserves_user_hooks(self):
        root = _adoption_workspace()
        try:
            (root / ".claude").mkdir()
            user_cfg = {"hooks": {"PreToolUse": [{"matcher": "Edit", "hooks": [{"type": "command", "command": "user-lint.sh"}]}]}}
            (root / ".claude" / "settings.json").write_text(json.dumps(user_cfg), encoding="utf-8")
            write_host_hooks(root)
            merged = json.loads((root / ".claude" / "settings.json").read_text(encoding="utf-8"))
            groups = merged["hooks"]["PreToolUse"]
            self.assertEqual(len(groups), 2, "user hook group must survive merge")
            self.assertIn("Edit", [g["matcher"] for g in groups if isinstance(g, dict) and "matcher" in g])
            managed = [g for g in groups if any(h.get("managed_by") == "ai-harness" for h in g.get("hooks", []))]
            self.assertEqual(len(managed), 1)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_idempotent_refresh(self):
        root = _adoption_workspace()
        try:
            write_host_hooks(root)
            write_host_hooks(root)
            claude = json.loads((root / ".claude" / "settings.json").read_text(encoding="utf-8"))
            groups = claude["hooks"]["PreToolUse"]
            managed_entries = [
                h for g in groups for h in g.get("hooks", []) if h.get("managed_by") == "ai-harness"
            ]
            self.assertEqual(len(managed_entries), 1, "refresh must not duplicate the managed entry")
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_skips_unreadable_user_config(self):
        root = _adoption_workspace()
        try:
            (root / ".cursor").mkdir()
            bad = root / ".cursor" / "hooks.json"
            bad.write_text("not json at all", encoding="utf-8")
            write_host_hooks(root)
            self.assertEqual(bad.read_text(encoding="utf-8"), "not json at all", "unreadable user file must be untouched")
        finally:
            shutil.rmtree(root, ignore_errors=True)


class AdoptionStatusTests(unittest.TestCase):
    """T5: adoption health in status."""

    def _ws(self):
        root = _adoption_workspace()
        attach_project(root, root / "configs/task.yaml")
        return root

    def test_not_attached(self):
        root = _adoption_workspace()
        try:
            self.assertEqual(_adoption_section(str(root))["status"], "not_attached")
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_fresh_workspace_no_activity(self):
        root = self._ws()
        try:
            res = _adoption_section(str(root))
            self.assertEqual(res["verdict"], "no_activity")
            self.assertEqual(res["experiments"], 0)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_sealed_run_healthy(self):
        root = self._ws()
        try:
            run = root / ".ai-harness" / "artifacts" / "adhoc" / "run_0001"
            run.mkdir(parents=True)
            (run / "artifact_manifest.json").write_text("{}", encoding="utf-8")
            res = _adoption_section(str(root))
            self.assertEqual(res["verdict"], "healthy")
            self.assertEqual(res["experiments"], 1)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_bypass_detected(self):
        root = self._ws()
        try:
            (root / "kaggle_sub_exp001_manual").mkdir()
            (root / "kaggle_sub_exp002_manual").mkdir()
            (root / "kaggle_sub_exp003_manual").mkdir()
            run = root / ".ai-harness" / "artifacts" / "adhoc" / "run_0001"
            run.mkdir(parents=True)
            (run / "artifact_manifest.json").write_text("{}", encoding="utf-8")
            res = _adoption_section(str(root))
            self.assertEqual(res["verdict"], "bypass_detected")
            self.assertEqual(res["bypassed_run_outputs"], 1)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_markdown_renders_adoption_line(self):
        root = self._ws()
        try:
            (root / "kaggle_sub_exp001_manual").mkdir()
            (root / "kaggle_sub_exp002_manual").mkdir()
            (root / "kaggle_sub_exp003_manual").mkdir()
            md = render_markdown(collect_status(str(root), include_cloud=False))
            self.assertIn("adoption:", md)
            self.assertIn("bypass_detected", md)
        finally:
            shutil.rmtree(root, ignore_errors=True)


class AdoptionGateTests(unittest.TestCase):
    """T3: session-start adoption gate in the Context HUD."""

    def setUp(self):
        self.root = _adoption_workspace()
        attach_project(self.root, self.root / "configs/task.yaml")
        self.bridge = AgentBridge(self.root)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_gate_shows_on_fresh_workspace_zero_experiments(self):
        md = self.bridge.render_context_markdown()
        self.assertIn("⛔ Adoption protocol", md)
        self.assertIn("0 harness experiments", md)

    def test_gate_shows_bypass_evidence(self):
        for i in (1, 2, 3):
            (self.root / f"kaggle_sub_exp{i:03d}_manual").mkdir()
        md = self.bridge.render_context_markdown()
        self.assertIn("⛔ Adoption protocol", md)
        self.assertIn("lack harness evidence", md)

    def test_gate_mentions_override_is_human_only(self):
        md = self.bridge.render_context_markdown()
        self.assertIn("AI_HARNESS_ALLOW_MANUAL=1", md)


# ---------------------------------------------------------------------------
# Section 2: Bootstrap Task Autonomy & Ingestion
# (from tests/test_bootstrap_task.py)
# ---------------------------------------------------------------------------


class TestBootstrapTask(unittest.TestCase):
    """Verifies that AI-Harness autonomously onboards arbitrary competitions and dataset formats."""

    def setUp(self) -> None:
        self.test_dir = Path(tempfile.mkdtemp(prefix="test_bootstrap_"))
        self.dataset_dir = self.test_dir / "dataset"
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.docs_dir = self.test_dir / "docs"
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        self.configs_dir = self.test_dir / "configs"
        self.configs_dir.mkdir(parents=True, exist_ok=True)
        self.skills_dir = self.test_dir / ".agents" / "skills" / "harness"
        self.skills_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_multi_format_loading_and_archive_extraction(self) -> None:
        """Verify that arbitrary formats (parquet, jsonl, zip) are seamlessly extracted and loaded."""
        df_sample = pd.DataFrame({
            "record_id": [f"rec_{i}" for i in range(25)],
            "feature_val": [i * 1.5 for i in range(25)],
            "group_key": [f"grp_{i % 5}" for i in range(25)],
            "target": [i % 2 for i in range(25)],
        })
        pq_path = self.dataset_dir / "raw_data.parquet"
        df_sample.to_parquet(pq_path, index=False)

        zip_path = self.dataset_dir / "data_package.zip"
        with zipfile.ZipFile(zip_path, "w") as z:
            z.write(pq_path, arcname="train.parquet")
        pq_path.unlink()

        extracted = extract_archives_in_directory(self.dataset_dir)
        self.assertTrue(len(extracted) > 0)
        extracted_pq = self.dataset_dir / "data_package" / "train.parquet"
        self.assertTrue(extracted_pq.exists())

        loaded_df = load_dataframe_file(extracted_pq)
        self.assertEqual(len(loaded_df), 25)
        self.assertIn("record_id", loaded_df.columns)

    def test_context_markdown_extraction(self) -> None:
        """Verify parsing of English criteria from TASK.md."""
        task_md = self.docs_dir / "TASK.md"
        task_md.write_text(
            """
# Competition & Task Specification Context

## 1. Competition Metadata
- **Competition Name**: Medical Imaging Risk 2027
- **Platform**: codabench
- **Problem Statement**: Predict diagnostic risk from structured telemetry.

## 2. Modality & Task Characteristics
- **Modality**: tabular
- **Task Type**: binary_classification
- **Target Columns**: risk_outcome
- **Class Labels**: [0, 1]

## 3. Data Identifiers & Anti-Leakage Invariants
- **Primary ID Column**: patient_uid
- **Anti-Leakage Group Column**: hospital_site_id

## 4. Evaluation Metrics & Optimization
- **Primary Metric**: roc_auc
- **Optimization Direction**: maximize
""",
            encoding="utf-8",
        )

        ctx = extract_context_from_markdown(task_md)
        self.assertEqual(ctx.get("name"), "Medical Imaging Risk 2027")
        self.assertEqual(ctx.get("platform"), "codabench")
        self.assertEqual(ctx.get("modality"), "tabular")
        self.assertEqual(ctx.get("task_type"), "binary_classification")
        self.assertEqual(ctx.get("id_column"), "patient_uid")
        self.assertEqual(ctx.get("group_column"), "hospital_site_id")
        self.assertEqual(ctx.get("primary_metric"), "roc_auc")
        self.assertEqual(ctx.get("direction"), "maximize")

    def test_end_to_end_bootstrap_pipeline(self) -> None:
        """Verify full end-to-end task onboarding (spec generation, GroupKFold, skill injection)."""
        train_jsonl = self.dataset_dir / "train.jsonl"
        records = [
            {
                "patient_uid": f"P_{i:04d}",
                "hospital_site_id": f"HOSP_{i % 5}",
                "age": 20 + i,
                "risk_outcome": i % 2,
            }
            for i in range(50)
        ]
        with open(train_jsonl, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        task_md = self.docs_dir / "TASK.md"
        task_md.write_text(
            """
- **Competition Name**: Hospital Risk AI
- **Platform**: kaggle
- **Modality**: tabular
- **Task Type**: binary_classification
- **Primary ID Column**: patient_uid
- **Anti-Leakage Group Column**: hospital_site_id
- **Target Columns**: risk_outcome
- **Primary Metric**: roc_auc
- **Optimization Direction**: maximize
""",
            encoding="utf-8",
        )

        out_cfg = self.configs_dir / "task.yaml"
        report = bootstrap_competition_task(
            dataset_dir=self.dataset_dir,
            context_file=task_md,
            output_config=out_cfg,
            project_root=self.test_dir,
        )

        self.assertEqual(report["status"], "SUCCESS")
        self.assertEqual(report["id_column"], "patient_uid")
        self.assertEqual(report["group_column"], "hospital_site_id")
        self.assertEqual(report["target_column"], "risk_outcome")
        self.assertEqual(report["primary_metric"], "roc_auc")

        spec = UniversalTaskSpec.from_yaml(out_cfg)
        self.assertEqual(spec.name, "Hospital Risk AI")
        self.assertEqual(spec.platform, "kaggle")
        self.assertEqual(spec.id_column, "patient_uid")
        self.assertEqual(spec.group_column, "hospital_site_id")

        splits_file = self.test_dir / "splits" / "fold_assignments.tsv"
        self.assertTrue(splits_file.exists())
        splits_df = pd.read_csv(splits_file, sep="\t")
        self.assertEqual(len(splits_df), 50)
        self.assertEqual(splits_df["fold"].nunique(), 5)

        merged = splits_df.merge(load_dataframe_file(train_jsonl), on="patient_uid")
        for fold_idx in range(5):
            val_groups = set(merged[merged["fold"] == fold_idx]["hospital_site_id"].unique())
            tr_groups = set(merged[merged["fold"] != fold_idx]["hospital_site_id"].unique())
            self.assertEqual(len(val_groups.intersection(tr_groups)), 0, "GroupKFold leakage detected!")

        baseline_file = self.configs_dir / "exp000_baseline.yaml"
        self.assertTrue(baseline_file.exists())
        base_data = yaml.safe_load(baseline_file.read_text(encoding="utf-8"))
        from harness.grounding_gate import GroundingGate
        passed, errors, _ = GroundingGate.verify_config(base_data, config_path=baseline_file, raise_on_error=False)
        self.assertTrue(passed, f"GroundingGate verification failed: {errors}")


# ---------------------------------------------------------------------------
# Section 3: Local Run Import & Quarantine
# (from tests/test_local_import.py)
# ---------------------------------------------------------------------------


def _spec_for(root: Path):
    return load_active_task_spec(HarnessProject.discover(root).task_config_path)


def _make_run(root: Path, name: str, *, stale_task: bool = False, stale_scorer: bool = False) -> Path:
    folder = root / name
    spec = _spec_for(root)
    seal_fixture(folder, spec)
    if stale_task or stale_scorer:
        manifest_path = folder / "artifact_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if stale_task:
            manifest["task_sha256"] = "0" * 64
            manifest["scorer_sha256"] = "1" * 64
        else:
            manifest["scorer_sha256"] = "1" * 64
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return folder


class ScanImportableRunsTests(unittest.TestCase):
    def setUp(self):
        self.root, _ = make_workspace()

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_scan_classifies_strict_stale_invalid_and_imported(self):
        _make_run(self.root, "kernel_output_exp001")
        _make_run(self.root, "kernel_output_exp002", stale_task=True)
        broken = self.root / "kernel_output_exp003"
        broken.mkdir()
        (broken / "artifact_manifest.json").write_text("{}", encoding="utf-8")
        entries = scan_importable_runs(self.root)
        by_dir = {e["dir"]: e for e in entries}
        self.assertEqual(by_dir["kernel_output_exp001"]["verdict"], "importable")
        self.assertEqual(by_dir["kernel_output_exp002"]["verdict"], "quarantine")
        self.assertIn("drift", by_dir["kernel_output_exp002"]["detail"])
        self.assertEqual(by_dir["kernel_output_exp003"]["verdict"], "invalid")
        self.assertNotIn("dataset", by_dir)

    def test_scan_marks_already_imported(self):
        source = _make_run(self.root, "kernel_output_exp001")
        result = import_verified_local_run(self.root, source)
        entries = scan_importable_runs(self.root)
        by_dir = {e["dir"]: e for e in entries}
        self.assertEqual(by_dir["kernel_output_exp001"]["verdict"], "already_imported")
        self.assertEqual(result["quarantined"], False)


class ImportVerifiedLocalRunTests(unittest.TestCase):
    def setUp(self):
        self.root, _ = make_workspace()

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_strict_import_lands_in_live_tree_with_receipt(self):
        source = _make_run(self.root, "kernel_output_exp001")
        result = import_verified_local_run(self.root, source)
        target = self.root / ".ai-harness" / "artifacts" / "fixture-task" / "kernel_output_exp001"
        self.assertEqual(result["target"], str(target))
        self.assertFalse(result["quarantined"])
        self.assertTrue((target / "metrics.json").is_file())
        receipt = json.loads((target / "IMPORT_RECEIPT.json").read_text(encoding="utf-8"))
        self.assertTrue(receipt["content_verified"])
        self.assertFalse(receipt["quarantined"])
        self.assertEqual(receipt["identity"]["task_sha256"], "match")
        journal = json.loads((self.root / ".ai-harness" / "artifacts" / "fixture-task" / "journal.json").read_text())
        self.assertEqual(journal[-1]["event"], "local_artifact_import")

    def test_stale_identity_quarantined_not_live(self):
        source = _make_run(self.root, "kernel_output_exp002", stale_task=True)
        result = import_verified_local_run(self.root, source)
        live = self.root / ".ai-harness" / "artifacts" / "fixture-task" / "kernel_output_exp002"
        quarantine = self.root / ".ai-harness" / "artifacts" / "fixture-task" / "quarantine" / "kernel_output_exp002"
        self.assertTrue(quarantine.is_dir())
        self.assertFalse(live.is_dir())
        self.assertTrue(result["quarantined"])
        self.assertIn("task_sha256", result["drift"])
        receipt = json.loads((quarantine / "IMPORT_RECEIPT.json").read_text(encoding="utf-8"))
        self.assertTrue(receipt["content_verified"])
        self.assertIn("drift", receipt["identity"]["task_sha256"])
        self.assertEqual(receipt["reason"], "identity drift vs attached task spec")

    def test_reject_stale_raises(self):
        source = _make_run(self.root, "kernel_output_exp002", stale_task=True)
        with self.assertRaisesRegex(ValueError, "identity drift"):
            import_verified_local_run(self.root, source, reject_stale=True)

    def test_tampered_file_rejected_nothing_copied(self):
        source = _make_run(self.root, "kernel_output_exp001")
        (source / "oof_predictions.csv").write_text("tampered\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            import_verified_local_run(self.root, source)
        target = self.root / ".ai-harness" / "artifacts" / "fixture-task" / "kernel_output_exp001"
        self.assertFalse(target.exists())

    def test_missing_manifest_file_rejected(self):
        source = _make_run(self.root, "kernel_output_exp001")
        (source / "metrics.json").unlink()
        with self.assertRaisesRegex(ValueError, "content integrity"):
            import_verified_local_run(self.root, source)

    def test_never_overwrites_existing_evidence(self):
        source = _make_run(self.root, "kernel_output_exp001")
        import_verified_local_run(self.root, source)
        source2 = self.root / "kernel_output_exp001_copy"
        shutil.copytree(source, source2)
        with self.assertRaisesRegex(ValueError, "never overwritten"):
            import_verified_local_run(self.root, source2)
        receipt = json.loads(
            (self.root / ".ai-harness" / "artifacts" / "fixture-task" / "kernel_output_exp001" / "IMPORT_RECEIPT.json").read_text()
        )
        self.assertEqual(receipt["source_dir"], str(source))

    def test_unsafe_manifest_filename_rejected(self):
        source = _make_run(self.root, "kernel_output_exp001")
        manifest = json.loads((source / "artifact_manifest.json").read_text(encoding="utf-8"))
        manifest["files"]["../evil.csv"] = "0" * 64
        (source / "artifact_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "unsafe"):
            import_verified_local_run(self.root, source)


class AdoptionScanTests(unittest.TestCase):
    def test_aggregates_workspaces_and_unimported_counts(self):
        parent = Path(tempfile.mkdtemp(prefix="aih-adoption-scan-"))
        ws_a_target = parent / "ws_a"
        ws_b_target = parent / "ws_b"
        try:
            ws_a, _ = make_workspace(prefix="aih-scan-a-")
            shutil.move(str(ws_a), str(ws_a_target))
            ws_b, _ = make_workspace(prefix="aih-scan-b-")
            shutil.move(str(ws_b), str(ws_b_target))
            _make_run(ws_a_target, "kernel_output_exp001", stale_task=True)

            from harness.cli import _run_adoption_scan

            args = argparse.Namespace(root=str(parent), include=None)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = _run_adoption_scan(args)
            self.assertEqual(rc, 0)
            payload = json.loads(buf.getvalue())
            self.assertEqual(payload["status"], "ok")
            by_name = {r["workspace"]: r for r in payload["workspaces"]}
            self.assertEqual(len(by_name), 2)
            row_a = by_name["ws_a"]
            self.assertEqual(row_a["unimported_count"], 1)
            self.assertEqual(row_a["unimported_runs"][0]["verdict"], "quarantine")
            self.assertEqual(by_name["ws_b"]["unimported_count"], 0)
            self.assertEqual(payload["summary"]["no_activity"], 2)
        finally:
            shutil.rmtree(str(parent), ignore_errors=True)

    def test_imported_source_record_written(self):
        root, _ = make_workspace()
        try:
            source = _make_run(root, "kernel_output_exp001", stale_task=True)
            import_verified_local_run(root, source)
            task_root = root / ".ai-harness" / "artifacts" / "fixture-task"
            payload = json.loads((task_root / "imported_sources.json").read_text(encoding="utf-8"))
            self.assertEqual(len(payload), 1)
            self.assertEqual(payload[0]["source_dir"], "kernel_output_exp001")
            self.assertTrue(payload[0]["quarantined"])

            import_verified_local_run(root, _make_run(root, "kaggle_results", stale_scorer=True))
            payload = json.loads((task_root / "imported_sources.json").read_text(encoding="utf-8"))
            self.assertEqual(len(payload), 2)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_quarantined_source_is_already_imported_in_scan(self):
        root, _ = make_workspace()
        try:
            source = _make_run(root, "kernel_output_exp001", stale_task=True)
            import_verified_local_run(root, source)
            entries = scan_importable_runs(root)
            by_dir = {e["dir"]: e for e in entries}
            self.assertEqual(by_dir["kernel_output_exp001"]["verdict"], "already_imported")
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_recovered_dir_not_flagged_as_bypass(self):
        from harness.hygiene import scan_workspace_hygiene

        root, _ = make_workspace()
        try:
            source = _make_run(root, "kernel_output_exp001", stale_task=True)
            (root / "kaggle_sub_exp002_apex").mkdir()
            (root / "kaggle_sub_exp999_apex").mkdir()
            (root / "kaggle_sub_exp999_apex" / "out.csv").write_text("x\n")
            import_verified_local_run(root, source)

            findings = scan_workspace_hygiene(root)["findings"]
            kinds = [f["kind"] for f in findings]
            self.assertIn("recovered_run_output", kinds)
            self.assertIn("bypassed_run_output", kinds)
            for f in findings:
                if f["kind"] == "bypassed_run_output":
                    self.assertNotIn("kernel_output_exp001", f["detail"])
                    self.assertIn("kaggle_sub_exp002_apex", f["detail"])
                    self.assertIn("kaggle_sub_exp999_apex", f["detail"])
        finally:
            shutil.rmtree(root, ignore_errors=True)


# ---------------------------------------------------------------------------
# Section 4: Single-Install Architecture & Sync
# (from tests/test_single_install.py)
# ---------------------------------------------------------------------------


def _make_ws_clone(ws: Path) -> Path:
    clone = ws / "ai-harness"
    (clone / "harness").mkdir(parents=True)
    (clone / ".git").mkdir()
    (clone / "harness" / "__init__.py").write_text("", encoding="utf-8")
    return clone


def _make_canonical(tmp: Path, name: str = "canonical") -> Path:
    canonical = tmp / name
    (canonical / "harness").mkdir(parents=True)
    (canonical / "harness" / "__init__.py").write_text("", encoding="utf-8")
    (canonical / ".git").mkdir()
    return canonical


class TestFindFrameworkRoot(unittest.TestCase):
    def test_env_var_precedence_over_clone(self) -> None:
        with tempfile.TemporaryDirectory(prefix="test_sif_root_") as tmp:
            tmp_path = Path(tmp)
            ws = tmp_path / "ws"
            (ws / "configs").mkdir(parents=True)
            _make_ws_clone(ws)
            canonical = _make_canonical(tmp_path)
            with mock.patch.dict(os.environ, {FRAMEWORK_DIR_ENV: str(canonical)}):
                self.assertEqual(find_framework_root(ws), canonical)

    def test_fallback_to_clone_when_no_env(self) -> None:
        with tempfile.TemporaryDirectory(prefix="test_sif_fallback_") as tmp:
            tmp_path = Path(tmp)
            ws = tmp_path / "ws"
            (ws / "configs").mkdir(parents=True)
            clone = _make_ws_clone(ws)
            env = {k: v for k, v in os.environ.items() if k != FRAMEWORK_DIR_ENV}
            with mock.patch.dict(os.environ, env, clear=True):
                self.assertEqual(find_framework_root(ws), clone)

    def test_none_when_nothing_present(self) -> None:
        with tempfile.TemporaryDirectory(prefix="test_sif_none_") as tmp:
            ws = Path(tmp) / "ws"
            ws.mkdir()
            env = {k: v for k, v in os.environ.items() if k != FRAMEWORK_DIR_ENV}
            with mock.patch.dict(os.environ, env, clear=True):
                self.assertIsNone(find_framework_root(ws))

    def test_invalid_env_raises_even_when_clone_exists(self) -> None:
        with tempfile.TemporaryDirectory(prefix="test_sif_invalid_") as tmp:
            tmp_path = Path(tmp)
            ws = tmp_path / "ws"
            (ws / "configs").mkdir(parents=True)
            _make_ws_clone(ws)
            bad = tmp_path / "not-a-framework"
            bad.mkdir()
            with mock.patch.dict(os.environ, {FRAMEWORK_DIR_ENV: str(bad)}):
                with self.assertRaises(FrameworkConfigurationError):
                    find_framework_root(ws)


class TestCanonicalPin(unittest.TestCase):
    def test_pin_targets_canonical_when_env_set(self) -> None:
        with tempfile.TemporaryDirectory(prefix="test_sip_pin_") as tmp:
            tmp_path = Path(tmp)
            ws = tmp_path / "ws"
            (ws / "configs").mkdir(parents=True)
            _make_ws_clone(ws)
            canonical = _make_canonical(tmp_path)
            fake_origin = tmp_path / "elsewhere" / "harness" / "__init__.py"
            captured: dict = {}

            def fake_runner(cmd, env=None, cwd=None):
                captured["env"] = env
                return subprocess.CompletedProcess(cmd, 0)

            env = {FRAMEWORK_DIR_ENV: str(canonical)}
            with mock.patch("harness.toolchain.current_origin", return_value=fake_origin), \
                    mock.patch.dict(os.environ, env, clear=False):
                result = ensure_pinned_toolchain(ws, "harness.runner", [], runner=fake_runner)
            self.assertIsNotNone(result)
            self.assertTrue(captured["env"]["PYTHONPATH"].startswith(str(canonical)))

    def test_no_pin_when_origin_is_already_canonical(self) -> None:
        with tempfile.TemporaryDirectory(prefix="test_sip_nopinx_") as tmp:
            tmp_path = Path(tmp)
            ws = tmp_path / "ws"
            (ws / "configs").mkdir(parents=True)
            _make_ws_clone(ws)
            canonical = _make_canonical(tmp_path)
            canonical_origin = (canonical / "harness" / "__init__.py").resolve()
            calls = {"n": 0}

            def counting_runner(cmd, env=None, cwd=None):  # pragma: no cover
                calls["n"] += 1
                return subprocess.CompletedProcess(cmd, 0)

            env = {FRAMEWORK_DIR_ENV: str(canonical)}
            with mock.patch("harness.toolchain.current_origin", return_value=canonical_origin), \
                    mock.patch.dict(os.environ, env, clear=False):
                needs, _o, _e = _needs_pin(ws)
                self.assertFalse(needs)
                result = ensure_pinned_toolchain(ws, "harness.runner", [], runner=counting_runner)
            self.assertIsNone(result)
            self.assertEqual(calls["n"], 0)


class TestCanonicalSync(unittest.TestCase):
    def test_canonical_mode_skips_pip(self) -> None:
        with tempfile.TemporaryDirectory(prefix="test_sis_nopip_") as tmp:
            tmp_path = Path(tmp)
            canonical = _make_canonical(tmp_path)
            ws = tmp_path / "workspace"
            (ws / "configs").mkdir(parents=True)
            env = {FRAMEWORK_DIR_ENV: str(canonical)}
            with mock.patch.dict(os.environ, env), \
                    mock.patch.object(sys, "stdout", new=io.StringIO()) as out:
                rc = run_sync(project_root=ws, no_pull=True, no_attach=True, as_json=True)
            report = json.loads(out.getvalue())
            self.assertEqual(rc, 0)
            self.assertEqual(report["mode"], "canonical")
            install = report["steps"]["install"]
            self.assertTrue(install.get("skipped"))
            self.assertIn("canonical", install.get("reason", ""))

    def test_workspace_clone_mode_when_no_env(self) -> None:
        with tempfile.TemporaryDirectory(prefix="test_sis_clone_") as tmp:
            tmp_path = Path(tmp)
            ws = tmp_path / "workspace"
            (ws / "configs").mkdir(parents=True)
            _make_ws_clone(ws)
            env = {k: v for k, v in os.environ.items() if k != FRAMEWORK_DIR_ENV}
            with mock.patch.dict(os.environ, env, clear=True), \
                    mock.patch.object(sys, "stdout", new=io.StringIO()) as out:
                rc = run_sync(project_root=ws, no_pull=True, no_install=True,
                              no_attach=True, as_json=True)
            report = json.loads(out.getvalue())
            self.assertEqual(rc, 0)
            self.assertEqual(report["mode"], "workspace-clone")

    def test_canonical_sync_writes_provenance_pointer(self) -> None:
        import harness as _pkg

        with tempfile.TemporaryDirectory(prefix="test_sis_ptr_") as tmp:
            tmp_path = Path(tmp)
            canonical = tmp_path / "canonical"
            git_env = {
                **os.environ,
                "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t",
                "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
            }
            subprocess.run(
                ["git", "clone", "--quiet",
                 str(Path(_pkg.__file__).resolve().parents[1]), str(canonical)],
                capture_output=True, text=True, check=True, env=git_env,
            )
            ws = tmp_path / "workspace"
            (ws / "configs").mkdir(parents=True)
            (ws / "configs" / "task.yaml").write_text(
                "task:\n  name: SingleInstall\n  primary_metric: accuracy\n",
                encoding="utf-8",
            )
            env = {
                FRAMEWORK_DIR_ENV: str(canonical),
                "HARNESS_PROJECT_ROOT": str(ws),
            }
            with mock.patch.dict(os.environ, env), \
                    mock.patch.object(sys, "stdout", new=io.StringIO()) as out:
                rc = run_sync(project_root=ws, no_pull=True, as_json=True)
            report = json.loads(out.getvalue())
            self.assertEqual(rc, 0, "canonical sync should succeed")
            self.assertEqual(report["mode"], "canonical")
            self.assertTrue(report["steps"]["install"].get("skipped"),
                            "canonical sync must not retarget the editable install")
            ptr = report.get("framework_pointer")
            self.assertIsNotNone(ptr)
            self.assertEqual(Path(ptr["path"]).resolve(), canonical.resolve())
            self.assertTrue(ptr.get("commit"), "pointer should record a commit")
            manifest = ws / ".ai-harness" / "project.yaml"
            self.assertTrue(manifest.is_file())
            got = read_framework_pointer(manifest)
            self.assertEqual(Path(got["path"]).resolve(), canonical.resolve())
            self.assertEqual(got.get("commit"), ptr.get("commit"))


class TestStatusFrameworkSection(unittest.TestCase):
    def test_single_install_true_for_canonical(self) -> None:
        with tempfile.TemporaryDirectory(prefix="test_sis_status_") as tmp:
            tmp_path = Path(tmp)
            ws = tmp_path / "ws"
            (ws / "configs").mkdir(parents=True)
            canonical = _make_canonical(tmp_path)
            env = {FRAMEWORK_DIR_ENV: str(canonical)}
            with mock.patch.dict(os.environ, env):
                section = _framework_section(str(ws))
            self.assertTrue(section["single_install"])
            self.assertEqual(Path(section["resolved_framework"]).resolve(), canonical.resolve())

    def test_single_install_false_for_clone(self) -> None:
        with tempfile.TemporaryDirectory(prefix="test_sis_status2_") as tmp:
            tmp_path = Path(tmp)
            ws = tmp_path / "ws"
            (ws / "configs").mkdir(parents=True)
            _make_ws_clone(ws)
            env = {k: v for k, v in os.environ.items() if k != FRAMEWORK_DIR_ENV}
            with mock.patch.dict(os.environ, env, clear=True):
                section = _framework_section(str(ws))
            self.assertFalse(section["single_install"])
            self.assertEqual(Path(section["resolved_framework"]).resolve(),
                             (ws / "ai-harness").resolve())

    def test_advisory_when_import_origin_mismatches(self) -> None:
        with tempfile.TemporaryDirectory(prefix="test_sis_adv_") as tmp:
            tmp_path = Path(tmp)
            ws = tmp_path / "ws"
            (ws / "configs").mkdir(parents=True)
            canonical = _make_canonical(tmp_path)
            stale = tmp_path / "stale" / "harness" / "__init__.py"
            env = {FRAMEWORK_DIR_ENV: str(canonical)}
            with mock.patch.dict(os.environ, env), \
                    mock.patch("harness.toolchain.current_origin", return_value=stale):
                section = _framework_section(str(ws))
            self.assertIn("pip install -e", section["advisory"])

    def test_pointer_fallback_when_no_env(self) -> None:
        with tempfile.TemporaryDirectory(prefix="test_sis_ptr_fb_") as tmp:
            tmp_path = Path(tmp)
            ws = tmp_path / "ws"
            (ws / "configs").mkdir(parents=True)
            canonical = _make_canonical(tmp_path)
            manifest = ws / ".ai-harness" / "project.yaml"
            write_framework_pointer(manifest, canonical, commit="abc123", branch="main")
            canonical_origin = (canonical / "harness" / "__init__.py").resolve()
            env = {k: v for k, v in os.environ.items() if k != FRAMEWORK_DIR_ENV}
            with mock.patch.dict(os.environ, env, clear=True), \
                    mock.patch("harness.toolchain.current_origin", return_value=canonical_origin):
                section = _framework_section(str(ws))
            self.assertTrue(section["single_install"], "pointer fallback should mark single-install")
            self.assertIsNone(section["resolved_framework"], "no env -> no resolved root")
            self.assertIsNotNone(section["workspace_pointer"])
            self.assertIsNone(section["advisory"], "origin matches pointer -> no advisory")

    def test_pointer_fallback_advisory_on_mismatch(self) -> None:
        with tempfile.TemporaryDirectory(prefix="test_sis_ptr_adv_") as tmp:
            tmp_path = Path(tmp)
            ws = tmp_path / "ws"
            (ws / "configs").mkdir(parents=True)
            canonical = _make_canonical(tmp_path)
            manifest = ws / ".ai-harness" / "project.yaml"
            write_framework_pointer(manifest, canonical, commit="abc123", branch="main")
            stale = tmp_path / "stale" / "harness" / "__init__.py"
            env = {k: v for k, v in os.environ.items() if k != FRAMEWORK_DIR_ENV}
            with mock.patch.dict(os.environ, env, clear=True), \
                    mock.patch("harness.toolchain.current_origin", return_value=stale):
                section = _framework_section(str(ws))
            self.assertTrue(section["single_install"])
            self.assertIn("pip install -e", section["advisory"])


# ---------------------------------------------------------------------------
# Section 5: Dynamic Skill Injector & Dashboard API
# (from tests/test_skill_injector.py)
# ---------------------------------------------------------------------------


class SkillInjectorTests(unittest.TestCase):
    def test_generated_skill_contains_confinement_and_candidate_citation_rules(self):
        content = generate_skill_content({
            "task_name": "Confinement fixture",
            "platform": "kaggle",
            "modality": "tabular",
            "primary_metric": "Accuracy",
            "direction": "maximize",
        })

        self.assertIn("Workspace Confinement & Anti-Dredging", content)
        self.assertIn("<appDataDir>\\brain\\<other-conversation-id>", content)
        self.assertIn("ai-harness-agent context", content)
        self.assertIn("validate_candidate_pitch", content)
        self.assertIn("verified 'preflight_id'", content)
        self.assertIn("Spearman rho >= 0.995", content)
        self.assertIn("verified paired OOF gain", content)
        self.assertIn("not an OS security boundary", content)

    def test_generate_skill_nlp(self):
        task_cfg = {
            "task_name": "My NLP Benchmark",
            "platform": "codabench",
            "task_type": "Multi-Task Text Classification",
            "modality": "text",
            "primary_metric": "Macro-F1 (CV)",
            "direction": "maximize",
            "primary_metric_baseline": 0.6459,
            "primary_metric_target": 0.85,
            "metrics": [
                {
                    "id": "subtask_a",
                    "name": "Label Detection",
                    "metric": "Macro-F1",
                    "direction": "maximize",
                    "baseline_score": 0.6962,
                    "target_score": 0.88,
                    "weight": 0.333,
                    "description": "Binary label detection",
                }
            ],
            "languages": ["EN", "IT", "NL"],
        }
        content = generate_skill_content(task_cfg)
        self.assertIn("name: harness", content)
        self.assertIn("My NLP Benchmark", content)
        self.assertIn("Codabench", content)
        self.assertIn("Macro-F1 (CV)", content)
        self.assertIn("0.6459", content)
        self.assertIn("0.85", content)
        self.assertIn("Natural Language Processing (NLP)", content)
        self.assertIn("microsoft/deberta-v3-base", content)
        self.assertIn("Focal Loss", content)
        self.assertIn("SWE-Agent Physical & Statistical Audit Gate", content)
        self.assertIn("Task Contract and Approval Gates", content)
        self.assertNotIn("MMES >= 2.0%", content)

    def test_generate_skill_tabular(self):
        task_cfg = {
            "task_name": "House Prices Benchmark",
            "platform": "kaggle",
            "task_type": "Tabular Regression",
            "modality": "tabular",
            "primary_metric": "RMSE (CV)",
            "direction": "minimize",
            "primary_metric_baseline": 0.145,
            "primary_metric_target": 0.110,
            "metrics": [
                {
                    "id": "rmse",
                    "name": "Root Mean Squared Error",
                    "metric": "RMSE",
                    "direction": "minimize",
                    "baseline_score": 0.145,
                    "target_score": 0.110,
                    "weight": 1.0,
                    "description": "Log-scale RMSE evaluation",
                }
            ],
        }
        content = generate_skill_content(task_cfg)
        self.assertIn("House Prices Benchmark", content)
        self.assertIn("Kaggle", content)
        self.assertIn("Tabular & Structured Data", content)
        self.assertIn("LightGBM", content)
        self.assertIn("CatBoost", content)
        self.assertIn("RMSE (CV)", content)
        self.assertIn("minimize", content)
        self.assertIn("Target encoding", content)

    def test_generate_skill_vision(self):
        task_cfg = {
            "task_name": "Plant Pathology Classification",
            "platform": "kaggle",
            "task_type": "Image Classification",
            "modality": "vision",
            "primary_metric": "Top-1 Accuracy (CV)",
            "direction": "maximize",
        }
        content = generate_skill_content(task_cfg)
        self.assertIn("Plant Pathology Classification", content)
        self.assertIn("Computer Vision", content)
        self.assertIn("convnext", content)
        self.assertIn("swin", content)
        self.assertIn("Mixup", content)

    def test_generate_skill_anti_defeatism_and_cli(self):
        task_cfg = {
            "task_name": "Test Benchmark",
            "platform": "kaggle",
            "modality": "tabular",
            "primary_metric": "ROC-AUC",
            "direction": "maximize",
        }
        content = generate_skill_content(task_cfg)
        self.assertIn("Competition Research Drive & Proactive Discussion Mining", content)
        self.assertIn("ai-harness discussion list", content)
        self.assertIn("ai-harness discussion view", content)
        self.assertIn("ai-harness submission list", content)
        self.assertIn("ai-harness codabench status", content)
        self.assertIn("there is no obligation to submit every slot", content)
        self.assertIn("registered bounded recovery probe", content)

    def test_generate_modular_skills(self):
        task_cfg = {
            "task_name": "Airline Delay Prediction",
            "platform": "kaggle",
            "modality": "tabular",
            "primary_metric": "ROC-AUC",
            "direction": "maximize",
            "competition_id": "airline-delay-prediction",
        }
        modular = generate_modular_skills(task_cfg)
        self.assertEqual(set(modular.keys()), {"harness", "debate", "research"})

        harness = modular["harness"]
        self.assertIn("name: harness", harness)
        self.assertIn("ML Experiment Harness", harness)
        self.assertIn("ai-harness-agent context", harness)
        self.assertIn("/debate", harness)
        self.assertIn("/research", harness)
        self.assertIn("Candidate Preflight & Diversity Check", harness)
        self.assertIn("registered promotion rule", harness)

        debate = modular["debate"]
        self.assertIn("name: debate", debate)
        self.assertIn("Three permitted outcomes", debate)
        self.assertIn("Grounded hypothesis", debate)
        self.assertIn("External reported result", debate)
        self.assertIn("Demonstrated local gain", debate)
        self.assertIn(r"\times", debate)
        self.assertIn(r"\text{SE}", debate)
        self.assertNotIn("reported_delta", debate)
        self.assertNotIn("N < 3", debate)
        self.assertIn("ai-harness audit-claims", debate)

        research = modular["research"]
        self.assertIn("name: research", research)
        self.assertIn("Airline Delay Prediction", research)
        self.assertIn("ROC-AUC", research)
        self.assertIn("/harness", research)
        self.assertIn("/debate", research)

    def test_install_global_skills(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            target = Path(tmp_dir) / "skills"
            (target / "ai-experiment-harness").mkdir(parents=True)
            (target / "ai-harness-resume").mkdir(parents=True)
            (target / "submit").mkdir(parents=True)
            written = install_global_skills(target_dir=target)
            self.assertEqual(len(written), 3)
            self.assertTrue((target / "harness" / "SKILL.md").exists())
            self.assertTrue((target / "debate" / "SKILL.md").exists())
            self.assertTrue((target / "research" / "SKILL.md").exists())
            self.assertFalse((target / "submit").exists())
            self.assertFalse((target / "ai-experiment-harness").exists())
            self.assertFalse((target / "ai-harness-resume").exists())

    def test_install_workspace_skills(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            task_cfg = {
                "task_name": "Workspace Task",
                "platform": "kaggle",
                "modality": "generic",
                "primary_metric": "F1",
                "direction": "maximize",
            }
            written = install_workspace_skills(project_root=root, task_config=task_cfg)
            self.assertEqual(len(written), 3)
            skills_dir = root / ".agents" / "skills"
            self.assertTrue((skills_dir / "harness" / "SKILL.md").exists())
            self.assertTrue((skills_dir / "debate" / "SKILL.md").exists())
            self.assertTrue((skills_dir / "research" / "SKILL.md").exists())
            self.assertFalse((skills_dir / "ai-harness").exists())
            self.assertFalse((skills_dir / "submit").exists())
            self.assertFalse((skills_dir / "ai-harness-resume").exists())

    def test_cli_install_skills(self):
        from harness.cli import main

        with tempfile.TemporaryDirectory() as tmp_dir:
            target = Path(tmp_dir) / "global_skills"
            prev_env = os.environ.get("GEMINI_SKILLS_DIR")
            os.environ["GEMINI_SKILLS_DIR"] = str(target)
            try:
                rc = main(["install-skills", "--global", "--json"])
                self.assertEqual(rc, 0)
                self.assertTrue((target / "harness" / "SKILL.md").exists())
                self.assertTrue((target / "debate" / "SKILL.md").exists())
                self.assertTrue((target / "research" / "SKILL.md").exists())
                self.assertFalse((target / "submit").exists())
            finally:
                if prev_env is None:
                    os.environ.pop("GEMINI_SKILLS_DIR", None)
                else:
                    os.environ["GEMINI_SKILLS_DIR"] = prev_env

    def test_inject_skill_file(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_path = Path(tmp_dir) / "SKILL.md"
            task_cfg = {
                "task_name": "Test Benchmark",
                "platform": "custom",
                "modality": "generic",
                "primary_metric": "Accuracy",
                "direction": "maximize",
            }
            res_path = inject_skill_file(task_config=task_cfg, output_path=out_path)
            self.assertEqual(res_path, out_path)
            self.assertTrue(out_path.exists())
            text = out_path.read_text(encoding="utf-8")
            self.assertIn("Test Benchmark", text)
            self.assertIn("UCB Is Advice", text)

    def test_validation_spec_uses_task_contract_before_legacy_data_fields(self):
        content = generate_skill_content({
            "task_name": "Validation fixture",
            "data_config": {"train_path": "train.csv", "n_folds": 3, "split_strategy": "legacy"},
            "validation": {"strategy": "kfold", "num_folds": 5},
        })
        self.assertIn("`kfold` (5 folds", content)
        self.assertNotIn("`legacy` (3 folds", content)

        legacy = generate_skill_content({
            "task_name": "Legacy fixture",
            "data_config": {"train_path": "train.csv", "n_folds": 3, "split_strategy": "legacy"},
        })
        self.assertIn("`legacy` (3 folds", legacy)

    def test_owned_alias_is_retired_in_both_scopes(self):
        legacy_content = generate_compatibility_skill_content({"task_name": "Old Task"})

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            legacy_dir = root / ".agents" / "skills" / "ai-harness"
            legacy_dir.mkdir(parents=True)
            (legacy_dir / "SKILL.md").write_text(legacy_content, encoding="utf-8")
            self.assertTrue((legacy_dir / "SKILL.md").is_file())

            written = install_workspace_skills(project_root=root, task_config={"task_name": "New Task"})
            self.assertEqual({p.parent.name for p in written}, {"harness", "debate", "research"})
            self.assertFalse(legacy_dir.exists())
            self.assertTrue((root / ".agents" / "skills" / "harness" / "SKILL.md").is_file())

        with tempfile.TemporaryDirectory() as tmp_dir:
            target = Path(tmp_dir) / "global_skills"
            legacy_dir = target / "ai-harness"
            legacy_dir.mkdir(parents=True)
            (legacy_dir / "SKILL.md").write_text(legacy_content, encoding="utf-8")
            self.assertTrue((legacy_dir / "SKILL.md").is_file())

            written = install_global_skills(target_dir=target, task_config={"task_name": "New Task"})
            self.assertEqual({p.parent.name for p in written}, {"harness", "debate", "research"})
            self.assertFalse(legacy_dir.exists())
            self.assertTrue((target / "harness" / "SKILL.md").is_file())

    def test_alias_retirement_preserves_custom_files(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            legacy_dir = root / ".agents" / "skills" / "ai-harness"
            legacy_dir.mkdir(parents=True)
            (legacy_dir / "SKILL.md").write_text("# My Custom Skill\nNot generated by harness", encoding="utf-8")
            (legacy_dir / "custom_notes.txt").write_text("User notes", encoding="utf-8")

            with self.assertWarnsRegex(UserWarning, "Preserved unrecognized legacy skill"):
                install_workspace_skills(project_root=root, task_config={"task_name": "T"})
            self.assertTrue(legacy_dir.exists())
            self.assertTrue((legacy_dir / "custom_notes.txt").is_file())
            self.assertTrue((legacy_dir / "SKILL.md").is_file())
            self.assertIn("My Custom Skill", (legacy_dir / "SKILL.md").read_text(encoding="utf-8"))

    def test_retired_alias_cannot_be_generated_explicitly(self):
        with self.assertRaisesRegex(ValueError, "/ai-harness is retired"):
            generate_skill_content({"task_name": "T"}, skill_name="ai-harness")

        for name in ("ai-harness", "AI-HARNESS"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                alias = Path(tmp) / name / "SKILL.md"
                with self.assertRaisesRegex(ValueError, "/ai-harness is retired"):
                    inject_skill_file(
                        task_config={"task_name": "Task"},
                        output_path=alias,
                    )
                self.assertFalse(alias.parent.exists())

    def test_reattach_refreshes_missing_skill_and_retires_alias(self):
        with (
            tempfile.TemporaryDirectory(prefix="aih-skill-migration-") as tmp,
            mock.patch("harness.skill_injector.install_global_skills", return_value=[]),
        ):
            root = Path(tmp)
            cfg_dir = root / "configs"
            cfg_dir.mkdir(parents=True)
            cfg_path = cfg_dir / "task.yaml"
            cfg_path.write_text("task_name: Reattach Task\nplatform: custom\nprimary_metric: Acc\ndirection: maximize\n", encoding="utf-8")

            res1 = attach_project(project_root=root, task_config=cfg_path, force=True)
            self.assertEqual(res1["status"], "ATTACHED")

            harness_skill = root / ".agents" / "skills" / "harness" / "SKILL.md"
            if harness_skill.exists():
                harness_skill.unlink()
            legacy_dir = root / ".agents" / "skills" / "ai-harness"
            legacy_dir.mkdir(parents=True, exist_ok=True)
            (legacy_dir / "SKILL.md").write_text(
                generate_compatibility_skill_content({"task_name": "Reattach Task"}),
                encoding="utf-8",
            )

            res2 = attach_project(project_root=root, task_config=cfg_path, force=False)
            self.assertEqual(res2["status"], "ALREADY_ATTACHED")
            self.assertFalse(legacy_dir.exists())
            self.assertTrue(harness_skill.is_file())


class SkillInjectorAPITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        prepare_dashboard_workspace(cls, "test_skill_injector_api_")
        cls.server = socketserver.TCPServer(("127.0.0.1", 0), DashboardHandler)
        cls.port = cls.server.server_address[1]
        cls.server_thread = threading.Thread(
            target=cls.server.serve_forever, daemon=True
        )
        cls.server_thread.start()
        time.sleep(0.2)

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cleanup_dashboard_workspace(cls)

    def test_api_skill_inject(self):
        url = f"http://127.0.0.1:{self.port}/api/skill/inject"
        payload = json.dumps({}).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode())
            self.assertTrue(data.get("success"))
            self.assertIn("skill_path", data)
            self.assertTrue((self.workspace_root / data["skill_path"]).is_file())

    def test_api_task_info_auto_inject(self):
        url = f"http://127.0.0.1:{self.port}/api/task_info"
        from tools.dashboard_server import load_task_config

        payload = json.dumps(load_task_config()).encode("utf-8")
        post_req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(post_req) as resp:
            self.assertEqual(resp.status, 200)
            data = json.loads(resp.read().decode("utf-8"))
            self.assertTrue(data.get("success"))
            self.assertTrue(data.get("skill_injected"))


# ---------------------------------------------------------------------------
# Section 6: Toolchain Pinning, Re-execution & Adoption Audit
# (from tests/test_toolchain_adoption.py)
# ---------------------------------------------------------------------------


class TestToolchainPin(unittest.TestCase):
    """Gap 1: CLI entry points must execute the workspace clone's code."""

    def setUp(self) -> None:
        # Isolate TestToolchainPin from any ambient HARNESS_FRAMEWORK_DIR
        env = {k: v for k, v in os.environ.items() if k != FRAMEWORK_DIR_ENV}
        patcher = mock.patch.dict(os.environ, env, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_clone_discovery(self) -> None:
        with tempfile.TemporaryDirectory(prefix="test_tc_clone_") as tmp:
            ws = Path(tmp)
            clone = ws / "ai-harness"
            (clone / "harness").mkdir(parents=True)
            (clone / "harness" / "__init__.py").write_text("", encoding="utf-8")
            self.assertEqual(find_workspace_clone(ws), clone)
            self.assertIsNone(find_workspace_clone(ws / "nonexistent"))

    def test_needs_pin_false_without_clone(self) -> None:
        with tempfile.TemporaryDirectory(prefix="test_tc_nopin_") as tmp:
            self.assertEqual(_needs_pin(tmp), (False, "", ""))

    def test_needs_pin_true_on_drift(self) -> None:
        with tempfile.TemporaryDirectory(prefix="test_tc_drift_") as tmp:
            ws = Path(tmp)
            clone = ws / "ai-harness"
            (clone / "harness").mkdir(parents=True)
            (clone / "harness" / "__init__.py").write_text("", encoding="utf-8")
            fake_origin = Path(tmp) / "other" / "harness" / "__init__.py"
            with mock.patch("harness.toolchain.current_origin", return_value=fake_origin):
                needs, origin, _expected = _needs_pin(ws)
            self.assertTrue(needs)
            self.assertIn("other", origin)

    def test_reexec_pins_pythonpath_and_preserves_argv(self) -> None:
        with tempfile.TemporaryDirectory(prefix="test_tc_reexec_") as tmp:
            ws = Path(tmp)
            clone = ws / "ai-harness"
            (clone / "harness").mkdir(parents=True)
            (clone / "harness" / "__init__.py").write_text("", encoding="utf-8")
            fake_origin = Path(tmp) / "elsewhere" / "harness" / "__init__.py"
            captured: dict = {}

            def fake_runner(cmd, env=None, cwd=None):
                captured["cmd"] = cmd
                captured["env"] = env
                captured["cwd"] = cwd
                proc = subprocess.CompletedProcess(cmd, 7)
                return proc

            with mock.patch("harness.toolchain.current_origin", return_value=fake_origin):
                result = ensure_pinned_toolchain(
                    ws, "harness.runner", ["--config", "x.yaml"], runner=fake_runner
                )
            self.assertIsNotNone(result)
            self.assertEqual(result.returncode, 7)
            cmd = captured["cmd"]
            self.assertEqual(cmd[0], sys.executable)
            self.assertEqual(cmd[1:3], ["-m", "harness.runner"])
            self.assertEqual(cmd[3:], ["--config", "x.yaml"])
            self.assertTrue(captured["env"]["PYTHONPATH"].startswith(str(clone)))

    def test_escape_hatch_disables_pin(self) -> None:
        with tempfile.TemporaryDirectory(prefix="test_tc_hatch_") as tmp:
            ws = Path(tmp)
            clone = ws / "ai-harness"
            (clone / "harness").mkdir(parents=True)
            (clone / "harness" / "__init__.py").write_text("", encoding="utf-8")
            fake_origin = Path(tmp) / "elsewhere" / "harness" / "__init__.py"
            with mock.patch("harness.toolchain.current_origin", return_value=fake_origin), mock.patch.dict(
                os.environ, {PIN_DISABLE_ENV: "1"}
            ):
                self.assertIsNone(
                    ensure_pinned_toolchain(ws, "harness.runner", [], runner=lambda *a, **k: None)
                )

    def test_fail_open_on_runner_error(self) -> None:
        with tempfile.TemporaryDirectory(prefix="test_tc_failopen_") as tmp:
            ws = Path(tmp)
            clone = ws / "ai-harness"
            (clone / "harness").mkdir(parents=True)
            (clone / "harness" / "__init__.py").write_text("", encoding="utf-8")
            fake_origin = Path(tmp) / "elsewhere" / "harness" / "__init__.py"

            def bad_runner(cmd, env=None, cwd=None):
                raise OSError("boom")

            with mock.patch("harness.toolchain.current_origin", return_value=fake_origin):
                self.assertIsNone(
                    ensure_pinned_toolchain(ws, "harness.runner", [], runner=bad_runner)
                )

    def test_reexec_bumps_pin_depth_in_child_env(self) -> None:
        with tempfile.TemporaryDirectory(prefix="test_tc_depth_") as tmp:
            ws = Path(tmp)
            clone = ws / "ai-harness"
            (clone / "harness").mkdir(parents=True)
            (clone / "harness" / "__init__.py").write_text("", encoding="utf-8")
            fake_origin = Path(tmp) / "elsewhere" / "harness" / "__init__.py"
            captured: dict = {}

            def fake_runner(cmd, env=None, cwd=None):
                captured["env"] = env
                return subprocess.CompletedProcess(cmd, 0)

            with mock.patch("harness.toolchain.current_origin", return_value=fake_origin):
                ensure_pinned_toolchain(ws, "harness.runner", [], runner=fake_runner)
            self.assertEqual(captured["env"][PIN_DEPTH_ENV], "1")

    def test_drift_in_pinned_child_fails_open_not_loop(self) -> None:
        with tempfile.TemporaryDirectory(prefix="test_tc_noloop_") as tmp:
            ws = Path(tmp)
            clone = ws / "ai-harness"
            (clone / "harness").mkdir(parents=True)
            (clone / "harness" / "__init__.py").write_text("", encoding="utf-8")
            fake_origin = Path(tmp) / "elsewhere" / "harness" / "__init__.py"
            calls = {"n": 0}

            def counting_runner(cmd, env=None, cwd=None):
                calls["n"] += 1
                return subprocess.CompletedProcess(cmd, 0)

            env = {PIN_DEPTH_ENV: "1"}
            with mock.patch("harness.toolchain.current_origin", return_value=fake_origin), \
                    mock.patch.dict(os.environ, env):
                result = ensure_pinned_toolchain(ws, "harness.runner", [], runner=counting_runner)
            self.assertIsNone(result)
            self.assertEqual(calls["n"], 0)

    def test_chain_bounded_to_single_reexec_when_drift_persists(self) -> None:
        with tempfile.TemporaryDirectory(prefix="test_tc_chain_") as tmp:
            ws = Path(tmp)
            clone = ws / "ai-harness"
            (clone / "harness").mkdir(parents=True)
            (clone / "harness" / "__init__.py").write_text("", encoding="utf-8")
            fake_origin = Path(tmp) / "elsewhere" / "harness" / "__init__.py"
            calls = {"n": 0}

            def emulating_runner(cmd, env=None, cwd=None):
                calls["n"] += 1
                if calls["n"] > 50:
                    raise AssertionError("unbounded re-exec chain")
                with mock.patch("harness.toolchain.current_origin", return_value=fake_origin), \
                        mock.patch.dict(os.environ, env, clear=False):
                    child_pin = ensure_pinned_toolchain(ws, "harness.runner", [], runner=counting_runner)
                return subprocess.CompletedProcess(cmd, 0 if child_pin is None else 3)

            def counting_runner(cmd, env=None, cwd=None):  # pragma: no cover
                raise AssertionError("child must not re-exec")

            with mock.patch("harness.toolchain.current_origin", return_value=fake_origin):
                result = ensure_pinned_toolchain(ws, "harness.runner", [], runner=emulating_runner)
            self.assertIsNotNone(result)
            self.assertEqual(result.returncode, 0)
            self.assertEqual(calls["n"], 1)


def _make_toolchain_workspace(tmp: Path) -> Path:
    import harness as _harness_pkg

    src = Path(_harness_pkg.__file__).resolve().parents[1]
    ws = tmp / "competition"
    (ws / "configs").mkdir(parents=True)
    shutil.copytree(
        src,
        ws / "ai-harness",
        ignore=shutil.ignore_patterns(".git", "__pycache__", ".ai-harness", "dataset", "dashboard"),
    )
    (ws / "configs" / "task.yaml").write_text(
        "task:" + chr(10) + "  name: AuditTask" + chr(10)
        + "  primary_metric: accuracy" + chr(10)
        + "  data:" + chr(10) + "    train: dataset/train.csv" + chr(10),
        encoding="utf-8",
    )
    return ws


class TestStartSessionGate(unittest.TestCase):
    """Gap 2: the adoption gate must appear in the start-session observation."""

    def test_start_session_carries_adoption_gate(self) -> None:
        with tempfile.TemporaryDirectory(prefix="test_ss_gate_") as tmp:
            ws = _make_toolchain_workspace(Path(tmp))
            env = {**os.environ, "HARNESS_PROJECT_ROOT": str(ws)}
            subprocess.run(
                [sys.executable, "-c",
                 "import json;"
                 "from pathlib import Path;"
                 "from harness.project import attach_project;"
                 "from harness.skill_injector import inject_skill_file;"
                 "r = attach_project(str(Path('COMP')), force=True);"
                 "r['skill_path'] = str(inject_skill_file(config_path=Path('COMP/configs/task.yaml')));"
                 "print(json.dumps(r))".replace("COMP", str(ws).replace(chr(92), "/"))],
                capture_output=True, text=True, check=True,
                env={**env, "PYTHONPATH": str(ws / "ai-harness")},
                cwd=ws,
            )
            for i in range(3):
                (ws / f"kaggle_sub_exp{i:03d}").mkdir()
            proc = subprocess.run(
                [sys.executable, "-m", "tools.agent_bridge_cli",
                 "start-session", "--goal", "audit test"],
                capture_output=True, text=True, check=True,
                env={**env, "PYTHONPATH": str(ws / "ai-harness")},
                cwd=ws,
            )
            payload = json.loads(proc.stdout)
            self.assertIn("adoption_gate", payload)
            gate = " ".join(payload["adoption_gate"])
            self.assertIn("Adoption protocol", gate)
            self.assertIn("3 run-output dir(s)", gate)


class TestGroundingNextStep(unittest.TestCase):
    """Gap 3: ungrounded create_experiment must say exactly what to do next."""

    def test_next_required_step_present_when_unverified(self) -> None:
        with tempfile.TemporaryDirectory(prefix="test_gnd_step_") as tmp:
            ws = _make_toolchain_workspace(Path(tmp))
            env = {**os.environ, "HARNESS_PROJECT_ROOT": str(ws)}
            subprocess.run(
                [sys.executable, "-c",
                 "from pathlib import Path;"
                 "from harness.project import attach_project;"
                 "attach_project(str(Path('COMP')), force=True);".replace("COMP", str(ws).replace(chr(92), "/"))],
                capture_output=True, text=True, check=True,
                env={**env, "PYTHONPATH": str(ws / "ai-harness")},
                cwd=ws,
            )
            out = subprocess.run(
                [sys.executable, "-c",
                 "import json;"
                 "from pathlib import Path;"
                 "from harness.agent_bridge import AgentBridge;"
                 "b = AgentBridge(str(Path('COMP')));"
                 "sess = b.start_session(goal='g');"
                 "sess2, obs = b.execute_action('create_experiment', params={'branch': 'gnd-test', 'descriptor': 'gndtest'}, session_id=sess.session_id);"
                 "print(json.dumps(obs.data, default=str))".replace("COMP", str(ws).replace(chr(92), "/"))],
                capture_output=True, text=True,
                env={**env, "PYTHONPATH": str(ws / "ai-harness")},
                cwd=ws,
            )
            if out.returncode != 0:
                self.fail("create_experiment child failed: " + out.stderr[-800:])
            data = json.loads(out.stdout.strip().splitlines()[-1])
            self.assertEqual(data["grounding_status"], "UNVERIFIED")
            self.assertIn("grounding_gate.py", data["next_required_step"])
            self.assertIn("verified_baseline", data["next_required_step"])


class TestSkillPolicyDocs(unittest.TestCase):
    """Gap 4: SKILL.md must document the bypass policy + human override."""

    def test_skill_contains_bypass_policy(self) -> None:
        with tempfile.TemporaryDirectory(prefix="test_skill_pol_") as tmp:
            ws = _make_toolchain_workspace(Path(tmp))
            env = {**os.environ, "HARNESS_PROJECT_ROOT": str(ws)}
            proc = subprocess.run(
                [sys.executable, "-c",
                 "from pathlib import Path;"
                 "from harness.project import attach_project;"
                 "from harness.skill_injector import inject_skill_file;"
                 "attach_project(str(Path('COMP')), force=True);"
                 "inject_skill_file(config_path=Path('COMP/configs/task.yaml'));"
                 "t = Path('COMP/.agents/skills/harness/SKILL.md').read_text(encoding='utf-8');"
                 "assert 'AI_HARNESS_ALLOW_MANUAL' in t, 'override missing';"
                 "assert 'report the failure' in t, 'report policy missing';"
                 "assert 'Manual runs & bypass policy' in t, 'section missing';"
                 "print('ok')".replace("COMP", str(ws).replace(chr(92), "/"))],
                capture_output=True, text=True,
                env={**env, "PYTHONPATH": str(ws / "ai-harness")},
                cwd=ws,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr[-800:])
            self.assertIn("ok", proc.stdout)


class TestBypassDirCount(unittest.TestCase):
    """Gap 5: counts, not finding counts."""

    def test_hygiene_and_status_report_dir_counts(self) -> None:
        with tempfile.TemporaryDirectory(prefix="test_byp_cnt_") as tmp:
            ws = Path(tmp)
            (ws / ".ai-harness").mkdir()
            for i in range(4):
                (ws / f"kaggle_sub_exp{i:03d}").mkdir()
            sys.path.insert(0, str(SOURCE_ROOT))
            from harness.hygiene import scan_workspace_hygiene

            findings = scan_workspace_hygiene(ws)["findings"]
            bypassed = [f for f in findings if f["kind"] == "bypassed_run_output"]
            self.assertEqual(len(bypassed), 1)
            self.assertEqual(bypassed[0]["dir_count"], 4)
            adoption = _adoption_section(str(ws))
            self.assertEqual(adoption["bypassed_run_outputs"], 1)
            self.assertEqual(adoption["bypassed_run_output_dirs"], 4)


class TestHarnessProjectAutoResolution(unittest.TestCase):
    def test_harness_project_auto_resolution(self):
        from harness.project import HarnessProject

        project = HarnessProject(".")
        self.assertTrue(project.root.is_absolute())
        self.assertTrue(project.task_config_path.is_absolute())
        self.assertTrue(project.artifacts_dir.is_absolute())
        self.assertIsNotNone(project.manifest)

        default_proj = HarnessProject()
        self.assertEqual(default_proj.root, project.root)


if __name__ == "__main__":
    unittest.main()


class CleanupCompatibilityTests(unittest.TestCase):
    def test_legacy_skill_cleanup_is_equivalent_locally_and_globally(self):
        from harness.skill_injector import LEGACY_SKILL_NAMES, install_global_skills, install_workspace_skills
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            global_dir = root / "global"
            local_dir = root / ".agents/skills"
            for target in (global_dir, local_dir):
                for name in (*LEGACY_SKILL_NAMES, "user-owned"):
                    path = target / name / "SKILL.md"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("custom")
            config = {"name": "Fixture"}
            install_global_skills(target_dir=global_dir, task_config=config)
            install_workspace_skills(project_root=root, task_config=config)
            for target in (global_dir, local_dir):
                self.assertTrue((target / "user-owned/SKILL.md").exists())
                for name in LEGACY_SKILL_NAMES:
                    self.assertFalse((target / name).exists())

    def test_timing_discovers_nested_paths_and_duplicate_basenames(self):
        from tools.timing._measure_modules import discover_modules
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "No test modules"):
                discover_modules(root)
            for name in ("tests/unit/test_same.py", "tests/integration/test_same.py"):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("")
            self.assertEqual(discover_modules(root), ["tests/integration/test_same.py", "tests/unit/test_same.py"])

    def test_agent_package_forwarding_survives_module_removal(self):
        from harness.agent import AgentBridge as CompatBridge
        from harness.agent.cli import main
        self.assertIs(CompatBridge, AgentBridge)
        self.assertTrue(callable(main))
