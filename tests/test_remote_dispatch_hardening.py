"""Remote-dispatch hardening (from the first live AIMO dispatch, 2026-10-16).

Regression coverage for failures observed during a real Kaggle dispatch:

1. ``CloudJobManager._pid_running`` used ``os.kill(pid, 0)`` — on Windows
   CPython maps every signal (including 0) to ``TerminateProcess``, so a
   status poll silently KILLED the worker it was checking. Verified
   empirically: a live ``sleep(60)`` process died with exit 3221225794.
2. A runner started inside a dispatched kernel re-applied the task
   contract's cloud backend and tried to dispatch a nested kernel (blocked
   by the upload-consent check, which only exists locally). Fix: the remote
   templates set ``AI_HARNESS_REMOTE_KERNEL=1``; the runner forces local
   execution under that flag; ``_prepare_remote_run`` refuses re-entry.
3. Dead workers without a terminal state stayed ``RUNNING`` forever (the
   monitor thread dies with the spawning process). ``get_job`` now
   reconciles: dead pid + no terminal state + no verified artifacts
   => FAILED with an actionable message.
4. ``ai-harness kaggle status/log --exp`` derived the slug without the
   job-hash suffix and 404'd on every real dispatch. Job state's
   ``remote_id`` is now the source of truth.
5. A constant ``fold`` column produced the misleading
   "Declared fold count differs from split file".
6. Per-workspace hygiene allow-lists (``task.hygiene.allowed_top_dirs``)
   stop the generic scanner flagging legitimate vendor directories
   (official ``getting-started``, model caches, prior solutions).
"""

from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from tests.fixtures import make_workspace, task_document

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestPidRunningNonlethal(unittest.TestCase):
    def test_liveness_check_does_not_kill_the_process(self) -> None:
        """Regression: the old os.kill(pid, 0) probe terminated the process
        on Windows the moment a status poll checked it."""
        from harness.cloud_job_runner import CloudJobManager

        proc = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"]
        )
        try:
            self.assertTrue(CloudJobManager._pid_running(proc.pid))
            # The pre-fix implementation killed the process on the first
            # call (Windows); it must still be alive after two probes.
            self.assertTrue(CloudJobManager._pid_running(proc.pid))
            self.assertIsNone(proc.poll(), "liveness probe killed the worker")
        finally:
            proc.kill()
            proc.wait()

    def test_dead_pid_reports_false(self) -> None:
        from harness.cloud_job_runner import CloudJobManager

        proc = subprocess.Popen([sys.executable, "-c", "pass"])
        proc.wait()
        self.assertFalse(CloudJobManager._pid_running(proc.pid))
        self.assertFalse(CloudJobManager._pid_running(None))
        self.assertFalse(CloudJobManager._pid_running(0))


class TestRemoteKernelReentrancyGuard(unittest.TestCase):
    def test_prepare_remote_run_refuses_reentry(self) -> None:
        from harness.cloud_runner import _prepare_remote_run

        os.environ["AI_HARNESS_REMOTE_KERNEL"] = "1"
        try:
            with self.assertRaises(RuntimeError) as ctx:
                _prepare_remote_run("configs/does-not-matter.yaml", "kaggle")
            self.assertIn("already inside a dispatched remote kernel", str(ctx.exception))
        finally:
            del os.environ["AI_HARNESS_REMOTE_KERNEL"]

    def test_runner_forces_local_under_remote_kernel_flag(self) -> None:
        """End-to-end: the runner CLI inside a flagged kernel must execute
        locally even when the contract is locked to kaggle (no dispatch, no
        lock violation)."""
        doc = task_document(task_id="reentry-task")
        doc["task"]["execution"] = {"backend": "kaggle", "locked": True}
        root, task_config = make_workspace(prefix="aih-reentry-", task=doc)
        try:
            (root / "configs" / "exp-guard.yaml").write_text(
                "task:\n  name: Fixture Task\nmodel:\n  name: logistic_regression\n",
                encoding="utf-8",
            )
            env = dict(os.environ, AI_HARNESS_REMOTE_KERNEL="1")
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "harness.runner",
                    "--project-root",
                    str(root),
                    "--config",
                    "configs/exp-guard.yaml",
                    "--debug",
                    "--fold",
                    "0",
                ],
                capture_output=True,
                text=True,
                env=env,
                cwd=str(REPO_ROOT),
                timeout=180,
            )
            combined = result.stdout + result.stderr
            self.assertIn(
                "inside a dispatched remote kernel",
                combined,
                f"guard notice missing; output:\n{combined}",
            )
            self.assertNotIn("Backend lock violation", combined)
            self.assertNotIn("kaggle push", combined.lower())
        finally:
            import shutil

            shutil.rmtree(root, ignore_errors=True)

    def test_kaggle_template_sets_remote_kernel_flag(self) -> None:
        from harness.templates import kaggle_template

        source = inspect.getsource(kaggle_template.KaggleKernelManager.setup_kernel_package)
        guard = 'os.environ["AI_HARNESS_REMOTE_KERNEL"] = "1"'
        self.assertIn(guard, source, "kaggle runner script must set the re-entrancy flag")
        # The flag must be set before the runner command is built/executed.
        self.assertLess(source.index(guard), source.index("harness.runner"))

    def test_modal_template_sets_remote_kernel_flag(self) -> None:
        from harness.templates import modal_template

        source = inspect.getsource(modal_template.generate_modal_script)
        guard = 'os.environ["AI_HARNESS_REMOTE_KERNEL"] = "1"'
        self.assertIn(guard, source, "modal remote script must set the re-entrancy flag")
        self.assertLess(source.index(guard), source.index("harness.runner"))


class TestDeadWorkerReconciliation(unittest.TestCase):
    def test_dead_worker_without_terminal_state_is_failed(self) -> None:
        from harness.cloud_job_runner import CloudJobManager

        root, _ = make_workspace(prefix="aih-deadworker-")
        try:
            from harness.cloud_job_runner import CloudJob

            manager = CloudJobManager(root)
            job = CloudJob(
                job_id="job_dead_worker",
                backend="kaggle",
                config_path="configs/exp.yaml",
                exp_id="exp_dead",
                task_id="fixture-task",
                config_sha256="0" * 64,
                artifact_dir=".ai-harness/artifacts/exp_dead",
            )
            # Simulate the observed failure: the worker process died
            # (killed/OOM/crash) without recording a terminal state, and the
            # spawning process is gone too (monitor thread lost).
            proc = subprocess.Popen([sys.executable, "-c", "pass"])
            proc.wait()
            job.pid = proc.pid
            job.status = "RUNNING"
            manager._persist_job(job)

            refreshed = CloudJobManager(root).get_job(job.job_id)
            self.assertIsNotNone(refreshed)
            self.assertEqual(refreshed["status"], "FAILED")
            self.assertIn("no longer running", refreshed["error_message"])
        finally:
            import shutil

            shutil.rmtree(root, ignore_errors=True)


class TestExpKernelSlugResolution(unittest.TestCase):
    def test_slug_comes_from_job_state_not_derivation(self) -> None:
        from harness.kaggle_inspect import _resolve_exp_kernel_slug
        from harness.project import HarnessProject

        root, _ = make_workspace(prefix="aih-slug-")
        old_cwd = os.getcwd()
        try:
            jobs_dir = HarnessProject.discover(root).artifacts_dir / "jobs"
            jobs_dir.mkdir(parents=True, exist_ok=True)
            newer = {
                "job_id": "job_newer",
                "status": "FAILED",
                "exp_id": "exp_1",
                "remote_id": "tester/fixture-task-exp-1-abc12345",
                "started_at": 200.0,
            }
            older = {
                "job_id": "job_older",
                "status": "COMPLETED",
                "exp_id": "exp_1",
                "remote_id": "tester/fixture-task-exp-1-old99999",
                "started_at": 100.0,
            }
            (jobs_dir / "job_newer.json").write_text(json.dumps(newer))
            (jobs_dir / "job_older.json").write_text(json.dumps(older))
            os.chdir(root)
            slug = _resolve_exp_kernel_slug("exp_1")
            self.assertEqual(slug, "tester/fixture-task-exp-1-abc12345")
            # Unknown exp falls back to derivation (no job state recorded).
            self.assertIsNone(_resolve_exp_kernel_slug("exp_unknown"))
        finally:
            os.chdir(old_cwd)
            import shutil

            shutil.rmtree(root, ignore_errors=True)


class TestConstantFoldColumnError(unittest.TestCase):
    def test_constant_fold_raises_actionable_error(self) -> None:
        import argparse

        import pandas as pd
        import yaml

        from harness.runner import Runner

        root = Path(tempfile.mkdtemp(prefix="aih-foldconst-"))
        try:
            task_yaml = root / "task.yaml"
            task_dict = task_document(
                task_id="fold-task",
                train_path="dataset/train.csv",
                num_folds=2,
            )
            task_dict["task"]["subtasks"] = {
                "label": {"task_type": "classification", "target_column": "label", "classes": [0, 1]}
            }
            task_yaml.write_text(yaml.safe_dump(task_dict), encoding="utf-8")
            exp_yaml = root / "exp.yaml"
            exp_dict = {
                "task": task_dict["task"],
                "model": {"name": "logistic_regression"},
                "validation": {"strategy": "kfold", "num_folds": 2},
            }
            exp_yaml.write_text(yaml.safe_dump(exp_dict), encoding="utf-8")

            args = argparse.Namespace(
                project_root=str(root),
                task_config=str(task_yaml),
                debug=True,
                probe_samples=None,
                fold=None,
                job_id="test_foldconst",
                force_import=False,
            )
            runner = Runner(exp_dict, exp_yaml, args)
            runner.task_plugin.build_validation_splits = lambda *a, **k: None

            df = pd.DataFrame(
                {
                    "id": range(10),
                    "text": [f"row {i}" for i in range(10)],
                    "label": [0, 1] * 5,
                    # The AIMO trap: the dataset ships a constant fold column.
                    "fold": [0] * 10,
                }
            )
            with self.assertRaises(ValueError) as ctx:
                runner._resolve_validation_splits(df, "kfold")
            message = str(ctx.exception)
            self.assertIn("constant 'fold' column", message)
            self.assertIn("Remove the column", message)
        finally:
            import shutil

            shutil.rmtree(root, ignore_errors=True)


class TestHygieneTaskAllowlist(unittest.TestCase):
    def test_task_spec_directories_are_not_flagged(self) -> None:
        from harness.hygiene import scan_workspace_hygiene

        doc = task_document()
        doc["task"]["hygiene"] = {"allowed_top_dirs": ["hf_local", "baselines"]}
        root, _ = make_workspace(prefix="aih-hygene-", task=doc)
        try:
            for name in ("hf_local", "baselines", "getting-started", "rogue_dir"):
                (root / name).mkdir()
                (root / name / "x.txt").write_text("x")
            report = scan_workspace_hygiene(root)
            flagged = {f["path"] for f in report["findings"]}
            self.assertNotIn("hf_local", flagged)
            self.assertNotIn("baselines", flagged)
            # 'getting-started' is still a generic unknown dir for THIS
            # workspace (not in its allow-list) => still flagged.
            self.assertIn("getting-started", flagged)
            self.assertIn("rogue_dir", flagged)
        finally:
            import shutil

            shutil.rmtree(root, ignore_errors=True)

    def test_allowlist_cache_invalidates_on_mtime_change(self) -> None:
        from harness.hygiene import scan_workspace_hygiene, task_allowed_dirs

        doc = task_document()
        doc["task"]["hygiene"] = {"allowed_top_dirs": ["cache_a"]}
        root, task_config = make_workspace(prefix="aih-hygcache-", task=doc)
        try:
            self.assertEqual(task_allowed_dirs(root), frozenset({"cache_a"}))
            doc2 = task_document()
            doc2["task"]["hygiene"] = {"allowed_top_dirs": ["cache_b"]}
            task_config.write_text(
                __import__("yaml").safe_dump(doc2, sort_keys=False), encoding="utf-8"
            )
            # Force a mtime delta on coarse-grained filesystems.
            future = time.time() + 5
            os.utime(task_config, (future, future))
            self.assertEqual(task_allowed_dirs(root), frozenset({"cache_b"}))
            self.assertNotIn(
                "hf_local",
                {f["path"] for f in scan_workspace_hygiene(root)["findings"]},
            )
            (root / "cache_b").mkdir()
            self.assertNotIn(
                "cache_b",
                {f["path"] for f in scan_workspace_hygiene(root)["findings"]},
            )
        finally:
            import shutil

            shutil.rmtree(root, ignore_errors=True)


class TestAuditGateOofIds(unittest.TestCase):
    """The AIMO baseline's predictions.csv is an OOF table: one row per
    (id, fold, input_variant), so `id` repeats by design. The audit gate
    rejected it with 'IDs id must be non-null and unique' and left the
    otherwise-verified run audit_passed=False."""

    def _oof_df(self, rows_per_key: int = 3, duplicate_row: bool = False):
        import pandas as pd

        rows = []
        for i in range(4):
            for j in range(rows_per_key):
                rows.append(
                    {
                        "id": f"case_{i:04d}",
                        "fold": j,
                        "input_variant": "problem_text",
                        "pred_robustness_prediction": bool(i % 2),
                    }
                )
        if duplicate_row:
            rows.append(dict(rows[0]))
        return pd.DataFrame(rows)

    def test_multi_variant_oof_table_passes_integrity(self) -> None:
        import sys

        sys.path.insert(0, str(REPO_ROOT / "tools"))
        import audit_gate

        df = self._oof_df()
        self.assertTrue(df["id"].duplicated().any(), "test data must repeat ids")
        self.assertEqual(audit_gate.audit_prediction_integrity(df, id_col="id"), [])

    def test_true_duplicate_within_scope_still_fails(self) -> None:
        import sys

        sys.path.insert(0, str(REPO_ROOT / "tools"))
        import audit_gate

        df = self._oof_df(duplicate_row=True)
        errors = audit_gate.audit_prediction_integrity(df, id_col="id")
        self.assertTrue(
            any("unique within" in e for e in errors), f"expected scope error, got {errors}"
        )

    def test_plain_table_still_requires_unique_ids(self) -> None:
        import sys

        sys.path.insert(0, str(REPO_ROOT / "tools"))
        import audit_gate

        df = self._oof_df().drop(columns=["fold", "input_variant"])
        errors = audit_gate.audit_prediction_integrity(df, id_col="id")
        self.assertTrue(
            any("must be non-null and unique" in e for e in errors), f"got {errors}"
        )

    def test_diff_metrics_composite_identity(self) -> None:
        import sys

        sys.path.insert(0, str(REPO_ROOT / "tools"))
        import audit_gate

        cand = self._oof_df()
        base = cand.copy()
        base["pred_robustness_prediction"] = base["pred_robustness_prediction"] ^ True
        counts, _ = audit_gate.compute_diff_metrics(cand, base, id_col="id")
        self.assertEqual(counts, len(cand))
        # Changed rows must still be detectable inside the composite scope.
        base2 = cand.copy()
        counts2, _ = audit_gate.compute_diff_metrics(cand, base2, id_col="id")
        self.assertEqual(counts2, 0)


if __name__ == "__main__":
    unittest.main()
