import io
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from harness.cloud_job_runner import CloudJob, CloudJobManager
from tests.fixtures import make_workspace


class CloudCancellationTests(unittest.TestCase):
    def setUp(self):
        kill_signal = patch("harness.cloud_job_runner.signal.SIGKILL", 9, create=True)
        kill_signal.start()
        self.addCleanup(kill_signal.stop)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root, _ = make_workspace(root_path=Path(self.tmp.name))
        self.manager = CloudJobManager(root)
        self.job = CloudJob("job_test", "local", "config.yaml", "exp1", status="RUNNING", pid=12345)
        self.manager._jobs[self.job.job_id] = self.job
        self.proc = Mock(pid=12345, returncode=-15)
        self.proc.poll.return_value = None
        self.manager._processes[self.job.job_id] = self.proc

    def stop(self):
        return self.manager.stop_job(self.job.job_id)

    def test_graceful_and_forced_exit_precede_terminal_accounting(self):
        for forced in (False, True):
            self.job.status = "RUNNING"
            self.manager._processes[self.job.job_id] = self.proc
            self.proc.poll.return_value = None
            def wait(timeout):
                if forced and timeout == 5.0 and not getattr(wait, "called", False):
                    wait.called = True
                    raise subprocess.TimeoutExpired("job", timeout)
                self.proc.poll.return_value = -15
                return -15
            self.proc.wait.side_effect = wait
            def persist(job):
                self.assertIsNotNone(self.proc.poll())
                self.assertEqual(job.status, "CANCELLED")
            with patch("harness.cloud_job_runner.sys.platform", "linux"), patch(
                    "harness.cloud_job_runner.os.killpg", create=True) as signaler, patch.object(
                    self.manager, "_persist_job", side_effect=persist) as persist_mock:
                self.assertTrue(self.stop()["success"])
                self.assertEqual(signaler.call_count, 2 if forced else 1)
                persist_mock.assert_called_once()

    def test_survivor_permission_and_unknown_liveness_preserve_nonterminal_state(self):
        for failure in (subprocess.TimeoutExpired("job", 5), PermissionError("denied")):
            self.proc.wait.side_effect = failure
            with patch("harness.cloud_job_runner.sys.platform", "linux"), patch(
                    "harness.cloud_job_runner.os.killpg", create=True), patch.object(self.manager, "_persist_job") as persist:
                self.assertFalse(self.stop()["success"])
                self.assertEqual(self.job.status, "RUNNING")
                persist.assert_not_called()
        self.manager._processes.clear()
        with patch.object(self.manager, "_probe_pid", return_value=None), patch.object(self.manager, "_persist_job") as persist:
            self.assertFalse(self.stop()["success"])
            persist.assert_not_called()

    def test_signal_permission_and_windows_taskkill_failures(self):
        for platform in ("linux", "win32"):
            with patch("harness.cloud_job_runner.sys.platform", platform), patch(
                    "harness.cloud_job_runner.os.killpg", side_effect=PermissionError, create=True), patch(
                    "harness.cloud_job_runner.subprocess.run", side_effect=PermissionError):
                self.assertFalse(self.stop()["success"])
                self.assertEqual(self.job.status, "RUNNING")

    def test_liveness_query_failure_is_structured_and_nonterminal(self):
        self.proc.poll.side_effect = PermissionError("query denied")
        with patch.object(self.manager, "_persist_job") as persist:
            result = self.stop()
        self.assertFalse(result["success"])
        self.assertIn("liveness unavailable", result["error"])
        self.assertEqual(self.job.status, "RUNNING")
        persist.assert_not_called()

    def test_recovered_pid_is_polled_until_confirmed_exit(self):
        self.manager._processes.clear()
        with patch.object(self.manager, "_probe_pid", side_effect=[True, True, False]), patch(
                "harness.cloud_job_runner.sys.platform", "linux"), patch(
                "harness.cloud_job_runner.os.killpg", create=True), patch.object(self.manager, "_persist_job") as persist:
            self.assertTrue(self.stop()["success"])
            persist.assert_called_once()

    def test_completed_and_remote_jobs_cannot_be_cancelled(self):
        for status, backend in (("COMPLETED", "local"), ("RUNNING", "kaggle"), ("RUNNING", "modal")):
            self.job.status, self.job.backend = status, backend
            with patch.object(self.manager, "_terminate_local_process") as terminate:
                self.assertFalse(self.stop()["success"])
                terminate.assert_not_called()

    def test_monitor_race_cannot_override_cancellation_or_run_hooks(self):
        exited = threading.Event()
        def wait(timeout=None):
            if timeout is None:
                exited.wait(2)
            else:
                self.proc.poll.return_value = -15
                exited.set()
            return -15
        self.proc.wait.side_effect = wait
        with patch("harness.cloud_job_runner.sys.platform", "linux"), patch(
                "harness.cloud_job_runner.os.killpg", create=True), patch.object(self.manager, "_persist_job") as persist, patch.object(
                self.manager, "_handle_post_run") as hooks:
            thread = threading.Thread(target=self.manager._monitor_process,
                args=(self.job.job_id, self.proc, io.StringIO()))
            thread.start()
            self.assertTrue(self.stop()["success"])
            thread.join(3)
            self.assertFalse(thread.is_alive())
            self.assertEqual(self.job.status, "CANCELLED")
            hooks.assert_not_called()
            persist.assert_called_once()

    def test_recovered_manager_cancellation_is_observed_by_original_monitor(self):
        self.manager._persist_job(self.job)
        recovered = CloudJobManager(self.manager.project_root)
        with patch.object(recovered, "_probe_pid", return_value=True), patch.object(
                recovered, "_terminate_local_process", return_value=True):
            self.assertTrue(recovered.stop_job(self.job.job_id)["success"])
        self.proc.wait.return_value = 0
        with patch.object(self.manager, "_handle_post_run") as hooks, patch.object(self.manager, "_persist_job") as persist:
            self.manager._monitor_process(self.job.job_id, self.proc, io.StringIO())
            self.assertEqual(self.job.status, "CANCELLED")
            hooks.assert_not_called()
            persist.assert_not_called()

    def test_real_local_process_exits_before_success(self):
        proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                start_new_session=sys.platform != "win32")
        def cleanup():
            if proc.poll() is None:
                proc.kill()
            proc.wait(timeout=5)
        self.addCleanup(cleanup)
        self.job.pid = proc.pid
        self.manager._processes[self.job.job_id] = proc
        # taskkill's RPC inspection is denied in restricted Windows CI. Inject
        # only the signal primitive; wait()/poll() still observe a real child.
        if sys.platform == "win32":
            def terminate_owned_child(command, **kwargs):
                self.assertEqual(command[command.index("/PID") + 1], str(proc.pid))
                proc.terminate()
                return subprocess.CompletedProcess(command, 0)
            with patch("harness.cloud_job_runner.subprocess.run", side_effect=terminate_owned_child):
                result = self.stop()
        else:
            result = self.stop()
        self.assertTrue(result["success"], result)
        self.assertIsNotNone(proc.poll())
        self.assertEqual(self.job.status, "CANCELLED")

    def test_monitor_wait_error_does_not_finalize_a_running_process(self):
        self.proc.wait.side_effect = OSError("inspection failed")
        with patch.object(self.manager, "_persist_job") as persist:
            self.manager._monitor_process(self.job.job_id, self.proc, io.StringIO())
        self.assertEqual(self.job.status, "RUNNING")
        persist.assert_not_called()
