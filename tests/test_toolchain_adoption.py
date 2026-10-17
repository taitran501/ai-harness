"""Regression tests for the Round-4 adoption audit (5 verified gaps):

1. CLI execution toolchain drift -> harness/toolchain.py self-healing re-exec
2. start-session must surface the T3 adoption gate (not only `context`)
3. create_experiment observation carries `next_required_step` when ungrounded
4. generated SKILL.md documents the manual-run/bypass policy
5. hygiene/status report dir COUNTS, not just finding counts
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from harness.toolchain import (
    PIN_DISABLE_ENV,
    PIN_DEPTH_ENV,
    _needs_pin,
    ensure_pinned_toolchain,
    find_workspace_clone,
)


class TestToolchainPin(unittest.TestCase):
    """Gap 1: CLI entry points must execute the workspace clone's code."""

    def test_clone_discovery(self) -> None:
        with tempfile.TemporaryDirectory(prefix="test_tc_clone_") as tmp:
            ws = Path(tmp)
            clone = ws / "ai-harness"
            (clone / "harness").mkdir(parents=True)
            (clone / "harness" / "__init__.py").write_text("", encoding="utf-8")
            self.assertEqual(find_workspace_clone(ws), clone)
            # a bare dir without a harness package is NOT a clone
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
                needs, origin, expected = _needs_pin(ws)
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
        """The pinned child must carry the generation counter (depth=1)."""
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
        """A child where drift PERSISTS must not re-exec again.

        Without the guard this case spawns an unbounded re-exec chain
        (observed live: 35+ generations of ``python -m harness.cli`` pegging
        the CPU when the command was run from the framework repo against a
        workspace clone — the child's ``-m`` path resolution always prefers
        the cwd's own harness package over PYTHONPATH).
        """
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
            self.assertIsNone(result)  # fail-open, no re-exec
            self.assertEqual(calls["n"], 0)

    def test_chain_bounded_to_single_reexec_when_drift_persists(self) -> None:
        """End-to-end simulation of the live incident: parent re-execs once,
        the simulated child still drifts (cwd keeps its own harness package),
        and the child's pin must fail open — total runner calls exactly 1.
        """
        with tempfile.TemporaryDirectory(prefix="test_tc_chain_") as tmp:
            ws = Path(tmp)
            clone = ws / "ai-harness"
            (clone / "harness").mkdir(parents=True)
            (clone / "harness" / "__init__.py").write_text("", encoding="utf-8")
            fake_origin = Path(tmp) / "elsewhere" / "harness" / "__init__.py"
            calls = {"n": 0}

            def emulating_runner(cmd, env=None, cwd=None):
                calls["n"] += 1
                if calls["n"] > 50:  # the unguarded bug
                    raise AssertionError("unbounded re-exec chain")
                # Emulate the child process: same interpreter, same cwd,
                # child env as its os.environ, and the SAME persistent drift.
                with mock.patch("harness.toolchain.current_origin", return_value=fake_origin), \
                        mock.patch.dict(os.environ, env, clear=False):
                    child_pin = ensure_pinned_toolchain(ws, "harness.runner", [], runner=counting_runner)
                # The child either fails open (guard) or would loop: the
                # guard guarantees it exits with its own result code.
                return subprocess.CompletedProcess(cmd, 0 if child_pin is None else 3)

            def counting_runner(cmd, env=None, cwd=None):  # pragma: no cover - unreachable
                raise AssertionError("child must not re-exec")

            with mock.patch("harness.toolchain.current_origin", return_value=fake_origin):
                result = ensure_pinned_toolchain(ws, "harness.runner", [], runner=emulating_runner)
            self.assertIsNotNone(result)
            self.assertEqual(result.returncode, 0)  # child failed open, exited 0
            self.assertEqual(calls["n"], 1)         # exactly one generation


def _make_workspace(tmp: Path) -> Path:
    """Throwaway workspace whose ai-harness dir is a COPY of the current
    working tree (git clone would only see the last commit, not uncommitted
    edits under test). .git is skipped - the pin only needs a real
    harness package."""
    import shutil

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
            ws = _make_workspace(Path(tmp))
            env = {**os.environ, "HARNESS_PROJECT_ROOT": str(ws)}
            # attach (writes manifest + hooks)
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
            # bypass evidence: 3 unsealed run-output dirs
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
            ws = _make_workspace(Path(tmp))
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
            ws = _make_workspace(Path(tmp))
            env = {**os.environ, "HARNESS_PROJECT_ROOT": str(ws)}
            proc = subprocess.run(
                [sys.executable, "-c",
                 "from pathlib import Path;"
                 "from harness.project import attach_project;"
                 "from harness.skill_injector import inject_skill_file;"
                 "attach_project(str(Path('COMP')), force=True);"
                 "inject_skill_file(config_path=Path('COMP/configs/task.yaml'));"
                 "t = Path('COMP/.agents/skills/ai-harness/SKILL.md').read_text(encoding='utf-8');"
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
            sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
            from harness.hygiene import scan_workspace_hygiene
            from harness.status_report import _adoption_section

            findings = scan_workspace_hygiene(ws)["findings"]
            bypassed = [f for f in findings if f["kind"] == "bypassed_run_output"]
            self.assertEqual(len(bypassed), 1)
            self.assertEqual(bypassed[0]["dir_count"], 4)
            adoption = _adoption_section(str(ws))
            self.assertEqual(adoption["bypassed_run_outputs"], 1)
            self.assertEqual(adoption["bypassed_run_output_dirs"], 4)


if __name__ == "__main__":
    unittest.main()
