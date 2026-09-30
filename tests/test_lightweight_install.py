"""Smoke tests for the workflow-only installation profile."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]


class TestLightweightInstall(unittest.TestCase):
    def test_workflow_commands_run_without_numerical_dependencies(self):
        script = r"""
import contextlib
import importlib.abc
import io
import json
import sys
import tempfile
from pathlib import Path

BLOCKED = {"numpy", "pandas", "sklearn", "scipy"}

class BlockNumericalImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in BLOCKED:
            raise ModuleNotFoundError(f"blocked by lightweight-install test: {fullname}")
        return None

sys.meta_path.insert(0, BlockNumericalImports())

from harness.cli import main as harness_main
from tools.agent_bridge_cli import main as agent_main
from tools.dashboard_cli import main as dashboard_main
from tools.optional_cli import run_main

assert not any(name.split(".", 1)[0] in BLOCKED for name in sys.modules)

with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    (root / "configs").mkdir()
    (root / "configs" / "task.yaml").write_text(
        "task:\n  name: lightweight-fixture\n  primary_metric: accuracy\n  direction: maximize\n",
        encoding="utf-8",
    )
    with contextlib.redirect_stdout(io.StringIO()):
        assert harness_main(["phase", "init", "--project-root", str(root)]) == 0
        assert harness_main(["phase", "validate", "--project-root", str(root)]) == 0
        assert harness_main(["plugins"]) == 0

    packet_output = io.StringIO()
    with contextlib.redirect_stdout(packet_output):
        assert agent_main(["--project-root", str(root), "context", "--format", "json"]) == 0
    packet = json.loads(packet_output.getvalue())
    assert packet["profile"] == "workflow"
    assert packet["task"]["name"] == "lightweight-fixture"
    assert packet["phase_status"]["next_phase"] == "setup"

    assert harness_main(["eda", "run", "--project-root", str(root)]) == 2
    assert dashboard_main(["--project-root", str(root)]) == 2
    assert run_main() == 2

assert not any(name.split(".", 1)[0] in BLOCKED for name in sys.modules)
"""
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + environment.get("PYTHONPATH", "")
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=REPO_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("requires the optional 'ml' dependencies", result.stderr)


if __name__ == "__main__":
    unittest.main()
