"""Tests for dynamic task runtime requirements and fast uv image building for Modal."""

import tempfile
import unittest
from pathlib import Path

from harness.cli import attach_project
from harness.remote_runtime import resolve_remote_runtime
from harness.task_spec import load_active_task_spec
from harness.templates.modal_template import generate_modal_script


class TestModalRuntimeRequirements(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="aih-test-reqs-")
        self.root = Path(self.tmp.name)
        self.configs = self.root / "configs"
        self.configs.mkdir(parents=True)
        (self.root / "dataset").mkdir(parents=True)
        (self.root / "dataset" / "train.csv").write_text("id,val\n1,2\n", encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def test_task_spec_loads_runtime_requirements(self):
        task_yaml = self.configs / "task.yaml"
        task_yaml.write_text(
            """task:
  name: llm_task
  primary_metric: accuracy
  runtime:
    requirements:
      - "torch>=2.5.0"
      - "transformers>=4.51.0"
      - "peft>=0.14.0"
""",
            encoding="utf-8",
        )
        spec = load_active_task_spec(task_yaml)
        self.assertEqual(
            spec.runtime_requirements,
            ["torch>=2.5.0", "transformers>=4.51.0", "peft>=0.14.0"],
        )

    def test_resolve_remote_runtime_includes_task_requirements(self):
        task_yaml = self.configs / "task.yaml"
        task_yaml.write_text(
            """task:
  name: llm_task
  primary_metric: accuracy
  runtime:
    requirements:
      - "torch>=2.5.0"
      - "transformers>=4.51.0"
""",
            encoding="utf-8",
        )
        attach_project(self.root, task_yaml, force=True)
        from harness.project import HarnessProject

        project = HarnessProject.discover(self.root)
        runtime = resolve_remote_runtime(project)
        self.assertIn("torch>=2.5.0", runtime["requirements"])
        self.assertIn("transformers>=4.51.0", runtime["requirements"])
        # Check deduplication: no requirement appears more than once
        self.assertEqual(len(runtime["requirements"]), len(set(runtime["requirements"])))

    def test_generate_modal_script_uses_uv_and_deduplicates(self):
        task_yaml = self.configs / "task.yaml"
        task_yaml.write_text(
            """task:
  name: llm_task
  primary_metric: accuracy
  runtime:
    requirements:
      - "torch>=2.5.0"
      - "transformers>=4.51.0"
      - "numpy>=1.24.0"
""",
            encoding="utf-8",
        )
        attach_project(self.root, task_yaml, force=True)

        exp_yaml = self.configs / "exp001.yaml"
        exp_yaml.write_text("experiment:\n  id: exp001\n", encoding="utf-8")

        script = generate_modal_script(
            config_path=str(exp_yaml),
            project_root=self.root,
            volume_name="test_vol",
        )

        # Check uv pip install is used
        self.assertIn(".pip_install(\"uv\")", script)
        self.assertIn("uv pip install --system", script)
        self.assertIn("'torch>=2.5.0'", script)
        self.assertIn("'transformers>=4.51.0'", script)

        # Compiles valid python code
        compile(script, "test_modal_script.py", "exec")


if __name__ == "__main__":
    unittest.main()
