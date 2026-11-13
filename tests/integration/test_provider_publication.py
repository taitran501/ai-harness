import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import json
import yaml

from harness.artifacts import publish_run_artifacts, contract_identity
from harness.plugins import TaskPlugin
from harness.templates.modal_template import ModalAppManager
from harness.templates.kaggle_template import KaggleKernelManager
from tests.evidence_fixtures import seal_fixture
from tests.fixtures import make_workspace, task


class _CustomScorerPlugin(TaskPlugin):
    def evaluate_predictions(self, context, truth, predictions):
        return {"primary_score": 0.75}


class ProviderPublicationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.spec = task()
        self.root, _ = make_workspace(root_path=Path(self.tmp.name) / "workspace", task={"task": self.spec.metadata})
        self.source = Path(self.tmp.name) / "source"
        seal_fixture(self.source, self.spec)
        self.dest = self.root / "artifacts/source"

    def test_kaggle_import_delegates_and_retries_exact_bytes(self):
        with patch("harness.artifacts.publish_run_artifacts", wraps=publish_run_artifacts) as publish:
            for _ in range(2):
                files = KaggleKernelManager.migrate_downloaded_artifacts(self.source, self.dest, project_root=self.root)
                self.assertIn("artifact_manifest.json", files)
            self.assertEqual(publish.call_count, 2)
        # Recovery must resolve the attached custom plugin rather than comparing
        # its scorer identity against the generic default implementation.
        project_file = self.root / ".ai-harness/project.yaml"
        document = yaml.safe_load(project_file.read_text(encoding="utf-8"))
        document["plugin_id"] = "publication-fixture"
        project_file.write_text(yaml.safe_dump(document), encoding="utf-8")
        plugin = _CustomScorerPlugin()
        path = self.source / "artifact_manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifest.update(contract_identity(self.spec, plugin))
        path.write_text(json.dumps(manifest), encoding="utf-8")
        with patch("harness.plugins.load_task_plugin", return_value=plugin) as resolve:
            KaggleKernelManager.migrate_downloaded_artifacts(self.source, self.root / "artifacts/custom", project_root=self.root)
            resolve.assert_called_once()

    def test_modal_recovery_delegates_and_preserves_conflict(self):
        def download(command, **kwargs):
            shutil.copytree(self.source, Path(command[5]) / "source")
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        with patch("harness.templates.modal_template.subprocess.run", side_effect=download), patch("harness.artifacts.publish_run_artifacts", wraps=publish_run_artifacts) as publish:
            result = ModalAppManager.pull_app_output("source", "volume", self.dest, {}, project_root=self.root)
            self.assertTrue(result["success"], result)
            self.assertEqual(publish.call_count, 1)
            result = ModalAppManager.pull_app_output("source", "volume", self.dest, {}, project_root=self.root)
            self.assertEqual(result["publication_disposition"], "already_present")
            sentinel = self.dest / "sentinel"
            sentinel.write_text("preserve", encoding="utf-8")
            result = ModalAppManager.pull_app_output("source", "volume", self.dest, {}, project_root=self.root)
            self.assertFalse(result["success"])
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "preserve")
