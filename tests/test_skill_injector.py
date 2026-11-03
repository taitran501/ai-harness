"""Unit tests for the Dynamic Skill Injector Engine in AI-Harness."""

import json
import socketserver
import tempfile
import threading
import time
import unittest
import urllib.request
from pathlib import Path

from harness.skill_injector import (
    generate_compatibility_skill_content,
    generate_modular_skills,
    generate_skill_content,
    inject_skill_file,
    install_global_skills,
    install_workspace_skills,
)
from tests.workspace_fixtures import (
    cleanup_dashboard_workspace,
    prepare_dashboard_workspace,
)
from tools.dashboard_server import DashboardHandler


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
        """Verify dynamic skill generation for NLP multi-task competition."""
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
        """Verify dynamic skill generation for Tabular regression competition."""
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
        """Verify dynamic skill generation for Computer Vision competition."""
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
        """Verify dynamic skill generation includes anti-defeatism and new discussion/codabench commands."""
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
        """Verify task-aware modular slash skill generation."""
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

        # Harness skill verification
        harness = modular["harness"]
        self.assertIn("name: harness", harness)
        self.assertIn("ML Experiment Harness", harness)
        self.assertIn("ai-harness-agent context", harness)
        self.assertIn("/debate", harness)
        self.assertIn("/research", harness)
        self.assertIn("Candidate Preflight & Diversity Check", harness)
        self.assertIn("registered promotion rule", harness)

        # Debate skill verification
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
        """Verify installing modular skills into global directory and purging legacy."""
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
        """Verify installing modular skills into workspace .agents/skills/."""
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
            self.assertEqual(len(written), 3)  # harness, debate, research (no legacy ai-harness alias)
            skills_dir = root / ".agents" / "skills"
            self.assertTrue((skills_dir / "harness" / "SKILL.md").exists())
            self.assertTrue((skills_dir / "debate" / "SKILL.md").exists())
            self.assertTrue((skills_dir / "research" / "SKILL.md").exists())
            self.assertFalse((skills_dir / "ai-harness").exists())
            self.assertFalse((skills_dir / "submit").exists())
            self.assertFalse((skills_dir / "ai-harness-resume").exists())

    def test_cli_install_skills(self):
        """Verify CLI subcommand ai-harness install-skills."""
        import os
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
        """Verify writing skill file to disk in a temporary directory."""
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
        """Verify that legacy generated ai-harness skills are retired in workspace and global scopes."""
        legacy_content = generate_compatibility_skill_content({"task_name": "Old Task"})

        # Workspace scope
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

        # Global scope
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
        """Verify that ai-harness directory is not deleted if it contains non-generated user files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            legacy_dir = root / ".agents" / "skills" / "ai-harness"
            legacy_dir.mkdir(parents=True)
            (legacy_dir / "SKILL.md").write_text("# My Custom Skill\nNot generated by harness", encoding="utf-8")
            (legacy_dir / "custom_notes.txt").write_text("User notes", encoding="utf-8")

            with self.assertWarnsRegex(UserWarning, "Preserved unrecognized legacy skill"):
                install_workspace_skills(project_root=root, task_config={"task_name": "T"})
            # Should NOT touch user files
            self.assertTrue(legacy_dir.exists())
            self.assertTrue((legacy_dir / "custom_notes.txt").is_file())
            self.assertTrue((legacy_dir / "SKILL.md").is_file())
            self.assertIn("My Custom Skill", (legacy_dir / "SKILL.md").read_text(encoding="utf-8"))

    def test_retired_alias_cannot_be_generated_explicitly(self):
        """Verify that attempting to generate or output to ai-harness raises ValueError."""
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
        """Verify that attach_project on an already-attached workspace retires legacy alias and ensures harness/SKILL.md exists."""
        from unittest.mock import patch
        from harness.project import attach_project
        with (
            tempfile.TemporaryDirectory(prefix="aih-skill-migration-") as tmp,
            patch("harness.skill_injector.install_global_skills", return_value=[]),
        ):
            root = Path(tmp)
            cfg_dir = root / "configs"
            cfg_dir.mkdir(parents=True)
            cfg_path = cfg_dir / "task.yaml"
            cfg_path.write_text("task_name: Reattach Task\nplatform: custom\nprimary_metric: Acc\ndirection: maximize\n", encoding="utf-8")

            # First attach
            res1 = attach_project(project_root=root, task_config=cfg_path, force=True)
            self.assertEqual(res1["status"], "ATTACHED")

            # Simulate legacy state: delete harness/SKILL.md and add legacy ai-harness alias
            harness_skill = root / ".agents" / "skills" / "harness" / "SKILL.md"
            if harness_skill.exists():
                harness_skill.unlink()
            legacy_dir = root / ".agents" / "skills" / "ai-harness"
            legacy_dir.mkdir(parents=True, exist_ok=True)
            (legacy_dir / "SKILL.md").write_text(
                generate_compatibility_skill_content({"task_name": "Reattach Task"}),
                encoding="utf-8",
            )

            # Re-attach (without force)
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
        """POST /api/skill/inject should re-generate and update SKILL.md."""
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
        """POST /api/task_info should save task config and automatically inject skill."""
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


if __name__ == "__main__":
    unittest.main()
