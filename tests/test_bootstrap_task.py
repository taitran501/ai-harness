"""Unit and Integration tests for Autonomous Task Bootstrapper and Multi-Format Ingestion."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
import zipfile

import pandas as pd
import yaml

from harness.agent_bridge import AgentBridge, TypedActionName
from harness.task_spec import UniversalTaskSpec, load_dataframe_file, extract_archives_in_directory
from tools.bootstrap_task import (
    analyze_dataset_schema,
    bootstrap_competition_task,
    extract_context_from_markdown,
    find_data_files,
    generate_validation_splits,
)


class TestBootstrapTask(unittest.TestCase):
    """Verifies that ML-Harness autonomously onboards arbitrary competitions and dataset formats."""

    def setUp(self) -> None:
        self.test_dir = Path(tempfile.mkdtemp(prefix="test_bootstrap_"))
        self.dataset_dir = self.test_dir / "dataset"
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.docs_dir = self.test_dir / "docs"
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        self.configs_dir = self.test_dir / "configs"
        self.configs_dir.mkdir(parents=True, exist_ok=True)
        self.skills_dir = self.test_dir / ".agents" / "skills" / "ml-experiment-harness"
        self.skills_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_multi_format_loading_and_archive_extraction(self) -> None:
        """Verify that arbitrary formats (parquet, jsonl, zip) are seamlessly extracted and loaded."""
        # 1. Create Parquet dataset
        df_sample = pd.DataFrame({
            "record_id": [f"rec_{i}" for i in range(25)],
            "feature_val": [i * 1.5 for i in range(25)],
            "group_key": [f"grp_{i % 5}" for i in range(25)],
            "target": [i % 2 for i in range(25)],
        })
        pq_path = self.dataset_dir / "raw_data.parquet"
        df_sample.to_parquet(pq_path, index=False)

        # 2. Pack it into a .zip archive
        zip_path = self.dataset_dir / "data_package.zip"
        with zipfile.ZipFile(zip_path, "w") as z:
            z.write(pq_path, arcname="train.parquet")
        pq_path.unlink()

        # 3. Test auto-extraction
        extracted = extract_archives_in_directory(self.dataset_dir)
        self.assertTrue(len(extracted) > 0)
        extracted_pq = self.dataset_dir / "data_package" / "train.parquet"
        self.assertTrue(extracted_pq.exists())

        # 4. Test universal loader
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
        # Create dataset (JSONL format)
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

        # Create context
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

        # Verify task.yaml conforms to UniversalTaskSpec
        spec = UniversalTaskSpec.from_yaml(out_cfg)
        self.assertEqual(spec.name, "Hospital Risk AI")
        self.assertEqual(spec.platform, "kaggle")
        self.assertEqual(spec.id_column, "patient_uid")
        self.assertEqual(spec.group_column, "hospital_site_id")

        # Verify splits/fold_assignments.tsv
        splits_file = self.test_dir / "splits" / "fold_assignments.tsv"
        self.assertTrue(splits_file.exists())
        splits_df = pd.read_csv(splits_file, sep="\t")
        self.assertEqual(len(splits_df), 50)
        self.assertEqual(splits_df["fold"].nunique(), 5)

        # Verify GroupKFold no leakage between train and val groups
        merged = splits_df.merge(load_dataframe_file(train_jsonl), on="patient_uid")
        for fold_idx in range(5):
            val_groups = set(merged[merged["fold"] == fold_idx]["hospital_site_id"].unique())
            tr_groups = set(merged[merged["fold"] != fold_idx]["hospital_site_id"].unique())
            self.assertEqual(len(val_groups.intersection(tr_groups)), 0, "GroupKFold leakage detected!")

        # Verify exp000_baseline.yaml exists and complies with GroundingGate
        baseline_file = self.configs_dir / "exp000_baseline.yaml"
        self.assertTrue(baseline_file.exists())
        base_data = yaml.safe_load(baseline_file.read_text(encoding="utf-8"))
        from harness.grounding_gate import GroundingGate
        passed, errors, _ = GroundingGate.verify_config(base_data, config_path=baseline_file, raise_on_error=False)
        self.assertTrue(passed, f"GroundingGate verification failed: {errors}")


if __name__ == "__main__":
    unittest.main()
