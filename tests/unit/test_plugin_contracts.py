"""Declarative submission formatting and test-data failures obey the task contract."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from harness.plugins import DeclarativeTaskPlugin, TaskPluginContext
from tests.fixtures import task


class DeclarativeSubmissionContractTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.spec = task()
        self.spec.metadata["submission_spec"] = {"required_columns": ["id", "target"]}
        self.spec.data_config["test_path"] = "test.csv"
        pd.DataFrame({"id": [1, 2]}).to_csv(self.root / "test.csv", index=False)
        self.context = TaskPluginContext(self.root, self.root / "task.yaml", self.spec, "default")
        self.plugin = DeclarativeTaskPlugin(self.context)
        self.predictions = pd.DataFrame({"id": [1, 2], "pred_target": [0.1, 0.2]})
        self.output = self.root / "submission.csv"

    def test_test_split_loading_errors_fail_closed(self):
        self.plugin.build_submission(self.context, self.predictions, self.output)
        for error in (RuntimeError("loader failed"), OSError("read failed"), ValueError("malformed data")):
            with self.subTest(error=type(error).__name__):
                with patch.object(self.spec, "load_dataset_split", side_effect=error) as loader:
                    with self.assertRaises(type(error)) as raised:
                        self.plugin.validate_submission(self.context, self.output)
                    self.assertIs(raised.exception, error)
                    loader.assert_called_once_with("test", project_root=self.root)

    def test_valid_submission_and_id_mismatches(self):
        self.plugin.build_submission(self.context, self.predictions, self.output)
        self.plugin.validate_submission(self.context, self.output)
        for ids, message in (([1], "row count"), ([1, 3], "IDs do not match")):
            with self.subTest(ids=ids):
                pd.DataFrame({"id": ids, "target": [0.1] * len(ids)}).to_csv(self.output, index=False)
                with self.assertRaisesRegex(ValueError, message):
                    self.plugin.validate_submission(self.context, self.output)

    def test_id_padding_is_declarative_with_geoid_compatibility(self):
        cases = (("id", 5, ["00001", "00012"]),
                 ("GEOID", None, ["00000000001", "00000000012"]),
                 ("GEOID", 4, ["0001", "0012"]),
                 ("id", None, ["1.0", "12.0"]))
        for required_columns in (True, False):
            for id_col, zfill, expected in cases:
                with self.subTest(required_columns=required_columns, id_col=id_col, zfill=zfill):
                    self.spec.data_config.update(id_column=id_col, id_zfill=zfill)
                    self.spec.metadata["submission_spec"] = (
                        {"required_columns": [id_col, "target"]} if required_columns else {"archive_format": "csv"}
                    )
                    predictions = pd.DataFrame({id_col: [1.0, 12.0], "pred_target": [0.1, 0.2]})
                    self.plugin.build_submission(self.context, predictions, self.output)
                    result = pd.read_csv(self.output, dtype={id_col: str})
                    self.assertEqual(result[id_col].tolist(), expected)
                    self.assertEqual(result["target"].tolist(), [0.1, 0.2])
