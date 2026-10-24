"""Execute exported notebook code: it must assemble real, ID-aligned predictions."""

import contextlib
import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from harness.kaggle_notebook_builder import build_kaggle_submission_notebook


class NotebookPredictionIntegrityTests(unittest.TestCase):
    def make_notebook(self, predictions=None, test=None):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        exp = root / "exp001"
        exp.mkdir()
        test = pd.DataFrame({"id": [10, 11]}) if test is None else test
        test_path = root / "chosen_test.csv"
        test.to_csv(test_path, index=False)
        if predictions is not None:
            predictions.to_csv(exp / "test_predictions.csv", index=False)
        output = root / "submission.ipynb"
        build_kaggle_submission_notebook(exp, output, test_path_override=str(test_path))
        return root, json.loads(output.read_text())

    def execute(self, notebook, glob_results=None):
        namespace = {}
        with (patch("glob.glob", side_effect=glob_results or (lambda *a, **kw: [])),
              patch.object(pd.DataFrame, "to_csv", autospec=True) as writer,
              contextlib.redirect_stdout(io.StringIO())):
            try:
                for cell in notebook["cells"]:
                    if cell["cell_type"] == "code":
                        exec("".join(cell["source"]), namespace)
            except Exception:
                writer.assert_not_called()
                raise
            self.assertEqual(writer.call_count, 1)
            self.assertEqual(writer.call_args.args[1], "submission.csv")
            return writer.call_args.args[0].copy()

    def test_missing_prediction_artifact_cannot_create_placeholder_submission(self):
        _, notebook = self.make_notebook()
        with self.assertRaisesRegex(FileNotFoundError, "prediction|inference"):
            self.execute(notebook)

    def test_shuffled_predictions_use_explicit_test_path_and_match_ids(self):
        _, notebook = self.make_notebook(pd.DataFrame({"id": [11, 10], "pred_target": [.8, .2],
                                                       "fold": [0, 0], "classes_target": "[0, 1]"}))
        result = self.execute(notebook)
        self.assertEqual(result.columns.tolist(), ["id", "target"])
        self.assertEqual(result.to_dict("list"), {"id": [10, 11], "target": [.2, .8]})

    def test_malformed_prediction_artifacts_fail_before_writing(self):
        for predictions in (
            pd.DataFrame({"id": [10, 10], "target": [.2, .8]}),
            pd.DataFrame({"id": [10, None], "target": [.2, .8]}),
            pd.DataFrame({"id": [10], "target": [.2]}),
            pd.DataFrame({"id": [10, 12], "target": [.2, .8]}),
            pd.DataFrame({"id": [10, 11, 12], "target": [.2, .8, .5]}),
            pd.DataFrame({"target": [.2, .8]}),
            pd.DataFrame({"id": [10, 11], "wrong_target": [.2, .8]}),
            pd.DataFrame({"id": [10, 11], "target": [.2, float("inf")]}),
        ):
            with self.subTest(predictions=predictions.to_dict()):
                _, notebook = self.make_notebook(predictions)
                with self.assertRaises((ValueError, KeyError)):
                    self.execute(notebook)

    def test_invalid_test_ids_cannot_be_fabricated_or_duplicated(self):
        for test in (pd.DataFrame({"feature": [1, 2]}), pd.DataFrame({"id": [10, 10]})):
            with self.subTest(test=test.to_dict()):
                _, notebook = self.make_notebook(pd.DataFrame({"id": [10, 11], "target": [.2, .8]}), test)
                with self.assertRaises(ValueError):
                    self.execute(notebook)

    def test_ambiguous_attached_predictions_require_explicit_selection(self):
        root, notebook = self.make_notebook()
        files = []
        for name in ("one", "two"):
            folder = root / name
            folder.mkdir()
            path = folder / "test_predictions.csv"
            pd.DataFrame({"id": [10, 11], "target": [.2, .8]}).to_csv(path, index=False)
            files.append(str(path))
        with self.assertRaisesRegex(ValueError, "Multiple|multiple|Ambiguous"):
            self.execute(notebook, lambda pattern, **kw: files if "/exp001/" in pattern else [])

    def test_failed_offline_install_is_not_reported_as_success(self):
        _, notebook = self.make_notebook()
        with (patch("glob.glob", return_value=["/offline/dependency.whl"]),
              patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "pip")) as run,
              contextlib.redirect_stdout(io.StringIO())):
            with self.assertRaises(subprocess.CalledProcessError):
                exec("".join(notebook["cells"][1]["source"]), {})
            self.assertTrue(run.call_args.kwargs["check"])
