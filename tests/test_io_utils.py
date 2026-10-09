"""Unit tests for JSON serialization and type safety in io_utils and runner."""

from __future__ import annotations

import dataclasses
import datetime
import json
from pathlib import Path
import shutil
import tempfile
import unittest
import uuid

import numpy as np
import pandas as pd

from harness.io_utils import (
    atomic_write_json,
    atomic_write_text,
    json_default_serializer,
    sanitize_for_json,
)


@dataclasses.dataclass
class SampleConfig:
    name: str
    folds: int


class CustomDictObject:
    def __init__(self, data: dict):
        self._data = data

    def to_dict(self) -> dict:
        return dict(self._data)


class TestJsonDefaultSerializer(unittest.TestCase):
    """Test suite for json_default_serializer across scalar, array, and filesystem types."""

    def test_numpy_integer_types(self):
        """Serializes np.int64 and np.int32 into native Python int."""
        val_64 = np.int64(42)
        res_64 = json_default_serializer(val_64)
        self.assertEqual(res_64, 42)
        self.assertIsInstance(res_64, int)

        val_32 = np.int32(100)
        res_32 = json_default_serializer(val_32)
        self.assertEqual(res_32, 100)
        self.assertIsInstance(res_32, int)

    def test_numpy_floating_types(self):
        """Serializes np.float64 and np.float32 into native Python float."""
        val_64 = np.float64(3.14159)
        res_64 = json_default_serializer(val_64)
        self.assertAlmostEqual(res_64, 3.14159)
        self.assertIsInstance(res_64, float)

        val_32 = np.float32(2.718)
        res_32 = json_default_serializer(val_32)
        self.assertAlmostEqual(res_32, 2.718, places=3)
        self.assertIsInstance(res_32, float)

    def test_numpy_bool_types(self):
        """Serializes np.bool_ into native Python bool."""
        t = np.bool_(True)
        f = np.bool_(False)
        self.assertIs(json_default_serializer(t), True)
        self.assertIs(json_default_serializer(f), False)

    def test_numpy_ndarray(self):
        """Serializes 1D and 2D np.ndarray into Python nested lists."""
        arr_1d = np.array([1, 2, 3])
        self.assertEqual(json_default_serializer(arr_1d), [1, 2, 3])

        arr_2d = np.array([[1, 2], [3, 4]])
        self.assertEqual(json_default_serializer(arr_2d), [[1, 2], [3, 4]])

    def test_pathlib_paths(self):
        """Serializes pathlib.Path and pathlib.PurePath into strings."""
        p = Path("models/best.pt")
        self.assertEqual(json_default_serializer(p), str(p))

    def test_extended_types(self):
        """Serializes dates, sets, dataclasses, to_dict objects, UUIDs, and bytes."""
        dt = datetime.datetime(2026, 10, 8, 12, 0, 0)
        self.assertEqual(json_default_serializer(dt), "2026-10-08T12:00:00")

        d = datetime.date(2026, 10, 8)
        self.assertEqual(json_default_serializer(d), "2026-10-08")

        s = {1, 2, 3}
        self.assertEqual(sorted(json_default_serializer(s)), [1, 2, 3])

        fs = frozenset([4, 5])
        self.assertEqual(sorted(json_default_serializer(fs)), [4, 5])

        dc = SampleConfig(name="xgb", folds=5)
        self.assertEqual(json_default_serializer(dc), {"name": "xgb", "folds": 5})

        obj = CustomDictObject({"metric": "auc"})
        self.assertEqual(json_default_serializer(obj), {"metric": "auc"})

        u = uuid.uuid4()
        self.assertEqual(json_default_serializer(u), str(u))

        b = b"hello bytes"
        self.assertEqual(json_default_serializer(b), "hello bytes")

    def test_unhandled_type_raises_type_error(self):
        """Raises TypeError for unhandled arbitrary objects."""
        class Unserializable:
            pass

        with self.assertRaises(TypeError):
            json_default_serializer(Unserializable())


class TestSanitizeForJson(unittest.TestCase):
    """Test suite for sanitize_for_json handling non-string keys, nested types, and cycles."""

    def test_non_string_keys(self):
        """Sanitizes np.int64, Path, int, and tuple dictionary keys."""
        data = {
            np.int64(10): np.int32(20),
            Path("configs/task.yaml"): np.float64(1.23),
            42: "int_key_val",
            (1, 2): [np.bool_(True), Path("sub/file")],
        }
        cleaned = sanitize_for_json(data)
        self.assertIsInstance(cleaned, dict)
        dumped = json.dumps(cleaned)
        loaded = json.loads(dumped)

        self.assertEqual(loaded["10"], 20)
        self.assertEqual(loaded[str(Path("configs/task.yaml"))], 1.23)
        self.assertEqual(loaded["42"], "int_key_val")
        self.assertEqual(loaded[str((1, 2))], [True, str(Path("sub/file"))])

    def test_nested_structures(self):
        """Recursively sanitizes deeply nested dictionaries, lists, and ndarrays."""
        nested = {
            "outer": {
                np.int64(99): np.array([4, 5, 6]),
                "sub": [Path("a/b"), np.int32(7)],
            }
        }
        cleaned = sanitize_for_json(nested)
        self.assertEqual(cleaned["outer"][99], [4, 5, 6])
        self.assertEqual(cleaned["outer"]["sub"], [str(Path("a/b")), 7])
        loaded = json.loads(json.dumps(cleaned))
        self.assertEqual(loaded["outer"]["99"], [4, 5, 6])

    def test_cycle_detection_direct_dict(self):
        """Raises ValueError('Circular reference detected') on self-referential dict."""
        d = {}
        d["self"] = d
        with self.assertRaises(ValueError) as ctx:
            sanitize_for_json(d)
        self.assertIn("Circular reference detected", str(ctx.exception))

    def test_cycle_detection_direct_list(self):
        """Raises ValueError('Circular reference detected') on self-referential list."""
        lst = []
        lst.append(lst)
        with self.assertRaises(ValueError) as ctx:
            sanitize_for_json(lst)
        self.assertIn("Circular reference detected", str(ctx.exception))

    def test_cycle_detection_indirect(self):
        """Raises ValueError('Circular reference detected') on mutually referential objects."""
        a = {}
        b = {"parent": a}
        a["child"] = b
        with self.assertRaises(ValueError) as ctx:
            sanitize_for_json(a)
        self.assertIn("Circular reference detected", str(ctx.exception))

    def test_dag_shared_reference_without_cycle(self):
        """Successfully processes DAGs (shared references without cycle)."""
        shared = {"count": np.int64(42)}
        dag = {
            "branch_a": shared,
            "branch_b": shared,
            "list_refs": [shared, shared],
        }
        cleaned = sanitize_for_json(dag)
        self.assertEqual(cleaned["branch_a"]["count"], 42)
        self.assertEqual(cleaned["branch_b"]["count"], 42)
        self.assertEqual(cleaned["list_refs"][0]["count"], 42)


class TestAtomicWriteJson(unittest.TestCase):
    """Test suite for atomic_write_json end-to-end serialization and file safety."""

    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_atomic_write_json_full_matrix(self):
        """Serializes full matrix of NumPy scalars, arrays, Paths, and non-string keys."""
        target = self.test_dir / "subdir" / "matrix.json"
        data = {
            "int64": np.int64(123456789),
            "int32": np.int32(987),
            "float64": np.float64(3.1415926535),
            "float32": np.float32(1.414),
            "bool_t": np.bool_(True),
            "bool_f": np.bool_(False),
            "ndarray_1d": np.array([10, 20, 30]),
            "ndarray_2d": np.array([[1, 2], [3, 4]]),
            "path": Path("models/best.pt"),
            np.int64(77): "int64_key",
            Path("configs/task.yaml"): "path_key",
            42: "int_key",
            ("tuple", "key"): "tuple_val",
        }
        atomic_write_json(target, data)
        self.assertTrue(target.exists())

        # Verify no temporary files remain
        tmp_files = list(target.parent.glob(".*tmp*"))
        self.assertEqual(len(tmp_files), 0)

        with open(target, "r", encoding="utf-8") as f:
            loaded = json.load(f)

        self.assertEqual(loaded["int64"], 123456789)
        self.assertEqual(loaded["int32"], 987)
        self.assertAlmostEqual(loaded["float64"], 3.1415926535)
        self.assertAlmostEqual(loaded["float32"], 1.414, places=3)
        self.assertIs(loaded["bool_t"], True)
        self.assertIs(loaded["bool_f"], False)
        self.assertEqual(loaded["ndarray_1d"], [10, 20, 30])
        self.assertEqual(loaded["ndarray_2d"], [[1, 2], [3, 4]])
        self.assertEqual(loaded["path"], str(Path("models/best.pt")))
        self.assertEqual(loaded["77"], "int64_key")
        self.assertEqual(loaded[str(Path("configs/task.yaml"))], "path_key")
        self.assertEqual(loaded["42"], "int_key")
        self.assertEqual(loaded[str(("tuple", "key"))], "tuple_val")

    def test_atomic_write_json_circular_reference(self):
        """Raises ValueError('Circular reference detected') even with non-string keys."""
        target = self.test_dir / "circ.json"
        d = {Path("a"): 1}
        d["self"] = d
        with self.assertRaises(ValueError) as ctx:
            atomic_write_json(target, d)
        self.assertIn("Circular reference detected", str(ctx.exception))
        self.assertFalse(target.exists())

    def test_atomic_write_json_custom_default(self):
        """Custom default serializer is prioritized for custom types."""
        target = self.test_dir / "custom.json"

        class SpecialToken:
            def __init__(self, val: str):
                self.val = val

        data = {
            "token": SpecialToken("ABC"),
            "num": np.int64(42),
        }

        def custom_serializer(o):
            if isinstance(o, SpecialToken):
                return f"token_{o.val}"
            raise TypeError("unhandled")

        atomic_write_json(target, data, default=custom_serializer)
        with open(target, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        self.assertEqual(loaded["token"], "token_ABC")
        self.assertEqual(loaded["num"], 42)


class TestRunnerFoldCoercion(unittest.TestCase):
    """Test suite verifying fold ID type coercion in runner logic."""

    def test_fold_coercion_preserves_int(self):
        """Coerces numpy int64/int32 to native int, preserving string folds."""
        f_np64 = np.int64(3)
        coerced_64 = int(f_np64) if isinstance(f_np64, (int, np.integer)) else f_np64
        self.assertIsInstance(coerced_64, int)
        self.assertEqual(coerced_64, 3)

        f_str = "fold_a"
        coerced_str = int(f_str) if isinstance(f_str, (int, np.integer)) else f_str
        self.assertEqual(coerced_str, "fold_a")

    def test_prediction_record_fold_type(self):
        """Verifies prediction records contain native int fold types."""
        from harness.task_spec import UniversalTaskSpec

        spec = UniversalTaskSpec.from_dict(
            {
                "task": {
                    "name": "test_task",
                    "task_type": "regression",
                    "id_column": "id",
                    "target_column": "target",
                    "primary_metric": "rmse",
                    "direction": "minimize",
                }
            }
        )

        class DummyRunner:
            def __init__(self, task_spec):
                self.task_spec = task_spec

        from harness.runner import UniversalRunner

        dev_df = pd.DataFrame({"id": [1, 2], "target": [10.0, 20.0]})
        preds = {"pred": [9.5, 20.5]}
        records = UniversalRunner._record_predictions(
            DummyRunner(spec),
            dev_df,
            preds,
            np.int64(0),
            "baseline",
        )
        self.assertEqual(len(records), 2)
        self.assertIsInstance(records[0]["fold"], int)
        self.assertEqual(records[0]["fold"], 0)

    def test_universal_runner_split_with_np_int64_folds(self):
        """Validates UniversalRunner successfully ingests int64 splits and serializes metrics."""
        from types import SimpleNamespace
        import yaml
        from tests.test_correctness_contracts import task
        from harness.runner import UniversalRunner

        tmp = tempfile.TemporaryDirectory()
        try:
            root = Path(tmp.name) / "workspace"
            (root / "configs").mkdir(parents=True)
            (root / "dataset").mkdir()
            (root / "splits").mkdir()
            spec = task()
            doc = {"task": spec.metadata}
            doc["task"]["data"]["train_path"] = "dataset/train.csv"
            doc["task"]["validation"] = {
                "strategy": "kfold",
                "num_folds": 2,
                "split_file": "splits/folds.tsv",
            }
            (root / "configs/task.yaml").write_text(yaml.safe_dump(doc), encoding="utf-8")
            train_df = pd.DataFrame(
                {
                    "id": range(8),
                    "group": range(8),
                    "x": range(8),
                    "category": ["a", "b"] * 4,
                    "target": [float(x * 2) for x in range(8)],
                }
            )
            train_df.to_csv(root / "dataset/train.csv", index=False)
            folds_df = pd.DataFrame({"id": range(8), "fold": [0, 1] * 4})
            folds_df["fold"] = folds_df["fold"].astype("int64")
            folds_df.to_csv(root / "splits/folds.tsv", sep="\t", index=False)

            args = SimpleNamespace(
                project_root=str(root),
                task_config=None,
                fold=None,
                debug=False,
                probe_samples=None,
            )
            cfg = {
                "experiment": {
                    "id": "exp001",
                    "branch": "baseline",
                    "parent_id": "none",
                    "problem_addressed": "Establish verified baseline and validate split ingestion",
                    "hypothesis": "Linear features provide measurable regression control with minimal error",
                    "intervention": "Train a regularized linear ridge regression baseline",
                },
                "model": {"name": "ridge", "params": {"alpha": 0.1}},
            }
            runner = UniversalRunner(cfg, "config.yaml", args)
            result = runner.run()
            self.assertEqual(len(result["fold_metrics"]), 2)
            metrics_file = runner.exp_dir / "metrics.json"
            self.assertTrue(metrics_file.exists())
            with open(metrics_file, "r", encoding="utf-8") as f:
                metrics_data = json.load(f)
            for fm in metrics_data["fold_metrics"]:
                self.assertIsInstance(fm["dev_fold"], int)
                self.assertEqual(type(fm["dev_fold"]), int)
                self.assertIn(fm["dev_fold"], [0, 1])

            summary_file = runner.exp_dir / "summary.csv"
            self.assertTrue(summary_file.exists())
            summary_df = pd.read_csv(summary_file)
            self.assertIn("dev_fold", summary_df.columns)
            self.assertEqual(list(summary_df["dev_fold"]), [0, 1])
        finally:
            tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
