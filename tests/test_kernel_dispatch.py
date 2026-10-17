"""Tests for ``harness.kernel_dispatch`` (Gap B: kernel wrap + dispatch).

Covers build_kernel_metadata (schema + defaults), dispatch_kernel (staging +
push + warnings + missing notebook), and top-level CLI registration of
``dispatch-kernel`` and ``submission``. All push calls are mocked, so the suite
runs offline.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from harness.kernel_dispatch import build_kernel_metadata, dispatch_kernel


class _MockKernelClient:
    """In-memory client exposing ``kernels_push`` (the only method used)."""

    def __init__(self, warnings=None):
        self._warnings = warnings or {}
        self.push_calls = []

    def kernels_push(self, folder, timeout=None, acc=None):
        self.push_calls.append({"folder": folder, "timeout": timeout, "acc": acc})
        resp = SimpleNamespace(
            ref=777,
            url="https://www.kaggle.com/kernels/own/kernel",
            kernel_id=555,
            version_number=2,
            error=None,
            invalid_dataset_sources=None,
            invalid_competition_sources=None,
            invalid_kernel_sources=None,
            invalid_model_sources=None,
        )
        for key, value in self._warnings.items():
            setattr(resp, key, value)
        return resp


class BuildKernelMetadataTests(unittest.TestCase):
    def test_default_schema(self):
        meta = build_kernel_metadata("own/kernel", "My Kernel")
        self.assertEqual(meta["id"], "own/kernel")
        self.assertEqual(meta["title"], "My Kernel")
        self.assertEqual(meta["code_file"], "code.py")
        self.assertEqual(meta["language"], "python")
        self.assertEqual(meta["kernel_type"], "notebook")
        self.assertTrue(meta["is_private"])
        self.assertFalse(meta["enable_internet"])
        self.assertFalse(meta["enable_gpu"])
        self.assertFalse(meta["enable_tpu"])
        self.assertEqual(meta["keywords"], [])
        self.assertEqual(meta["dataset_sources"], [])
        self.assertEqual(meta["competition_sources"], [])
        self.assertEqual(meta["kernel_sources"], [])
        self.assertEqual(meta["model_sources"], [])
        self.assertNotIn("machine_shape", meta)

    def test_machine_shape_included_when_set(self):
        meta = build_kernel_metadata(
            "own/k", "T", machine_shape="KaggleGPU", enable_internet=True
        )
        self.assertEqual(meta["machine_shape"], "KaggleGPU")
        self.assertTrue(meta["enable_internet"])

    def test_sources_and_keywords(self):
        meta = build_kernel_metadata(
            "own/k",
            "T",
            dataset_sources=["own/data"],
            competition_sources=["comp"],
            kernel_sources=["own/base"],
            model_sources=["own/model"],
            keywords=["ml", "ensemble"],
        )
        self.assertEqual(meta["dataset_sources"], ["own/data"])
        self.assertEqual(meta["competition_sources"], ["comp"])
        self.assertEqual(meta["kernel_sources"], ["own/base"])
        self.assertEqual(meta["model_sources"], ["own/model"])
        self.assertEqual(meta["keywords"], ["ml", "ensemble"])


class DispatchKernelTests(unittest.TestCase):
    def _notebook(self, folder: Path):
        nb = folder / "train.ipynb"
        nb.write_text(json.dumps({"cells": []}))
        return nb

    def test_dispatch_stages_and_pushes(self):
        with tempfile.TemporaryDirectory() as tmp:
            nb = self._notebook(Path(tmp))
            client = _MockKernelClient()
            result = dispatch_kernel(
                str(nb), "own/kernel", "My Kernel", machine_shape="KaggleGPU", client=client
            )
            self.assertTrue(result["ok"])
            self.assertEqual(result["ref"], 777)
            self.assertEqual(result["kernel_id"], 555)
            self.assertEqual(result["version_number"], 2)
            self.assertEqual(len(client.push_calls), 1)
            self.assertEqual(client.push_calls[0]["acc"], "KaggleGPU")
            self.assertEqual(result["metadata"]["id"], "own/kernel")
            self.assertEqual(result["metadata"]["machine_shape"], "KaggleGPU")

    def test_dispatch_staging_contents(self):
        with tempfile.TemporaryDirectory() as tmp:
            nb = self._notebook(Path(tmp))
            client = _MockKernelClient()
            result = dispatch_kernel(str(nb), "own/k", "T", keep_staging=True, client=client)
            staging = Path(result["staging_dir"])
            self.assertTrue(staging.exists())
            self.assertTrue((staging / "train.ipynb").exists())
            meta_file = staging / "kernel-metadata.json"
            self.assertTrue(meta_file.exists())
            meta = json.loads(meta_file.read_text())
            self.assertEqual(meta["id"], "own/k")
            self.assertEqual(meta["code_file"], "train.ipynb")

    def test_dispatch_surfaces_warnings(self):
        with tempfile.TemporaryDirectory() as tmp:
            nb = self._notebook(Path(tmp))
            client = _MockKernelClient(warnings={"invalid_dataset_sources": ["bad/source"]})
            result = dispatch_kernel(str(nb), "own/k", "T", client=client)
            self.assertTrue(result["ok"])
            self.assertEqual(result["invalid_dataset_sources"], ["bad/source"])

    def test_dispatch_error_field_surfaced(self):
        with tempfile.TemporaryDirectory() as tmp:
            nb = self._notebook(Path(tmp))
            client = _MockKernelClient(warnings={"error": "invalid slug"})
            result = dispatch_kernel(str(nb), "own/k", "T", client=client)
            self.assertEqual(result.get("error"), "invalid slug")

    def test_missing_notebook_raises(self):
        with self.assertRaises(FileNotFoundError):
            dispatch_kernel("/nonexistent/train.ipynb", "own/k", "T", client=_MockKernelClient())

    def test_default_code_file_is_notebook_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            nb = Path(tmp) / "my_model.ipynb"
            nb.write_text(json.dumps({"cells": []}))
            client = _MockKernelClient()
            result = dispatch_kernel(str(nb), "own/k", "T", client=client)
            self.assertEqual(result["metadata"]["code_file"], "my_model.ipynb")

    def test_staging_cleaned_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            nb = self._notebook(Path(tmp))
            client = _MockKernelClient()
            result = dispatch_kernel(str(nb), "own/k", "T", client=client)
            self.assertNotIn("staging_dir", result)


class CliRegistrationTests(unittest.TestCase):
    def test_new_subcommands_registered(self):
        from harness.cli import build_parser

        parser = build_parser()
        choices = list(parser._subparsers._group_actions[0].choices.keys())
        self.assertIn("dispatch-kernel", choices)
        self.assertIn("submission", choices)


if __name__ == "__main__":
    unittest.main()
