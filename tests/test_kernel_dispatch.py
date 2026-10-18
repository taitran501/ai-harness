"""Tests for ``harness.kernel_dispatch`` (Gap B: kernel wrap + dispatch).

Covers build_kernel_metadata (schema + defaults), dispatch_kernel (staging +
push + warnings + missing notebook), the main() CLI entry point (arg
parsing, JSON-on-stdout contract, exit codes), and top-level CLI
registration of ``dispatch-kernel`` and ``submission``. All push calls are
mocked, so the suite runs offline.
"""
from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

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


class KernelDispatchMainCliTests(unittest.TestCase):
    """``main()`` CLI entry point: arg parsing, JSON-on-stdout contract, exit
    codes. ``dispatch_kernel`` is mocked — no staging, client, or network.
    The JSON printed to stdout is what the agent reads back, so it is pinned.
    """

    def _run_main(self, argv, dispatch_result=None, dispatch_exc=None):
        from harness.kernel_dispatch import main

        with mock.patch("harness.kernel_dispatch.dispatch_kernel") as dk:
            if dispatch_exc is not None:
                dk.side_effect = dispatch_exc
            else:
                dk.return_value = dispatch_result or {
                    "ok": True, "ref": 9, "url": "https://k/own/k",
                }
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = main(argv)
        return rc, json.loads(buf.getvalue()), dk

    def test_happy_path_exit_zero_and_json(self):
        rc, out, dk = self._run_main(["nb.ipynb", "-i", "owner/k", "-t", "Title"])
        self.assertEqual(rc, 0)
        self.assertTrue(out["ok"])
        self.assertEqual(out["ref"], 9)
        dk.assert_called_once()

    def test_failure_exit_one_and_error_json(self):
        rc, out, _ = self._run_main(
            ["nb.ipynb", "-i", "owner/k", "-t", "T"],
            dispatch_exc=RuntimeError("boom"),
        )
        self.assertEqual(rc, 1)
        self.assertFalse(out["ok"])
        self.assertEqual(out["error"], "boom")

    def test_all_flags_forwarded_to_dispatch(self):
        rc, _, dk = self._run_main([
            "nb.ipynb", "-i", "o/k", "-t", "T",
            "--code-file", "x.py",
            "--machine-shape", "T4",
            "--dataset", "d1", "--dataset", "d2",
            "--competition", "c1",
            "--kernel-source", "ks1",
            "--model", "m1",
            "--keyword", "kw1",
            "--enable-internet",
            "--timeout", "1200",
            "--keep-staging",
        ])
        self.assertEqual(rc, 0)
        args, kwargs = dk.call_args
        self.assertEqual(args[:3], ("nb.ipynb", "o/k", "T"))
        self.assertEqual(kwargs["code_file"], "x.py")
        self.assertEqual(kwargs["machine_shape"], "T4")
        self.assertEqual(kwargs["dataset_sources"], ["d1", "d2"])
        self.assertEqual(kwargs["competition_sources"], ["c1"])
        self.assertEqual(kwargs["kernel_sources"], ["ks1"])
        self.assertEqual(kwargs["model_sources"], ["m1"])
        self.assertEqual(kwargs["keywords"], ["kw1"])
        self.assertTrue(kwargs["enable_internet"])
        self.assertEqual(kwargs["timeout"], 1200)
        self.assertTrue(kwargs["keep_staging"])

    def test_missing_required_args_exits_two(self):
        from harness.kernel_dispatch import main

        with mock.patch("harness.kernel_dispatch.dispatch_kernel") as dk, \
             contextlib.redirect_stdout(io.StringIO()), \
             self.assertRaises(SystemExit) as cm:
            main(["nb.ipynb", "-t", "T"])  # --kernel-id missing
        self.assertEqual(cm.exception.code, 2)
        dk.assert_not_called()


class CliRegistrationTests(unittest.TestCase):
    def test_new_subcommands_registered(self):
        from harness.cli import build_parser

        parser = build_parser()
        choices = list(parser._subparsers._group_actions[0].choices.keys())
        self.assertIn("dispatch-kernel", choices)
        self.assertIn("submission", choices)


if __name__ == "__main__":
    unittest.main()
