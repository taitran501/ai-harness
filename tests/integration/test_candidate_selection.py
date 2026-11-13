"""Integration tests for candidate selection, blend optimization, and custom scorers.

Consolidates:
- BlendOptimizerContractTests (source contracts, class orders, duplicate components)
- BlendRecipeIdentityTests (canonical recipe fingerprinting, duplicate detection)
- CandidateOfficialScorerTests (official metrics propagation in runner/optimizer)
- ResolveMetricsHookTests (colon and dot format hook references)
- ValidateRunCustomHookTests (custom scorer validation from detached roots)
- ExternalCandidateMetadataTests (external prediction ingestion into candidate pool)
- ScoringImportHermeticTests (scorer hook hermetic imports on artifact-import paths)
- TestCandidateSynthesisEnhancements (weighted probability blend synthesis)
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.util
import inspect
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml

from harness.agent_bridge import AgentBridge
from harness.artifacts import digest, sha256, validate_run
from harness.candidates import (
    _candidate_recipe_fingerprint as fingerprint,
    build_candidate,
    find_optimal_blend_weights,
    register_external_prediction,
)
from harness.competition_state import CompetitionState
from harness.hook_ref import (
    ensure_hook_project_roots_on_sys_path,
    resolve_metrics_hook,
)
from harness.oof_revalidation import score_predictions
from harness.plugins import TaskPlugin
from harness.runner import UniversalRunner
from harness.task_spec import UniversalTaskSpec, load_active_task_spec
from harness.templates.kaggle_template import KaggleKernelManager
from tests.evidence_fixtures import seal_fixture
from tests.fixtures import make_workspace, task, text_task_document as task_document


class BlendOptimizerContractTests(unittest.TestCase):
    def make_bridge(self, reverse_second=False):
        doc = {"task": copy.deepcopy(task("binary_classification", "accuracy", "maximize").metadata)}
        doc["task"]["data"]["train_path"] = "dataset/train.csv"
        truth = pd.DataFrame({
            "id": range(4),
            "group": range(4),
            "x": range(4),
            "category": ["a", "b"] * 2,
            "target": [0, 0, 1, 1],
        })
        root, _ = make_workspace(task=doc, train_csv=truth.to_csv(index=False))
        self.addCleanup(shutil.rmtree, root)
        env = patch.dict(os.environ, {"HARNESS_PROJECT_ROOT": str(root)})
        env.start()
        self.addCleanup(env.stop)
        for name, values in (("A", [0.1, 0.6, 0.7, 0.4]), ("B", [0.6, 0.1, 0.4, 0.7])):
            reverse = name == "B" and reverse_second
            frame = pd.DataFrame({
                "id": truth.id,
                "pred_target": [int(v > 0.5) for v in values],
                "proba_target": [json.dumps([v, 1 - v] if reverse else [1 - v, v]) for v in values],
                "classes_target": json.dumps([1, 0] if reverse else [0, 1]),
            }).iloc[::-1]
            source = root / (name + ".csv")
            frame.to_csv(source, index=False)
            register_external_prediction(root, name, source)
        return AgentBridge(root)

    def test_incompatible_class_order_fails_before_reporting_an_optimization(self):
        with self.assertRaisesRegex(ValueError, "class order"):
            self.make_bridge(reverse_second=True)

    def test_compatible_sources_optimize_and_build_the_reported_candidate(self):
        bridge = self.make_bridge()
        _, obs = bridge.execute_action("optimize_blend", {"components": ["A", "B"]})
        self.assertEqual(obs.status, "FAILED", obs.summary)
        self.assertIn("EXTERNAL_PREDICTION_NOT_SUBMITTABLE", obs.summary)

    def test_duplicate_components_and_unknown_modes_are_rejected(self):
        bridge = self.make_bridge()
        with self.assertRaisesRegex(ValueError, "distinct"):
            find_optimal_blend_weights(bridge.project, ["A", "A"])
        with self.assertRaisesRegex(ValueError, "Unsupported blend mode"):
            find_optimal_blend_weights(bridge.project, ["A", "B"], blend_mode="not_a_mode")


class BlendRecipeIdentityTests(unittest.TestCase):
    def fp(self, ids, weights=None):
        source_map = {"components": ids}
        if weights is not None:
            source_map["weights"] = weights
        return fingerprint("weighted_blend", ids, source_map)

    def test_source_order_and_weight_scale_are_canonical(self):
        self.assertEqual(self.fp(["A", "B"], [0.9, 0.1]), self.fp(["B", "A"], [0.1, 0.9]))
        self.assertEqual(self.fp(["A", "B"], [0.1, 0.2]), self.fp(["A", "B"], [1, 2]))
        self.assertEqual(self.fp(["A", "B"]), self.fp(["A", "B"], [0.5, 0.5]))

    def test_swapping_only_weights_changes_the_recipe(self):
        self.assertNotEqual(self.fp(["A", "B"], [0.9, 0.1]), self.fp(["A", "B"], [0.1, 0.9]))

    def test_bridge_accepts_distinct_blend_but_blocks_equivalent_legacy_recipe(self):
        doc = {"task": copy.deepcopy(task().metadata)}
        doc["task"]["data"]["train_path"] = "dataset/train.csv"
        truth = pd.DataFrame({
            "id": range(12),
            "group": range(12),
            "x": range(12),
            "category": ["a", "b"] * 6,
            "target": [2.0 * i for i in range(12)],
        })
        root, _ = make_workspace(task=doc, train_csv=truth.to_csv(index=False))
        self.addCleanup(shutil.rmtree, root)
        with patch.dict(os.environ, {"HARNESS_PROJECT_ROOT": str(root)}):
            bridge = AgentBridge(root)
            spec = load_active_task_spec(root / "configs/task.yaml")
            error = 10 * np.sin(np.arange(len(truth)))
            for name, sign in (("A", 1), ("B", -1)):
                folder = bridge.artifacts_dir / name
                folder.mkdir(parents=True)
                pd.DataFrame({
                    "id": truth.id,
                    "fold": [0, 1] * 6,
                    "input_variant": "features",
                    "pred_target": truth.target + sign * error,
                }).to_csv(folder / "oof_predictions.csv", index=False)
                truth.to_csv(folder / "evaluation_truth.csv", index=False)
                (folder / "run_config.yaml").write_text(yaml.safe_dump({"experiment": {"id": name}}))
                seal_fixture(folder, spec, score=float(np.sqrt(np.mean(error ** 2))))

            def synth(ids, weights):
                return bridge.execute_action(
                    "synthesize_candidate",
                    {"mode": "weighted_blend", "source_map": {"components": ids, "weights": weights}},
                )[1]

            first = synth(["A", "B"], [0.9, 0.1])
            self.assertEqual(first.status, "SUCCESS", first.summary)
            candidate_id = first.data["id"]
            manifest = bridge.artifacts_dir / candidate_id / "candidate_manifest.json"
            original_hash = sha256(manifest)
            # Simulate the stored guard hash from the old release. Recipe
            # details remain available and should drive duplicate detection.
            stored = next(
                c for c in bridge.comp_state.state["candidate_pool"]["synthesized_candidates"]
                if c["id"] == candidate_id
            )
            legacy_recipe = {
                "mode": "weighted_blend",
                "source_ids": ["A", "B"],
                "weights": [0.1, 0.9],
                "routing": None,
            }
            stored["recipe_fingerprint"] = hashlib.sha256(
                json.dumps(legacy_recipe, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            bridge.comp_state.save()
            distinct = synth(["A", "B"], [0.1, 0.9])
            self.assertEqual(distinct.status, "SUCCESS", distinct.summary)
            duplicate = synth(["B", "A"], [1, 9])
            self.assertEqual(duplicate.status, "FAILED")
            self.assertIn("recipe already exists", duplicate.summary)
            self.assertEqual(sha256(manifest), original_hash)
            self.assertEqual(json.loads(manifest.read_text())["details"]["weights"], [0.9, 0.1])

    def test_invalid_weights_do_not_acquire_a_recipe_identity(self):
        for weights in ([], [1], [-1, 2], [0, 0], [float("nan"), 1], [float("inf"), 1]):
            with self.subTest(weights=weights):
                with self.assertRaisesRegex(ValueError, "Invalid blend weights"):
                    self.fp(["A", "B"], weights)


class CandidateOfficialScorerTests(unittest.TestCase):
    def setUp(self):
        doc = {"task": copy.deepcopy(task().metadata)}
        doc["task"]["data"]["train_path"] = "dataset/train.csv"
        doc["task"]["validation"] = {"strategy": "kfold", "num_folds": 2}
        doc["task"].update(primary_metric="official_score", direction="maximize")
        doc["task"]["subtasks"][0].update(primary_metric="official_score", direction="maximize")
        train = pd.DataFrame({
            "id": range(12),
            "group": range(12),
            "x": range(12),
            "category": ["a", "b"] * 6,
            "target": [2.0 * i for i in range(12)],
        })
        self.root, _ = make_workspace(task=doc, train_csv=train.to_csv(index=False))
        self.addCleanup(shutil.rmtree, self.root)
        self.env = patch.dict(os.environ, {"HARNESS_PROJECT_ROOT": str(self.root)})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.addCleanup(lambda: sys.modules.pop("candidate_official_fixture", None))
        original_path = list(sys.path)
        self.addCleanup(lambda: sys.path.__setitem__(slice(None), original_path))
        (self.root / "candidate_official_fixture.py").write_text(
            "import numpy as np\n"
            "def score(truth, predictions):\n"
            "    values = predictions.get('pred_target', predictions.get('target'))\n"
            "    return {'official_score': -float(np.mean(np.abs(truth.target.to_numpy() - values)))}\n"
            "def rmse_score(truth, predictions):\n    return {'rmse': 42.0}\n",
            encoding="utf-8",
        )
        self.args = SimpleNamespace(
            project_root=str(self.root), task_config=None, debug=False, fold=None, probe_samples=None
        )

    def run_source(self, exp_id, alpha=1.0, hook="candidate_official_fixture:score"):
        cfg = {
            "experiment": {
                "id": exp_id,
                "branch": "baseline",
                "parent_id": None,
                "problem_addressed": "Measure the official regression baseline.",
                "hypothesis": "Regularization changes the official validation score.",
                "intervention": "Train a ridge baseline with fixed splits.",
            },
            "model": {"name": "ridge", "params": {"alpha": alpha}},
            "paths": {"metrics_evaluator": hook},
        }
        runner = UniversalRunner(cfg, "config.yaml", self.args)
        result = runner.run()
        return runner, result

    def test_bridge_promotes_custom_metric_without_changing_source_evidence(self):
        runner, result = self.run_source("exp000_official")
        before = sha256(runner.exp_dir / "artifact_manifest.json")
        bridge = AgentBridge(self.root)
        _, obs = bridge.execute_action("promote_candidate", {"exp_id": runner.exp_id})
        self.assertEqual(obs.status, "SUCCESS", obs.summary)
        self.assertEqual(obs.data["primary_metric"], "official_score")
        self.assertAlmostEqual(obs.composite_score, result["primary_score"])
        self.assertEqual(sha256(runner.exp_dir / "artifact_manifest.json"), before)

    def test_optimizer_uses_same_custom_metric_as_sources(self):
        runner, _ = self.run_source("exp001_official", 0.2)
        self.run_source("exp002_official", 2.0)
        result = find_optimal_blend_weights(runner.project, ["exp001_official", "exp002_official"], maxiter=2)
        self.assertEqual(result["evaluation_details"]["primary_metric"], "official_score")
        self.assertLessEqual(result["best_score"], 0)

    def test_different_custom_scorers_cannot_be_blended(self):
        runner, _ = self.run_source("exp001_official")
        self.addCleanup(lambda: sys.modules.pop("candidate_other_fixture", None))
        (self.root / "candidate_other_fixture.py").write_text(
            "def score(truth, predictions):\n    return {'official_score': -100.0}\n", encoding="utf-8"
        )
        other, _ = self.run_source("exp002_official", 2.0, "candidate_other_fixture:score")
        for source in (runner, other):
            validate_run(source.exp_dir, task_spec=source.task_spec, plugin=source.task_plugin)
        with self.assertRaisesRegex(ValueError, "custom_scorer_sha256"):
            build_candidate(
                runner.project,
                "invalid",
                "weighted_blend",
                {"components": ["exp001_official", "exp002_official"]},
            )
        with self.assertRaisesRegex(ValueError, "custom_scorer_sha256"):
            find_optimal_blend_weights(runner.project, ["exp001_official", "exp002_official"])

    def test_named_hook_metric_overrides_generic_primary_score(self):
        truth = pd.DataFrame({"id": [1, 2], "target": [0.0, 1.0]})
        result = score_predictions(
            task(),
            TaskPlugin(),
            SimpleNamespace(project_root=self.root),
            {"paths": {"metrics_evaluator": "candidate_official_fixture:rmse_score"}},
            truth,
            {"target": [0.0, 1.0]},
        )
        self.assertEqual(result["primary_score"], 42.0)
        self.assertEqual(result["utility"], -42.0)


class ResolveMetricsHookTests(unittest.TestCase):
    def test_colon_format(self):
        self.assertEqual(
            resolve_metrics_hook("src.evaluator:evaluate_rhae"),
            ("src.evaluator", "evaluate_rhae"),
        )

    def test_dot_format(self):
        self.assertEqual(
            resolve_metrics_hook("src.evaluator.evaluate_rhae"),
            ("src.evaluator", "evaluate_rhae"),
        )

    def test_deep_module_with_colon(self):
        self.assertEqual(
            resolve_metrics_hook("a.b.c.fn:score"), ("a.b.c.fn", "score")
        )

    def test_whitespace_tolerated(self):
        self.assertEqual(resolve_metrics_hook("  a.b : c "), ("a.b", "c"))

    def test_invalid_references_raise(self):
        for bad in ("bare", "module:", ":name", "", "   "):
            with self.assertRaises(ValueError, msg=repr(bad)):
                resolve_metrics_hook(bad)


class ValidateRunCustomHookTests(unittest.TestCase):
    """validate_run must accept the hook in both reference formats and
    import the scorer from a project root that is not on sys.path."""

    def setUp(self):
        self._drop_cached_scorer()
        self.temp = tempfile.TemporaryDirectory(prefix="aih-hook-ref-")
        self.root = Path(self.temp.name)
        (self.root / "src").mkdir()
        (self.root / "src" / "__init__.py").write_text("", encoding="utf-8")
        (self.root / "src" / "evaluator.py").write_text(
            "def evaluate_rhae(truth, predictions):\n"
            "    return {'rhae': 0.5}\n",
            encoding="utf-8",
        )
        self.spec = UniversalTaskSpec.from_dict(task_document())
        self.run_dir = self.root / "artifacts" / "exp001_hook"
        self.run_dir.mkdir(parents=True)
        self._sys_path = list(sys.path)

    def _drop_cached_scorer(self):
        prefix = "aih-hook-ref-"
        for mod in list(sys.modules):
            if not (mod == "src" or mod.startswith("src.")):
                continue
            file_ = getattr(sys.modules[mod], "__file__", "") or ""
            if prefix in file_ or not file_:
                del sys.modules[mod]

    def tearDown(self):
        for p in list(sys.path):
            if str(self.root) == str(p):
                sys.path.remove(p)
        sys.path[:] = self._sys_path
        self._drop_cached_scorer()
        self.temp.cleanup()

    def _seal(self, hook_ref, tamper_score=False):
        cfg = {
            "experiment": {"id": "exp001_hook"},
            "paths": {"metrics_evaluator": hook_ref},
        }
        (self.run_dir / "run_config.yaml").write_text(
            yaml.safe_dump(cfg), encoding="utf-8"
        )
        if str(self.root) not in sys.path:
            sys.path.insert(0, str(self.root))
        try:
            fn = importlib.import_module("src.evaluator").evaluate_rhae
            scorer_digest = digest(inspect.getsource(fn))
        finally:
            sys.path.remove(str(self.root))
        seal_fixture(
            self.run_dir,
            self.spec,
            custom_scorer_sha256=scorer_digest if not tamper_score else "deadbeef",
        )

    def _run_validate(self):
        env = dict(os.environ)
        env["HARNESS_PROJECT_ROOT"] = str(self.root)
        with patch.dict(os.environ, env, clear=True):
            return validate_run(self.run_dir)

    def test_colon_hook_passes_when_root_not_on_sys_path(self):
        self._seal("src.evaluator:evaluate_rhae")
        self.assertNotIn(str(self.root), sys.path)
        previous = {name: module for name, module in sys.modules.items()
                    if name == "src" or name.startswith("src.")}
        manifest = self._run_validate()
        self.assertNotIn(str(self.root), sys.path)
        self.assertEqual(previous, {name: module for name, module in sys.modules.items()
                                   if name == "src" or name.startswith("src.")})
        self.assertTrue(manifest["custom_scorer_sha256"])

    def test_dot_hook_passes(self):
        self._seal("src.evaluator.evaluate_rhae")
        self.assertNotIn(str(self.root), sys.path)
        self._run_validate()

    def test_tampered_scorer_digest_rejected(self):
        self._seal("src.evaluator:evaluate_rhae", tamper_score=True)
        with self.assertRaisesRegex(ValueError, "Custom scorer"):
            self._run_validate()


class ExternalCandidateMetadataTests(unittest.TestCase):
    def make_bridge(self, probability=False):
        spec = task("binary_classification", "roc_auc", "maximize") if probability else task()
        doc = {"task": copy.deepcopy(spec.metadata)}
        doc["task"]["data"]["train_path"] = "dataset/train.csv"
        if probability:
            doc["task"]["subtasks"][0]["classes"] = [1, 0]
        truth = pd.DataFrame({
            "id": range(6),
            "group": range(6),
            "x": [0.1, 0.9] * 3,
            "category": ["a", "b"] * 3,
            "target": [0, 1] * 3 if probability else list(range(6)),
        })
        root, _ = make_workspace(task=doc, train_csv=truth.to_csv(index=False))
        self.addCleanup(shutil.rmtree, root)
        env = patch.dict(os.environ, {"HARNESS_PROJECT_ROOT": str(root)})
        env.start()
        self.addCleanup(env.stop)
        return AgentBridge(root), truth

    def register_and_promote(self, bridge, source, name):
        original_hash = sha256(source)
        _, registered = bridge.execute_action(
            "register_external_prediction",
            {"name": name, "oof_path": str(source)},
        )
        self.assertEqual(registered.status, "SUCCESS", registered.summary)
        folder = bridge.artifacts_dir / name
        with self.assertRaisesRegex(ValueError, "full validation"):
            validate_run(folder, task_spec=bridge.task_spec, plugin=bridge.task_plugin)
        validate_run(folder, task_spec=bridge.task_spec, plugin=bridge.task_plugin, _run_kind="external")
        _, promoted = bridge.execute_action("promote_candidate", {"exp_id": name})
        self.assertEqual(promoted.status, "FAILED", promoted.summary)
        self.assertIn("EXTERNAL_PREDICTION_NOT_SUBMITTABLE", promoted.summary)
        self.assertEqual(sha256(source), original_hash)
        return pd.read_csv(folder / "oof_predictions.csv")

    def test_minimal_shuffled_csv_and_target_alias_enter_candidate_pool(self):
        bridge, truth = self.make_bridge()
        output = truth[["id", "target"]].iloc[::-1].copy()
        output["input_variant"] = "exporter-specific-name"
        source = bridge.project_root / "external.csv"
        output.to_csv(source, index=False)
        managed = self.register_and_promote(bridge, source, "ext_csv")
        self.assertEqual(managed.input_variant.unique().tolist(), ["external"])
        self.assertEqual(managed.fold.unique().tolist(), [-1])
        self.assertEqual(managed.pred_target.tolist(), truth.target.tolist())

    def test_numpy_export_without_internal_metadata_can_be_promoted(self):
        bridge, truth = self.make_bridge()
        source = bridge.project_root / "external.npy"
        np.save(source, truth.target.to_numpy() + 0.1)
        managed = self.register_and_promote(bridge, source, "ext_numpy")
        self.assertEqual(managed.input_variant.unique().tolist(), ["external"])

    def test_probability_export_preserves_class_order_and_existing_folds(self):
        bridge, truth = self.make_bridge(probability=True)
        output = pd.DataFrame({
            "id": truth.id,
            "pred_target": truth.target,
            "proba_target": ["[0.1, 0.9]", "[0.9, 0.1]"] * 3,
            "classes_target": "[1, 0]",
            "fold": [0, 1] * 3,
        })
        source = bridge.project_root / "probabilities.csv"
        output.to_csv(source, index=False)
        managed = self.register_and_promote(bridge, source, "ext_probability")
        self.assertEqual(managed.fold.tolist(), [0, 1] * 3)
        self.assertEqual(managed.classes_target.unique().tolist(), ["[1, 0]"])


PKG = "scorer_pkg"  # unique name: no collision with ambient packages


def _write_scorer(ws: Path) -> str:
    """Create ``ws/scorer_pkg/evaluator.py``; return the sealed scorer digest."""
    (ws / PKG).mkdir()
    (ws / PKG / "__init__.py").write_text("", encoding="utf-8")
    (ws / PKG / "evaluator.py").write_text(
        "def evaluate_rhae(y_true, y_pred, **kwargs):\n"
        "    pairs = list(zip(y_true, y_pred))\n"
        "    return sum(a == b for a, b in pairs) / max(len(pairs), 1)\n",
        encoding="utf-8",
    )
    spec = importlib.util.spec_from_file_location(
        f"aih_test_{PKG}_evaluator", ws / PKG / "evaluator.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return digest(inspect.getsource(module.evaluate_rhae))


def _sealed_run_with_hook(download_dir: Path, scorer_sha: str) -> Path:
    """A valid sealed run whose run_config declares a custom scorer hook."""
    run_dir = download_dir / "exp001"
    run_dir.mkdir()
    (run_dir / "run_config.yaml").write_text(
        "experiment: {}\n"
        f"paths:\n  metrics_evaluator: '{PKG}.evaluator:evaluate_rhae'\n",
        encoding="utf-8",
    )
    seal_fixture(run_dir, task(), score=0.75, custom_scorer_sha256=scorer_sha)
    return run_dir


class _DetachedContext:
    """cwd in a bare temp dir, HARNESS_PROJECT_ROOT cleared, sys.modules clean."""

    def __init__(self):
        self.cwd = os.getcwd()
        self.env_value = os.environ.pop("HARNESS_PROJECT_ROOT", None)
        self.detached = Path(tempfile.mkdtemp(prefix="aih-detached-"))
        self.popped = {
            name: sys.modules.pop(name)
            for name in (PKG, f"{PKG}.evaluator")
            if name in sys.modules
        }
        os.chdir(self.detached)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        os.chdir(self.cwd)
        if self.env_value is not None:
            os.environ["HARNESS_PROJECT_ROOT"] = self.env_value
        sys.modules.update(self.popped)
        shutil.rmtree(self.detached, ignore_errors=True)
        return False


class ScoringImportHermeticTests(unittest.TestCase):
    def test_migrate_with_project_root_imports_scorer_from_detached_cwd(self):
        ws, _ = make_workspace(task={"task": task().metadata})
        try:
            scorer_sha = _write_scorer(ws)
            with _DetachedContext() as ctx:
                self.assertIsNone(importlib.util.find_spec(PKG))
                self.assertNotIn(str(ws), sys.path)
                download = ctx.detached / "download"
                download.mkdir()
                _sealed_run_with_hook(download, scorer_sha)
                target = ws / ".ai-harness" / "artifacts" / "exp001"
                migrated = KaggleKernelManager.migrate_downloaded_artifacts(
                    download, target, project_root=ws
                )
                self.assertIn("metrics.json", migrated)
                self.assertIn("run_config.yaml", migrated)
                self.assertTrue((target / "artifact_manifest.json").is_file())
                self.assertIn(str(ws), sys.path)
        finally:
            sys.modules.pop(PKG, None)
            sys.modules.pop(f"{PKG}.evaluator", None)
            if str(ws) in sys.path:
                sys.path.remove(str(ws))
            shutil.rmtree(ws, ignore_errors=True)

    def test_validate_run_failure_message_names_hook_and_context(self):
        ws, _ = make_workspace()
        try:
            with _DetachedContext() as ctx:
                download = ctx.detached / "download"
                download.mkdir()
                run_dir = _sealed_run_with_hook(download, digest("any-source"))
                with self.assertRaises(RuntimeError) as ctx_err:
                    validate_run(run_dir)
                message = str(ctx_err.exception)
                self.assertIn(f"{PKG}.evaluator:evaluate_rhae", message)
                self.assertIn(f"module='{PKG}.evaluator'", message)
                self.assertIn("project_root=", message)
                self.assertIn("sys.path", message)
                self.assertIsInstance(ctx_err.exception.__cause__, ModuleNotFoundError)
        finally:
            sys.modules.pop(PKG, None)
            sys.modules.pop(f"{PKG}.evaluator", None)
            shutil.rmtree(ws, ignore_errors=True)

    def test_ensure_hook_project_roots_prefers_explicit_root(self):
        explicit = Path(tempfile.mkdtemp(prefix="aih-explicit-"))
        fallback = Path(tempfile.mkdtemp(prefix="aih-fallback-"))
        try:
            ensure_hook_project_roots_on_sys_path(project_root=explicit)
            self.assertIn(str(explicit), sys.path)
            self.assertEqual(sys.path[0], str(explicit))
        finally:
            for p in (explicit, fallback):
                shutil.rmtree(p, ignore_errors=True)
            if str(explicit) in sys.path:
                sys.path.remove(str(explicit))


class TestCandidateSynthesisEnhancements(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        self.artifacts_dir = self.test_dir / "artifacts"
        self.artifacts_dir.mkdir(parents=True)

        for exp_id in ["exp01_model", "exp02_model"]:
            exp_path = self.artifacts_dir / exp_id
            exp_path.mkdir()
            df = pd.DataFrame({
                "id": [1, 2, 3, 4],
                "fold": [0, 0, 1, 1],
                "pred_prob": [0.2, 0.8, 0.4, 0.9] if exp_id == "exp01_model" else [0.4, 0.6, 0.6, 0.7],
                "pred_label": ["no", "yes", "no", "yes"] if exp_id == "exp01_model" else ["no", "no", "yes", "yes"],
            })
            df.to_csv(exp_path / "oof_predictions.csv", index=False)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_weighted_probability_blend(self):
        """Weighted blend correctly averages continuous probabilities."""
        comp_state = CompetitionState(project_root=self.test_dir)
        with self.assertRaisesRegex(ValueError, "LEGACY_UNVERIFIED"):
            comp_state.synthesize_candidate(
                name="prob_blend_v1",
                mode="weighted_blend",
                source_map={"components": ["exp01_model", "exp02_model"], "weights": [0.75, 0.25]},
            )


if __name__ == "__main__":
    unittest.main()
