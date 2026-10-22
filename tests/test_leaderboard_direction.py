"""An agent's best-submission lookup must follow the competition metric direction."""

import contextlib
import copy
import io
import json
import os
import shutil
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from harness import submission_dispatch
from harness.endgame_submission import KaggleLeaderboardTracker
from tests.fixtures import make_workspace
from tests.test_correctness_contracts import task


def tracker(scores=(.1, .5)):
    submissions = [SimpleNamespace(ref=i, status="COMPLETE", public_score=str(value))
                   for i, value in enumerate(scores, 1)]
    return KaggleLeaderboardTracker(client=SimpleNamespace(list_submissions=lambda competition: submissions))


class LeaderboardDirectionTests(unittest.TestCase):
    def minimizing_workspace(self):
        doc = {"task": copy.deepcopy(task().metadata)}
        doc["task"]["data"]["train_path"] = "dataset/train.csv"
        frame = pd.DataFrame({"id": range(4), "group": range(4), "x": range(4),
                              "category": ["a", "b"] * 2, "target": range(4)})
        root, _ = make_workspace(task=doc, train_csv=frame.to_csv(index=False))
        self.addCleanup(shutil.rmtree, root)
        env = patch.dict(os.environ, {"HARNESS_PROJECT_ROOT": str(root)})
        env.start()
        self.addCleanup(env.stop)

    def test_api_defaults_to_the_attached_task_direction(self):
        self.minimizing_workspace()
        result = tracker().latest("rmse-competition")
        self.assertEqual(result["best"]["ref"], 1)
        self.assertEqual(result["direction"], "minimize")
        self.assertEqual(result["direction_source"], "task")

    def test_explicit_direction_overrides_the_workspace(self):
        self.minimizing_workspace()
        result = tracker().latest("other-competition", direction="maximize")
        self.assertEqual(result["best"]["ref"], 2)
        self.assertEqual(result["direction_source"], "explicit")

    def test_cli_uses_the_task_and_exposes_an_explicit_override(self):
        self.minimizing_workspace()
        for extra, expected in (([], 1), (["--direction", "maximize"], 2)):
            with self.subTest(extra=extra):
                output = io.StringIO()
                with (patch.object(submission_dispatch, "KaggleLeaderboardTracker", return_value=tracker()),
                      contextlib.redirect_stdout(output)):
                    code = submission_dispatch.main(["latest", "rmse-competition", *extra])
                self.assertEqual(code, 0, output.getvalue())
                self.assertEqual(json.loads(output.getvalue())["best"]["ref"], expected)

    def test_without_a_task_the_legacy_maximize_default_is_explicit(self):
        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ, {"HARNESS_PROJECT_ROOT": root}):
            result = tracker().latest("comp")
        self.assertEqual(result["best"]["ref"], 2)
        self.assertEqual(result["direction"], "maximize")
        self.assertEqual(result["direction_source"], "default")

    def test_zero_is_a_valid_minimum_and_nonfinite_scores_are_not_ranked(self):
        result = tracker([float("nan"), float("inf"), 0, .5]).latest("comp", direction="minimize")
        self.assertEqual(result["scored"], 2)
        self.assertEqual(result["best"]["ref"], 3)
        self.assertEqual(result["best"]["public_score_f"], 0)
        with self.assertRaisesRegex(ValueError, "direction"):
            tracker().latest("comp", direction="sideways")
