"""Unit tests for Kaggle competition discovery, container indexing, and compat stubs."""
from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from harness.kaggle_compat import KaggleClient, _NormalisedObject


class TestNormalisedObjectContainersAndAliases(unittest.TestCase):
    def test_container_indexing_and_iteration(self):
        comp1 = SimpleNamespace(title="Comp 1", ref="comp-1")
        comp2 = SimpleNamespace(title="Comp 2", ref="comp-2")
        raw = SimpleNamespace(competitions=[comp1, comp2])
        norm = _NormalisedObject(raw, "kagglesdk")

        self.assertEqual(len(norm), 2)
        self.assertEqual(norm[0].title, "Comp 1")
        self.assertEqual(norm[1].title, "Comp 2")

        titles = [item.title for item in norm]
        self.assertEqual(titles, ["Comp 1", "Comp 2"])

    def test_aliases_size_total_bytes_and_replies(self):
        raw = {
            "totalBytes": 1024,
            "totalReplies": 15,
            "teamId": 42,
            "teamName": "WinningTeam",
            "isKernelsSubmissionsOnly": True,
        }
        norm = _NormalisedObject(raw, "kagglesdk")
        self.assertEqual(norm.total_bytes, 1024)
        self.assertEqual(norm.size, 1024)
        self.assertEqual(norm.total_replies, 15)
        self.assertEqual(norm.comment_count, 15)
        self.assertEqual(norm.team_id, 42)
        self.assertEqual(norm.team_name, "WinningTeam")
        self.assertTrue(norm.is_code_competition)


class TestKaggleClientCompetitionDiscovery(unittest.TestCase):
    def test_get_competition_extracts_metadata(self):
        client = KaggleClient()
        fake_comp = SimpleNamespace(
            id=123,
            ref="https://www.kaggle.com/competitions/test-comp",
            title="Test Challenge",
            url="https://www.kaggle.com/competitions/test-comp",
            description="A test competition",
            deadline="2026-11-01 23:59:00",
            category="Featured",
            reward="$50,000",
            teamCount=500,
            userRank=12,
            userHasEntered=True,
            maxDailySubmissions=5,
            evaluationMetric="Log Loss",
            isKernelsSubmissionsOnly=True,
            submissionsDisabled=False,
        )
        fake_resp = SimpleNamespace(competitions=[fake_comp])

        mock_api = MagicMock()
        mock_api.competitions_list.return_value = fake_resp

        with patch.object(client, "_legacy_api_instance", return_value=mock_api):
            info = client.get_competition("test-comp")

        self.assertIsNotNone(info)
        self.assertEqual(info["ref"], "test-comp")
        self.assertEqual(info["title"], "Test Challenge")
        self.assertEqual(info["deadline"], "2026-11-01 23:59:00")
        self.assertEqual(info["evaluation_metric"], "Log Loss")
        self.assertTrue(info["is_code_competition"])
        self.assertTrue(info["is_ongoing"])
        self.assertEqual(info["user_rank"], 12)

    def test_list_competitions_wraps_objects(self):
        client = KaggleClient()
        fake_comp = SimpleNamespace(id=1, title="Alpha Comp")
        fake_resp = SimpleNamespace(competitions=[fake_comp])

        mock_api = MagicMock()
        mock_api.competitions_list.return_value = fake_resp

        with patch.object(client, "_legacy_api_instance", return_value=mock_api):
            comps = client.list_competitions(search="alpha")

        self.assertEqual(len(comps), 1)
        self.assertEqual(comps[0].title, "Alpha Comp")

    def test_competition_aliases(self):
        client = KaggleClient()
        self.assertEqual(client.competition_submissions, client.list_submissions)
        self.assertEqual(client.competition_view, client.get_competition)
        self.assertEqual(client.competitions_list, client.list_competitions)
        self.assertEqual(client.kernel_status, client.kernels_status)
        self.assertEqual(client.kernel_logs, client.kernels_logs)


class TestCompetitionCliAndCompatStubs(unittest.TestCase):
    def test_competition_cli_info(self):
        from harness.competition_cli import main

        fake_info = {
            "ref": "test-comp",
            "title": "Test Comp",
            "deadline": "2026-12-01",
            "is_ongoing": True,
        }
        with patch("harness.competition_cli.KaggleClient") as mock_cls:
            mock_inst = MagicMock()
            mock_inst.get_competition.return_value = fake_info
            mock_cls.return_value = mock_inst

            ret = main(["info", "test-comp"])
            self.assertEqual(ret, 0)
            mock_inst.get_competition.assert_called_once_with("test-comp")

    def test_submission_cli_info(self):
        from harness.submission_dispatch import main

        fake_info = {
            "ref": "test-comp",
            "title": "Test Comp",
            "deadline": "2026-12-01",
        }
        with patch("harness.kaggle_compat.KaggleClient") as mock_cls:
            mock_inst = MagicMock()
            mock_inst.get_competition.return_value = fake_info
            mock_cls.return_value = mock_inst

            ret = main(["info", "test-comp"])
            self.assertEqual(ret, 0)

    def test_compat_stubs_importable(self):
        from harness.agent_bridge import typed_actions
        from harness.endgame_submission import SubmissionValidator

        self.assertIsInstance(typed_actions, list)
        self.assertIn("create_experiment", typed_actions)
        self.assertIsNotNone(SubmissionValidator)


if __name__ == "__main__":
    unittest.main()
