import unittest
import datetime
from unittest.mock import MagicMock
from harness.kaggle_compat import _NormalisedObject
from harness.endgame_submission import KaggleLeaderboardTracker


class TestKaggleCompatAliasesAndWrapping(unittest.TestCase):
    def test_datetime_objects_not_wrapped(self):
        dt = datetime.datetime(2026, 10, 21, 10, 30, 0)
        norm = _NormalisedObject({"dt": dt}, "kagglesdk")
        self.assertIsInstance(norm.dt, datetime.datetime)
        self.assertEqual(type(norm.dt), datetime.datetime)

    def test_topic_and_message_aliases(self):
        # Raw topic-like dict or object
        raw_topic = {
            "id": 12345,
            "title": "Discussion Topic",
            "votes": 15,
            "author_name": "researcher1",
            "post_date": datetime.datetime(2026, 10, 20, 12, 0, 0),
            "comment_count": 8,
        }
        topic = _NormalisedObject(raw_topic, "kagglesdk")
        self.assertEqual(topic.votes, 15)
        self.assertEqual(topic.total_votes, 15)
        self.assertEqual(topic.totalVotes, 15)
        self.assertEqual(topic.score, 15)
        self.assertEqual(topic.author, "researcher1")
        self.assertEqual(topic.author_name, "researcher1")
        self.assertEqual(topic.post_date, raw_topic["post_date"])
        self.assertEqual(topic.date, raw_topic["post_date"])
        self.assertEqual(topic.created_at, raw_topic["post_date"])
        self.assertEqual(topic.comments, 8)
        self.assertEqual(topic.comment_count, 8)

    def test_message_content_aliases(self):
        raw_msg = {
            "id": 999,
            "author_name": "expert",
            "content": "Here is the key insight",
            "post_date": datetime.datetime(2026, 10, 20, 14, 0, 0),
            "votes": 5,
        }
        msg = _NormalisedObject(raw_msg, "kagglesdk")
        self.assertEqual(msg.author, "expert")
        self.assertEqual(msg.author_name, "expert")
        self.assertEqual(msg.content, "Here is the key insight")
        self.assertEqual(msg.message, "Here is the key insight")
        self.assertEqual(msg.body, "Here is the key insight")
        self.assertEqual(msg.date, raw_msg["post_date"])
        self.assertEqual(msg.created_at, raw_msg["post_date"])
        self.assertEqual(msg.votes, 5)
        self.assertEqual(msg.total_votes, 5)


class TestSubmissionListSubcommand(unittest.TestCase):
    def test_tracker_list_submissions(self):
        mock_client = MagicMock()
        mock_sub = MagicMock()
        mock_sub.ref = 123456
        mock_sub.status = "COMPLETE"
        mock_sub.public_score = "0.950"
        mock_sub.private_score = "0.940"
        mock_sub.date = datetime.datetime(2026, 10, 21, 8, 0, 0)
        mock_sub.description = "Test sub"
        mock_sub.file_name = "sub.csv"
        mock_sub.error_description = ""
        mock_client.list_submissions.return_value = [mock_sub]

        tracker = KaggleLeaderboardTracker(client=mock_client)
        subs = tracker.list_submissions("test-comp", limit=10)
        self.assertEqual(len(subs), 1)
        self.assertEqual(subs[0]["ref"], 123456)
        self.assertEqual(subs[0]["status"], "COMPLETE")
        self.assertEqual(subs[0]["public_score"], "0.950")
        self.assertEqual(subs[0]["public_score_f"], 0.950)
        self.assertEqual(subs[0]["description"], "Test sub")

    def test_submission_dict_metadata_parity(self):
        mock_client = MagicMock()
        mock_sub = MagicMock()
        mock_sub.ref = 57510673
        mock_sub.status = "ERROR"
        mock_sub.public_score = None
        mock_sub.private_score = None
        mock_sub.date = datetime.datetime(2026, 10, 21, 12, 0, 0)
        mock_sub.description = "V7 Hearth Idle Timeout"
        mock_sub.file_name = "submission.csv"
        mock_sub.error_description = "Hearth Gateway connection timeout after 15m idle"
        mock_sub.submitted_by = "test_user"
        mock_sub.team_name = "AlphaTeam"
        mock_sub.url = "https://www.kaggle.com/code/test_user/hearth-v7?scriptVersionId=123"

        tracker = KaggleLeaderboardTracker(client=mock_client)
        info = tracker._submission_dict(mock_sub)
        self.assertEqual(info["ref"], 57510673)
        self.assertEqual(info["submitted_by"], "test_user")
        self.assertEqual(info["team_name"], "AlphaTeam")
        self.assertEqual(info["url"], "https://www.kaggle.com/code/test_user/hearth-v7?scriptVersionId=123")
        self.assertIn("Hearth Gateway", info["error_description"])

    def test_tracker_wait_for_scores_multi(self):
        mock_client = MagicMock()
        sub1 = MagicMock()
        sub1.ref = 111
        sub1.status = "COMPLETE"
        sub1.public_score = "0.85"
        sub1.private_score = None
        sub1.date = None
        sub1.description = "Sub 1"
        sub1.file_name = "sub1.csv"
        sub1.error_description = ""
        sub1.submitted_by = "alice"
        sub1.team_name = "team"
        sub1.url = ""

        sub2 = MagicMock()
        sub2.ref = 222
        sub2.status = "COMPLETE"
        sub2.public_score = "0.88"
        sub2.private_score = None
        sub2.date = None
        sub2.description = "Sub 2"
        sub2.file_name = "sub2.csv"
        sub2.error_description = ""
        sub2.submitted_by = "bob"
        sub2.team_name = "team"
        sub2.url = ""

        mock_client.list_submissions.return_value = [sub1, sub2]
        tracker = KaggleLeaderboardTracker(client=mock_client)
        res = tracker.wait_for_scores("test-comp", [111, 222], timeout_s=10.0, poll_interval_s=0.01)

        self.assertTrue(res["all_final"])
        self.assertEqual(res["count"], 2)
        self.assertEqual(len(res["submissions"]), 2)
        scores = {s["ref"]: s["public_score_f"] for s in res["submissions"]}
        self.assertEqual(scores[111], 0.85)
        self.assertEqual(scores[222], 0.88)




class TestDiscussionCLI(unittest.TestCase):
    def test_discussion_cli_list(self):
        from unittest.mock import patch
        from harness.discussion_cli import main

        mock_topic = MagicMock()
        mock_topic.id = 101
        mock_topic.title = "Important Topic"
        mock_topic.author = "alice"
        mock_topic.votes = 10
        mock_topic.comments = 4
        mock_topic.date = "2026-10-21"
        mock_topic.topic_url = "/discussion/101"

        with patch("harness.discussion_cli.KaggleClient") as mock_cls:
            mock_inst = MagicMock()
            mock_inst.competition_topics.return_value = [mock_topic]
            mock_cls.return_value = mock_inst
            ret = main(["list", "test-comp"])
            self.assertEqual(ret, 0)
            mock_inst.competition_topics.assert_called_once()

    def test_discussion_cli_view(self):
        from unittest.mock import patch
        from harness.discussion_cli import main

        mock_topic_payload = {
            "id": 101,
            "title": "Topic Title",
            "author": "bob",
            "date": "2026-10-21",
            "votes": 3,
            "content": "Insightful response",
            "comments_count": 0,
            "comments": [],
        }

        with patch("harness.discussion_cli.KaggleClient") as mock_cls:
            mock_inst = MagicMock()
            mock_inst.get_topic.return_value = mock_topic_payload
            mock_cls.return_value = mock_inst
            ret = main(["view", "101", "--competition", "test-comp"])
            self.assertEqual(ret, 0)
            mock_inst.get_topic.assert_called_once_with(101, competition="test-comp")


class TestSubmissionDispatchCLI(unittest.TestCase):
    def test_submission_dispatch_list(self):
        from unittest.mock import patch
        from harness.submission_dispatch import main

        with patch("harness.submission_dispatch.KaggleLeaderboardTracker") as mock_cls:
            mock_inst = MagicMock()
            mock_inst.list_submissions.return_value = [{"ref": 111, "status": "COMPLETE", "public_score": "0.9"}]
            mock_cls.return_value = mock_inst
            ret = main(["list", "test-comp", "--limit", "5"])
            self.assertEqual(ret, 0)
            mock_inst.list_submissions.assert_called_once_with("test-comp", limit=5)

    def test_submission_dispatch_watch_single(self):
        from unittest.mock import patch
        from harness.submission_dispatch import main

        with patch("harness.submission_dispatch.KaggleLeaderboardTracker") as mock_cls:
            mock_inst = MagicMock()
            mock_inst.wait_for_score.return_value = {"ref": "111", "status": "COMPLETE", "final": True}
            mock_cls.return_value = mock_inst
            ret = main(["watch", "test-comp", "111"])
            self.assertEqual(ret, 0)
            mock_inst.wait_for_score.assert_called_once()
            self.assertEqual(mock_inst.wait_for_score.call_args[0][0], "test-comp")
            self.assertEqual(mock_inst.wait_for_score.call_args[0][1], "111")

    def test_submission_dispatch_watch_multi_args(self):
        from unittest.mock import patch
        from harness.submission_dispatch import main

        with patch("harness.submission_dispatch.KaggleLeaderboardTracker") as mock_cls:
            mock_inst = MagicMock()
            mock_inst.wait_for_scores.return_value = {"all_final": True, "count": 2, "submissions": []}
            mock_cls.return_value = mock_inst
            ret = main(["watch", "test-comp", "111", "222"])
            self.assertEqual(ret, 0)
            mock_inst.wait_for_scores.assert_called_once()
            self.assertEqual(mock_inst.wait_for_scores.call_args[0][0], "test-comp")
            self.assertEqual(mock_inst.wait_for_scores.call_args[0][1], ["111", "222"])

    def test_submission_dispatch_watch_multi_comma(self):
        from unittest.mock import patch
        from harness.submission_dispatch import main

        with patch("harness.submission_dispatch.KaggleLeaderboardTracker") as mock_cls:
            mock_inst = MagicMock()
            mock_inst.wait_for_scores.return_value = {"all_final": True, "count": 2, "submissions": []}
            mock_cls.return_value = mock_inst
            ret = main(["watch", "test-comp", "111,222"])
            self.assertEqual(ret, 0)
            mock_inst.wait_for_scores.assert_called_once()
            self.assertEqual(mock_inst.wait_for_scores.call_args[0][0], "test-comp")
            self.assertEqual(mock_inst.wait_for_scores.call_args[0][1], ["111", "222"])

    def test_submission_dispatch_quota_cli(self):
        from unittest.mock import patch
        from harness.submission_dispatch import main

        with patch("harness.submission_dispatch.KaggleLeaderboardTracker") as mock_cls:
            mock_inst = MagicMock()
            mock_inst.get_quota.return_value = {
                "competition": "test-comp",
                "num_today": 2,
                "num_allowed_now": 3,
                "num_total": 33,
                "has_remaining_quota": True,
                "daily_quota": 5,
            }
            mock_cls.return_value = mock_inst
            ret = main(["quota", "test-comp"])
            self.assertEqual(ret, 0)
            mock_inst.get_quota.assert_called_once_with("test-comp")

            # Also verify alias limits
            ret2 = main(["limits", "test-comp"])
            self.assertEqual(ret2, 0)
            self.assertEqual(mock_inst.get_quota.call_count, 2)

    def test_tracker_get_quota_and_caching(self):
        import tempfile
        import shutil
        import json
        from pathlib import Path
        from harness.endgame_submission import KaggleLeaderboardTracker

        tmp_dir = Path(tempfile.mkdtemp())
        try:
            dot_ai = tmp_dir / ".ai-harness"
            dot_ai.mkdir()

            mock_client = MagicMock()
            mock_client.get_submission_limits.return_value = {
                "competition": "test-comp",
                "num_today": 1,
                "num_allowed_now": 4,
                "num_total": 10,
                "has_remaining_quota": True,
                "daily_quota": 5,
            }

            tracker = KaggleLeaderboardTracker(client=mock_client)
            quota = tracker.get_quota("test-comp", project_root=tmp_dir)

            self.assertEqual(quota["num_today"], 1)
            self.assertEqual(quota["num_allowed_now"], 4)
            self.assertTrue(quota["has_remaining_quota"])

            # Verify cached file written
            cached_file = dot_ai / "submission_limits.json"
            self.assertTrue(cached_file.is_file())
            cached_data = json.loads(cached_file.read_text(encoding="utf-8"))
            self.assertEqual(cached_data["num_allowed_now"], 4)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
