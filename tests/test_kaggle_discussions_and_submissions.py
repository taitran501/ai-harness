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

        mock_msg = MagicMock()
        mock_msg.id = 501
        mock_msg.author = "bob"
        mock_msg.date = "2026-10-21"
        mock_msg.votes = 3
        mock_msg.content = "Insightful response"

        with patch("harness.discussion_cli.KaggleClient") as mock_cls:
            mock_inst = MagicMock()
            mock_inst.competition_topic_messages.return_value = [mock_msg]
            mock_cls.return_value = mock_inst
            ret = main(["view", "101", "--competition", "test-comp"])
            self.assertEqual(ret, 0)
            mock_inst.competition_topic_messages.assert_called_once_with(101, competition="test-comp")


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

if __name__ == "__main__":
    unittest.main()
