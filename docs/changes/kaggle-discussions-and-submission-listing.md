# Kaggle Discussions Facade, Native CLI, and Submission Listing

- **Date**: 2026-10-21
- **PR**: [#114](https://github.com/taitran501/ai-harness/pull/114)
- **Branch**: `feat/agent-friction-and-platform-integrations`
- **Status**: merged
- **Commits**: `966a561` (squash, on `main`)

## Why

Deep forensic analysis across the latest active Antigravity conversations on 2026-10-21 (`arc-prize-2026-arc-agi-3`, `airline_prediction`, `rsna-knee`, `aimo`) revealed recurrent agent friction:
1. **Repeated User Directives for Discussion Insights**: In all four competitions, the user repeatedly instructed agents to inspect competition discussions ("check discussions for latest context", "check thêm discussions để get insights"). Agents struggled to query Kaggle discussions due to missing CLI commands, raw protobuf response types (`ApiListCompetitionTopicsResponse` raising `TypeError: object has no len()`), signature mismatches (`competition_list_topic_messages(competition, topic_id)` missing arguments), and unmapped fields (`votes`, `post_date`, `author_name`).
2. **Missing `submission list` Subcommand**: When agents needed to review previous submission IDs, scores, and timestamps, `ai-harness submission` only exposed `dispatch|status|watch|latest`. Agents resorted to fragile Python one-liners calling internal methods and hitting attribute errors.
3. **Datetime Normalisation Wrapping**: Datetime objects were unnecessarily wrapped inside `_NormalisedObject`, breaking `isinstance(..., datetime.datetime)` checks and formatting ergonomics.

## What changed

- **`harness/kaggle_compat.py`**:
  - Expanded `_LEGACY_ALIASES` to provide bidirectional resolution between:
    - `votes` <-> `total_votes` <-> `totalVotes` <-> `vote_count` <-> `score`
    - `author` <-> `author_name` <-> `authorName` <-> `user` <-> `username`
    - `date` <-> `created_at` <-> `createdAt` <-> `post_date` <-> `postDate` <-> `submittedDate` <-> `submitted_date`
    - `content` <-> `message` <-> `body` <-> `raw_markdown`
    - `comment_count` <-> `commentCount` <-> `comments`
  - In `_NormalisedObject._wrap`: Exempt `datetime.date`, `datetime.datetime`, and `datetime.time` from being wrapped in `_NormalisedObject`, returning them natively.
  - In `KaggleClient`:
    - Updated `competition_topics` to safely unpack `resp.topics` or list-like responses.
    - Added `competition_topic_messages(topic_id, competition=None, **kwargs)` with automatic fallback between `competition_list_topic_messages` and `forums_topic_show`.
- **`harness/endgame_submission.py`**:
  - Added `list_submissions(competition, limit=None)` returning normalised projection dictionaries with `date`, `ref`, `status`, `public_score`, `public_score_f`, `private_score`, `private_score_f`, `description`, `file_name`.
- **`harness/submission_dispatch.py`**:
  - Added `list` subcommand (`ai-harness submission list <competition> [--limit N]`).
- **`harness/discussion_cli.py` & `harness/cli.py`**:
  - Added native `ai-harness discussion` CLI with `list` and `view` subcommands.
- **`tests/test_kaggle_discussions_and_submissions.py`**:
  - Added 7 unit tests covering datetime wrapping exemption, bidirectional alias resolution, submission listing, and discussion CLI handlers.

## Approach

- Maintained zero breaking changes for existing SDK consumers.
- Reused `KaggleClient` unified facade and `_NormalisedObject` proxy pattern to absorb discrepancies between legacy `kaggle` and new `kagglesdk` protobuf objects.

## Evidence

- `tests/test_kaggle_discussions_and_submissions.py`: 7/7 tests PASS.
- `tools/run_tests.py --smoke`: 47/47 tests PASS.
- `tools/run_tests.py --fast`: 915/915 tests PASS (89 modules).
- `ruff check harness/ tools/ tests/`: 100% clean.

## Impact & follow-ups

None. All competition workspaces seamlessly benefit via Single-Install (`HARNESS_FRAMEWORK_DIR`).
