# Codabench Platform Integration and Enhanced Discussion Viewer

- **Date**: 2026-10-21
- **PR**: [#114](https://github.com/taitran501/ai-harness/pull/114)
- **Branch**: `feat/agent-friction-and-platform-integrations`
- **Status**: merged
- **Commits**: `966a561` (squash, on `main`)

## Why

1. **Codabench Artifact Audit Friction**: In the AIMO / Codabench competition workspace (`Documents/aimo`), agents evaluating submissions (`1007196`, `1007195`) had to hand-craft 70-line Python scripts using raw `requests` and `zipfile` to query status, download detailed execution logs, and unpack prediction / scoring zip files.
2. **Kaggle Discussion Topic Content & Author Completeness**: In the RSNA Knee competition workspace (`Documents/rsna-knee`), topic messages queried via the legacy `competition_list_topic_messages` API returned empty author strings (`authorName: null`) and omitted root post details. The agent was forced to write a custom script reaching into `kagglesdk.discussions.discussion_api_client` directly to read topic titles, authors, and comments.

## What changed

- **`harness/codabench_compat.py`**:
  - Implemented `CodabenchClient` supporting session token resolution from CLI, environment variable `CODABENCH_SESSION_ID`, or `.codabench_session` / `.codabench_token` files in the workspace root.
  - Implemented `get_submission(sub_id)`: fetches metadata, primary score, status, and individual metric columns.
  - Implemented `get_details(sub_id)`: fetches log URLs, prediction result URLs, and scoring result URLs.
  - Implemented `download_submission(sub_id, output_dir)`: downloads and unzips logs, `meta.json`, `details.json`, predictions, and score breakdowns into the target directory.
- **`harness/codabench_cli.py` & `harness/cli.py`**:
  - Registered `ai-harness codabench` CLI with `status` and `download` subcommands.
- **`harness/kaggle_compat.py`**:
  - Added `KaggleClient.get_topic(topic_id, competition=None)`: queries the modern `discussions.discussion_api_client` to retrieve complete topic metadata, author name, clean text content, and all comments with commenter names, with automatic fallback to legacy message listing.
- **`harness/discussion_cli.py`**:
  - Updated `ai-harness discussion view <topic_id>` to output structured topic title, author, date, votes, clean content, comments count, and all comments.
- **`tests/test_codabench_compat.py`**:
  - Added unit tests for Codabench API responses, artifact downloading/unzipping, and CLI invocations.

## Evidence

- `tests/test_codabench_compat.py`: 3/3 tests PASS.
- `tests/test_kaggle_discussions_and_submissions.py`: 7/7 tests PASS.
- Live CLI verification:
  - `ai-harness codabench status 1007196 --session 288nmtzcsxx26hnrl8xa7wrqpdamcgta` returned status `Finished`, score `0.7142857143`.
  - `ai-harness codabench download 1007196` successfully downloaded and unpacked 15 artifact/log files.
  - `ai-harness discussion view 751021` successfully retrieved topic title `Shake up/down or stable private LB`, author `DeepUnderstanding`, and comments.
- `tools/run_tests.py --smoke`: 47/47 tests PASS.
- `ruff check harness/ tools/ tests/`: 100% clean.

## Impact & follow-ups

None. Workspaces on Codabench or Kaggle instantly inherit these capabilities via Single-Install (`HARNESS_FRAMEWORK_DIR`).
