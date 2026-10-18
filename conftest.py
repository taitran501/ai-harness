"""Pytest bootstrap for the ai-harness unit/regression suite.

1. CPU-only seeding: the suite never exercises CUDA, but the dev machine has a
   GPU, so ``set_seed`` would otherwise import the full PyTorch stack (~4-7s
   cold) on every runner invocation. We mark torch-seeding as disabled for the
   whole session here, keeping the CPU seeded values identical while removing
   the cost. Production/GPU runs do not load this file, so their torch/cuDNN
   seeding is unaffected (set_seed defaults to seeding when the flag is
   unset).

2. Environment isolation (mirrors ``tools/run_tests.py``): drop ANY globally-set
   ``HARNESS_*`` / ``AI_HARNESS_*`` var so a plain ``pytest`` invocation
   behaves exactly like the canonical runner. The suite is written to run in a
   clean environment — tests that need a specific var set it explicitly per
   test (via mock.patch.dict or direct os.environ assignment). Without this, a
   machine with harness env vars set globally (e.g. the one-time single-install
   migration sets ``HARNESS_FRAMEWORK_DIR``) contaminates the suite with false
   failures in env-sensitive tests (framework discovery, sync, adoption,
   status). The prefix sweep is systemic: a future env var is isolated
   automatically without touching each test module.
"""
import os
from pathlib import Path

import pytest

# Sweep first, then re-apply the one HARNESS_* var this suite itself relies on.
for k in [k for k in os.environ if k.startswith(("HARNESS_", "AI_HARNESS_"))]:
    del os.environ[k]
os.environ["HARNESS_DISABLE_TORCH_SEED"] = "1"


# ---------------------------------------------------------------------------
# 3. Suite side-effect guard: the suite must never write artifact-state files
#    into this checkout. TreeJournal()/AgentBridge() constructed without
#    explicit temp paths fall back to project discovery (package root when
#    nothing is attached to the CWD) and have leaked journal.json / locks /
#    research state into the real checkout (see
#    docs/changes/test-isolation-and-sync-cache.md). Snapshot the checkout's
#    own artifact-state directories at session start and fail at session end
#    if the suite itself created anything there. Pre-existing files (a dev
#    who attached this checkout as a workspace) are the baseline, not a
#    failure — only files the suite adds are reported.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent
_ARTIFACT_STATE_DIRS = ("artifacts", ".ai-harness")


def _artifact_state_files() -> set:
    files = set()
    for rel in _ARTIFACT_STATE_DIRS:
        base = _REPO_ROOT / rel
        if base.is_dir():
            for p in base.rglob("*"):
                if p.is_file():
                    files.add(str(p))
    return files


@pytest.fixture(scope="session", autouse=True)
def _no_suite_artifact_leak():
    before = _artifact_state_files()
    yield
    added = sorted(_artifact_state_files() - before)
    assert not added, (
        "Test suite wrote artifact-state files into this checkout:\n"
        + "\n".join(added)
        + "\nTests must use explicit temp paths (journal_path/storage_path/artifacts_dir "
        "or a project root), never default-path project discovery."
    )

