#!/usr/bin/env python3
"""Compatibility CLI; the installed runtime owns all audit logic."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from harness.audit_gate import (  # noqa: F401,E402
    compute_sha256, get_git_commit, audit_prediction_integrity, compute_diff_metrics,
    compute_continuous_significance, compute_mcnemar_significance,
    check_prediction_diversity, align_prediction_frames,
)
from harness.audit_cli import main  # noqa: E402
if __name__ == '__main__':
    main()
