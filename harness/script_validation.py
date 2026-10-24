"""Read the execution scope provided to a custom script or pipeline stage."""

import json
import os
from pathlib import Path


def load_validation_manifest(path=None):
    """Return selected fold train/validation IDs, run kind and sample bound.

    Use these partitions instead of independently reconstructing CV or looping
    over all folds in ``splits.tsv``. A probe's selected folds and sample limits
    are already resolved by the parent runner.
    """
    path = path or os.environ.get("HARNESS_VALIDATION_MANIFEST")
    if not path:
        raise ValueError("Custom execution requires HARNESS_VALIDATION_MANIFEST")
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or not manifest.get("partitions"):
        raise ValueError("Invalid script validation manifest")
    return manifest
