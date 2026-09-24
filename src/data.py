"""Data loading, validation, and preprocessing utilities for StereoQueerEval."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Tuple

import pandas as pd

REQUIRED_COLUMNS = [
    "StereoQueerEval_id",
    "yt_title",
    "yt_description",
    "yt_comment",
    "stereotype",
    "hate_speech",
    "target",
]
STEREOTYPE_LABELS = {"no", "yes"}
HATE_LABELS = {"no", "yes_explicit", "yes_implicit"}
TARGET_TYPES = ["individual", "group"]
IDENTITIES = ["l", "g", "b", "t", "q", "i", "a", "lgbtqia+", "nb"]
LANG_PATTERN = re.compile(r"StereoQueerEval_([A-Z]{2})_training\.tsv$")
INPUT_VARIANTS = ["comment", "title_comment", "full_context"]


def sha256(path: Path) -> str:
    """Compute the SHA-256 checksum of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_target(value: str) -> Tuple[str | None, list[str]]:
    """Parse target string (e.g., 'group_l,g' -> ('group', ['l', 'g'])).

    Args:
        value: Raw target annotation string.

    Returns:
        Tuple of (target_type, list_of_identities).

    Raises:
        ValueError: If target string is malformed or contains invalid identifiers.
    """
    if value == "none":
        return None, []
    parts = value.split(",")
    if "_" not in parts[0]:
        raise ValueError(f"Malformed target: {value!r}")
    target_type, first_identity = parts[0].split("_", 1)
    identities = [first_identity, *parts[1:]]
    if target_type not in TARGET_TYPES:
        raise ValueError(f"Unknown target type in {value!r}: {target_type!r}")
    unknown = sorted(set(identities) - set(IDENTITIES))
    if unknown:
        raise ValueError(f"Unknown identities in {value!r}: {unknown}")
    if len(identities) != len(set(identities)):
        raise ValueError(f"Duplicate identities in {value!r}")
    return target_type, identities


def load_one_language(path: Path) -> pd.DataFrame:
    """Load and validate a single language TSV file."""
    match = LANG_PATTERN.fullmatch(path.name)
    if not match:
        raise ValueError(f"Unexpected training filename format: {path.name}")
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    if list(df.columns) != REQUIRED_COLUMNS:
        raise ValueError(
            f"{path.name}: expected columns {REQUIRED_COLUMNS}, got {list(df.columns)}"
        )
    if df.empty:
        raise ValueError(f"{path.name}: dataset is empty")
    empty = {column: int(df[column].eq("").sum()) for column in REQUIRED_COLUMNS}
    if any(empty.values()):
        raise ValueError(f"{path.name}: empty required values: {empty}")
    if df["StereoQueerEval_id"].duplicated().any():
        raise ValueError(f"{path.name}: duplicate IDs detected")
    if not set(df["stereotype"]).issubset(STEREOTYPE_LABELS):
        raise ValueError(f"{path.name}: invalid stereotype labels")
    if not set(df["hate_speech"]).issubset(HATE_LABELS):
        raise ValueError(f"{path.name}: invalid hate_speech labels")

    parsed = [parse_target(value) for value in df["target"]]
    df["target_type"] = [item[0] or "none" for item in parsed]
    for identity in IDENTITIES:
        df[f"identity_{identity}"] = [int(identity in item[1]) for item in parsed]

    non_hateful = df["hate_speech"].eq("no")
    if not df.loc[non_hateful, "target"].eq("none").all():
        raise ValueError(f"{path.name}: non-hateful rows with non-empty target")
    if not df.loc[~non_hateful, "target"].ne("none").all():
        raise ValueError(f"{path.name}: hateful rows without target")

    df["language"] = match.group(1)
    df["video_key"] = [
        hashlib.sha256(f"{title}\0{description}".encode("utf-8")).hexdigest()
        for title, description in zip(df["yt_title"], df["yt_description"])
    ]
    return df


def load_dataset(data_dir: Path | str) -> tuple[pd.DataFrame, dict[str, str]]:
    """Discover, load, and validate all training datasets in data_dir."""
    data_path = Path(data_dir)
    paths = sorted(data_path.glob("StereoQueerEval_*_training.tsv"))
    if not paths:
        raise FileNotFoundError(f"No training TSV files found in {data_path}")
    frames = [load_one_language(p) for p in paths]
    combined = pd.concat(frames, ignore_index=True)
    if combined["StereoQueerEval_id"].duplicated().any():
        raise ValueError("Duplicate IDs across language files")
    checksums = {p.name: sha256(p) for p in paths}
    return combined, checksums


def load_splits(split_path: Path | str) -> pd.DataFrame:
    """Load canonical fold assignments from TSV."""
    path = Path(split_path)
    if not path.is_file():
        raise FileNotFoundError(f"Split file not found: {path}")
    df = pd.read_csv(path, sep="\t", dtype={"fold": int, "StereoQueerEval_id": str})
    expected = {"StereoQueerEval_id", "language", "video_key", "fold"}
    if not expected.issubset(set(df.columns)):
        raise ValueError(f"Split file missing expected columns: {expected - set(df.columns)}")
    return df


def merge_data_and_splits(
    data_df: pd.DataFrame, splits_df: pd.DataFrame
) -> pd.DataFrame:
    """Merge dataset with fold assignments with strict 1:1 validation."""
    return data_df.merge(
        splits_df[["StereoQueerEval_id", "fold"]],
        on="StereoQueerEval_id",
        validate="one_to_one",
    )


def build_text(df: pd.DataFrame, variant: str) -> pd.Series:
    """Construct formatted text inputs based on the selected variant."""
    if variant == "comment":
        return df["yt_comment"]
    if variant == "title_comment":
        return "[TITLE] " + df["yt_title"] + " [COMMENT] " + df["yt_comment"]
    if variant == "full_context":
        return (
            "[TITLE] "
            + df["yt_title"]
            + " [DESCRIPTION] "
            + df["yt_description"]
            + " [COMMENT] "
            + df["yt_comment"]
        )
    raise ValueError(f"Unknown input variant: {variant}. Expected one of {INPUT_VARIANTS}")
