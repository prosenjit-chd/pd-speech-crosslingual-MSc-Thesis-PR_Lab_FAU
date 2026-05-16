"""
dataset_index.py

Builds a dataset index for multilingual Parkinson's disease speech datasets.

This module creates a single metadata table where every row corresponds to
one audio file and includes language, task, label, and speaker identifier.

The index is the first reproducibility step of the thesis pipeline:
audio files -> dataset index -> embeddings -> visualization/classification.

Author: Prosenjit Chowdhury
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd


SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".m4a", ".ogg", ".aac"}


@dataclass(frozen=True)
class AudioRecord:
    """One row in the dataset index."""

    file_path: str
    speaker_id: str
    language: str
    task: str
    label: str
    dataset: str


def infer_label_from_path(path: Path) -> Optional[str]:
    """
    Infer PD/HC label from folder names.

    Expected folder names can vary between datasets:
    - Spanish: PD / HC
    - German: pd / control / hc / ips

    Returns
    -------
    "PD", "HC", or None
    """
    parts = {p.lower() for p in path.parts}

    if "pd" in parts or "ips" in parts or "parkinson" in parts:
        return "PD"

    if "hc" in parts or "control" in parts or "healthy" in parts:
        return "HC"

    return None


def infer_speaker_id(path: Path) -> str:
    """
    Infer a speaker ID from the file name.

    This function intentionally uses a conservative generic strategy because
    dataset naming conventions can differ. It can later be replaced by
    metadata-based speaker mapping when the final metadata sheet is available.
    """
    stem = path.stem

    for sep in ["_", "-", "."]:
        if sep in stem:
            return stem.split(sep)[0]

    return stem


def scan_audio_files(
    root_dir: str | Path,
    language: str,
    task: str,
    dataset: str,
) -> List[AudioRecord]:
    """
    Recursively scan a folder and create AudioRecord entries.

    Parameters
    ----------
    root_dir:
        Folder containing audio files.
    language:
        Language label, for example "Spanish" or "German".
    task:
        Speech task label, for example "readtext".
    dataset:
        Dataset name/source.
    """
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset folder does not exist: {root}")

    records: List[AudioRecord] = []

    for audio_path in sorted(root.rglob("*")):
        if audio_path.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
            continue

        label = infer_label_from_path(audio_path)
        if label is None:
            # Keep the scan strict to avoid silently mixing unknown labels.
            continue

        records.append(
            AudioRecord(
                file_path=str(audio_path.resolve()),
                speaker_id=infer_speaker_id(audio_path),
                language=language,
                task=task,
                label=label,
                dataset=dataset,
            )
        )

    return records


def build_dataset_index(records: Iterable[AudioRecord]) -> pd.DataFrame:
    """
    Convert records to a clean DataFrame and validate required columns.
    """
    df = pd.DataFrame([asdict(record) for record in records])

    required = ["file_path", "speaker_id", "language", "task", "label", "dataset"]
    if df.empty:
        return pd.DataFrame(columns=required)

    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required dataset index columns: {missing}")

    df = df[required].copy()
    df["label"] = df["label"].str.upper()
    df["language"] = df["language"].str.strip()
    df["task"] = df["task"].str.strip()

    return df.sort_values(["language", "label", "speaker_id", "file_path"]).reset_index(drop=True)


def summarize_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize number of files and speakers per language/task/label.
    """
    if df.empty:
        return pd.DataFrame()

    return (
        df.groupby(["language", "task", "label"])
        .agg(
            n_files=("file_path", "count"),
            n_speakers=("speaker_id", "nunique"),
        )
        .reset_index()
        .sort_values(["language", "task", "label"])
    )


def save_index(df: pd.DataFrame, output_csv: str | Path) -> None:
    """
    Save the dataset index to CSV.
    """
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
