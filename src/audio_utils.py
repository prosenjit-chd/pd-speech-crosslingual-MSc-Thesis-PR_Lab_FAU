"""
audio_utils.py

Reusable audio loading and preprocessing utilities for multilingual
Parkinson's disease speech experiments.

The functions in this module intentionally do not depend on the private
medical datasets. They provide a clean, reusable foundation for loading
audio files from Spanish/German/Czech PD speech corpora.

Author: Prosenjit Chowdhury
Thesis: Multilingual Parkinson's Disease Detection from Speech
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torchaudio


TARGET_SAMPLE_RATE = 16_000


class AudioLoadError(RuntimeError):
    """Raised when an audio file cannot be loaded or converted safely."""


def load_audio(
    file_path: str | Path,
    target_sample_rate: int = TARGET_SAMPLE_RATE,
    mono: bool = True,
    normalize: bool = True,
) -> Tuple[torch.Tensor, int]:
    """
    Load an audio file and convert it into the format expected by
    self-supervised speech models such as Wav2Vec2, XLSR, and WavLM.

    Parameters
    ----------
    file_path:
        Path to the audio file.
    target_sample_rate:
        Target sampling rate. Most Hugging Face speech models expect 16 kHz.
    mono:
        If True, convert multi-channel audio to mono.
    normalize:
        If True, normalize waveform amplitude to [-1, 1].

    Returns
    -------
    waveform:
        Tensor with shape [num_samples].
    sample_rate:
        Sampling rate after resampling.
    """
    path = Path(file_path)

    if not path.exists():
        raise AudioLoadError(f"Audio file not found: {path}")

    try:
        waveform, sample_rate = torchaudio.load(str(path))
    except Exception as exc:
        raise AudioLoadError(f"Could not load audio file: {path}") from exc

    if waveform.numel() == 0:
        raise AudioLoadError(f"Audio file is empty: {path}")

    if mono and waveform.ndim == 2:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate,
            new_freq=target_sample_rate,
        )
        waveform = resampler(waveform)
        sample_rate = target_sample_rate

    waveform = waveform.squeeze(0).float()

    if normalize:
        max_abs = torch.max(torch.abs(waveform))
        if max_abs > 0:
            waveform = waveform / max_abs

    return waveform, sample_rate


def audio_duration_seconds(file_path: str | Path) -> float:
    """
    Return the duration of an audio file in seconds without loading the
    full waveform into memory.

    Parameters
    ----------
    file_path:
        Path to the audio file.

    Returns
    -------
    float
        Duration in seconds.
    """
    path = Path(file_path)

    if not path.exists():
        raise AudioLoadError(f"Audio file not found: {path}")

    try:
        info = torchaudio.info(str(path))
    except Exception as exc:
        raise AudioLoadError(f"Could not inspect audio file: {path}") from exc

    return float(info.num_frames) / float(info.sample_rate)


def waveform_to_numpy(waveform: torch.Tensor) -> np.ndarray:
    """
    Convert a PyTorch waveform tensor into a NumPy array.

    This is useful for plotting, logging, or compatibility with classical
    machine learning libraries.
    """
    return waveform.detach().cpu().numpy().astype(np.float32)
