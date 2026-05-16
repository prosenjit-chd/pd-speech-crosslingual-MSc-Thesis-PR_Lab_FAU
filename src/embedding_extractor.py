"""
embedding_extractor.py

Layer-wise embedding extraction using Hugging Face speech models.

This module supports the first technical milestone of the thesis:
extract self-supervised speech representations from Spanish/German PD speech.

Models planned:
- XLSR / Wav2Vec2 multilingual
- Wav2Vec2 base
- WavLM

No private audio files are included in this repository.

Author: Prosenjit Chowdhury
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np
import torch
from transformers import AutoModel, AutoProcessor


MODEL_REGISTRY: Dict[str, str] = {
    "xlsr": "facebook/wav2vec2-large-xlsr-53",
    "wav2vec2": "facebook/wav2vec2-base",
    "wavlm": "microsoft/wavlm-base",
}


@dataclass
class EmbeddingConfig:
    """Configuration for layer-wise embedding extraction."""

    model_name: str = "xlsr"
    layers: tuple[int, ...] = (0, 4, 8, 11)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    pooling: str = "mean"


class SpeechEmbeddingExtractor:
    """
    Extract hidden-state embeddings from pretrained speech models.

    The output for each layer is one fixed-size vector per audio file.
    Mean pooling over time is used by default.
    """

    def __init__(self, config: EmbeddingConfig) -> None:
        if config.model_name not in MODEL_REGISTRY:
            valid = ", ".join(MODEL_REGISTRY)
            raise ValueError(f"Unknown model '{config.model_name}'. Valid options: {valid}")

        self.config = config
        hf_model_id = MODEL_REGISTRY[config.model_name]

        self.processor = AutoProcessor.from_pretrained(hf_model_id)
        self.model = AutoModel.from_pretrained(hf_model_id, output_hidden_states=True)
        self.model.to(config.device)
        self.model.eval()

    @torch.no_grad()
    def extract_from_waveform(
        self,
        waveform: torch.Tensor,
        sample_rate: int = 16_000,
    ) -> Dict[int, np.ndarray]:
        """
        Extract selected layer embeddings from a waveform.

        Parameters
        ----------
        waveform:
            Tensor with shape [num_samples].
        sample_rate:
            Sampling rate. Expected: 16 kHz.

        Returns
        -------
        dict
            Mapping: layer index -> embedding vector.
        """
        inputs = self.processor(
            waveform.cpu().numpy(),
            sampling_rate=sample_rate,
            return_tensors="pt",
            padding=True,
        )

        inputs = {key: value.to(self.config.device) for key, value in inputs.items()}
        outputs = self.model(**inputs)

        hidden_states = outputs.hidden_states
        embeddings: Dict[int, np.ndarray] = {}

        for layer in self.config.layers:
            if layer >= len(hidden_states):
                raise IndexError(
                    f"Layer {layer} is not available. "
                    f"Model returned {len(hidden_states)} hidden-state tensors."
                )

            layer_tensor = hidden_states[layer].squeeze(0)

            if self.config.pooling == "mean":
                pooled = layer_tensor.mean(dim=0)
            elif self.config.pooling == "max":
                pooled = layer_tensor.max(dim=0).values
            else:
                raise ValueError(f"Unsupported pooling method: {self.config.pooling}")

            embeddings[layer] = pooled.detach().cpu().numpy().astype(np.float32)

        return embeddings


def embedding_to_feature_row(
    metadata: dict,
    embedding: np.ndarray,
) -> dict:
    """
    Combine metadata and a numerical embedding vector into one flat row.

    Example columns:
    speaker_id, language, label, feature_0, feature_1, ...
    """
    row = dict(metadata)
    for idx, value in enumerate(embedding):
        row[f"feature_{idx}"] = float(value)
    return row
