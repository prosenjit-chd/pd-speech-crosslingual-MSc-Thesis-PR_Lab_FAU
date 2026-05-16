"""
02_extract_embeddings.py

Extract layer-wise XLSR/Wav2Vec2/WavLM embeddings from a dataset index.

Example:
    python scripts/02_extract_embeddings.py \
        --index metadata/dataset_index_readtext.csv \
        --model xlsr \
        --layers 0 4 8 11 \
        --output-dir features/xlsr
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.audio_utils import load_audio
from src.embedding_extractor import (
    EmbeddingConfig,
    SpeechEmbeddingExtractor,
    embedding_to_feature_row,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract self-supervised speech embeddings.")
    parser.add_argument("--index", type=Path, required=True, help="Dataset index CSV.")
    parser.add_argument("--model", type=str, default="xlsr", choices=["xlsr", "wav2vec2", "wavlm"])
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 4, 8, 11])
    parser.add_argument("--output-dir", type=Path, default=Path("features/xlsr"))
    parser.add_argument("--task", type=str, default="readtext")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.index)
    if df.empty:
        raise ValueError("Dataset index is empty.")

    config = EmbeddingConfig(model_name=args.model, layers=tuple(args.layers))
    extractor = SpeechEmbeddingExtractor(config)

    rows_by_layer = {layer: [] for layer in args.layers}

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting embeddings"):
        waveform, sample_rate = load_audio(row["file_path"])
        embeddings = extractor.extract_from_waveform(waveform, sample_rate=sample_rate)

        metadata = {
            "file_path": row["file_path"],
            "speaker_id": row["speaker_id"],
            "language": row["language"],
            "task": row["task"],
            "label": row["label"],
            "dataset": row["dataset"],
        }

        for layer, embedding in embeddings.items():
            rows_by_layer[layer].append(embedding_to_feature_row(metadata, embedding))

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for layer, rows in rows_by_layer.items():
        output_path = args.output_dir / f"{args.model}_{args.task}_layer{layer}.csv"
        pd.DataFrame(rows).to_csv(output_path, index=False)
        print(f"Saved layer {layer} features to: {output_path}")


if __name__ == "__main__":
    main()
