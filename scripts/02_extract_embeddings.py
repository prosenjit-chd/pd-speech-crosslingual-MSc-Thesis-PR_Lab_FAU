import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import pandas as pd
import logging
import numpy as np
from tqdm import tqdm

from src.utils import load_config, setup_logging, ensure_dir, get_device, set_seed
from src.audio_utils import load_and_preprocess_audio
from src.embedding_extractor import FeatureExtractor

def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    config = load_config()
    set_seed(config['random_seed'])
    
    parser = argparse.ArgumentParser(description="Extract XLSR embeddings")
    parser.add_argument("--model", type=str, default=config['model']['type'], help="Model type (e.g. xlsr)")
    parser.add_argument("--layers", type=int, nargs='+', default=config['model']['target_layers'], help="Layers to extract")
    parser.add_argument("--subset", type=int, default=None, help="Process only N files for a dry run")
    args = parser.parse_args()

    target_task = config['data']['target_task']
    index_file = os.path.join(config['paths']['metadata_out_dir'], f"dataset_index_{target_task}.csv")
    
    if not os.path.exists(index_file):
        logger.error(f"Dataset index not found at {index_file}. Run script 01 first.")
        sys.exit(1)
        
    df = pd.read_csv(index_file)
    if len(df) == 0:
        logger.error("Dataset index is empty.")
        sys.exit(1)
        
    if args.subset is not None:
        logger.info(f"Subset mode active: Limiting to {args.subset} files.")
        df = df.head(args.subset)
        
    features_out_dir = os.path.join(config['paths']['features_dir'], args.model)
    from pathlib import Path
    Path(features_out_dir).mkdir(parents=True, exist_ok=True)
    
    device = get_device()
    extractor = FeatureExtractor(model_name=config['model']['name'], device=device)
    target_sr = config['data']['target_sr']
    
    # We will collect features layer by layer
    layer_data = {layer: [] for layer in args.layers}
    valid_indices = []
    
    logger.info(f"Extracting features for {len(df)} files...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        file_path = row['file_path']
        waveform = load_and_preprocess_audio(file_path, target_sr=target_sr)
        
        if waveform is None:
            continue
            
        features = extractor.extract_features(waveform, target_layers=args.layers)
        if features is None:
            continue
            
        valid_indices.append(idx)
        for layer in args.layers:
            if layer in features:
                layer_data[layer].append(features[layer])
                
    # Save a CSV for each layer
    valid_df = df.iloc[valid_indices].reset_index(drop=True)
    
    for layer in args.layers:
        if len(layer_data[layer]) == 0:
            continue
            
        feature_matrix = np.array(layer_data[layer])
        num_features = feature_matrix.shape[1]
        feature_cols = [f"feature_{i}" for i in range(num_features)]
        
        feature_df = pd.DataFrame(feature_matrix, columns=feature_cols)
        final_df = pd.concat([valid_df, feature_df], axis=1)
        
        out_file = os.path.join(features_out_dir, f"{args.model}_{target_task}_layer{layer}.csv")
        final_df.to_csv(out_file, index=False)
        logger.info(f"Saved layer {layer} features to {out_file}")

if __name__ == "__main__":
    main()
