import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import pandas as pd
import logging

from src.utils import load_config, setup_logging, ensure_dir, set_seed
from src.visualization import plot_embeddings

def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    config = load_config()
    set_seed(config['random_seed'])
    
    parser = argparse.ArgumentParser(description="Visualize Embeddings")
    parser.add_argument("--features", type=str, help="Path to specific features CSV (optional)")
    args = parser.parse_args()

    figures_out_dir = config['paths']['figures_out_dir']
    from pathlib import Path
    Path(figures_out_dir).mkdir(parents=True, exist_ok=True)
    
    # If a specific feature file is provided, visualize only that
    feature_files = []
    if args.features:
        if os.path.exists(args.features):
            feature_files.append(args.features)
        else:
            logger.error(f"File not found: {args.features}")
            sys.exit(1)
    else:
        # Otherwise find all feature files in features/xlsr/
        features_dir = os.path.join(config['paths']['features_dir'], config['model']['type'])
        if os.path.exists(features_dir):
            for file in os.listdir(features_dir):
                if file.endswith('.csv'):
                    feature_files.append(os.path.join(features_dir, file))
                    
    if not feature_files:
        logger.error("No feature files found. Run extraction script first.")
        sys.exit(1)
        
    for file_path in feature_files:
        logger.info(f"Generating plots for {file_path}")
        df = pd.read_csv(file_path)
        
        # Identify feature columns
        feature_cols = [col for col in df.columns if col.startswith('feature_')]
        if not feature_cols:
            logger.warning(f"No feature columns found in {file_path}. Skipping.")
            continue
            
        # Prefix for output files, e.g., 'xlsr_readtext_layer4'
        prefix = os.path.splitext(os.path.basename(file_path))[0]
        
        plot_embeddings(df, feature_cols, figures_out_dir, prefix)
        logger.info(f"Finished plotting for {prefix}")

if __name__ == "__main__":
    main()
