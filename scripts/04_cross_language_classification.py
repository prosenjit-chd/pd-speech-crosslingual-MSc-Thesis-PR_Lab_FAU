import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse
import pandas as pd
import logging

from src.utils import load_config, setup_logging, ensure_dir, set_seed
from src.classification import run_classification_scenarios

def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    config = load_config()
    set_seed(config['random_seed'])
    
    parser = argparse.ArgumentParser(description="Cross-language Classification")
    parser.add_argument("--features", type=str, help="Path to specific features CSV (optional)")
    args = parser.parse_args()

    tables_out_dir = config['paths']['tables_out_dir']
    from pathlib import Path
    Path(tables_out_dir).mkdir(parents=True, exist_ok=True)
    
    classifiers = config['evaluation']['classifiers']
    outer_folds = config['evaluation']['outer_folds']
    inner_folds = config['evaluation']['inner_folds']
    
    # If a specific feature file is provided, process only that
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
        
    all_results = []
        
    for file_path in feature_files:
        logger.info(f"Running classification on {file_path}")
        df = pd.read_csv(file_path)
        
        feature_cols = [col for col in df.columns if col.startswith('feature_')]
        if not feature_cols:
            logger.warning(f"No feature columns found in {file_path}. Skipping.")
            continue
            
        prefix = os.path.splitext(os.path.basename(file_path))[0]
        
        try:
            parts = prefix.split('_')
            model_name = parts[0]
            layer = parts[-1].replace('layer', '')
        except:
            model_name = config['model']['type']
            layer = "unknown"
            
        results_df = run_classification_scenarios(df, feature_cols, classifiers, outer_folds, inner_folds)
        
        if not results_df.empty:
            results_df.insert(0, 'layer', layer)
            results_df.insert(0, 'model', model_name)
            
            out_file = os.path.join(tables_out_dir, f"classification_results_{prefix}.csv")
            results_df.to_csv(out_file, index=False)
            logger.info(f"Saved results to {out_file}")
            
            all_results.append(results_df)

    if all_results:
        final_comparison = pd.concat(all_results, ignore_index=True)
        comparison_out = os.path.join(tables_out_dir, "model_layer_comparison.csv")
        final_comparison.to_csv(comparison_out, index=False)
        logger.info(f"Saved full comparison to {comparison_out}")

if __name__ == "__main__":
    main()
