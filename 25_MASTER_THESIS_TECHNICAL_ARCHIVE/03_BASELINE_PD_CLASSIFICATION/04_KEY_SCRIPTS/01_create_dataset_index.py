import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import logging
from src.utils import load_config, setup_logging, ensure_dir
from src.dataset_index import create_dataset_index

def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    
    config = load_config()
    input_dir = config['paths']['input_dir']
    metadata_out_dir = config['paths']['metadata_out_dir']
    target_task = config['data']['target_task']
    
    from pathlib import Path
    Path(metadata_out_dir).mkdir(parents=True, exist_ok=True)
    out_file = os.path.join(metadata_out_dir, f"dataset_index_{target_task}.csv")
    
    logger.info(f"Scanning {input_dir} for task: {target_task}")
    df = create_dataset_index(input_dir, target_task=target_task)
    
    if len(df) > 0:
        df.to_csv(out_file, index=False)
        logger.info(f"Saved dataset index to {out_file}")
    else:
        logger.error("No data found. Index not created.")
        sys.exit(1)

if __name__ == "__main__":
    main()
