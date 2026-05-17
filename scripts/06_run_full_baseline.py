import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import subprocess
import argparse
import logging
from src.utils import setup_logging, ensure_project_directories

def run_script(script_path):
    logger = logging.getLogger(__name__)
    logger.info(f"========================================")
    logger.info(f"Running {script_path}")
    logger.info(f"========================================")
    
    result = subprocess.run([sys.executable, script_path], capture_output=False)
    if result.returncode != 0:
        logger.error(f"Script {script_path} failed with return code {result.returncode}")
        return False
    return True

def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Initialize project directories
    ensure_project_directories()
    
    parser = argparse.ArgumentParser(description="Run Full Baseline")
    parser.add_argument("--subset", type=int, default=None, help="Process only N files for a dry run")
    args = parser.parse_args()
    
    logger.info("Starting Full Baseline Pipeline")
    
    # Base scripts
    script_01 = "scripts/01_create_dataset_index.py"
    
    if args.subset:
        script_02 = f"scripts/02_extract_embeddings.py --subset {args.subset}"
    else:
        script_02 = "scripts/02_extract_embeddings.py"
        
    scripts = [
        script_01,
        script_02,
        "scripts/03_visualize_embeddings.py",
        "scripts/04_cross_language_classification.py",
        "scripts/05_experiment_summary.py"
    ]
    
    for script_cmd in scripts:
        # Split command to handle args properly if any
        cmd_parts = script_cmd.split()
        script_path = cmd_parts[0]
        
        if not os.path.exists(script_path):
            logger.error(f"Script not found: {script_path}")
            return
            
        # Run script with arguments
        result = subprocess.run([sys.executable] + cmd_parts, capture_output=False)
        if result.returncode != 0:
            logger.error(f"Pipeline halted due to error in {script_path}.")
            return
            
    logger.info("Full Baseline Pipeline completed successfully.")

if __name__ == "__main__":
    main()
