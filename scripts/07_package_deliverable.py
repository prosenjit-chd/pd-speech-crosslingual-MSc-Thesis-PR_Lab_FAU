import os
import zipfile
import logging

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)
    
    zip_filename = "baseline_outputs_for_tomas.zip"
    
    files_to_include = [
        "metadata/dataset_index_readtext.csv",
        "outputs/reports/baseline_summary.md",
        "outputs/tables/model_layer_comparison.csv",
    ]
    
    # Add all layer classification tables
    if os.path.exists("outputs/tables"):
        for file in os.listdir("outputs/tables"):
            if file.startswith("classification_results_") and file.endswith(".csv"):
                files_to_include.append(os.path.join("outputs/tables", file))
                
    # Add all figures
    if os.path.exists("outputs/figures"):
        for file in os.listdir("outputs/figures"):
            if file.endswith(".png"):
                files_to_include.append(os.path.join("outputs/figures", file))
                
    logger.info(f"Creating zip archive: {zip_filename}")
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in files_to_include:
            if os.path.exists(file):
                zipf.write(file)
                logger.info(f"Added {file}")
            else:
                logger.warning(f"File not found, skipping: {file}")
                
    logger.info("Successfully packaged safe deliverables. You can now share this ZIP with Tomas.")

if __name__ == "__main__":
    main()
