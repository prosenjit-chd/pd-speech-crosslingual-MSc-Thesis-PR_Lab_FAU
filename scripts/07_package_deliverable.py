import os
import shutil
import zipfile
import logging

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)
    
    out_dir = "baseline_outputs_for_tomas"
    zip_filename = f"{out_dir}.zip"
    
    # Recreate the output directory safely
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    
    # 1. Create the README
    readme_content = """# Baseline Outputs for Tomás

This package contains the first baseline results for the Multilingual Parkinson's Speech Detection project.

## Contents
- **baseline_summary.md**: A comprehensive overview of the dataset, experimental setup, best results, and cross-lingual performance drops.
- **model_layer_comparison.csv**: Aggregated results for all cross-lingual classification scenarios (UAR, Accuracy, Sensitivity, Specificity, AUC).
- **classification_results_*.csv**: Detailed classification results for each individual XLSR layer (0, 4, 8, 11).
- **tsne_*.png / pca_*.png**: Visualizations showing feature separability by clinical label and by language.

*Note: For data privacy, raw audio files, private clinical metadata, and the raw embeddings have been strictly excluded from this deliverable.*
"""
    readme_path = os.path.join(out_dir, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)
    logger.info("Generated README.md")

    # 2. Copy the safe files
    safe_files = [
        ("outputs/reports/baseline_summary.md", "baseline_summary.md"),
        ("outputs/tables/model_layer_comparison.csv", "model_layer_comparison.csv"),
    ]
    
    # Add layer CSVs
    if os.path.exists("outputs/tables"):
        for file in os.listdir("outputs/tables"):
            if file.startswith("classification_results_") and file.endswith(".csv"):
                safe_files.append((os.path.join("outputs/tables", file), file))
                
    # Add plots
    if os.path.exists("outputs/figures"):
        for file in os.listdir("outputs/figures"):
            if file.endswith(".png"):
                safe_files.append((os.path.join("outputs/figures", file), file))
                
    for src, dst_name in safe_files:
        if os.path.exists(src):
            shutil.copy(src, os.path.join(out_dir, dst_name))
            logger.info(f"Copied {src}")
        else:
            logger.warning(f"File not found, skipping: {src}")
            
    # 3. Create the ZIP archive
    logger.info(f"Creating zip archive: {zip_filename}")
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(out_dir):
            for file in files:
                file_path = os.path.join(root, file)
                # Store the file in the zip relative to the out_dir
                arcname = os.path.relpath(file_path, os.path.dirname(out_dir))
                zipf.write(file_path, arcname)
                
    logger.info(f"Successfully packaged safe deliverables into {out_dir}/ and {zip_filename}.")

if __name__ == "__main__":
    main()
