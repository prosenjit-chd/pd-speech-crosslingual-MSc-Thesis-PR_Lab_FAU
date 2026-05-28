import os
import shutil
import zipfile
import logging

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)
    
    out_dir = "full_baseline_outputs_for_tomas"
    zip_filename = f"{out_dir}.zip"
    
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)
    
    # 1. Create the README
    readme_content = """# Full Baseline Outputs for Tomás

This package contains the complete first baseline results for the Multilingual Parkinson's Speech Detection project, comparing XLSR, Wav2Vec2, and WavLM.

## Contents
- **outputs/reports/**: Summaries for each individual model and the combined comprehensive comparison report.
- **outputs/tables/**: Raw layer-by-layer classification results and combined model comparisons (UAR, Accuracy, Sensitivity, Specificity, AUC).
- **outputs/figures/**: Visualizations (t-SNE and PCA) for all models showing feature separability.

*Note: For data privacy, raw audio files, private clinical metadata, dataset indexes with speaker IDs, and large raw embeddings have been strictly excluded from this deliverable.*
"""
    readme_path = os.path.join(out_dir, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)
    logger.info("Generated README.md")

    # Ensure internal directories exist in zip staging
    os.makedirs(os.path.join(out_dir, "outputs/reports"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "outputs/tables"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "outputs/figures"), exist_ok=True)

    # 2. Find and copy the safe files
    # We want specific reports, tables, and figures.
    
    # Reports
    reports_src = "outputs/reports"
    if os.path.exists(reports_src):
        for f in os.listdir(reports_src):
            if f.endswith(".md"):
                shutil.copy(os.path.join(reports_src, f), os.path.join(out_dir, "outputs/reports", f))
                
    # Tables
    tables_src = "outputs/tables"
    if os.path.exists(tables_src):
        for f in os.listdir(tables_src):
            if f.endswith(".csv") and ("classification_results_" in f or "model_comparison" in f):
                shutil.copy(os.path.join(tables_src, f), os.path.join(out_dir, "outputs/tables", f))
                
    # Figures
    figures_src = "outputs/figures"
    if os.path.exists(figures_src):
        for f in os.listdir(figures_src):
            if f.endswith(".png"):
                shutil.copy(os.path.join(figures_src, f), os.path.join(out_dir, "outputs/figures", f))
                
    # 3. Create the ZIP archive
    logger.info(f"Creating zip archive: {zip_filename}")
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(out_dir):
            for file in files:
                file_path = os.path.join(root, file)
                # Store relative to out_dir to avoid deep nesting
                arcname = os.path.relpath(file_path, os.path.dirname(out_dir))
                zipf.write(file_path, arcname)
                
    logger.info(f"Successfully packaged full safe deliverables into {out_dir}/ and {zip_filename}.")

if __name__ == "__main__":
    main()
