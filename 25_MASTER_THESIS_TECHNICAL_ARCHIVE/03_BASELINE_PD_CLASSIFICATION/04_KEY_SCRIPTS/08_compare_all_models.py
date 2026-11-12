import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import logging
import glob

from src.utils import load_config, setup_logging

def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    config = load_config()
    
    tables_dir = config['paths']['tables_out_dir']
    reports_dir = config['paths']['reports_out_dir']
    
    # 1. Collect all model classification results
    result_files = glob.glob(os.path.join(tables_dir, "classification_results_*.csv"))
    
    if not result_files:
        logger.error("No classification results found. Please run baseline for at least one model.")
        return
        
    all_dfs = []
    for f in result_files:
        try:
            df = pd.read_csv(f)
            # Ensure model column exists just in case old files lack it
            if 'model' not in df.columns:
                prefix = os.path.basename(f).replace("classification_results_", "").split("_")[0]
                df.insert(0, 'model', prefix)
            all_dfs.append(df)
        except Exception as e:
            logger.error(f"Failed to read {f}: {e}")
            
    full_df = pd.concat(all_dfs, ignore_index=True)
    
    # Save the combined comparison
    out_csv = os.path.join(tables_dir, "full_model_comparison.csv")
    full_df.to_csv(out_csv, index=False)
    logger.info(f"Saved full model comparison to {out_csv}")
    
    # 2. Generate Full Summary
    md_content = "# Full Baseline Model Comparison Summary\n\n"
    
    md_content += "## 1. Dataset Summary\n"
    md_content += f"- **Task**: {config['data']['target_task']}\n"
    md_content += f"- **Total Recordings**: 276\n"
    md_content += f"- **Languages**: Spanish (100), German (176)\n"
    md_content += f"- **Labels**: PD (138), HC (138)\n\n"
    
    available_models = full_df['model'].unique().tolist()
    md_content += "## 2. Models Evaluated\n"
    for m in available_models:
        md_content += f"- {m}\n"
    md_content += "\n"
    
    md_content += "## 3. Layers & Classifiers\n"
    md_content += f"- **Layers Evaluated**: {config['model']['target_layers']}\n"
    md_content += f"- **Classifiers**: Linear SVM, Logistic Regression\n\n"
    
    md_content += "## 4. Best Results Per Scenario\n\n"
    
    scenarios = [
        ("Spanish", "Spanish", "Best Within-Language (Spanish)"),
        ("German", "German", "Best Within-Language (German)"),
        ("Spanish", "German", "Best Cross-Language (Spanish -> German)"),
        ("German", "Spanish", "Best Cross-Language (German -> Spanish)"),
        ("Spanish+German", "Spanish+German", "Best Combined Language (Spanish+German)")
    ]
    
    for train_l, test_l, title in scenarios:
        sub_df = full_df[(full_df['train_language'] == train_l) & (full_df['test_language'] == test_l)]
        if not sub_df.empty:
            best_row = sub_df.loc[sub_df['uar'].idxmax()]
            md_content += f"### {title}\n"
            md_content += f"- **Best UAR**: {best_row['uar']:.4f}\n"
            md_content += f"- **Model**: {best_row['model']}\n"
            md_content += f"- **Layer**: {best_row['layer']}\n"
            md_content += f"- **Classifier**: {best_row['classifier']}\n"
            md_content += f"- **Accuracy**: {best_row['accuracy']:.4f}\n"
            md_content += f"- **Sensitivity**: {best_row['sensitivity']:.4f}\n"
            md_content += f"- **Specificity**: {best_row['specificity']:.4f}\n"
            md_content += f"- **AUC**: {best_row['auc']:.4f}\n\n"
            
    md_content += "## 5. Model Rankings & Interpretation\n\n"
    
    # Rank overall models by their max UAR across all scenarios
    overall_best = full_df.loc[full_df['uar'].idxmax()]
    md_content += f"**Overall Best Performing Configuration**:\n"
    md_content += f"- Model: {overall_best['model']}, Layer: {overall_best['layer']}, Scenario: {overall_best['train_language']}->{overall_best['test_language']} (UAR: {overall_best['uar']:.4f})\n\n"
    
    md_content += "**Language Mismatch Conclusion**:\n"
    # Simple check for mismatch drop
    es_es = full_df[(full_df['train_language'] == 'Spanish') & (full_df['test_language'] == 'Spanish')]['uar'].max()
    es_de = full_df[(full_df['train_language'] == 'Spanish') & (full_df['test_language'] == 'German')]['uar'].max()
    if pd.notna(es_es) and pd.notna(es_de):
        drop = es_es - es_de
        if drop > 0.05:
            md_content += f"Yes, a domain mismatch is clearly visible. For instance, the best Spanish->Spanish UAR is {es_es:.4f}, but drops to {es_de:.4f} when transferring to German.\n\n"
        else:
            md_content += f"Language mismatch seems minimal. Spanish->Spanish UAR is {es_es:.4f} and Spanish->German is {es_de:.4f}.\n\n"
    else:
        md_content += "Not enough data to calculate mismatch drop.\n\n"
        
    md_content += "**Final Recommendation for Baseline**:\n"
    md_content += f"The **{overall_best['model']}** model provides the strongest overall representations for speech classification in this dataset. This model should be used as the definitive reference baseline prior to introducing Voice Conversion.\n\n"
    
    out_md = os.path.join(reports_dir, "full_baseline_model_comparison_summary.md")
    with open(out_md, 'w', encoding='utf-8') as f:
        f.write(md_content)
        
    logger.info(f"Saved summary report to {out_md}")

if __name__ == "__main__":
    main()
