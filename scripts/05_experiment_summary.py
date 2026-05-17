import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import logging
from src.utils import load_config, setup_logging, ensure_dir

def main():
    setup_logging()
    logger = logging.getLogger(__name__)
    config = load_config()
    
    tables_dir = config['paths']['tables_out_dir']
    reports_dir = config['paths']['reports_out_dir']
    from pathlib import Path
    Path(reports_dir).mkdir(parents=True, exist_ok=True)
    
    comp_file = os.path.join(tables_dir, "model_layer_comparison.csv")
    if not os.path.exists(comp_file):
        logger.error(f"Comparison file not found: {comp_file}")
        return
        
    df = pd.read_csv(comp_file)
    
    # Load dataset index for summary
    index_file = os.path.join(config['paths']['metadata_out_dir'], f"dataset_index_{config['data']['target_task']}.csv")
    if os.path.exists(index_file):
        df_index = pd.read_csv(index_file)
        total_rec = len(df_index)
        es_rec = len(df_index[df_index['language'] == 'Spanish'])
        de_rec = len(df_index[df_index['language'] == 'German'])
        pd_rec = len(df_index[df_index['label'].str.upper() == 'PD'])
        hc_rec = len(df_index[df_index['label'].str.upper() == 'HC'])
    else:
        total_rec = es_rec = de_rec = pd_rec = hc_rec = "Unknown"

    md_content = "# Multilingual Parkinson's Speech Detection Baseline\n\n"
    md_content += "## 1. Dataset Summary\n"
    md_content += f"- **Task**: {config['data']['target_task']}\n"
    md_content += f"- **Total Recordings**: {total_rec}\n"
    md_content += f"- **Languages**: Spanish ({es_rec}), German ({de_rec})\n"
    md_content += f"- **Labels**: PD ({pd_rec}), HC ({hc_rec})\n\n"
    
    md_content += "## 2. Experimental Setup\n"
    md_content += f"- **Model Used**: {config['model']['name']}\n"
    md_content += f"- **Layers Evaluated**: {config['model']['target_layers']}\n"
    md_content += f"- **Classifiers Used**: {config['evaluation']['classifiers']}\n\n"
    
    md_content += "## 3. Best Performing Configurations (by UAR)\n\n"
    
    scenarios = df[['train_language', 'test_language']].drop_duplicates()
    
    best_overall_row = df.loc[df['uar'].idxmax()]
    md_content += f"**Best Overall Layer**: Layer {best_overall_row['layer']} (UAR: {best_overall_row['uar']:.4f} via {best_overall_row['train_language']}->{best_overall_row['test_language']})\n\n"

    for _, row in scenarios.iterrows():
        train_l = row['train_language']
        test_l = row['test_language']
        sub_df = df[(df['train_language'] == train_l) & (df['test_language'] == test_l)]
        
        if not sub_df.empty:
            best_row = sub_df.loc[sub_df['uar'].idxmax()]
            
            # Specific titles for Tomas' requested sections
            if train_l == test_l and train_l in ['Spanish', 'German']:
                md_content += f"### Best Within-Language Result ({train_l} → {test_l})\n"
            elif train_l == 'Spanish' and test_l == 'German':
                md_content += f"### Cross-lingual: Spanish to German\n"
            elif train_l == 'German' and test_l == 'Spanish':
                md_content += f"### Cross-lingual: German to Spanish\n"
            else:
                md_content += f"### {train_l} → {test_l}\n"
                
            md_content += f"- **UAR**: {best_row['uar']:.4f}\n"
            md_content += f"- **Model/Layer**: {best_row['model']} / Layer {best_row['layer']}\n"
            md_content += f"- **Classifier**: {best_row['classifier']}\n"
            md_content += f"- **Sensitivity**: {best_row['sensitivity']:.4f}\n"
            md_content += f"- **Specificity**: {best_row['specificity']:.4f}\n"
            md_content += f"- **AUC**: {best_row['auc']:.4f}\n\n"
            
    md_content += "## 4. Interpretation on Language Mismatch\n"
    
    # Simple logic to determine mismatch: compare cross UAR to within UAR
    try:
        es_es_uar = df[(df['train_language'] == 'Spanish') & (df['test_language'] == 'Spanish')]['uar'].max()
        es_de_uar = df[(df['train_language'] == 'Spanish') & (df['test_language'] == 'German')]['uar'].max()
        drop = es_es_uar - es_de_uar
        if drop > 0.05:
            interp = f"Yes, there appears to be a notable domain mismatch. Training on Spanish and testing on German caused a performance drop of {drop:.4f} UAR compared to within-language testing."
        else:
            interp = f"Language mismatch appears minimal or manageable. Training on Spanish and testing on German only caused a UAR difference of {drop:.4f} compared to within-language."
    except:
        interp = "Could not automatically interpret mismatch due to missing data."
        
    md_content += f"{interp}\n\n"

    md_content += "## 5. Generated Deliverable Files\n"
    md_content += "- `metadata/dataset_index_readtext.csv` (Full dataset index)\n"
    md_content += "- `features/xlsr/*.csv` (Layer-wise embedded features)\n"
    md_content += "- `outputs/figures/tsne_*.png` (t-SNE plots by PD/HC label and language)\n"
    md_content += "- `outputs/figures/pca_*.png` (PCA plots by PD/HC label and language)\n"
    md_content += "- `outputs/tables/classification_results_*.csv` (Raw result tables per layer)\n"
    md_content += "- `outputs/tables/model_layer_comparison.csv` (Aggregated cross-lingual classification results)\n\n"
    
    md_content += "## 6. Full Results Table\n\n"
    
    md_df = df.copy()
    for col in ['uar', 'accuracy', 'sensitivity', 'specificity', 'auc']:
        md_df[col] = md_df[col].apply(lambda x: f"{x:.4f}")
        
    md_content += md_df.to_markdown(index=False)
    
    out_file = os.path.join(reports_dir, "baseline_summary.md")
    with open(out_file, 'w', encoding='utf-8') as f:
        f.write(md_content)
        
    logger.info(f"Summary written to {out_file}")

if __name__ == "__main__":
    main()
