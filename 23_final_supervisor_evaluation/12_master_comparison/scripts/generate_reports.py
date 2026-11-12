import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

base_dir = r"C:\pd-speech-crosslingual\23_final_supervisor_evaluation"
class_res = os.path.join(base_dir, "05_classification", "summaries", "all_classification_results.csv")
out_tables = os.path.join(base_dir, "12_master_comparison", "tables")
out_figs = os.path.join(base_dir, "12_master_comparison", "figures")
out_rep = os.path.join(base_dir, "12_master_comparison", "report")
os.makedirs(out_tables, exist_ok=True)
os.makedirs(out_figs, exist_ok=True)
os.makedirs(out_rep, exist_ok=True)

def generate_reports():
    if not os.path.exists(class_res):
        print("Classification results not yet available. Please run run_classification.py first.")
        return
        
    df = pd.read_csv(class_res)
    
    # original_vs_converted_by_classifier
    # For A: Train DE, Test ES
    df_es_orig = df[(df['scenario'] == 'DE -> ES')]
    df_es_conv = df[(df['scenario'] == 'DE -> ES_to_DE')]
    
    # For B: Train ES, Test DE
    df_de_orig = df[(df['scenario'] == 'ES -> DE')]
    df_de_conv = df[(df['scenario'] == 'ES -> DE_to_ES')]
    
    # Merge on classifier, model, layer
    merge_cols = ['classifier', 'representation_model', 'layer']
    
    # A
    merged_a = pd.merge(df_es_orig, df_es_conv, on=merge_cols, suffixes=('_orig', '_conv'))
    merged_a['Delta_UAR'] = merged_a['UAR_conv'] - merged_a['UAR_orig']
    merged_a['Direction'] = 'ES_to_DE'
    
    # B
    merged_b = pd.merge(df_de_orig, df_de_conv, on=merge_cols, suffixes=('_orig', '_conv'))
    merged_b['Delta_UAR'] = merged_b['UAR_conv'] - merged_b['UAR_orig']
    merged_b['Direction'] = 'DE_to_ES'
    
    merged_all = pd.concat([merged_a, merged_b])
    merged_all.to_csv(os.path.join(out_tables, "original_vs_converted_full_matrix.csv"), index=False)
    
    # Group by classifier
    by_clf = merged_all.groupby('classifier')[['UAR_orig', 'UAR_conv', 'Delta_UAR']].mean().reset_index()
    by_clf.to_csv(os.path.join(out_tables, "original_vs_converted_by_classifier.csv"), index=False)
    
    # Group by model
    by_mod = merged_all.groupby('representation_model')[['UAR_orig', 'UAR_conv', 'Delta_UAR']].mean().reset_index()
    by_mod.to_csv(os.path.join(out_tables, "original_vs_converted_by_model.csv"), index=False)

    # Group by layer
    by_lay = merged_all.groupby('layer')[['UAR_orig', 'UAR_conv', 'Delta_UAR']].mean().reset_index()
    by_lay.to_csv(os.path.join(out_tables, "original_vs_converted_by_layer.csv"), index=False)
    
    # Figures
    # 1. Delta UAR by classifier
    plt.figure(figsize=(10, 6))
    sns.barplot(data=by_clf, x='classifier', y='Delta_UAR')
    plt.title('Average Delta UAR by Classifier (Converted - Original)')
    plt.savefig(os.path.join(out_figs, "Delta_UAR_by_Classifier.png"), dpi=300)
    plt.close()
    
    # 2. UAR by Model
    plt.figure(figsize=(10, 6))
    melted = by_mod.melt(id_vars='representation_model', value_vars=['UAR_orig', 'UAR_conv'])
    sns.barplot(data=melted, x='representation_model', y='value', hue='variable')
    plt.title('Original vs Converted UAR by Representation Model')
    plt.savefig(os.path.join(out_figs, "UAR_by_Model.png"), dpi=300)
    plt.close()
    
    # Write master report
    with open(os.path.join(out_rep, "FINAL_SUPERVISOR_EVALUATION_REPORT.md"), "w") as f:
        f.write("# Final Supervisor Evaluation Report\n\n")
        f.write("## 1. Classification Results\n")
        f.write("Summary of UAR changes after voice conversion:\n\n")
        f.write(by_clf.to_markdown(index=False))
        f.write("\n\n")
        f.write("## 2. Conclusion\n")
        f.write("Voice conversion shows classifier-dependent and model-dependent behavior on cross-language PD detection tasks. Refer to the tables and figures for specific performance changes.\n")
        
    print("Master comparison generated.")

if __name__ == "__main__":
    generate_reports()
