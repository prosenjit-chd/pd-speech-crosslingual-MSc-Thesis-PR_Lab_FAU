import pandas as pd
import numpy as np
import os
from sklearn.metrics import recall_score
from scipy import stats

base_dir = "/mnt/c/pd-speech-crosslingual/23_final_supervisor_evaluation"
out_dir = os.path.join(base_dir, "13_thesis_ready")
os.makedirs(out_dir, exist_ok=True)
manifest_dir = os.path.join(base_dir, "14_supervisor_package")
os.makedirs(manifest_dir, exist_ok=True)

def uar(y_true, y_pred):
    return recall_score(y_true, y_pred, average='macro')

def compute_bootstrap_ci(y_true, y_pred, n_bootstraps=1000, random_state=42):
    np.random.seed(random_state)
    uars = []
    indices = np.arange(len(y_true))
    for _ in range(n_bootstraps):
        boot_idx = np.random.choice(indices, size=len(indices), replace=True)
        # ensure both classes are in bootstrap sample to avoid warnings/NaNs
        if len(np.unique(y_true[boot_idx])) > 1:
            uars.append(uar(y_true[boot_idx], y_pred[boot_idx]))
    if not uars:
        return 0.0, 0.0
    return np.percentile(uars, 2.5), np.percentile(uars, 97.5)

def bh_fdr(p_values):
    n = len(p_values)
    if n == 0: return np.array([])
    ranked_p_indices = np.argsort(p_values)
    ranked_p = np.array(p_values)[ranked_p_indices]
    q_values = np.zeros(n)
    for i in range(n):
        q = (ranked_p[i] * n) / (i + 1)
        q_values[i] = q
    for i in range(n-2, -1, -1):
        q_values[i] = min(q_values[i], q_values[i+1])
    orig_order_q = np.zeros(n)
    orig_order_q[ranked_p_indices] = q_values
    return np.clip(orig_order_q, 0, 1)

print("Loading predictions...")
preds_276 = pd.read_csv(os.path.join(base_dir, "05_classification", "predictions", "sample_level_predictions_NESTED_CV_COMPACT.csv"))

print("Computing scenario performance and CIs...")
# Let's find the best model/layer/classifier for each scenario in N=276
# We group by scenario, model, layer, classifier and calculate exact UAR from predictions
results = []
for (scenario, model, layer, clf), group in preds_276.groupby(['scenario', 'model', 'layer', 'classifier']):
    y_true = group['true_label'].values
    y_pred = group['predicted_label'].values
    current_uar = uar(y_true, y_pred)
    results.append({
        'scenario': scenario,
        'model': model,
        'layer': layer,
        'classifier': clf,
        'uar': current_uar
    })

df_res = pd.DataFrame(results)
best_configs = df_res.loc[df_res.groupby('scenario')['uar'].idxmax()]

# Now we compute CIs and perform paired tests between original and converted for the best models
# E.g. DE_orig vs DE_conv, ES_orig vs ES_conv
p_values = []
test_names = []

final_table = []
for idx, row in best_configs.iterrows():
    scenario = row['scenario']
    group = preds_276[(preds_276['scenario'] == scenario) & 
                      (preds_276['model'] == row['model']) & 
                      (preds_276['layer'] == row['layer']) & 
                      (preds_276['classifier'] == row['classifier'])]
    y_true = group['true_label'].values
    y_pred = group['predicted_label'].values
    ci_low, ci_high = compute_bootstrap_ci(y_true, y_pred)
    
    final_table.append({
        'Scenario': scenario,
        'Best_Model': row['model'],
        'Best_Layer': row['layer'],
        'Best_Classifier': row['classifier'],
        'UAR': round(row['uar'], 4),
        'CI_95_Low': round(ci_low, 4),
        'CI_95_High': round(ci_high, 4)
    })

df_final = pd.DataFrame(final_table)
df_final.to_csv(os.path.join(out_dir, "Table1_Best_Classifiers_CIs_N276.csv"), index=False)
df_final.to_markdown(os.path.join(out_dir, "Table1_Best_Classifiers_CIs_N276.md"), index=False)

# McNemar's test for paired proportions or simply independent t-tests on bootstraps?
# Standard for ML is McNemar's test for paired classification.
def mcnemar_test(y_true, y_pred1, y_pred2):
    # b = 1 right, 2 wrong; c = 1 wrong, 2 right
    b = np.sum((y_pred1 == y_true) & (y_pred2 != y_true))
    c = np.sum((y_pred1 != y_true) & (y_pred2 == y_true))
    if b + c == 0:
        return 1.0
    statistic = ((np.abs(b - c) - 1)**2) / (b + c)
    p = stats.distributions.chi2.sf(statistic, 1)
    return p

# Compare Orig vs Converted for the best configuration of the 'Converted' scenario
print("Computing pairwise statistical tests...")
pairs = [
    ("DE_train__ES_original_test", "DE_train__ES_to_DE_converted_test"),
    ("ES_train__DE_original_test", "ES_train__DE_to_ES_converted_test")
]

test_results = []
for orig_scen, conv_scen in pairs:
    # Find best model for conv_scen
    best_conv = best_configs[best_configs['scenario'] == conv_scen].iloc[0]
    # Get predictions for both scenarios using the SAME configuration
    model, layer, clf = best_conv['model'], best_conv['layer'], best_conv['classifier']
    
    g_orig = preds_276[(preds_276['scenario'] == orig_scen) & (preds_276['model'] == model) & (preds_276['layer'] == layer) & (preds_276['classifier'] == clf)]
    g_conv = preds_276[(preds_276['scenario'] == conv_scen) & (preds_276['model'] == model) & (preds_276['layer'] == layer) & (preds_276['classifier'] == clf)]
    
    if len(g_orig) == len(g_conv) and len(g_orig) > 0:
        # Align by sample_id
        g_orig = g_orig.sort_values('sample_id')
        g_conv = g_conv.sort_values('sample_id')
        p_val = mcnemar_test(g_orig['true_label'].values, g_orig['predicted_label'].values, g_conv['predicted_label'].values)
        test_results.append({
            'Comparison': f"{orig_scen} vs {conv_scen}",
            'Model_Used': f"{model}_L{layer}_{clf}",
            'UAR_Orig': round(uar(g_orig['true_label'].values, g_orig['predicted_label'].values), 4),
            'UAR_Conv': round(uar(g_conv['true_label'].values, g_conv['predicted_label'].values), 4),
            'p_value': p_val
        })

if test_results:
    df_tests = pd.DataFrame(test_results)
    df_tests['q_value'] = bh_fdr(df_tests['p_value'].values)
    df_tests.to_csv(os.path.join(out_dir, "Table3_Statistical_Tests_FDR.csv"), index=False)
    df_tests.to_markdown(os.path.join(out_dir, "Table3_Statistical_Tests_FDR.md"), index=False)

print("Running N=276 vs N=264 stability check...")
preds_264 = pd.read_csv(os.path.join(base_dir, "06_sensitivity_analysis_264", "predictions", "sensitivity_sample_predictions_NESTED_CV_COMPACT.csv"))

results_264 = []
for (scenario, model, layer, clf), group in preds_264.groupby(['scenario', 'model', 'layer', 'classifier']):
    results_264.append({
        'scenario': scenario,
        'model': model,
        'layer': layer,
        'classifier': clf,
        'uar': uar(group['true_label'].values, group['predicted_label'].values)
    })
df_res_264 = pd.DataFrame(results_264)
best_264 = df_res_264.loc[df_res_264.groupby('scenario')['uar'].idxmax()]
best_276_summary = df_res.loc[df_res.groupby('scenario')['uar'].idxmax()]

stability_df = pd.merge(best_276_summary, best_264, on=['scenario'], suffixes=('_N276', '_N264'))
stability_df['Diff'] = stability_df['uar_N276'] - stability_df['uar_N264']
stability_df.to_csv(os.path.join(out_dir, "Table2_Sensitivity_N264.csv"), index=False)
stability_df.to_markdown(os.path.join(out_dir, "Table2_Sensitivity_N264.md"), index=False)

print("Preparing MOS blinding package...")
mos_csv_path = os.path.join(base_dir, "10_mos_listening_test", "mos_blinded_package.csv")
paired_csv = pd.read_csv(os.path.join(base_dir, "02_dataset_index", "paired_original_converted_index.csv"))
np.random.seed(42)
sampled = paired_csv.sample(n=20) if len(paired_csv) > 20 else paired_csv
sampled['blinded_id'] = [f"SUBJECT_{i:03d}" for i in range(len(sampled))]
sampled['MOS_Score'] = "PENDING HUMAN RATINGS"
sampled[['blinded_id', 'MOS_Score']].to_csv(mos_csv_path, index=False)

manifest = """# Authoritative Results Manifest

## FINAL VALID OUTPUTS (USE FOR THESIS)
1. **Classification N=276**: `13_thesis_ready/Table1_Best_Classifiers_CIs_N276.csv`
2. **Sensitivity N=264**: `13_thesis_ready/Table2_Sensitivity_N264.csv`
3. **Statistical Significance**: `13_thesis_ready/Table3_Statistical_Tests_FDR.csv`
4. **ECAPA Speaker Verification**: `08_speaker_similarity_ecapa/results/ecapa_16khz_results.csv`
5. **PCA & t-SNE figures**: `07_pca_tsne/figures/`
6. **ViSQOL (Exploratory)**: `09_visqol/results/visqol_per_pair.csv`
7. **MOS Listening Test Form**: `10_mos_listening_test/mos_blinded_package.csv`

## OBSOLETE / INVALID OUTPUTS (DO NOT CITE)
- Old 25-50 hour CPU single-thread grid runs.
- 24 kHz ECAPA results (invalid due to sampling rate mismatch, strictly audit-only).
- Any failed GPU RAPIDS logs.
"""
with open(os.path.join(manifest_dir, "AUTHORITATIVE_MANIFEST.md"), "w") as f:
    f.write(manifest)

print("Aggregation and evaluation complete.")
