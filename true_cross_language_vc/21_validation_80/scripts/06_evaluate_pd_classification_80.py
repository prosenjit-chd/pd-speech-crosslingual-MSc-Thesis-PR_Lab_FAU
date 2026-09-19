"""
Script 06: PD Classification Validation and Statistical Analysis.

Implements Sections 17, 18, 19, 20, 21:
1. Extracts wav2vec2-large-xlsr-53 Layer 11 features for all 80 true cross-language generated recordings.
2. Loads verified original XLSR Layer 11 features.
3. Evaluates frozen baseline classifiers (Linear SVM and Logistic Regression):
   - A. Within-Language Baseline:
        * German-trained -> Real German Test (20 HC, 20 PD)
        * Spanish-trained -> Real Spanish Test (20 HC, 20 PD)
   - B. Original Cross-Language:
        * German-trained -> Original Spanish Test (20 HC, 20 PD)
        * Spanish-trained -> Original German Test (20 HC, 20 PD)
   - C. True Cross-Language Generated Speech:
        * German-trained -> Spanish-to-German Generated Test (20 HC, 20 PD)
        * Spanish-trained -> German-to-Spanish Generated Test (20 HC, 20 PD)
4. Computes UAR, balanced accuracy, sensitivity, specificity, confusion matrix, and 1000-iteration bootstrap 95% CIs.
5. Performs statistical tests on delta UAR and FDR corrections.
6. Saves:
   - 12_pd_classification/classification_original_crosslanguage.csv
   - 12_pd_classification/classification_generated_crosslanguage.csv
   - 12_pd_classification/classification_comparison.csv
   - 12_pd_classification/classification_report.md
   - 13_statistics/statistical_analysis_plan.md
   - 13_statistics/statistical_results.csv
   - 13_statistics/fdr_results.csv
"""

from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import balanced_accuracy_score, recall_score, confusion_matrix
from scipy import stats

PROJECT_ROOT = Path(r"C:\pd-speech-crosslingual")
WORKSPACE = PROJECT_ROOT / "true_cross_language_vc"
VAL_DIR = WORKSPACE / "21_validation_80"

SELECT_CSV = VAL_DIR / "01_selection" / "validation_80_selection.csv"
SYNTH_DIR = VAL_DIR / "05_synthesis"
ES_DE_DIR = SYNTH_DIR / "spanish_to_german"
DE_ES_DIR = SYNTH_DIR / "german_to_spanish"

ORIG_XLSR_CSV = (
    PROJECT_ROOT / "voice_conversion" / "stage5_embedding_conditioned_vc" /
    "stage5c_full_276" / "features_original_stage5c" / "xlsr" / "xlsr_layer11_original.csv"
)

CLASS_DIR = VAL_DIR / "12_pd_classification"
STAT_DIR = VAL_DIR / "13_statistics"

def extract_xlsr_layer11(audio_paths, device="cuda" if torch.cuda.is_available() else "cpu"):
    """Extracts mean-pooled Layer 11 features from facebook/wav2vec2-large-xlsr-53."""
    print(f"Loading facebook/wav2vec2-large-xlsr-53 on {device.upper()}...")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-xlsr-53").to(device)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large-xlsr-53")
    model.eval()

    features = []
    for idx, p in enumerate(audio_paths):
        y, sr = sf.read(str(p))
        if y.ndim > 1:
            y = y[:, 0]
        if sr != 16000:
            t_y = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
            y = resampler(t_y).squeeze(0).numpy()

        inputs = feature_extractor(y, sampling_rate=16000, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            # Layer 11 hidden state: index 11
            hidden_11 = outputs.hidden_states[11]
            pooled = torch.mean(hidden_11, dim=1).squeeze(0).cpu().numpy()
        features.append(pooled)

        if (idx + 1) % 20 == 0 or (idx + 1) == len(audio_paths):
            print(f"  Extracted XLSR Layer 11 for {idx + 1}/{len(audio_paths)} recordings...")

    return np.array(features)

def train_and_eval_classifier(X_train, y_train, X_test, y_test, clf_type="linear_svm"):
    """Trains frozen pipeline with CV-tuned C and evaluates on test set."""
    if clf_type == "linear_svm":
        clf = SVC(kernel="linear", probability=True, class_weight="balanced", random_state=42)
        param_grid = {"clf__C": [0.01, 0.1, 1.0, 10.0]}
    elif clf_type == "logistic_regression":
        clf = LogisticRegression(solver="liblinear", class_weight="balanced", random_state=42)
        param_grid = {"clf__C": [0.01, 0.1, 1.0, 10.0]}
    else:
        raise ValueError(f"Unknown classifier type: {clf_type}")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", clf)
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(pipeline, param_grid, cv=cv, scoring="balanced_accuracy", n_jobs=-1)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    best_C = grid.best_params_["clf__C"]

    # Test evaluation
    preds = best_model.predict(X_test)
    uar = float(balanced_accuracy_score(y_test, preds))
    sens = float(recall_score(y_test, preds, pos_label=1))
    spec = float(recall_score(y_test, preds, pos_label=0))
    cm = confusion_matrix(y_test, preds, labels=[0, 1])

    return {
        "best_C": best_C,
        "uar": uar,
        "sensitivity": sens,
        "specificity": spec,
        "confusion_matrix": cm,
        "preds": preds
    }

def bootstrap_ci(y_true, y_pred, n_bootstraps=1000, seed=42):
    """Computes 95% bootstrap confidence interval for UAR."""
    rng = np.random.RandomState(seed)
    uars = []
    n = len(y_true)
    for _ in range(n_bootstraps):
        idx = rng.choice(n, size=n, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        uars.append(balanced_accuracy_score(y_true[idx], y_pred[idx]))
    uars = np.array(uars)
    ci_low = float(np.percentile(uars, 2.5))
    ci_high = float(np.percentile(uars, 97.5))
    return ci_low, ci_high, uars

def main():
    CLASS_DIR.mkdir(parents=True, exist_ok=True)
    STAT_DIR.mkdir(parents=True, exist_ok=True)

    df_cohort = pd.read_csv(SELECT_CSV)
    print(f"Loaded {len(df_cohort)} cohort records.")

    # 1. Load Original Features (all 276 recordings)
    print(f"\n=== Loading Original XLSR Layer 11 Features ===")
    df_orig_all = pd.read_csv(ORIG_XLSR_CSV)
    feat_cols = [c for c in df_orig_all.columns if c.startswith("feature_")]
    print(f"Loaded original features: {len(df_orig_all)} files x {len(feat_cols)} dimensions.")

    # 2. Extract XLSR Layer 11 for 80 Generated Recordings
    print(f"\n=== Extracting XLSR Layer 11 for 80 Generated Recordings ===")
    gen_audio_paths = []
    for _, r in df_cohort.iterrows():
        s_lang = r["language"]
        t_fn = r["target_filename"]
        gen_dir = ES_DE_DIR if s_lang == "Spanish" else DE_ES_DIR
        gen_audio_paths.append(gen_dir / t_fn)

    gen_features_matrix = extract_xlsr_layer11(gen_audio_paths)
    df_gen_feats = pd.DataFrame(gen_features_matrix, columns=feat_cols)
    df_gen_feats["safe_id"] = df_cohort["safe_id"].values
    df_gen_feats["language"] = df_cohort["language"].values
    df_gen_feats["diagnosis"] = df_cohort["diagnosis"].values
    df_gen_feats["target_language"] = df_cohort["target_language"].values

    # Validation cohorts
    val_sids = set(df_cohort["safe_id"])
    df_orig_cohort = df_orig_all[df_orig_all["file"].isin(val_sids)].copy()
    df_orig_cohort = df_orig_cohort.sort_values(by="file").reset_index(drop=True)

    # Separate training sets (out-of-sample: original recordings NOT in the 80 validation cohort)
    df_train_de = df_orig_all[(df_orig_all["language"] == "German") & (~df_orig_all["file"].isin(val_sids))].copy()
    df_train_es = df_orig_all[(df_orig_all["language"] == "Spanish") & (~df_orig_all["file"].isin(val_sids))].copy()

    print(f"\nTraining pool sizes (strict out-of-sample):")
    print(f"  German training set: {len(df_train_de)} recordings ({np.sum(df_train_de['group']=='HC')} HC, {np.sum(df_train_de['group']=='PD')} PD)")
    print(f"  Spanish training set: {len(df_train_es)} recordings ({np.sum(df_train_es['group']=='HC')} HC, {np.sum(df_train_es['group']=='PD')} PD)")

    # Test sets (all 40 recordings each, perfectly balanced 20 HC / 20 PD)
    # Real German validation test
    df_test_real_de = df_orig_cohort[df_orig_cohort["language"] == "German"].sort_values(by="file").reset_index(drop=True)
    # Real Spanish validation test
    df_test_real_es = df_orig_cohort[df_orig_cohort["language"] == "Spanish"].sort_values(by="file").reset_index(drop=True)
    # Generated German validation test (Spanish speakers converted to German)
    df_test_gen_de = df_gen_feats[df_gen_feats["target_language"] == "German"].sort_values(by="safe_id").reset_index(drop=True)
    # Generated Spanish validation test (German speakers converted to Spanish)
    df_test_gen_es = df_gen_feats[df_gen_feats["target_language"] == "Spanish"].sort_values(by="safe_id").reset_index(drop=True)

    def get_X_y(df, label_col="group"):
        X = df[feat_cols].values.astype(np.float32)
        y = (df[label_col] == "PD").values.astype(int)
        return X, y

    X_train_de, y_train_de = get_X_y(df_train_de)
    X_train_es, y_train_es = get_X_y(df_train_es)

    X_test_real_de, y_test_real_de = get_X_y(df_test_real_de)
    X_test_real_es, y_test_real_es = get_X_y(df_test_real_es)
    X_test_gen_de, y_test_gen_de = get_X_y(df_test_gen_de, label_col="diagnosis")
    X_test_gen_es, y_test_gen_es = get_X_y(df_test_gen_es, label_col="diagnosis")

    classifiers = ["linear_svm", "logistic_regression"]
    all_results = []
    comparisons = []
    statistical_tests = []

    print("\n=== Commencing Controlled Classification Experiments ===")
    for clf_name in classifiers:
        print(f"\n--- Classifier: {clf_name.upper()} ---")

        # -------------------------------------------------------------
        # Direction 1: German-Trained Model
        # Tested on: Real German (Within), Real Spanish (Orig Cross), Gen German (True Cross)
        # -------------------------------------------------------------
        # Within-Language
        res_within_de = train_and_eval_classifier(X_train_de, y_train_de, X_test_real_de, y_test_real_de, clf_name)
        ci_l_w_de, ci_h_w_de, boot_w_de = bootstrap_ci(y_test_real_de, res_within_de["preds"])

        # Original Cross-Language (Train German -> Test Real Spanish)
        res_orig_es = train_and_eval_classifier(X_train_de, y_train_de, X_test_real_es, y_test_real_es, clf_name)
        ci_l_o_es, ci_h_o_es, boot_o_es = bootstrap_ci(y_test_real_es, res_orig_es["preds"])

        # Generated Cross-Language (Train German -> Test Spanish Converted to German)
        res_gen_de = train_and_eval_classifier(X_train_de, y_train_de, X_test_gen_de, y_test_gen_de, clf_name)
        ci_l_g_de, ci_h_g_de, boot_g_de = bootstrap_ci(y_test_gen_de, res_gen_de["preds"])

        delta_es_to_de = res_gen_de["uar"] - res_orig_es["uar"]
        p_val_de = float(np.mean(boot_g_de <= boot_o_es)) # One-sided empirical bootstrap p-value

        comparisons.append({
            "classifier": clf_name,
            "direction": "Spanish -> German",
            "training_language": "German",
            "within_language_uar": round(res_within_de["uar"], 4),
            "original_crosslanguage_uar": round(res_orig_es["uar"], 4),
            "generated_crosslanguage_uar": round(res_gen_de["uar"], 4),
            "absolute_delta_uar": round(delta_es_to_de, 4),
            "relative_improvement_pct": round((delta_es_to_de / max(1e-4, res_orig_es["uar"])) * 100.0, 2),
            "bootstrap_p_value": round(p_val_de, 4),
            "significantly_improved": (delta_es_to_de > 0 and p_val_de < 0.05)
        })

        # -------------------------------------------------------------
        # Direction 2: Spanish-Trained Model
        # Tested on: Real Spanish (Within), Real German (Orig Cross), Gen Spanish (True Cross)
        # -------------------------------------------------------------
        # Within-Language
        res_within_es = train_and_eval_classifier(X_train_es, y_train_es, X_test_real_es, y_test_real_es, clf_name)
        ci_l_w_es, ci_h_w_es, boot_w_es = bootstrap_ci(y_test_real_es, res_within_es["preds"])

        # Original Cross-Language (Train Spanish -> Test Real German)
        res_orig_de = train_and_eval_classifier(X_train_es, y_train_es, X_test_real_de, y_test_real_de, clf_name)
        ci_l_o_de, ci_h_o_de, boot_o_de = bootstrap_ci(y_test_real_de, res_orig_de["preds"])

        # Generated Cross-Language (Train Spanish -> Test German Converted to Spanish)
        res_gen_es = train_and_eval_classifier(X_train_es, y_train_es, X_test_gen_es, y_test_gen_es, clf_name)
        ci_l_g_es, ci_h_g_es, boot_g_es = bootstrap_ci(y_test_gen_es, res_gen_es["preds"])

        delta_de_to_es = res_gen_es["uar"] - res_orig_de["uar"]
        p_val_es = float(np.mean(boot_g_es <= boot_o_de))

        comparisons.append({
            "classifier": clf_name,
            "direction": "German -> Spanish",
            "training_language": "Spanish",
            "within_language_uar": round(res_within_es["uar"], 4),
            "original_crosslanguage_uar": round(res_orig_de["uar"], 4),
            "generated_crosslanguage_uar": round(res_gen_es["uar"], 4),
            "absolute_delta_uar": round(delta_de_to_es, 4),
            "relative_improvement_pct": round((delta_de_to_es / max(1e-4, res_orig_de["uar"])) * 100.0, 2),
            "bootstrap_p_value": round(p_val_es, 4),
            "significantly_improved": (delta_de_to_es > 0 and p_val_es < 0.05)
        })

        # Append detailed rows
        scenario_data = [
            ("Within-Language (German->German)", "German", "Real German", res_within_de, ci_l_w_de, ci_h_w_de),
            ("Original Cross-Language (German->Spanish)", "German", "Real Spanish", res_orig_es, ci_l_o_es, ci_h_o_es),
            ("Generated True Cross-Language (Spanish in German)", "German", "Generated German", res_gen_de, ci_l_g_de, ci_h_g_de),
            ("Within-Language (Spanish->Spanish)", "Spanish", "Real Spanish", res_within_es, ci_l_w_es, ci_h_w_es),
            ("Original Cross-Language (Spanish->German)", "Spanish", "Real German", res_orig_de, ci_l_o_de, ci_h_o_de),
            ("Generated True Cross-Language (German in Spanish)", "Spanish", "Generated Spanish", res_gen_es, ci_l_g_es, ci_h_g_es)
        ]

        for sc_name, tr_lang, te_dom, r, cl, ch in scenario_data:
            cm = r["confusion_matrix"]
            all_results.append({
                "classifier": clf_name,
                "scenario": sc_name,
                "training_language": tr_lang,
                "test_domain": te_dom,
                "uar": round(r["uar"], 4),
                "ci_95_low": round(cl, 4),
                "ci_95_high": round(ch, 4),
                "sensitivity": round(r["sensitivity"], 4),
                "specificity": round(r["specificity"], 4),
                "tn": int(cm[0, 0]),
                "fp": int(cm[0, 1]),
                "fn": int(cm[1, 0]),
                "tp": int(cm[1, 1]),
                "best_C": r["best_C"]
            })

    # Save detailed tables
    df_all_res = pd.DataFrame(all_results)
    df_orig_cross = df_all_res[df_all_res["scenario"].str.contains("Original Cross-Language")].copy()
    df_gen_cross = df_all_res[df_all_res["scenario"].str.contains("Generated True Cross-Language")].copy()
    df_comp = pd.DataFrame(comparisons)

    df_orig_cross.to_csv(CLASS_DIR / "classification_original_crosslanguage.csv", index=False)
    df_gen_cross.to_csv(CLASS_DIR / "classification_generated_crosslanguage.csv", index=False)
    df_comp.to_csv(CLASS_DIR / "classification_comparison.csv", index=False)

    # FDR correction across comparisons
    p_vals = df_comp["bootstrap_p_value"].values
    n = len(p_vals)
    sorted_idx = np.argsort(p_vals)
    sorted_p = p_vals[sorted_idx]
    q_vals = np.zeros(n)
    cum_min = 1.0
    for i in range(n - 1, -1, -1):
        q = min(1.0, sorted_p[i] * n / (i + 1))
        cum_min = min(cum_min, q)
        q_vals[i] = cum_min
    orig_q = np.zeros(n)
    orig_q[sorted_idx] = q_vals
    df_comp["fdr_q_value"] = np.round(orig_q, 4)

    df_comp.to_csv(STAT_DIR / "fdr_results.csv", index=False)
    df_all_res.to_csv(STAT_DIR / "statistical_results.csv", index=False)

    # Generate Statistical Analysis Plan
    stat_plan_md = STAT_DIR / "statistical_analysis_plan.md"
    stat_plan_content = """# Statistical Analysis Plan: 80-File True Cross-Language Validation

## 1. Experimental Hypotheses
- **Primary Hypothesis $H_1$**: True cross-language voice conversion improves cross-language Parkinson's disease classification performance relative to unconverted speech:
  $$\\text{UAR}(\\text{Generated Cross}) > \\text{UAR}(\\text{Original Cross})$$
- **Secondary Hypothesis $H_2$**: True cross-language synthesis preserves pathology-relevant acoustic features without complete normalization:
  $$\\text{Preservation Rate} \\ge 70\\%$$
- **Null Hypothesis $H_0$**: True cross-language synthesis yields no change in cross-language PD detection ($\Delta \\text{UAR} \\le 0$).

## 2. Statistical Methodology
1. **Primary Metric**: Unweighted Average Recall (UAR / Balanced Accuracy) across strictly balanced 20 HC / 20 PD cohorts.
2. **Confidence Intervals**: Non-parametric percentile bootstrap with $B=1,000$ resamples.
3. **Hypothesis Testing**: Empirical paired bootstrap difference distribution:
   $$p = \\frac{1}{B} \\sum_{b=1}^B \\mathbb{I}\\left(\\text{UAR}_{\\text{gen}}^{(b)} \\le \\text{UAR}_{\\text{orig}}^{(b)}\\right)$$
4. **Multiple Comparison Correction**: Benjamini-Hochberg False Discovery Rate (BH-FDR) applied across tested directions and classifiers ($\alpha=0.05$).
5. **Pathology Feature Comparisons**: Two-sided Mann-Whitney U tests with Cliff's delta effect sizes and BH-FDR correction.
6. **Acoustic Normalization Testing**: Paired Wilcoxon signed-rank test and paired Student's t-test comparing Euclidean distances to Healthy Control centroids ($D_{\\text{orig}}$ vs $D_{\\text{gen}}$).
"""
    stat_plan_md.write_text(stat_plan_content, encoding="utf-8")

    # Generate master classification report
    report_md = CLASS_DIR / "classification_report.md"
    report_content = f"""# Cross-Language Parkinson's Disease Classification Report: 80-File Validation Cohort

## 1. Executive Summary
- **Baseline Feature Extractor**: `facebook/wav2vec2-large-xlsr-53` (Layer 11, frozen).
- **Classifiers**: Linear Support Vector Machine (`SVC`) & Logistic Regression (`liblinear`).
- **Validation Design**: Strict out-of-sample testing across 80 subjects (20 Spanish HC, 20 Spanish PD, 20 German HC, 20 German PD).

### Critical Comparison Table

| Direction | Classifier | Within-Language Baseline UAR | Original Foreign Cross-Language UAR | Generated True Cross-Language UAR | Absolute $\\Delta$ UAR | Relative Gain | Bootstrap $p$-value | BH-FDR $q$-value |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for _, r in df_comp.iterrows():
        report_content += (
            f"| **{r['direction']}** | `{r['classifier']}` | {r['within_language_uar']:.4f} | "
            f"{r['original_crosslanguage_uar']:.4f} | **{r['generated_crosslanguage_uar']:.4f}** | "
            f"**{r['absolute_delta_uar']:+.4f}** | **{r['relative_improvement_pct']:+.1f}%** | "
            f"{r['bootstrap_p_value']:.4f} | {r['fdr_q_value']:.4f} |\n"
        )

    report_content += """
---

## 2. Complete Scenario Results Table

| Classifier | Scenario | Training Domain | Test Domain | UAR [95% CI] | Sensitivity (PD Recall) | Specificity (HC Recall) | Confusion Matrix [TN, FP, FN, TP] | Best $C$ |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
"""
    for _, r in df_all_res.iterrows():
        report_content += (
            f"| `{r['classifier']}` | {r['scenario']} | {r['training_language']} | {r['test_domain']} | "
            f"**{r['uar']:.4f}** [{r['ci_95_low']:.4f}, {r['ci_95_high']:.4f}] | "
            f"{r['sensitivity']:.4f} | {r['specificity']:.4f} | [{r['tn']}, {r['fp']}, {r['fn']}, {r['tp']}] | {r['best_C']} |\n"
        )

    report_md.write_text(report_content, encoding="utf-8")
    print(f"\nSaved classification results and reports to {CLASS_DIR} and {STAT_DIR}")

if __name__ == "__main__":
    main()
