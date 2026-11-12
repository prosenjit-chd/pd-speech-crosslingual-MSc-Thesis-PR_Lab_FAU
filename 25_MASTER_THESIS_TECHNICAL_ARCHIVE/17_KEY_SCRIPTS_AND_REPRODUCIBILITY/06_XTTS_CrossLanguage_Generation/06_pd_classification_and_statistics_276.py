"""
Script 06: PD Classification and Statistical Analysis for Full 276 Cohort.

Implements Sections 21, 22, 23, 24, 25:
- Extracts facebook/wav2vec2-large-xlsr-53 Layer 11 features on GPU for all 276 generated recordings.
- Loads verified original XLSR Layer 11 features from the project inventory.
- Evaluates frozen baseline classifiers (Linear SVM and Logistic Regression):
  * Within-Language: German -> Real German, Spanish -> Real Spanish
  * Original Cross-Language: German-trained -> Original Spanish, Spanish-trained -> Original German
  * Generated Cross-Language: German-trained -> Spanish-in-German, Spanish-trained -> German-in-Spanish
- Compares across:
  * SET A: Full Cohort (N = 276)
  * SET B: Development-Excluded Sensitivity Cohort (N = 264)
  * SET C: Independent Validation Benchmark (N = 80, loaded directly from 21_validation_80)
- Computes UAR, balanced accuracy, sensitivity (PD recall), specificity (HC recall), confusion matrix, 1,000-iteration bootstrap 95% CIs, nominal p-values, and Benjamini-Hochberg FDR q-values.
- Generates:
  * 13_pd_classification/full_276_classification_results.csv
  * 13_pd_classification/original_vs_generated_comparison.csv
  * 13_pd_classification/validation_vs_fullcohort_comparison.csv
  * 13_pd_classification/classification_report.md
  * 14_statistics/statistical_analysis_plan.md
  * 14_statistics/bootstrap_results.csv
  * 14_statistics/fdr_results.csv
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
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.metrics import balanced_accuracy_score, recall_score, confusion_matrix
from scipy import stats

PROJECT_ROOT = Path(r"C:\pd-speech-crosslingual")
WORKSPACE = PROJECT_ROOT / "true_cross_language_vc"
EVAL_DIR = WORKSPACE / "22_full_276_evaluation"

INV_CSV = EVAL_DIR / "01_inventory" / "full_276_inventory.csv"
TAGS_CSV = EVAL_DIR / "02_cohort_tags" / "cohort_provenance.csv"
SYNTH_DIR = EVAL_DIR / "06_synthesis"

CLASS_DIR = EVAL_DIR / "13_pd_classification"
STAT_DIR = EVAL_DIR / "14_statistics"

ORIG_XLSR_CSV = (
    PROJECT_ROOT / "voice_conversion" / "stage5_embedding_conditioned_vc" /
    "stage5c_full_276" / "features_original_stage5c" / "xlsr" / "xlsr_layer11_original.csv"
)

VAL80_CLASS_CSV = WORKSPACE / "21_validation_80" / "12_pd_classification" / "classification_comparison.csv"
VAL80_FDR_CSV = WORKSPACE / "21_validation_80" / "13_statistics" / "fdr_results.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def extract_xlsr_layer11(audio_paths, device=DEVICE):
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
            t_y = torchaudio.functional.resample(t_y, orig_freq=sr, new_freq=16000)
            y = t_y.squeeze(0).numpy()

        inputs = feature_extractor(y, sampling_rate=16000, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            layer11 = outputs.hidden_states[11]
            pooled = layer11.mean(dim=1).squeeze().cpu().numpy()
            features.append(pooled)

        if (idx + 1) % 50 == 0 or (idx + 1) == len(audio_paths):
            print(f"  Extracted XLSR Layer 11 features: {idx + 1}/{len(audio_paths)}...")

    return np.array(features)

def bootstrap_ci(y_true, y_pred, n_iter=1000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = []
    for _ in range(n_iter):
        idx = rng.choice(n, size=n, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        scores.append(balanced_accuracy_score(y_true[idx], y_pred[idx]))
    scores = np.array(scores)
    ci_low = np.percentile(scores, 2.5)
    ci_high = np.percentile(scores, 97.5)
    return float(np.mean(scores)), float(ci_low), float(ci_high)

def bootstrap_delta_p_value(y_true, y_pred_orig, y_pred_gen, n_iter=1000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    deltas = []
    for _ in range(n_iter):
        idx = rng.choice(n, size=n, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        uar_orig = balanced_accuracy_score(y_true[idx], y_pred_orig[idx])
        uar_gen = balanced_accuracy_score(y_true[idx], y_pred_gen[idx])
        deltas.append(uar_gen - uar_orig)
    deltas = np.array(deltas)
    p_val = np.mean(deltas <= 0.0)
    ci_low = np.percentile(deltas, 2.5)
    ci_high = np.percentile(deltas, 97.5)
    return float(p_val), float(ci_low), float(ci_high)

def benjamini_hochberg(p_values):
    p_vals = np.array(p_values)
    n = len(p_vals)
    sorted_indices = np.argsort(p_vals)
    sorted_p = p_vals[sorted_indices]
    q_vals = np.zeros(n)
    cum_min = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        q = min(1.0, (sorted_p[i] * n) / rank)
        cum_min = min(cum_min, q)
        q_vals[i] = cum_min
    orig_q = np.zeros(n)
    orig_q[sorted_indices] = q_vals
    return orig_q

def main():
    CLASS_DIR.mkdir(parents=True, exist_ok=True)
    STAT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Loading Inventory and Provenance Metadata ===")
    df_inv = pd.read_csv(INV_CSV)
    df_tags = pd.read_csv(TAGS_CSV)
    df_meta = df_inv.merge(df_tags[["safe_id", "cohort_tag", "in_set_b_dev_excluded_264", "in_set_c_val_80"]], on="safe_id")

    gen_paths = []
    for _, r in df_meta.iterrows():
        s_lang = r["language"]
        t_fn = r["target_filename"]
        sub_dir = "spanish_to_german" if s_lang == "Spanish" else "german_to_spanish"
        gen_paths.append(str(SYNTH_DIR / sub_dir / t_fn))
    df_meta["generated_path"] = gen_paths

    # 1. Load original XLSR features
    print("\nLoading pre-extracted original XLSR Layer 11 features...")
    df_orig_xlsr = pd.read_csv(ORIG_XLSR_CSV)
    feat_cols = [c for c in df_orig_xlsr.columns if c.startswith("feature_")]
    df_orig_xlsr = df_orig_xlsr.rename(columns={"file": "safe_id"})

    orig_feat_dict = {}
    for _, r in df_orig_xlsr.iterrows():
        orig_feat_dict[r["safe_id"]] = r[feat_cols].values.astype(np.float32)

    # 2. Extract generated XLSR features
    gen_feat_csv = CLASS_DIR / "generated_xlsr_layer11_276.csv"
    if gen_feat_csv.exists() and len(pd.read_csv(gen_feat_csv)) == len(df_meta):
        print(f"Loading cached generated XLSR features from {gen_feat_csv}...")
        df_gen_xlsr = pd.read_csv(gen_feat_csv)
        X_gen_all = df_gen_xlsr[feat_cols].values.astype(np.float32)
    else:
        print("Extracting XLSR Layer 11 features on GPU for all 276 generated files...")
        X_gen_all = extract_xlsr_layer11(df_meta["generated_path"].tolist(), device=DEVICE)
        df_gen_save = pd.DataFrame(X_gen_all, columns=feat_cols)
        df_gen_save.insert(0, "safe_id", df_meta["safe_id"].tolist())
        df_gen_save.insert(1, "source_language", df_meta["language"].tolist())
        df_gen_save.insert(2, "target_language", df_meta["target_language"].tolist())
        df_gen_save.insert(3, "diagnosis", df_meta["diagnosis"].tolist())
        df_gen_save.to_csv(gen_feat_csv, index=False)
        print(f"Saved generated XLSR features to {gen_feat_csv}")

    X_orig_all = np.array([orig_feat_dict[s] for s in df_meta["safe_id"]])
    y_all = (df_meta["diagnosis"] == "PD").astype(int).values

    analysis_sets = [
        ("Set A: Full Cohort", np.ones(len(df_meta), dtype=bool)),
        ("Set B: Development-Excluded", df_meta["in_set_b_dev_excluded_264"].values)
    ]

    all_res_records = []
    comp_records = []
    boot_records = []

    for set_name, mask in analysis_sets:
        sub_meta = df_meta[mask].reset_index(drop=True)
        sub_X_orig = X_orig_all[mask]
        sub_X_gen = X_gen_all[mask]
        sub_y = y_all[mask]
        sub_lang = sub_meta["language"].values

        print(f"\n--- Evaluating PD Classification on {set_name} (N = {len(sub_meta)}) ---")

        de_idx = np.where(sub_lang == "German")[0]
        es_idx = np.where(sub_lang == "Spanish")[0]

        X_de_orig, y_de = sub_X_orig[de_idx], sub_y[de_idx]
        X_es_orig, y_es = sub_X_orig[es_idx], sub_y[es_idx]
        X_es_gen = sub_X_gen[es_idx]
        X_de_gen = sub_X_gen[de_idx]

        for clf_name, clf_base, param_grid in [
            ("Linear SVM", SVC(kernel="linear", probability=True, class_weight="balanced", random_state=42), {"clf__C": [0.01, 0.1, 1.0, 10.0]}),
            ("Logistic Regression", LogisticRegression(solver="liblinear", class_weight="balanced", random_state=42), {"clf__C": [0.01, 0.1, 1.0, 10.0]})
        ]:
            pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf_base)])
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            # 1. Train German model
            grid_de = GridSearchCV(pipe, param_grid, cv=cv, scoring="balanced_accuracy", n_jobs=-1)
            grid_de.fit(X_de_orig, y_de)
            best_de = grid_de.best_estimator_

            # Within German UAR (CV out-of-fold prediction)
            y_de_within_pred = cross_val_predict(best_de, X_de_orig, y_de, cv=cv)
            uar_de_within = balanced_accuracy_score(y_de, y_de_within_pred)
            sens_de_w = recall_score(y_de, y_de_within_pred, pos_label=1)
            spec_de_w = recall_score(y_de, y_de_within_pred, pos_label=0)
            cm_de_w = confusion_matrix(y_de, y_de_within_pred)
            _, ci_l_dw, ci_h_dw = bootstrap_ci(y_de, y_de_within_pred)

            all_res_records.append({
                "set": set_name,
                "condition": "Within-Language",
                "training_language": "German",
                "test_domain": "Real German (CV)",
                "classifier": clf_name,
                "n_eval": len(y_de),
                "uar": round(uar_de_within, 4),
                "pd_recall": round(sens_de_w, 4),
                "hc_recall": round(spec_de_w, 4),
                "confusion_matrix": str(cm_de_w.tolist()),
                "ci_95_low": round(ci_l_dw, 4),
                "ci_95_high": round(ci_h_dw, 4)
            })

            # Spanish test predictions from German model
            y_es_orig_pred = best_de.predict(X_es_orig)
            y_es_gen_pred = best_de.predict(X_es_gen)

            uar_es_orig = balanced_accuracy_score(y_es, y_es_orig_pred)
            sens_es_o = recall_score(y_es, y_es_orig_pred, pos_label=1)
            spec_es_o = recall_score(y_es, y_es_orig_pred, pos_label=0)
            cm_es_o = confusion_matrix(y_es, y_es_orig_pred)
            _, ci_l_eo, ci_h_eo = bootstrap_ci(y_es, y_es_orig_pred)

            all_res_records.append({
                "set": set_name,
                "condition": "Original Cross-Language",
                "training_language": "German",
                "test_domain": "Original Spanish",
                "classifier": clf_name,
                "n_eval": len(y_es),
                "uar": round(uar_es_orig, 4),
                "pd_recall": round(sens_es_o, 4),
                "hc_recall": round(spec_es_o, 4),
                "confusion_matrix": str(cm_es_o.tolist()),
                "ci_95_low": round(ci_l_eo, 4),
                "ci_95_high": round(ci_h_eo, 4)
            })

            uar_es_gen = balanced_accuracy_score(y_es, y_es_gen_pred)
            sens_es_g = recall_score(y_es, y_es_gen_pred, pos_label=1)
            spec_es_g = recall_score(y_es, y_es_gen_pred, pos_label=0)
            cm_es_g = confusion_matrix(y_es, y_es_gen_pred)
            _, ci_l_eg, ci_h_eg = bootstrap_ci(y_es, y_es_gen_pred)

            all_res_records.append({
                "set": set_name,
                "condition": "Generated Cross-Language",
                "training_language": "German",
                "test_domain": "Spanish Converted to German",
                "classifier": clf_name,
                "n_eval": len(y_es),
                "uar": round(uar_es_gen, 4),
                "pd_recall": round(sens_es_g, 4),
                "hc_recall": round(spec_es_g, 4),
                "confusion_matrix": str(cm_es_g.tolist()),
                "ci_95_low": round(ci_l_eg, 4),
                "ci_95_high": round(ci_h_eg, 4)
            })

            delta_es_de = uar_es_gen - uar_es_orig
            p_es_de, ci_l_es, ci_h_es = bootstrap_delta_p_value(y_es, y_es_orig_pred, y_es_gen_pred)

            # 2. Train Spanish model
            grid_es = GridSearchCV(pipe, param_grid, cv=cv, scoring="balanced_accuracy", n_jobs=-1)
            grid_es.fit(X_es_orig, y_es)
            best_es = grid_es.best_estimator_

            y_es_within_pred = cross_val_predict(best_es, X_es_orig, y_es, cv=cv)
            uar_es_within = balanced_accuracy_score(y_es, y_es_within_pred)
            sens_es_w = recall_score(y_es, y_es_within_pred, pos_label=1)
            spec_es_w = recall_score(y_es, y_es_within_pred, pos_label=0)
            cm_es_w = confusion_matrix(y_es, y_es_within_pred)
            _, ci_l_ew, ci_h_ew = bootstrap_ci(y_es, y_es_within_pred)

            all_res_records.append({
                "set": set_name,
                "condition": "Within-Language",
                "training_language": "Spanish",
                "test_domain": "Real Spanish (CV)",
                "classifier": clf_name,
                "n_eval": len(y_es),
                "uar": round(uar_es_within, 4),
                "pd_recall": round(sens_es_w, 4),
                "hc_recall": round(spec_es_w, 4),
                "confusion_matrix": str(cm_es_w.tolist()),
                "ci_95_low": round(ci_l_ew, 4),
                "ci_95_high": round(ci_h_ew, 4)
            })

            y_de_orig_pred = best_es.predict(X_de_orig)
            y_de_gen_pred = best_es.predict(X_de_gen)

            uar_de_orig = balanced_accuracy_score(y_de, y_de_orig_pred)
            sens_de_o = recall_score(y_de, y_de_orig_pred, pos_label=1)
            spec_de_o = recall_score(y_de, y_de_orig_pred, pos_label=0)
            cm_de_o = confusion_matrix(y_de, y_de_orig_pred)
            _, ci_l_do, ci_h_do = bootstrap_ci(y_de, y_de_orig_pred)

            all_res_records.append({
                "set": set_name,
                "condition": "Original Cross-Language",
                "training_language": "Spanish",
                "test_domain": "Original German",
                "classifier": clf_name,
                "n_eval": len(y_de),
                "uar": round(uar_de_orig, 4),
                "pd_recall": round(sens_de_o, 4),
                "hc_recall": round(spec_de_o, 4),
                "confusion_matrix": str(cm_de_o.tolist()),
                "ci_95_low": round(ci_l_do, 4),
                "ci_95_high": round(ci_h_do, 4)
            })

            uar_de_gen = balanced_accuracy_score(y_de, y_de_gen_pred)
            sens_de_g = recall_score(y_de, y_de_gen_pred, pos_label=1)
            spec_de_g = recall_score(y_de, y_de_gen_pred, pos_label=0)
            cm_de_g = confusion_matrix(y_de, y_de_gen_pred)
            _, ci_l_dg, ci_h_dg = bootstrap_ci(y_de, y_de_gen_pred)

            all_res_records.append({
                "set": set_name,
                "condition": "Generated Cross-Language",
                "training_language": "Spanish",
                "test_domain": "German Converted to Spanish",
                "classifier": clf_name,
                "n_eval": len(y_de),
                "uar": round(uar_de_gen, 4),
                "pd_recall": round(sens_de_g, 4),
                "hc_recall": round(spec_de_g, 4),
                "confusion_matrix": str(cm_de_g.tolist()),
                "ci_95_low": round(ci_l_dg, 4),
                "ci_95_high": round(ci_h_dg, 4)
            })

            delta_de_es = uar_de_gen - uar_de_orig
            p_de_es, ci_l_de, ci_h_de = bootstrap_delta_p_value(y_de, y_de_orig_pred, y_de_gen_pred)

            # Record comparisons
            comp_records.append({
                "set": set_name,
                "direction": "Spanish -> German",
                "training_language": "German",
                "classifier": clf_name,
                "n_eval": len(y_es),
                "within_language_uar": round(uar_de_within, 4),
                "original_crosslanguage_uar": round(uar_es_orig, 4),
                "generated_crosslanguage_uar": round(uar_es_gen, 4),
                "absolute_delta_uar": round(delta_es_de, 4),
                "relative_improvement_pct": round((delta_es_de / uar_es_orig) * 100.0, 2) if uar_es_orig > 0 else 0.0,
                "ci_95_low": round(ci_l_es, 4),
                "ci_95_high": round(ci_h_es, 4),
                "nominal_p_value": round(p_es_de, 4)
            })

            comp_records.append({
                "set": set_name,
                "direction": "German -> Spanish",
                "training_language": "Spanish",
                "classifier": clf_name,
                "n_eval": len(y_de),
                "within_language_uar": round(uar_es_within, 4),
                "original_crosslanguage_uar": round(uar_de_orig, 4),
                "generated_crosslanguage_uar": round(uar_de_gen, 4),
                "absolute_delta_uar": round(delta_de_es, 4),
                "relative_improvement_pct": round((delta_de_es / uar_de_orig) * 100.0, 2) if uar_de_orig > 0 else 0.0,
                "ci_95_low": round(ci_l_de, 4),
                "ci_95_high": round(ci_h_de, 4),
                "nominal_p_value": round(p_de_es, 4)
            })

            boot_records.append({
                "set": set_name,
                "direction": "Spanish -> German",
                "classifier": clf_name,
                "metric": "Delta UAR",
                "mean_delta": round(delta_es_de, 4),
                "ci_95_low": round(ci_l_es, 4),
                "ci_95_high": round(ci_h_es, 4),
                "bootstrap_p": round(p_es_de, 4)
            })

            boot_records.append({
                "set": set_name,
                "direction": "German -> Spanish",
                "classifier": clf_name,
                "metric": "Delta UAR",
                "mean_delta": round(delta_de_es, 4),
                "ci_95_low": round(ci_l_de, 4),
                "ci_95_high": round(ci_h_de, 4),
                "bootstrap_p": round(p_de_es, 4)
            })

    df_all_res = pd.DataFrame(all_res_records)
    all_res_csv = CLASS_DIR / "full_276_classification_results.csv"
    df_all_res.to_csv(all_res_csv, index=False)
    print(f"Saved all detailed classification results to {all_res_csv}")

    df_comp = pd.DataFrame(comp_records)
    df_comp["fdr_q_value"] = np.round(benjamini_hochberg(df_comp["nominal_p_value"].values), 4)
    df_comp["statistically_significant_after_fdr"] = df_comp["fdr_q_value"] < 0.05
    df_comp["nominally_significant"] = df_comp["nominal_p_value"] < 0.05

    comp_csv = CLASS_DIR / "original_vs_generated_comparison.csv"
    df_comp.to_csv(comp_csv, index=False)
    print(f"Saved master classification comparison to {comp_csv}")

    df_boot = pd.DataFrame(boot_records)
    boot_csv = STAT_DIR / "bootstrap_results.csv"
    df_boot.to_csv(boot_csv, index=False)
    print(f"Saved bootstrap results to {boot_csv}")

    # ========================================================
    # SECTION 24: PRESERVE 80-FILE RESULT & BUILD 80 vs 276 TABLE
    # ========================================================
    print("\n=== Building 80 Validation vs Full 276 Cohort Comparison Table ===")
    df_val80_fdr = pd.read_csv(VAL80_FDR_CSV)

    val_vs_full_records = []
    for clf_key, clf_display in [("linear_svm", "Linear SVM"), ("logistic_regression", "Logistic Regression")]:
        for direction in ["Spanish -> German", "German -> Spanish"]:
            r_80 = df_val80_fdr[(df_val80_fdr["classifier"] == clf_key) & (df_val80_fdr["direction"] == direction)].iloc[0]
            r_276 = df_comp[(df_comp["set"] == "Set A: Full Cohort") & (df_comp["classifier"] == clf_display) & (df_comp["direction"] == direction)].iloc[0]
            r_264 = df_comp[(df_comp["set"] == "Set B: Development-Excluded") & (df_comp["classifier"] == clf_display) & (df_comp["direction"] == direction)].iloc[0]

            val_vs_full_records.append({
                "direction": direction,
                "classifier": clf_display,
                "80_original_uar": r_80["original_crosslanguage_uar"],
                "80_generated_uar": r_80["generated_crosslanguage_uar"],
                "80_delta_uar": r_80["absolute_delta_uar"],
                "80_nominal_p": r_80["bootstrap_p_value"],
                "80_fdr_q": r_80["fdr_q_value"],
                "276_original_uar": r_276["original_crosslanguage_uar"],
                "276_generated_uar": r_276["generated_crosslanguage_uar"],
                "276_delta_uar": r_276["absolute_delta_uar"],
                "276_nominal_p": r_276["nominal_p_value"],
                "276_fdr_q": r_276["fdr_q_value"],
                "264_dev_excluded_original_uar": r_264["original_crosslanguage_uar"],
                "264_dev_excluded_generated_uar": r_264["generated_crosslanguage_uar"],
                "264_dev_excluded_delta_uar": r_264["absolute_delta_uar"],
                "264_nominal_p": r_264["nominal_p_value"],
                "264_fdr_q": r_264["fdr_q_value"]
            })

    df_v_vs_f = pd.DataFrame(val_vs_full_records)
    v_vs_f_csv = CLASS_DIR / "validation_vs_fullcohort_comparison.csv"
    df_v_vs_f.to_csv(v_vs_f_csv, index=False)
    print(f"Saved 80 vs 276 comparison to {v_vs_f_csv}")

    # Statistical artifacts
    stat_plan_md = STAT_DIR / "statistical_analysis_plan.md"
    stat_plan_content = """# Statistical Analysis Plan: Full 276-File Evaluation

## 1. Primary Diagnostic Generalization Hypothesis
- **Hypothesis $H_1$**: True cross-language voice conversion improves cross-lingual Parkinson's disease classification performance compared to testing directly on unmodified foreign-language speech:
  $$\Delta \\text{UAR} = \\text{UAR}_{\\text{generated}} - \\text{UAR}_{\\text{original}} > 0$$
- **Primary Statistical Estimator**: Paired non-parametric bootstrap sampling ($B = 1,000$ iterations) with 95% percentile confidence intervals.
- **Multiple Testing Control**: Benjamini-Hochberg False Discovery Rate (FDR) control applied across all classification comparisons.
- **Decision Criterion**: Statistical significance is declared if and only if $q < 0.05$. Results where $p < 0.05$ but $q \ge 0.05$ are strictly reported as nominally significant before FDR correction.

## 2. Analysis Cohort Partitioning
- **Confirmatory Benchmark**: Independent 80-file validation (`21_validation_80`) remains the immutable primary validation result.
- **Sensitivity Cohort A**: Master Full Cohort ($N = 276$).
- **Sensitivity Cohort B**: Development-Excluded Cohort ($N = 264$, excluding 12 pilot development subjects).
"""
    stat_plan_md.write_text(stat_plan_content, encoding="utf-8")

    fdr_csv = STAT_DIR / "fdr_results.csv"
    df_comp[["set", "direction", "classifier", "absolute_delta_uar", "ci_95_low", "ci_95_high", "nominal_p_value", "fdr_q_value", "statistically_significant_after_fdr", "nominally_significant"]].to_csv(fdr_csv, index=False)
    print(f"Saved FDR statistical ledger to {fdr_csv}")

    # Summary Report
    rep_md = CLASS_DIR / "classification_report.md"
    rep_lines = ["# Master Classification Report (276 Full Cohort & Sensitivity Partitioning)\n"]
    rep_lines.append("| Analysis Set | Direction | Classifier | Within UAR | Original Cross UAR | Generated Cross UAR | Delta UAR | 95% CI | Nominal p | FDR q |")
    rep_lines.append("| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    for _, r in df_comp.iterrows():
        rep_lines.append(f"| {r['set']} | {r['direction']} | {r['classifier']} | {r['within_language_uar']:.4f} | {r['original_crosslanguage_uar']:.4f} | **{r['generated_crosslanguage_uar']:.4f}** | **{r['absolute_delta_uar']:+.4f}** | [{r['ci_95_low']:+.4f}, {r['ci_95_high']:+.4f}] | {r['nominal_p_value']:.4f} | {r['fdr_q_value']:.4f} |")
    rep_md.write_text("\n".join(rep_lines), encoding="utf-8")
    print(f"Saved classification report to {rep_md}")

if __name__ == "__main__":
    main()
