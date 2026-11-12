import os
import yaml
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.metrics import balanced_accuracy_score, recall_score, confusion_matrix, roc_auc_score
from scipy import stats

base_dir = r"C:\pd-speech-crosslingual\23_final_supervisor_evaluation"
paired_csv = os.path.join(base_dir, "02_dataset_index", "paired_original_converted_index.csv")
grids_yaml = os.path.join(base_dir, "01_config", "classifier_grids.yaml")
seeds_yaml = os.path.join(base_dir, "01_config", "random_seeds.yaml")

with open(grids_yaml, 'r') as f:
    grids = yaml.safe_load(f)

with open(seeds_yaml, 'r') as f:
    seeds = yaml.safe_load(f)

RANDOM_STATE = seeds.get('sklearn', 42)

MODELS = ["xlsr", "wav2vec2", "wavlm"]
LAYERS = [0, 4, 8, 11]

CLASSIFIERS = {
    "Logistic Regression": LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
    "Linear SVM": SVC(kernel="linear", probability=True, random_state=RANDOM_STATE),
    "RBF SVM": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE),
    "XGBoost": XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss"),
    "MLP": MLPClassifier(random_state=RANDOM_STATE, early_stopping=True, max_iter=1000)
}

def load_data(model, layer):
    orig_file = os.path.join(base_dir, "04_embeddings", model, f"{model}_layer{layer}_original.csv")
    conv_file = os.path.join(base_dir, "04_embeddings", model, f"{model}_layer{layer}_converted.csv")
    df_orig = pd.read_csv(orig_file)
    df_conv = pd.read_csv(conv_file)
    return df_orig, df_conv

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
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))

def run_evaluation():
    df_meta = pd.read_csv(paired_csv)
    
    all_results = []
    predictions = []
    
    for representation in MODELS:
        for layer in LAYERS:
            print(f"Processing {representation} layer {layer}...")
            df_orig_feats, df_conv_feats = load_data(representation, layer)
            
            # Merge with meta
            merged_orig = pd.merge(df_meta, df_orig_feats, on='sample_id')
            merged_conv = pd.merge(df_meta, df_conv_feats, on='sample_id')
            
            feat_cols = [c for c in merged_orig.columns if c.startswith('feature_')]
            
            X_orig = merged_orig[feat_cols].values
            X_conv = merged_conv[feat_cols].values
            y = (merged_orig['diagnosis'] == 'PD').astype(int).values
            lang = merged_orig['source_language'].values
            
            de_idx = np.where(lang == "German")[0]
            es_idx = np.where(lang == "Spanish")[0]
            
            for clf_name, clf_base in CLASSIFIERS.items():
                print(f"  {clf_name}")
                param_grid = grids[clf_name]
                
                # We need scaling for linear models and MLP. 
                # RF and XGBoost generally don't need it, but using it doesn't hurt much and ensures uniform pipeline.
                pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf_base)])
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
                
                grid_de = GridSearchCV(pipe, param_grid, cv=cv, scoring="balanced_accuracy", n_jobs=4)
                grid_es = GridSearchCV(pipe, param_grid, cv=cv, scoring="balanced_accuracy", n_jobs=4)
                grid_all = GridSearchCV(pipe, param_grid, cv=cv, scoring="balanced_accuracy", n_jobs=4)
                
                # --- Train German Model ---
                X_de_orig, y_de = X_orig[de_idx], y[de_idx]
                grid_de.fit(X_de_orig, y_de)
                best_de = grid_de.best_estimator_
                
                # Evaluate Within German (CV)
                y_de_within_pred = cross_val_predict(best_de, X_de_orig, y_de, cv=cv)
                uar = balanced_accuracy_score(y_de, y_de_within_pred)
                sens = recall_score(y_de, y_de_within_pred, pos_label=1)
                spec = recall_score(y_de, y_de_within_pred, pos_label=0)
                ci_l, ci_h = bootstrap_ci(y_de, y_de_within_pred)
                
                all_results.append({
                    "speech_condition": "original",
                    "scenario": "DE -> DE",
                    "source_cohort": "German",
                    "acoustic_language": "German",
                    "representation_model": representation,
                    "layer": layer,
                    "classifier": clf_name,
                    "UAR": uar, "PD_recall": sens, "HC_recall": spec, "ci_low": ci_l, "ci_high": ci_h, "N": len(y_de)
                })
                
                # Test German Model on Spanish (A1)
                X_es_orig, y_es = X_orig[es_idx], y[es_idx]
                y_es_orig_pred = best_de.predict(X_es_orig)
                uar = balanced_accuracy_score(y_es, y_es_orig_pred)
                sens = recall_score(y_es, y_es_orig_pred, pos_label=1)
                spec = recall_score(y_es, y_es_orig_pred, pos_label=0)
                ci_l, ci_h = bootstrap_ci(y_es, y_es_orig_pred)
                
                all_results.append({
                    "speech_condition": "original",
                    "scenario": "DE -> ES",
                    "source_cohort": "Spanish",
                    "acoustic_language": "Spanish",
                    "representation_model": representation,
                    "layer": layer,
                    "classifier": clf_name,
                    "UAR": uar, "PD_recall": sens, "HC_recall": spec, "ci_low": ci_l, "ci_high": ci_h, "N": len(y_es)
                })
                
                # Test German Model on Converted Spanish (A2)
                X_es_conv = X_conv[es_idx]
                y_es_conv_pred = best_de.predict(X_es_conv)
                uar = balanced_accuracy_score(y_es, y_es_conv_pred)
                sens = recall_score(y_es, y_es_conv_pred, pos_label=1)
                spec = recall_score(y_es, y_es_conv_pred, pos_label=0)
                ci_l, ci_h = bootstrap_ci(y_es, y_es_conv_pred)
                
                all_results.append({
                    "speech_condition": "converted",
                    "scenario": "DE -> ES_to_DE",
                    "source_cohort": "Spanish",
                    "acoustic_language": "German",
                    "representation_model": representation,
                    "layer": layer,
                    "classifier": clf_name,
                    "UAR": uar, "PD_recall": sens, "HC_recall": spec, "ci_low": ci_l, "ci_high": ci_h, "N": len(y_es)
                })
                
                # --- Train Spanish Model ---
                grid_es.fit(X_es_orig, y_es)
                best_es = grid_es.best_estimator_
                
                # Evaluate Within Spanish (CV)
                y_es_within_pred = cross_val_predict(best_es, X_es_orig, y_es, cv=cv)
                uar = balanced_accuracy_score(y_es, y_es_within_pred)
                sens = recall_score(y_es, y_es_within_pred, pos_label=1)
                spec = recall_score(y_es, y_es_within_pred, pos_label=0)
                ci_l, ci_h = bootstrap_ci(y_es, y_es_within_pred)
                
                all_results.append({
                    "speech_condition": "original",
                    "scenario": "ES -> ES",
                    "source_cohort": "Spanish",
                    "acoustic_language": "Spanish",
                    "representation_model": representation,
                    "layer": layer,
                    "classifier": clf_name,
                    "UAR": uar, "PD_recall": sens, "HC_recall": spec, "ci_low": ci_l, "ci_high": ci_h, "N": len(y_es)
                })
                
                # Test Spanish Model on German (B1)
                y_de_orig_pred = best_es.predict(X_de_orig)
                uar = balanced_accuracy_score(y_de, y_de_orig_pred)
                sens = recall_score(y_de, y_de_orig_pred, pos_label=1)
                spec = recall_score(y_de, y_de_orig_pred, pos_label=0)
                ci_l, ci_h = bootstrap_ci(y_de, y_de_orig_pred)
                
                all_results.append({
                    "speech_condition": "original",
                    "scenario": "ES -> DE",
                    "source_cohort": "German",
                    "acoustic_language": "German",
                    "representation_model": representation,
                    "layer": layer,
                    "classifier": clf_name,
                    "UAR": uar, "PD_recall": sens, "HC_recall": spec, "ci_low": ci_l, "ci_high": ci_h, "N": len(y_de)
                })
                
                # Test Spanish Model on Converted German (B2)
                X_de_conv = X_conv[de_idx]
                y_de_conv_pred = best_es.predict(X_de_conv)
                uar = balanced_accuracy_score(y_de, y_de_conv_pred)
                sens = recall_score(y_de, y_de_conv_pred, pos_label=1)
                spec = recall_score(y_de, y_de_conv_pred, pos_label=0)
                ci_l, ci_h = bootstrap_ci(y_de, y_de_conv_pred)
                
                all_results.append({
                    "speech_condition": "converted",
                    "scenario": "ES -> DE_to_ES",
                    "source_cohort": "German",
                    "acoustic_language": "Spanish",
                    "representation_model": representation,
                    "layer": layer,
                    "classifier": clf_name,
                    "UAR": uar, "PD_recall": sens, "HC_recall": spec, "ci_low": ci_l, "ci_high": ci_h, "N": len(y_de)
                })
                
                # --- Combined Model ES+DE -> ES+DE (CV) ---
                grid_all.fit(X_orig, y)
                best_all = grid_all.best_estimator_
                y_all_pred = cross_val_predict(best_all, X_orig, y, cv=cv)
                uar = balanced_accuracy_score(y, y_all_pred)
                sens = recall_score(y, y_all_pred, pos_label=1)
                spec = recall_score(y, y_all_pred, pos_label=0)
                ci_l, ci_h = bootstrap_ci(y, y_all_pred)
                all_results.append({
                    "speech_condition": "original",
                    "scenario": "ES+DE -> ES+DE",
                    "source_cohort": "Spanish+German",
                    "acoustic_language": "Spanish+German",
                    "representation_model": representation,
                    "layer": layer,
                    "classifier": clf_name,
                    "UAR": uar, "PD_recall": sens, "HC_recall": spec, "ci_low": ci_l, "ci_high": ci_h, "N": len(y)
                })
                
    df_results = pd.DataFrame(all_results)
    out_dir = os.path.join(base_dir, "05_classification", "summaries")
    os.makedirs(out_dir, exist_ok=True)
    df_results.to_csv(os.path.join(out_dir, "all_classification_results.csv"), index=False)
    print("Saved all classification results.")

if __name__ == "__main__":
    run_evaluation()
