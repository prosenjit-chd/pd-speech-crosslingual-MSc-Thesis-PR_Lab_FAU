import os
import yaml
import json
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import balanced_accuracy_score, recall_score
from sklearn.base import BaseEstimator, ClassifierMixin
import warnings

# GPU libraries
import cupy as cp
from cuml.linear_model import LogisticRegression as cuml_LR
from cuml.svm import SVC as cuml_SVC
from cuml.ensemble import RandomForestClassifier as cuml_RF
from xgboost import XGBClassifier as gpu_XGB
import torch
import torch.nn as nn
import torch.optim as optim

warnings.filterwarnings("ignore")

class TorchMLP(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden_layer_sizes=(100,), alpha=0.0001, learning_rate_init=0.001, max_iter=200):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def fit(self, X, y):
        X = np.ascontiguousarray(X, dtype=np.float32)
        y = np.ascontiguousarray(y, dtype=np.int64)
        
        layers = []
        input_dim = X.shape[1]
        for h in self.hidden_layer_sizes:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.ReLU())
            input_dim = h
        layers.append(nn.Linear(input_dim, 2))
        
        self.model = nn.Sequential(*layers).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate_init, weight_decay=self.alpha)
        self.criterion = nn.CrossEntropyLoss()
        
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y, dtype=torch.long).to(self.device)
        
        for epoch in range(self.max_iter):
            self.model.train()
            self.optimizer.zero_grad()
            outputs = self.model(X_t)
            loss = self.criterion(outputs, y_t)
            loss.backward()
            self.optimizer.step()
        return self
        
    def predict(self, X):
        X = np.ascontiguousarray(X, dtype=np.float32)
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_t)
            _, predicted = torch.max(outputs.data, 1)
        return predicted.cpu().numpy()

base_dir = "/mnt/c/pd-speech-crosslingual/23_final_supervisor_evaluation"
paired_csv = os.path.join(base_dir, "02_dataset_index", "paired_original_converted_index.csv")
grids_yaml = os.path.join(base_dir, "01_config", "classifier_grids_GPU.yaml")
seeds_yaml = os.path.join(base_dir, "01_config", "random_seeds.yaml")

with open(grids_yaml, 'r') as f:
    grids = yaml.safe_load(f)

with open(seeds_yaml, 'r') as f:
    seeds = yaml.safe_load(f)

RANDOM_STATE = seeds.get('sklearn', 42)

MODELS = ["xlsr", "wav2vec2", "wavlm"]
LAYERS = [0, 4, 8, 11]

class CUMLWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, clf):
        self.clf = clf
    def fit(self, X, y):
        X = np.ascontiguousarray(X, dtype=np.float32)
        y = np.ascontiguousarray(y, dtype=np.float32)
        self.clf.fit(X, y)
        return self
    def predict(self, X):
        X = np.ascontiguousarray(X, dtype=np.float32)
        return self.clf.predict(X)
    def set_params(self, **params):
        self.clf.set_params(**params)
        return self
    def get_params(self, deep=True):
        return self.clf.get_params(deep)

CLASSIFIERS = {
    "Logistic Regression": CUMLWrapper(cuml_LR(max_iter=1000)),
    "Linear SVM": CUMLWrapper(cuml_SVC(kernel="linear", probability=True)),
    "RBF SVM": CUMLWrapper(cuml_SVC(kernel="rbf", probability=True)),
    "Random Forest": CUMLWrapper(cuml_RF(random_state=RANDOM_STATE)),
    "XGBoost": gpu_XGB(tree_method="hist", device="cuda", random_state=RANDOM_STATE, eval_metric="logloss"),
    "MLP": TorchMLP(max_iter=200)
}

TOTAL_BLOCKS = len(MODELS) * len(LAYERS) * len(CLASSIFIERS)

out_dir = os.path.join(base_dir, "06_sensitivity_analysis_264")
chk_dir = os.path.join(out_dir, "checkpoints_gpu")
summ_dir = os.path.join(out_dir, "summaries")
pred_dir = os.path.join(out_dir, "predictions")

for d in [chk_dir, summ_dir, pred_dir]:
    os.makedirs(d, exist_ok=True)

progress_file = os.path.join(chk_dir, "progress.json")

progress_state = {
    "completed_blocks": [],
    "total_blocks": TOTAL_BLOCKS,
    "current_model": "",
    "current_layer": "",
    "current_classifier": "",
    "current_scenario": "",
    "current_outer_fold": "",
    "timestamp": ""
}

def update_progress(scenario, fold=None):
    progress_state["current_scenario"] = scenario
    progress_state["current_outer_fold"] = str(fold) if fold is not None else "ALL"
    progress_state["timestamp"] = datetime.now().isoformat()
    if torch.cuda.is_available():
        mem = torch.cuda.memory_allocated() / (1024 ** 2)
        progress_state["gpu_memory_mb"] = mem
    with open(progress_file, 'w') as f:
        json.dump(progress_state, f, indent=4)

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
    if len(scores) == 0:
        return np.nan, np.nan
    scores = np.array(scores)
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))

def get_nested_cv_preds(X, y, sample_ids, pipe, param_grid, cv_outer, cv_inner, 
                        speech_cond, scenario, clf_name, representation, layer):
    y_pred = np.zeros_like(y)
    preds_log = []
    hyperparams_log = []
    
    for fold, (train_idx, test_idx) in enumerate(cv_outer.split(X, y)):
        update_progress(scenario, fold)
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        test_ids = sample_ids[test_idx]
        
        grid = GridSearchCV(pipe, param_grid, cv=cv_inner, scoring="balanced_accuracy", n_jobs=1)
        grid.fit(X_train, y_train)
        
        preds = grid.predict(X_test)
        y_pred[test_idx] = preds
        
        best_params = str(grid.best_params_)
        
        hyperparams_log.append({
            "speech_condition": speech_cond,
            "scenario": scenario,
            "classifier": clf_name,
            "model": representation,
            "layer": layer,
            "outer_fold": fold,
            "selected_hyperparameters": best_params
        })
        
        for i, sid in enumerate(test_ids):
            preds_log.append({
                "sample_id": sid,
                "true_label": int(y_test[i]),
                "predicted_label": int(preds[i]),
                "speech_condition": speech_cond,
                "scenario": scenario,
                "classifier": clf_name,
                "model": representation,
                "layer": layer,
                "outer_fold": fold
            })
            
    return y_pred, preds_log, hyperparams_log

def get_test_preds(grid, X_test, y_test, test_ids, speech_cond, scenario, clf_name, representation, layer):
    update_progress(scenario, "ALL")
    
    preds = grid.predict(X_test)
    preds_log = []
    hyperparams_log = []
    best_params = str(grid.best_params_)
    
    hyperparams_log.append({
        "speech_condition": speech_cond,
        "scenario": scenario,
        "classifier": clf_name,
        "model": representation,
        "layer": layer,
        "outer_fold": "ALL",
        "selected_hyperparameters": best_params
    })
    for i, sid in enumerate(test_ids):
        preds_log.append({
            "sample_id": sid,
            "true_label": int(y_test[i]),
            "predicted_label": int(preds[i]),
            "speech_condition": speech_cond,
            "scenario": scenario,
            "classifier": clf_name,
            "model": representation,
            "layer": layer,
            "outer_fold": "ALL"
        })
    return preds, preds_log, hyperparams_log

def run_evaluation():
    global progress_state
    
    df_meta = pd.read_csv(paired_csv)
    # Sensitivity exclusion
    df_meta = df_meta[df_meta['development_subject'] == False].reset_index(drop=True)
    
    all_results = []
    all_predictions_log = []
    all_hyperparams_log = []
    
    chk_res_file = os.path.join(chk_dir, "checkpoint_results_GPU.csv")
    chk_pred_file = os.path.join(chk_dir, "checkpoint_preds_GPU.csv")
    chk_hyp_file = os.path.join(chk_dir, "checkpoint_hyperparams_GPU.csv")
    
    if os.path.exists(progress_file):
        print("Found existing GPU progress file. Loading checkpoints...")
        with open(progress_file, 'r') as f:
            progress_state = json.load(f)
        if os.path.exists(chk_res_file):
            all_results = pd.read_csv(chk_res_file).to_dict('records')
        if os.path.exists(chk_pred_file):
            all_predictions_log = pd.read_csv(chk_pred_file).to_dict('records')
        if os.path.exists(chk_hyp_file):
            all_hyperparams_log = pd.read_csv(chk_hyp_file).to_dict('records')
        print(f"Resuming with {len(progress_state['completed_blocks'])} / {TOTAL_BLOCKS} blocks completed.")
    else:
        print("No GPU checkpoint found. Starting from scratch.")

    for representation in MODELS:
        progress_state["current_model"] = representation
        for layer in LAYERS:
            progress_state["current_layer"] = layer
            print(f"Loading data for {representation} layer {layer} (N={len(df_meta)})...")
            df_orig_feats, df_conv_feats = load_data(representation, layer)
            
            merged_orig = pd.merge(df_meta, df_orig_feats, on='sample_id')
            merged_conv = pd.merge(df_meta, df_conv_feats, on='sample_id')
            
            feat_cols = [c for c in merged_orig.columns if c.startswith('feature_')]
            
            X_orig = merged_orig[feat_cols].values.astype(np.float32)
            X_conv = merged_conv[feat_cols].values.astype(np.float32)
            y = (merged_orig['diagnosis'] == 'PD').astype(int).values
            lang = merged_orig['source_language'].values
            sample_ids = merged_orig['sample_id'].values
            
            de_idx = np.where(lang == "German")[0]
            es_idx = np.where(lang == "Spanish")[0]
            
            for clf_name, clf_base in CLASSIFIERS.items():
                block_id = f"{representation}_layer{layer}_{clf_name}"
                if block_id in progress_state["completed_blocks"]:
                    print(f"  Skipping {clf_name} (already completed)")
                    continue
                
                print(f"  Running {clf_name} ...")
                progress_state["current_classifier"] = clf_name
                
                param_grid = grids[clf_name]
                if clf_name in ["Logistic Regression", "Linear SVM", "RBF SVM", "MLP"]:
                    pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf_base)])
                else:
                    pipe = Pipeline([("clf", clf_base)])
                
                inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
                outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
                
                # --- German Model (Within-Language Nested CV) ---
                X_de_orig, y_de, sid_de = X_orig[de_idx], y[de_idx], sample_ids[de_idx]
                y_de_pred, log_de, hyp_de = get_nested_cv_preds(X_de_orig, y_de, sid_de, pipe, param_grid, outer_cv, inner_cv, 
                                                        "original", "DE_train__DE_original_test", clf_name, representation, layer)
                all_predictions_log.extend(log_de)
                all_hyperparams_log.extend(hyp_de)
                uar = balanced_accuracy_score(y_de, y_de_pred)
                sens = recall_score(y_de, y_de_pred, pos_label=1)
                spec = recall_score(y_de, y_de_pred, pos_label=0)
                ci_l, ci_h = bootstrap_ci(y_de, y_de_pred)
                all_results.append({
                    "speech_condition": "original", "scenario": "DE_train__DE_original_test",
                    "source_cohort": "German", "acoustic_language": "German",
                    "representation_model": representation, "layer": layer,
                    "classifier": clf_name, "UAR": uar, "PD_recall": sens, "HC_recall": spec,
                    "ci_low": ci_l, "ci_high": ci_h, "N": len(y_de)
                })
                
                # --- Train German Model on ALL DE (for Cross-Language Testing) ---
                update_progress("DE_train__ALL", "ALL")
                grid_de = GridSearchCV(pipe, param_grid, cv=inner_cv, scoring="balanced_accuracy", n_jobs=1)
                grid_de.fit(X_de_orig, y_de)
                
                # Test German Model on Spanish Original
                X_es_orig, y_es, sid_es = X_orig[es_idx], y[es_idx], sample_ids[es_idx]
                y_es_orig_pred, log_es_orig, hyp_es_orig = get_test_preds(grid_de, X_es_orig, y_es, sid_es, "original", "DE_train__ES_original_test", clf_name, representation, layer)
                all_predictions_log.extend(log_es_orig)
                all_hyperparams_log.extend(hyp_es_orig)
                uar = balanced_accuracy_score(y_es, y_es_orig_pred)
                sens = recall_score(y_es, y_es_orig_pred, pos_label=1)
                spec = recall_score(y_es, y_es_orig_pred, pos_label=0)
                ci_l, ci_h = bootstrap_ci(y_es, y_es_orig_pred)
                all_results.append({
                    "speech_condition": "original", "scenario": "DE_train__ES_original_test",
                    "source_cohort": "Spanish", "acoustic_language": "Spanish",
                    "representation_model": representation, "layer": layer,
                    "classifier": clf_name, "UAR": uar, "PD_recall": sens, "HC_recall": spec,
                    "ci_low": ci_l, "ci_high": ci_h, "N": len(y_es)
                })
                
                # Test German Model on Converted Spanish
                X_es_conv = X_conv[es_idx]
                y_es_conv_pred, log_es_conv, hyp_es_conv = get_test_preds(grid_de, X_es_conv, y_es, sid_es, "converted", "DE_train__ES_to_DE_converted_test", clf_name, representation, layer)
                all_predictions_log.extend(log_es_conv)
                all_hyperparams_log.extend(hyp_es_conv)
                uar = balanced_accuracy_score(y_es, y_es_conv_pred)
                sens = recall_score(y_es, y_es_conv_pred, pos_label=1)
                spec = recall_score(y_es, y_es_conv_pred, pos_label=0)
                ci_l, ci_h = bootstrap_ci(y_es, y_es_conv_pred)
                all_results.append({
                    "speech_condition": "converted", "scenario": "DE_train__ES_to_DE_converted_test",
                    "source_cohort": "Spanish", "acoustic_language": "German",
                    "representation_model": representation, "layer": layer,
                    "classifier": clf_name, "UAR": uar, "PD_recall": sens, "HC_recall": spec,
                    "ci_low": ci_l, "ci_high": ci_h, "N": len(y_es)
                })
                
                # --- Spanish Model (Within-Language Nested CV) ---
                y_es_pred, log_es, hyp_es = get_nested_cv_preds(X_es_orig, y_es, sid_es, pipe, param_grid, outer_cv, inner_cv, 
                                                        "original", "ES_train__ES_original_test", clf_name, representation, layer)
                all_predictions_log.extend(log_es)
                all_hyperparams_log.extend(hyp_es)
                uar = balanced_accuracy_score(y_es, y_es_pred)
                sens = recall_score(y_es, y_es_pred, pos_label=1)
                spec = recall_score(y_es, y_es_pred, pos_label=0)
                ci_l, ci_h = bootstrap_ci(y_es, y_es_pred)
                all_results.append({
                    "speech_condition": "original", "scenario": "ES_train__ES_original_test",
                    "source_cohort": "Spanish", "acoustic_language": "Spanish",
                    "representation_model": representation, "layer": layer,
                    "classifier": clf_name, "UAR": uar, "PD_recall": sens, "HC_recall": spec,
                    "ci_low": ci_l, "ci_high": ci_h, "N": len(y_es)
                })
                
                # --- Train Spanish Model on ALL ES (for Cross-Language Testing) ---
                update_progress("ES_train__ALL", "ALL")
                grid_es = GridSearchCV(pipe, param_grid, cv=inner_cv, scoring="balanced_accuracy", n_jobs=1)
                grid_es.fit(X_es_orig, y_es)
                
                # Test Spanish Model on German Original
                y_de_orig_pred, log_de_orig, hyp_de_orig = get_test_preds(grid_es, X_de_orig, y_de, sid_de, "original", "ES_train__DE_original_test", clf_name, representation, layer)
                all_predictions_log.extend(log_de_orig)
                all_hyperparams_log.extend(hyp_de_orig)
                uar = balanced_accuracy_score(y_de, y_de_orig_pred)
                sens = recall_score(y_de, y_de_orig_pred, pos_label=1)
                spec = recall_score(y_de, y_de_orig_pred, pos_label=0)
                ci_l, ci_h = bootstrap_ci(y_de, y_de_orig_pred)
                all_results.append({
                    "speech_condition": "original", "scenario": "ES_train__DE_original_test",
                    "source_cohort": "German", "acoustic_language": "German",
                    "representation_model": representation, "layer": layer,
                    "classifier": clf_name, "UAR": uar, "PD_recall": sens, "HC_recall": spec,
                    "ci_low": ci_l, "ci_high": ci_h, "N": len(y_de)
                })
                
                # Test Spanish Model on Converted German
                X_de_conv = X_conv[de_idx]
                y_de_conv_pred, log_de_conv, hyp_de_conv = get_test_preds(grid_es, X_de_conv, y_de, sid_de, "converted", "ES_train__DE_to_ES_converted_test", clf_name, representation, layer)
                all_predictions_log.extend(log_de_conv)
                all_hyperparams_log.extend(hyp_de_conv)
                uar = balanced_accuracy_score(y_de, y_de_conv_pred)
                sens = recall_score(y_de, y_de_conv_pred, pos_label=1)
                spec = recall_score(y_de, y_de_conv_pred, pos_label=0)
                ci_l, ci_h = bootstrap_ci(y_de, y_de_conv_pred)
                all_results.append({
                    "speech_condition": "converted", "scenario": "ES_train__DE_to_ES_converted_test",
                    "source_cohort": "German", "acoustic_language": "Spanish",
                    "representation_model": representation, "layer": layer,
                    "classifier": clf_name, "UAR": uar, "PD_recall": sens, "HC_recall": spec,
                    "ci_low": ci_l, "ci_high": ci_h, "N": len(y_de)
                })
                
                # --- Combined Model Nested CV ---
                y_all_pred, log_all, hyp_all = get_nested_cv_preds(X_orig, y, sample_ids, pipe, param_grid, outer_cv, inner_cv, 
                                                        "original", "Combined_train__Combined_original_test", clf_name, representation, layer)
                all_predictions_log.extend(log_all)
                all_hyperparams_log.extend(hyp_all)
                uar = balanced_accuracy_score(y, y_all_pred)
                sens = recall_score(y, y_all_pred, pos_label=1)
                spec = recall_score(y, y_all_pred, pos_label=0)
                ci_l, ci_h = bootstrap_ci(y, y_all_pred)
                all_results.append({
                    "speech_condition": "original", "scenario": "Combined_train__Combined_original_test",
                    "source_cohort": "Spanish+German", "acoustic_language": "Spanish+German",
                    "representation_model": representation, "layer": layer,
                    "classifier": clf_name, "UAR": uar, "PD_recall": sens, "HC_recall": spec,
                    "ci_low": ci_l, "ci_high": ci_h, "N": len(y)
                })
                
                # End of block checkpointing
                progress_state["completed_blocks"].append(block_id)
                pd.DataFrame(all_results).to_csv(chk_res_file, index=False)
                pd.DataFrame(all_predictions_log).to_csv(chk_pred_file, index=False)
                pd.DataFrame(all_hyperparams_log).to_csv(chk_hyp_file, index=False)
                update_progress("BLOCK_DONE", "ALL")
                print(f"  Saved checkpoint for block {block_id}")
                
    # Final output saving
    df_results = pd.DataFrame(all_results)
    df_preds = pd.DataFrame(all_predictions_log)
    df_hyp = pd.DataFrame(all_hyperparams_log)
    
    df_results.to_csv(os.path.join(summ_dir, "sensitivity_classification_results_NESTED_CV_GPU.csv"), index=False)
    df_preds.to_csv(os.path.join(pred_dir, "sensitivity_sample_predictions_NESTED_CV_GPU.csv"), index=False)
    df_hyp.to_csv(os.path.join(summ_dir, "sensitivity_selected_hyperparameters_NESTED_CV_GPU.csv"), index=False)
    print("Saved final GPU classification results, predictions, and hyperparameters (NESTED CV).")

if __name__ == "__main__":
    run_evaluation()
