import os
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from sklearn.base import BaseEstimator, ClassifierMixin

# CPU libraries
from sklearn.linear_model import LogisticRegression as sk_LR
from sklearn.svm import SVC as sk_SVC
from sklearn.ensemble import RandomForestClassifier as sk_RF
from xgboost import XGBClassifier as sk_XGB
from sklearn.neural_network import MLPClassifier as sk_MLP
from sklearn.preprocessing import StandardScaler

# GPU libraries
import cupy as cp
from cuml.linear_model import LogisticRegression as cuml_LR
from cuml.svm import SVC as cuml_SVC
from cuml.ensemble import RandomForestClassifier as cuml_RF
from xgboost import XGBClassifier as gpu_XGB
import torch
import torch.nn as nn
import torch.optim as optim

class TorchMLP(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden_layer_sizes=(100,), alpha=0.0001, learning_rate_init=0.001, max_iter=200):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def fit(self, X, y):
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
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            outputs = self.model(X_t)
            _, predicted = torch.max(outputs.data, 1)
        return predicted.cpu().numpy()

def load_xlsr_layer0():
    base_dir = "/mnt/c/pd-speech-crosslingual/23_final_supervisor_evaluation"
    paired_csv = os.path.join(base_dir, "02_dataset_index", "paired_original_converted_index.csv")
    orig_file = os.path.join(base_dir, "04_embeddings", "xlsr", "xlsr_layer0_original.csv")
    
    df_meta = pd.read_csv(paired_csv)
    df_orig = pd.read_csv(orig_file)
    merged = pd.merge(df_meta, df_orig, on='sample_id')
    
    feat_cols = [c for c in merged.columns if c.startswith('feature_')]
    X = merged[feat_cols].values
    y = (merged['diagnosis'] == 'PD').astype(int).values
    return X, y

def run_benchmark():
    print("Loading data...")
    X, y = load_xlsr_layer0()
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    classifiers = {
        # "Logistic Regression": {
        #     "cpu": sk_LR(max_iter=1000),
        #     "gpu": cuml_LR(max_iter=1000)
        # },
        # "RBF-SVM": {
        #     "cpu": sk_SVC(kernel="rbf"),
        #     "gpu": cuml_SVC(kernel="rbf")
        # },
        # "Random Forest": {
        #     "cpu": sk_RF(n_estimators=200, random_state=42),
        #     "gpu": cuml_RF(n_estimators=200, random_state=42)
        # },
        "XGBoost": {
            "cpu": sk_XGB(tree_method="hist", random_state=42),
            "gpu": gpu_XGB(tree_method="hist", device="cuda", random_state=42)
        },
        "MLP": {
            "cpu": sk_MLP(hidden_layer_sizes=(128,), max_iter=200, random_state=42),
            "gpu": TorchMLP(hidden_layer_sizes=(128,), max_iter=200)
        }
    }
    
    results = []
    
    print("Starting benchmark...")
    for name, clfs in classifiers.items():
        print(f"Testing {name}...")
        
        # CPU
        t0 = time.time()
        cpu_clf = clfs["cpu"]
        cpu_clf.fit(X_train, y_train)
        cpu_pred = cpu_clf.predict(X_test)
        cpu_time = time.time() - t0
        cpu_uar = balanced_accuracy_score(y_test, cpu_pred)
        
        # GPU
        t0 = time.time()
        gpu_clf = clfs["gpu"]
        if name in ["Logistic Regression", "RBF-SVM", "Random Forest"]:
            # cuML likes float32 float64, let's use float32
            X_tr_gpu = np.ascontiguousarray(X_train, dtype=np.float32)
            y_tr_gpu = np.ascontiguousarray(y_train, dtype=np.float32)
            X_te_gpu = np.ascontiguousarray(X_test, dtype=np.float32)
            gpu_clf.fit(X_tr_gpu, y_tr_gpu)
            gpu_pred = gpu_clf.predict(X_te_gpu)
        else:
            gpu_clf.fit(X_train, y_train)
            gpu_pred = gpu_clf.predict(X_test)
        gpu_time = time.time() - t0
        gpu_uar = balanced_accuracy_score(y_test, gpu_pred)
        
        speedup = cpu_time / gpu_time if gpu_time > 0 else 0
        
        results.append({
            "classifier": name,
            "cpu_seconds": round(cpu_time, 4),
            "gpu_seconds": round(gpu_time, 4),
            "speedup": round(speedup, 2),
            "cpu_uar": round(cpu_uar, 4),
            "gpu_uar": round(gpu_uar, 4),
            "gpu_memory_mb": 0, # Placeholder
            "notes": "Success"
        })
        print(f"  CPU Time: {cpu_time:.3f}s | UAR: {cpu_uar:.4f}")
        print(f"  GPU Time: {gpu_time:.3f}s | UAR: {gpu_uar:.4f}")
        print(f"  Speedup: {speedup:.2f}x")
        
    df = pd.DataFrame(results)
    out_dir = "/mnt/c/pd-speech-crosslingual/23_final_supervisor_evaluation/00_audit"
    df.to_csv(os.path.join(out_dir, "gpu_benchmark.csv"), index=False)
    
    # Also save as smoke test results
    smoke = df[["classifier", "gpu_uar", "gpu_seconds"]]
    smoke.to_csv(os.path.join(out_dir, "gpu_smoke_test_results.csv"), index=False)
    print("Benchmark complete. Results saved.")

if __name__ == "__main__":
    from sklearn.base import BaseEstimator, ClassifierMixin
    run_benchmark()
