"""
classification.py

Baseline classification utilities for PD vs HC speech experiments.

The first thesis milestone uses extracted speech embeddings as input features
and evaluates within-language and cross-language PD classification.

Author: Prosenjit Chowdhury
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC


FEATURE_PREFIX = "feature_"


@dataclass
class EvaluationResult:
    """Classification result summary."""

    train_language: str
    test_language: str
    classifier: str
    uar: float
    accuracy: float
    sensitivity: float
    specificity: float
    auc: float | None
    n_train: int
    n_test: int


def split_features_labels(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract feature matrix X and label vector y from an embedding DataFrame.
    """
    feature_cols = [col for col in df.columns if col.startswith(FEATURE_PREFIX)]
    if not feature_cols:
        raise ValueError("No feature columns found. Expected columns named feature_0, feature_1, ...")

    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy()

    return X, y


def make_classifier(name: str = "svm") -> Tuple[Pipeline, Dict[str, list]]:
    """
    Create a classifier pipeline and hyperparameter grid.
    """
    name = name.lower()

    if name == "svm":
        pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("clf", SVC(kernel="linear", probability=True, class_weight="balanced")),
            ]
        )
        param_grid = {"clf__C": [0.01, 0.1, 1.0, 10.0]}

    elif name in {"logreg", "logistic_regression"}:
        pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=5_000,
                        class_weight="balanced",
                        solver="liblinear",
                    ),
                ),
            ]
        )
        param_grid = {"clf__C": [0.01, 0.1, 1.0, 10.0]}

    else:
        raise ValueError("Supported classifiers: svm, logreg")

    return pipeline, param_grid


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray | None = None) -> dict:
    """
    Compute core biomedical classification metrics.

    Sensitivity is calculated for PD.
    Specificity is calculated for HC.
    """
    labels = ["HC", "PD"]
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    auc = None
    if y_score is not None and len(np.unique(y_true)) == 2:
        binary_true = np.array([1 if label == "PD" else 0 for label in y_true])
        auc = roc_auc_score(binary_true, y_score)

    return {
        "uar": balanced_accuracy_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "auc": auc,
    }


def train_and_test_language_transfer(
    features_df: pd.DataFrame,
    train_language: str,
    test_language: str,
    classifier_name: str = "svm",
    inner_folds: int = 5,
) -> EvaluationResult:
    """
    Train on one language and test on another language.

    Example:
    train_language="Spanish", test_language="German"
    """
    train_df = features_df[features_df["language"] == train_language].copy()
    test_df = features_df[features_df["language"] == test_language].copy()

    if train_df.empty or test_df.empty:
        raise ValueError("Train or test subset is empty. Check language labels.")

    X_train, y_train = split_features_labels(train_df)
    X_test, y_test = split_features_labels(test_df)

    pipeline, param_grid = make_classifier(classifier_name)

    cv = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=42)
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring="balanced_accuracy",
        cv=cv,
        n_jobs=-1,
    )

    search.fit(X_train, y_train)

    y_pred = search.predict(X_test)
    y_score = None

    if hasattr(search.best_estimator_, "predict_proba"):
        proba = search.best_estimator_.predict_proba(X_test)
        class_order = list(search.best_estimator_.classes_)
        if "PD" in class_order:
            y_score = proba[:, class_order.index("PD")]

    metrics = compute_metrics(y_test, y_pred, y_score)

    return EvaluationResult(
        train_language=train_language,
        test_language=test_language,
        classifier=classifier_name,
        uar=metrics["uar"],
        accuracy=metrics["accuracy"],
        sensitivity=metrics["sensitivity"],
        specificity=metrics["specificity"],
        auc=metrics["auc"],
        n_train=len(train_df),
        n_test=len(test_df),
    )
