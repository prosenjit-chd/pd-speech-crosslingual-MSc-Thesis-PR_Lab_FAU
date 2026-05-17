import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from src.metrics import calculate_metrics
import logging

logger = logging.getLogger(__name__)

def get_classifier_pipeline(clf_name):
    """Returns a scikit-learn pipeline with scaler and classifier."""
    if clf_name == "linear_svm":
        clf = SVC(kernel='linear', probability=True, class_weight='balanced', random_state=42)
        param_grid = {'clf__C': [0.01, 0.1, 1, 10]}
    elif clf_name == "logistic_regression":
        clf = LogisticRegression(solver='liblinear', class_weight='balanced', random_state=42)
        param_grid = {'clf__C': [0.01, 0.1, 1, 10]}
    else:
        raise ValueError(f"Unknown classifier: {clf_name}")
        
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', clf)
    ])
    
    return pipeline, param_grid

def run_nested_cv(X_train, y_train, groups_train, X_test, y_test, clf_name, outer_folds=10, inner_folds=9):
    """
    Runs nested cross-validation (or a simple train/test split if X_test is provided for cross-language).
    If X_test and y_test are provided, it treats X_train/y_train as the source language for training/tuning,
    and tests strictly on X_test/y_test.
    If X_test is None, it performs standard nested CV on X_train/y_train (within-language or mixed).
    """
    
    pipeline, param_grid = get_classifier_pipeline(clf_name)
    
    # Cross-language scenario: Train on source, Test on target
    if X_test is not None:
        # Determine inner CV strategy for hyperparameter tuning
        n_samples = len(y_train)
        cv_inner_splits = min(inner_folds, sum(y_train == 0), sum(y_train == 1))
        
        if cv_inner_splits < 2:
            logger.warning("Not enough samples for inner CV. Training directly.")
            pipeline.fit(X_train, y_train)
            best_model = pipeline
        else:
            if groups_train is not None and len(np.unique(groups_train)) > cv_inner_splits:
                inner_cv = StratifiedGroupKFold(n_splits=cv_inner_splits)
                grid = GridSearchCV(pipeline, param_grid, cv=inner_cv, scoring='balanced_accuracy', n_jobs=-1)
                grid.fit(X_train, y_train, groups=groups_train)
            else:
                inner_cv = StratifiedKFold(n_splits=cv_inner_splits, shuffle=True, random_state=42)
                grid = GridSearchCV(pipeline, param_grid, cv=inner_cv, scoring='balanced_accuracy', n_jobs=-1)
                grid.fit(X_train, y_train)
            best_model = grid.best_estimator_
            
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, 'predict_proba') else None
        
        metrics = calculate_metrics(y_test, y_pred, y_prob)
        return metrics

    # Within-language or mixed scenario: Nested CV
    else:
        n_samples = len(y_train)
        cv_outer_splits = min(outer_folds, sum(y_train == 0), sum(y_train == 1))
        
        if cv_outer_splits < 2:
            logger.warning("Not enough samples for outer CV. Cannot evaluate reliably.")
            return None
            
        if groups_train is not None and len(np.unique(groups_train)) > cv_outer_splits:
            outer_cv = StratifiedGroupKFold(n_splits=cv_outer_splits)
            splits = outer_cv.split(X_train, y_train, groups=groups_train)
        else:
            outer_cv = StratifiedKFold(n_splits=cv_outer_splits, shuffle=True, random_state=42)
            splits = outer_cv.split(X_train, y_train)

        all_y_true = []
        all_y_pred = []
        all_y_prob = []

        for train_idx, test_idx in splits:
            X_tr, X_te = X_train[train_idx], X_train[test_idx]
            y_tr, y_te = y_train[train_idx], y_train[test_idx]
            g_tr = groups_train[train_idx] if groups_train is not None else None
            
            cv_inner_splits = min(inner_folds, sum(y_tr == 0), sum(y_tr == 1))
            
            if cv_inner_splits < 2:
                pipeline.fit(X_tr, y_tr)
                best_model = pipeline
            else:
                if g_tr is not None and len(np.unique(g_tr)) > cv_inner_splits:
                    inner_cv = StratifiedGroupKFold(n_splits=cv_inner_splits)
                    grid = GridSearchCV(pipeline, param_grid, cv=inner_cv, scoring='balanced_accuracy', n_jobs=-1)
                    grid.fit(X_tr, y_tr, groups=g_tr)
                else:
                    inner_cv = StratifiedKFold(n_splits=cv_inner_splits, shuffle=True, random_state=42)
                    grid = GridSearchCV(pipeline, param_grid, cv=inner_cv, scoring='balanced_accuracy', n_jobs=-1)
                    grid.fit(X_tr, y_tr)
                best_model = grid.best_estimator_
                
            y_pred = best_model.predict(X_te)
            y_prob = best_model.predict_proba(X_te)[:, 1] if hasattr(best_model, 'predict_proba') else None
            
            all_y_true.extend(y_te)
            all_y_pred.extend(y_pred)
            if y_prob is not None:
                all_y_prob.extend(y_prob)

        metrics = calculate_metrics(np.array(all_y_true), np.array(all_y_pred), np.array(all_y_prob) if all_y_prob else None)
        return metrics

def run_classification_scenarios(df, feature_cols, classifiers, outer_folds, inner_folds):
    """
    Runs all defined Spanish/German cross-language scenarios.
    """
    scenarios = [
        ("Spanish", "Spanish"),
        ("Spanish", "German"),
        ("German", "German"),
        ("German", "Spanish"),
        ("Spanish+German", "Spanish+German")
    ]
    
    results = []
    
    # Map labels to binary
    # Assumes label mapping: PD -> 1, HC -> 0
    df['target'] = df['label'].apply(lambda x: 1 if x.upper() == 'PD' else 0)
    
    for train_lang, test_lang in scenarios:
        for clf_name in classifiers:
            logger.info(f"Running scenario: Train={train_lang}, Test={test_lang}, Classifier={clf_name}")
            
            if train_lang == "Spanish+German":
                # Mixed scenario
                mask_train = df['language'].isin(['Spanish', 'German'])
                X_train = df.loc[mask_train, feature_cols].values
                y_train = df.loc[mask_train, 'target'].values
                groups_train = df.loc[mask_train, 'speaker_id'].values
                X_test, y_test = None, None
            else:
                # Specific language scenario
                mask_train = df['language'] == train_lang
                X_train = df.loc[mask_train, feature_cols].values
                y_train = df.loc[mask_train, 'target'].values
                groups_train = df.loc[mask_train, 'speaker_id'].values
                
                if train_lang == test_lang:
                    X_test, y_test = None, None
                else:
                    mask_test = df['language'] == test_lang
                    X_test = df.loc[mask_test, feature_cols].values
                    y_test = df.loc[mask_test, 'target'].values
                    
            if len(y_train) == 0 or (X_test is not None and len(y_test) == 0):
                logger.warning(f"Skipping {train_lang}->{test_lang} due to missing data.")
                continue
                
            metrics = run_nested_cv(X_train, y_train, groups_train, X_test, y_test, clf_name, outer_folds, inner_folds)
            
            if metrics:
                results.append({
                    "train_language": train_lang,
                    "test_language": test_lang,
                    "classifier": clf_name,
                    "uar": metrics['uar'],
                    "accuracy": metrics['accuracy'],
                    "sensitivity": metrics['sensitivity'],
                    "specificity": metrics['specificity'],
                    "auc": metrics['auc'],
                    "n_train": len(y_train),
                    "n_test": len(y_test) if y_test is not None else len(y_train)
                })
                
    return pd.DataFrame(results)
