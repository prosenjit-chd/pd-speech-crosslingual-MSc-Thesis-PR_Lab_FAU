import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, confusion_matrix, recall_score
import logging

logger = logging.getLogger(__name__)

def calculate_metrics(y_true, y_pred, y_prob=None):
    """
    Calculates essential classification metrics for PD vs HC.
    y_true: 1 for PD, 0 for HC.
    """
    try:
        acc = accuracy_score(y_true, y_pred)
        # UAR / Balanced Accuracy
        uar = balanced_accuracy_score(y_true, y_pred)
        
        # Confusion matrix gives tn, fp, fn, tp when labels are 0, 1
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        else:
            sensitivity = 0.0
            specificity = 0.0
            logger.warning(f"Unexpected confusion matrix shape: {cm.shape}")

        auc = np.nan
        if y_prob is not None and len(np.unique(y_true)) > 1:
            auc = roc_auc_score(y_true, y_prob)
            
        return {
            "accuracy": acc,
            "uar": uar,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "auc": auc
        }
    except Exception as e:
        logger.error(f"Error calculating metrics: {e}")
        return {
            "accuracy": np.nan, "uar": np.nan, "sensitivity": np.nan, 
            "specificity": np.nan, "auc": np.nan
        }
