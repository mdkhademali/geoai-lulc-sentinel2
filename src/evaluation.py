"""
evaluation.py
=============
Classification evaluation: accuracy metrics, confusion matrices,
per-class report, multi-class ROC/AUC, prediction confidence analysis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                              confusion_matrix, classification_report,
                              roc_curve, auc, cohen_kappa_score)
from sklearn.preprocessing import label_binarize


def full_metrics(y_true, y_pred, class_names) -> pd.DataFrame:
    """Per-class precision/recall/F1/support + overall accuracy & kappa."""
    p, r, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=range(len(class_names)))
    df = pd.DataFrame({
        "Class": class_names, "Precision": p, "Recall": r, "F1-Score": f1, "Support": support,
    })
    overall = pd.DataFrame([{
        "Class": "OVERALL",
        "Precision": precision_recall_fscore_support(y_true, y_pred, average="macro")[0],
        "Recall": precision_recall_fscore_support(y_true, y_pred, average="macro")[1],
        "F1-Score": precision_recall_fscore_support(y_true, y_pred, average="macro")[2],
        "Support": len(y_true),
    }])
    return pd.concat([df, overall], ignore_index=True)


def get_confusion_matrix(y_true, y_pred, n_classes: int) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=range(n_classes))


def get_classification_report(y_true, y_pred, class_names) -> str:
    return classification_report(y_true, y_pred, target_names=class_names, zero_division=0)


def overall_accuracy_kappa(y_true, y_pred):
    return accuracy_score(y_true, y_pred), cohen_kappa_score(y_true, y_pred)


def multiclass_roc_auc(y_true, y_proba, n_classes):
    """One-vs-rest ROC curves + AUC for each class."""
    y_bin = label_binarize(y_true, classes=range(n_classes))
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    fpr["macro"], tpr["macro"], _ = roc_curve(y_bin.ravel(), y_proba.ravel())
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    return fpr, tpr, roc_auc


def prediction_confidence(y_proba: np.ndarray) -> np.ndarray:
    """Max class probability per sample = model confidence."""
    return y_proba.max(axis=1)


def error_analysis(y_true, y_pred, X_df: pd.DataFrame, class_names) -> pd.DataFrame:
    """Return a DataFrame of misclassified samples with true/predicted labels."""
    mis_idx = np.where(np.array(y_true) != np.array(y_pred))[0]
    out = X_df.iloc[mis_idx].copy()
    out["true_label"] = [class_names[i] for i in np.array(y_true)[mis_idx]]
    out["pred_label"] = [class_names[i] for i in np.array(y_pred)[mis_idx]]
    return out
