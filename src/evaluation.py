"""
evaluation.py

Plotting and metric helpers shared across notebooks.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)


def print_classification_report(y_true, y_pred, labels=None, title=""):
    if title:
        print(f"=== {title} ===")
    print(classification_report(y_true, y_pred, labels=labels, digits=3))


def plot_confusion_matrix(y_true, y_pred, labels, title="Confusion Matrix",
                           normalize=None, ax=None, figsize=(7, 6)):
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fmt = ".2f" if normalize else "d"
    disp.plot(ax=ax, cmap="Blues", values_format=fmt, xticks_rotation=45)
    ax.set_title(title)
    plt.tight_layout()
    return ax


def plot_roc_curve(y_true, y_score, pos_label, title="ROC Curve", ax=None):
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=pos_label)
    auc = roc_auc_score((np.array(y_true) == pos_label).astype(int), y_score)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    return ax, auc


def plot_feature_importance(model, feature_names, top_n=20,
                             title="Feature Importance", ax=None):
    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in idx]
    top_importances = importances[idx]

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.3)))
    sns.barplot(x=top_importances, y=top_features, ax=ax, orient="h")
    ax.set_title(title)
    ax.set_xlabel("Importance")
    plt.tight_layout()
    return ax


def summarize_results(results: dict[str, dict]) -> pd.DataFrame:
    """
    `results` maps model name -> dict of metric_name: value
    Returns a tidy comparison dataframe.
    """
    return pd.DataFrame(results).T
