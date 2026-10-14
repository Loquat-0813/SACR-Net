from __future__ import annotations

import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> dict:
    labels = list(range(num_classes))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    total = cm.sum()
    oa = float(np.trace(cm) / total) if total else 0.0
    per_class = []
    for i in range(num_classes):
        denom = cm[i].sum()
        per_class.append(float(cm[i, i] / denom) if denom else np.nan)
    aa = float(np.nanmean(per_class)) if per_class else 0.0
    macro_f1 = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    class_f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    kappa = float(cohen_kappa_score(y_true, y_pred, labels=labels)) if total else 0.0
    return {
        "oa": oa,
        "aa": aa,
        "kappa": kappa,
        "macro_f1": macro_f1,
        "per_class_f1": class_f1.tolist(),
        "per_class_support": cm.sum(axis=1).tolist(),
        "per_class_acc": per_class,
        "confusion_matrix": cm.tolist(),
        "num_samples": int(total),
    }

