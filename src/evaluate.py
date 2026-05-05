"""
evaluate.py — 성능 지표 계산

macro-averaged Accuracy / Precision / Recall / F1 반환.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)


def compute_metrics(
    preds: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
) -> dict:
    """
    반환 dict 키:
        accuracy, precision, recall, f1  → float
        report                           → str (per-class 상세)
    """
    acc    = float((preds == labels).mean())
    prec   = float(precision_score(labels, preds, average="macro", zero_division=0))
    rec    = float(recall_score   (labels, preds, average="macro", zero_division=0))
    f1     = float(f1_score       (labels, preds, average="macro", zero_division=0))
    report = classification_report(
        labels, preds, target_names=class_names, zero_division=0,
    )
    return dict(accuracy=acc, precision=prec, recall=rec, f1=f1, report=report)
