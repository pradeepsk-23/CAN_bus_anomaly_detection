"""
src/evaluation/metrics.py

Evaluation suite for the anomaly detection system.

Outputs:
  - Precision, Recall, F1 at calibrated threshold
  - ROC-AUC, Average Precision (area under PR curve)
  - Per-anomaly-type breakdown
  - Precision-Recall curve plot
  - Anomaly score distribution plot
  - Confusion matrix
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from loguru import logger
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------
def compute_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    anom_types: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute all evaluation metrics at the specified threshold.

    Args:
        y_true     : (N,) binary labels — 1=anomaly
        scores     : (N,) continuous anomaly scores ∈ [0, 1]
        threshold  : decision boundary
        anom_types : (N,) optional — for per-type breakdown

    Returns dict with keys: precision, recall, f1, roc_auc, average_precision,
                            tp, fp, tn, fn, and per-type F1 if anom_types provided.
    """
    y_pred = (scores >= threshold).astype(int)
    n_anomaly = y_true.sum()
    n_normal = (y_true == 0).sum()

    metrics = {
        "precision":          float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":             float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":                 float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc":            float(roc_auc_score(y_true, scores)),
        "average_precision":  float(average_precision_score(y_true, scores)),
        "threshold":          float(threshold),
        "n_total":            int(len(y_true)),
        "n_anomaly":          int(n_anomaly),
        "n_normal":           int(n_normal),
        "anomaly_rate":       float(n_anomaly / len(y_true)),
        "predicted_rate":     float(y_pred.mean()),
    }

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})

    # Per-anomaly-type breakdown
    if anom_types is not None:
        per_type: Dict[str, float] = {}
        for atype in np.unique(anom_types):
            if atype == "normal":
                continue
            mask = anom_types == atype
            if mask.sum() == 0:
                continue
            per_type[atype] = float(f1_score(y_true[mask], y_pred[mask], zero_division=0))
        metrics["per_type_f1"] = per_type

    _log_summary(metrics)
    return metrics


def _log_summary(m: Dict) -> None:
    logger.info(
        "Evaluation | P={:.3f} R={:.3f} F1={:.3f} | ROC-AUC={:.3f} | AP={:.3f} | threshold={:.4f}",
        m["precision"], m["recall"], m["f1"], m["roc_auc"], m["average_precision"], m["threshold"],
    )
    if "per_type_f1" in m:
        for atype, f1 in m["per_type_f1"].items():
            logger.info("  ↳ {} F1={:.3f}", atype, f1)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_pr_curve(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Precision-Recall curve with operating point marked."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    ap = average_precision_score(y_true, scores)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.step(recall, precision, where="post", color="#2196F3", linewidth=2, label=f"PR curve (AP={ap:.3f})")
    ax.fill_between(recall, precision, alpha=0.15, color="#2196F3")

    # Mark operating point
    y_pred = (scores >= threshold).astype(int)
    op_p = precision_score(y_true, y_pred, zero_division=0)
    op_r = recall_score(y_true, y_pred, zero_division=0)
    ax.scatter([op_r], [op_p], marker="*", s=200, color="#FF5722", zorder=5,
               label=f"Operating point (thr={threshold:.3f})")

    # Baseline
    baseline = y_true.mean()
    ax.axhline(baseline, linestyle="--", color="gray", linewidth=1, label=f"No-skill baseline ({baseline:.3f})")

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve — Vehicle Anomaly Detection", fontsize=13)
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        logger.info("PR curve saved → {}", save_path)

    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    scores: np.ndarray,
    save_path: Optional[str] = None,
) -> plt.Figure:
    fpr, tpr, _ = roc_curve(y_true, scores)
    auc = roc_auc_score(y_true, scores)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#4CAF50", linewidth=2, label=f"ROC (AUC={auc:.3f})")
    ax.fill_between(fpr, tpr, alpha=0.15, color="#4CAF50")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random classifier")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve", fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        logger.info("ROC curve saved → {}", save_path)

    return fig


def plot_score_distribution(
    scores: np.ndarray,
    y_true: np.ndarray,
    threshold: float,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Score distribution for normal vs anomalous windows — key diagnostic plot."""
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.hist(scores[y_true == 0], bins=60, color="#42A5F5", alpha=0.7, label="Normal", density=True)
    ax.hist(scores[y_true == 1], bins=60, color="#EF5350", alpha=0.7, label="Anomalous", density=True)
    ax.axvline(threshold, color="black", linestyle="--", linewidth=2, label=f"Threshold ({threshold:.3f})")

    ax.set_xlabel("Anomaly Score", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Anomaly Score Distribution", fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        logger.info("Score distribution saved → {}", save_path)

    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[str] = None,
) -> plt.Figure:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=["Predicted Normal", "Predicted Anomaly"],
        yticklabels=["True Normal", "True Anomaly"],
    )
    ax.set_title("Confusion Matrix", fontsize=13)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        logger.info("Confusion matrix saved → {}", save_path)

    return fig


def plot_per_type_f1(
    per_type_f1: Dict[str, float],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Bar chart of per-anomaly-type F1 scores."""
    types = list(per_type_f1.keys())
    scores = [per_type_f1[t] for t in types]

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = plt.cm.Set2(np.linspace(0, 1, len(types)))
    bars = ax.barh(types, scores, color=colors, edgecolor="white", height=0.6)

    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{score:.3f}", va="center", fontsize=10)

    ax.set_xlim([0, 1.15])
    ax.set_xlabel("F1 Score", fontsize=12)
    ax.set_title("Per-Anomaly-Type F1 Score", fontsize=13)
    ax.axvline(0.9, color="green", linestyle="--", alpha=0.5, label="Target (0.90)")
    ax.legend()
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        logger.info("Per-type F1 chart saved → {}", save_path)

    return fig


def plot_training_history(
    history: Dict[str, list],
    save_path: Optional[str] = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4))
    epochs = range(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], color="#2196F3", label="Train loss", linewidth=2)
    ax.plot(epochs, history["val_loss"], color="#FF9800", label="Val loss", linewidth=2, linestyle="--")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("MSE Loss", fontsize=12)
    ax.set_title("Training History", fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        logger.info("Training history saved → {}", save_path)

    return fig
