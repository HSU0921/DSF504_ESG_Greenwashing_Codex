"""
Generate fixed holdout evaluation artifacts for the DSF504 ESG dashboard.

This script is additive only. It does not train or tune any model. It loads the
saved LightGBM final scorer and reconstructs the same 70/30 stratified real
holdout split used by holdout_robustness_experiment.py.
"""

from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = PROJECT_ROOT / "outputs"
MODELS = PROJECT_ROOT / "models"
RANDOM_STATE = 42


def _load_project_data():
    sys.path.insert(0, str(PROJECT_ROOT))
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    try:
        from src.data_loader import load_merged_dataset
        from src.feature_engineering import build_feature_matrix
    except ModuleNotFoundError:
        from data_loader import load_merged_dataset
        from feature_engineering import build_feature_matrix

    df = load_merged_dataset(verbose=False)
    try:
        X, y, _ = build_feature_matrix(df, feature_set="baseline", verbose=False)
    except TypeError:
        X, y, _ = build_feature_matrix(df)
    return X.copy(), pd.Series(y).astype(int).reset_index(drop=True)


def _load_model():
    model_path = MODELS / "lgbm_best.pkl"
    try:
        return joblib.load(model_path)
    except Exception:
        with open(model_path, "rb") as f:
            return pickle.load(f)


def _predict_proba(model, X: pd.DataFrame) -> np.ndarray:
    try:
        return np.asarray(model.predict_proba(X)[:, 1], dtype=float)
    except Exception:
        return np.asarray(model.predict_proba(X.to_numpy())[:, 1], dtype=float)


def _save_roc_curve(y_true: np.ndarray, y_score: np.ndarray, roc_auc: float) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=(7, 5), facecolor="white")
    ax.plot(fpr, tpr, color="#1F5C3A", linewidth=2.4, label=f"ROC-AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], color="#A9B9AD", linestyle="--", linewidth=1.4)
    ax.set_title("Holdout ROC Curve", fontsize=15, weight="bold")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.22)
    fig.tight_layout()
    fig.savefig(OUTPUTS / "holdout_roc_curve.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def _save_pr_curve(y_true: np.ndarray, y_score: np.ndarray, pr_auc: float) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    fig, ax = plt.subplots(figsize=(7, 5), facecolor="white")
    ax.plot(recall, precision, color="#D99A2B", linewidth=2.4, label=f"PR-AUC = {pr_auc:.4f}")
    ax.set_title("Holdout Precision-Recall Curve", fontsize=15, weight="bold")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left")
    ax.grid(alpha=0.22)
    fig.tight_layout()
    fig.savefig(OUTPUTS / "holdout_pr_curve.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def _save_confusion_matrix(cm: np.ndarray, threshold: float, top_n: int) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 5.2), facecolor="white")
    im = ax.imshow(cm, cmap="Greens")
    ax.set_title("Holdout Review-Priority Confusion Matrix", fontsize=14, weight="bold")
    ax.set_xticks([0, 1], labels=["Predicted Low\nPriority", "Predicted Review\nPriority"])
    ax.set_yticks([0, 1], labels=["Actual Low\nRisk", "Actual High\nRisk"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", fontsize=18, weight="bold", color="#173D2A")
    ax.set_xlabel("Predicted review-priority band")
    ax.set_ylabel("Actual holdout label")
    ax.text(
        0.5,
        -0.23,
        f"Threshold = top 10% review-priority cutoff ({top_n} companies), score >= {threshold:.4f}",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=10,
        color="#5F7165",
    )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(OUTPUTS / "holdout_confusion_matrix.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUTS.mkdir(exist_ok=True)
    X, y = _load_project_data()
    _, X_holdout, _, y_holdout = train_test_split(
        X,
        y,
        test_size=0.30,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    y_true = y_holdout.reset_index(drop=True).to_numpy(dtype=int)
    model = _load_model()
    y_score = _predict_proba(model, X_holdout)

    roc_auc = float(roc_auc_score(y_true, y_score))
    pr_auc = float(average_precision_score(y_true, y_score))
    top_n = int(np.ceil(len(y_true) * 0.10))
    top_n = max(top_n, 1)
    ranked_idx = np.argsort(-y_score)
    selected_idx = ranked_idx[:top_n]
    y_pred = np.zeros_like(y_true)
    y_pred[selected_idx] = 1
    threshold = float(np.min(y_score[selected_idx]))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]

    _save_roc_curve(y_true, y_score, roc_auc)
    _save_pr_curve(y_true, y_score, pr_auc)
    _save_confusion_matrix(cm, threshold, top_n)

    summary = pd.DataFrame(
        [
            {
                "holdout_n_real_companies": int(len(y_true)),
                "holdout_positive_count": int(np.sum(y_true == 1)),
                "holdout_negative_count": int(np.sum(y_true == 0)),
                "roc_auc": round(roc_auc, 4),
                "pr_auc": round(pr_auc, 4),
                "review_priority_threshold": round(threshold, 6),
                "review_priority_top_percent": 0.10,
                "number_companies_reviewed": int(top_n),
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
                "threshold_note": "Review-priority cutoff selecting the top 10% of the fixed holdout ranking; not a legal classification threshold.",
            }
        ]
    )
    summary.to_csv(OUTPUTS / "holdout_eval_summary.csv", index=False)
    print("Generated holdout evaluation artifacts:")
    for name in [
        "holdout_roc_curve.png",
        "holdout_pr_curve.png",
        "holdout_confusion_matrix.png",
        "holdout_eval_summary.csv",
    ]:
        path = OUTPUTS / name
        print(f"- {path} ({'exists' if path.exists() else 'missing'})")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
