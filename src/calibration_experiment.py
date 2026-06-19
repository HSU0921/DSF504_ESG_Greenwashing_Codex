"""
Isolated probability calibration experiment for the saved LightGBM scorer.

This script does not retrain the base model, change the production pickle,
modify the formal dataset, or alter the Streamlit scoring path. It loads the
existing models/lgbm_best.pkl, rebuilds the formal 257-company / 14-feature
matrix, reconstructs the fixed 70/30 holdout split, and fits calibration
layers only on the training portion of that split.
"""

from __future__ import annotations

import os
import pickle
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "outputs" / ".mplconfig"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODEL_PATH = MODELS_DIR / "lgbm_best.pkl"
RANDOM_STATE = 42
HOLDOUT_TEST_SIZE = 0.30
CALIBRATION_SPLITS = 3
FEATURE_COUNT = 14
EXPECTED_COMPANY_COUNT = 257


@dataclass(frozen=True)
class HoldoutSplit:
    X_train: pd.DataFrame
    X_holdout: pd.DataFrame
    y_train: pd.Series
    y_holdout: pd.Series
    train_meta: pd.DataFrame
    holdout_meta: pd.DataFrame


@dataclass
class FrozenCVCalibratedModel:
    method: str
    cv_description: str
    calibrators: list[CalibratedClassifierCV]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        fold_probabilities = [calibrator.predict_proba(X) for calibrator in self.calibrators]
        return np.mean(fold_probabilities, axis=0)


def _ensure_project_imports() -> None:
    sys.path.insert(0, str(PROJECT_ROOT))
    sys.path.insert(0, str(PROJECT_ROOT / "src"))


def load_formal_training_matrix() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Load the formal 257-company, 14-feature matrix and labels."""
    _ensure_project_imports()
    from src.data_loader import load_merged_dataset
    from src.feature_engineering import build_feature_matrix

    df = load_merged_dataset(verbose=False)
    X, y, meta = build_feature_matrix(df, feature_set="baseline", verbose=False)
    y = pd.Series(y, index=X.index, name="greenwashing_label").astype(int)

    if X.shape != (EXPECTED_COMPANY_COUNT, FEATURE_COUNT):
        raise ValueError(
            f"Expected formal matrix {(EXPECTED_COMPANY_COUNT, FEATURE_COUNT)}, got {X.shape}. "
            "Stopping to avoid running calibration on a changed dataset."
        )

    meta_df = pd.DataFrame(
        {
            "ticker": meta["ticker"],
            "sector": meta["sector"],
            "industry": meta["industry"],
            "year": meta["year"],
        },
        index=X.index,
    )
    return X.copy(), y.copy(), meta_df


def reconstruct_holdout_split(X: pd.DataFrame, y: pd.Series, meta: pd.DataFrame) -> HoldoutSplit:
    """Rebuild the same fixed 78-company stratified holdout used elsewhere."""
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X,
        y,
        test_size=HOLDOUT_TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    train_meta = meta.loc[X_train.index].reset_index(drop=True)
    holdout_meta = meta.loc[X_holdout.index].reset_index(drop=True)
    return HoldoutSplit(
        X_train=X_train.reset_index(drop=True),
        X_holdout=X_holdout.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        y_holdout=y_holdout.reset_index(drop=True),
        train_meta=train_meta,
        holdout_meta=holdout_meta,
    )


def load_saved_model():
    """Load the existing fitted LightGBM pipeline without modifying it."""
    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)


def predict_positive_probability(model, X: pd.DataFrame) -> np.ndarray:
    """Return class-1 probabilities while preserving DataFrame feature names."""
    probabilities = model.predict_proba(X)
    return np.asarray(probabilities[:, 1], dtype=float)


def fit_frozen_cv_calibrator(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    method: str,
) -> FrozenCVCalibratedModel:
    """
    Fit a 3-fold calibration ensemble around the already-trained estimator.

    Each fold fits only the probability mapping on that fold's held-in
    calibration slice. FrozenEstimator prevents the saved LightGBM pipeline
    from being cloned, refit, or retuned.
    """
    cv = StratifiedKFold(n_splits=CALIBRATION_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    calibrators: list[CalibratedClassifierCV] = []
    for _, calibration_idx in cv.split(X_train, y_train):
        X_cal = X_train.iloc[calibration_idx].reset_index(drop=True)
        y_cal = y_train.iloc[calibration_idx].reset_index(drop=True)
        calibrator = CalibratedClassifierCV(
            estimator=FrozenEstimator(model),
            method=method,
            cv="prefit",
            ensemble="auto",
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The `cv='prefit'` option is deprecated", category=FutureWarning)
            calibrator.fit(X_cal, y_cal)
        calibrators.append(calibrator)

    cv_description = (
        f"FrozenEstimator prefit calibration; StratifiedKFold(n_splits={CALIBRATION_SPLITS}, "
        f"shuffle=True, random_state={RANDOM_STATE}) over the 179-company training portion. "
        "The base LightGBM model is never retrained; each outer fold calibrates on its "
        "held-in training slice only, using CalibratedClassifierCV(cv='prefit') internally."
    )
    return FrozenCVCalibratedModel(method=method, cv_description=cv_description, calibrators=calibrators)


def evaluate_probabilities(y_true: pd.Series, probabilities: dict[str, np.ndarray]) -> pd.DataFrame:
    """Compute ranking and probability-quality metrics for each probability set."""
    rows = []
    y_arr = y_true.to_numpy(dtype=int)
    for model_name, y_score in probabilities.items():
        rows.append(
            {
                "model": model_name,
                "roc_auc": roc_auc_score(y_arr, y_score),
                "pr_auc": average_precision_score(y_arr, y_score),
                "brier_score": brier_score_loss(y_arr, y_score),
                "log_loss": log_loss(y_arr, y_score, labels=[0, 1]),
                "min_probability": float(np.min(y_score)),
                "median_probability": float(np.median(y_score)),
                "max_probability": float(np.max(y_score)),
            }
        )
    return pd.DataFrame(rows)


def build_probability_examples(
    y_true: pd.Series,
    probabilities: dict[str, np.ndarray | list[float]],
    holdout_meta: pd.DataFrame,
    n_negatives: int = 6,
) -> pd.DataFrame:
    """Return all positive holdout companies plus a few high-scored negatives."""
    y_arr = y_true.to_numpy(dtype=int)
    raw_scores = np.asarray(probabilities["uncalibrated"], dtype=float)
    positive_idx = np.where(y_arr == 1)[0]
    negative_idx = np.where(y_arr == 0)[0]
    hardest_negatives = negative_idx[np.argsort(-raw_scores[negative_idx])[:n_negatives]]
    selected_idx = np.concatenate([positive_idx, hardest_negatives])

    examples = holdout_meta.iloc[selected_idx].copy().reset_index(drop=True)
    examples.insert(1, "label", y_arr[selected_idx])
    for name, values in probabilities.items():
        examples[f"{name}_probability"] = np.asarray(values, dtype=float)[selected_idx]
    return examples.sort_values(["label", "uncalibrated_probability"], ascending=[False, False]).reset_index(drop=True)


def save_reliability_curve(y_true: pd.Series, probabilities: dict[str, np.ndarray], path: Path) -> None:
    """Save a holdout reliability diagram for raw and calibrated probabilities."""
    fig, ax = plt.subplots(figsize=(7.5, 5.4), facecolor="white")
    colors = {
        "uncalibrated": "#B91C1C",
        "sigmoid": "#2563EB",
        "isotonic": "#15803D",
    }
    labels = {
        "uncalibrated": "Uncalibrated LightGBM",
        "sigmoid": "Sigmoid / Platt calibrated",
        "isotonic": "Isotonic calibrated",
    }

    for name, y_score in probabilities.items():
        prob_true, prob_pred = calibration_curve(
            y_true,
            y_score,
            n_bins=5,
            strategy="quantile",
        )
        ax.plot(
            prob_pred,
            prob_true,
            marker="o",
            linewidth=2.2,
            color=colors.get(name),
            label=labels.get(name, name),
        )

    ax.plot([0, 1], [0, 1], linestyle="--", color="#6B7280", linewidth=1.2, label="Perfect calibration")
    ax.set_title("Holdout Reliability Diagram", fontsize=15, weight="bold")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed positive rate")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _print_interpretation(metrics: pd.DataFrame) -> None:
    raw = metrics.loc[metrics["model"] == "uncalibrated"].iloc[0]
    for model_name in ["sigmoid", "isotonic"]:
        calibrated = metrics.loc[metrics["model"] == model_name].iloc[0]
        print(
            f"{model_name:>9}: ΔROC-AUC={calibrated['roc_auc'] - raw['roc_auc']:+.6f}, "
            f"ΔPR-AUC={calibrated['pr_auc'] - raw['pr_auc']:+.6f}, "
            f"ΔBrier={calibrated['brier_score'] - raw['brier_score']:+.6f}, "
            f"ΔLogLoss={calibrated['log_loss'] - raw['log_loss']:+.6f}"
        )


def run_experiment() -> tuple[pd.DataFrame, pd.DataFrame]:
    OUTPUTS_DIR.mkdir(exist_ok=True)
    (OUTPUTS_DIR / ".mplconfig").mkdir(exist_ok=True)

    X, y, meta = load_formal_training_matrix()
    split = reconstruct_holdout_split(X, y, meta)
    model = load_saved_model()

    print("=" * 78)
    print("DSF504 ESG Greenwashing — isolated probability calibration experiment")
    print("=" * 78)
    print(f"Formal matrix: {X.shape[0]} companies x {X.shape[1]} features")
    print(f"Label count: {int(y.sum())} positives / {int((y == 0).sum())} negatives")
    print(
        f"Fixed holdout: {len(split.y_holdout)} companies, "
        f"{int(split.y_holdout.sum())} positives / {int((split.y_holdout == 0).sum())} negatives"
    )
    print(
        "Holdout positives: "
        + ", ".join(split.holdout_meta.loc[split.y_holdout.to_numpy() == 1, "ticker"].tolist())
    )
    print(
        f"Calibration CV: StratifiedKFold(n_splits={CALIBRATION_SPLITS}, shuffle=True, "
        f"random_state={RANDOM_STATE}) on TRAINING portion only"
    )
    print("Base estimator: existing models/lgbm_best.pkl wrapped with sklearn.frozen.FrozenEstimator")

    sigmoid_model = fit_frozen_cv_calibrator(model, split.X_train, split.y_train, method="sigmoid")
    isotonic_model = fit_frozen_cv_calibrator(model, split.X_train, split.y_train, method="isotonic")

    probabilities = {
        "uncalibrated": predict_positive_probability(model, split.X_holdout),
        "sigmoid": predict_positive_probability(sigmoid_model, split.X_holdout),
        "isotonic": predict_positive_probability(isotonic_model, split.X_holdout),
    }
    metrics = evaluate_probabilities(split.y_holdout, probabilities)
    metrics_rounded = metrics.copy()
    numeric_cols = metrics_rounded.select_dtypes(include=[np.number]).columns
    metrics_rounded[numeric_cols] = metrics_rounded[numeric_cols].round(6)
    metrics_rounded.to_csv(OUTPUTS_DIR / "calibration_metrics.csv", index=False)

    examples = build_probability_examples(split.y_holdout, probabilities, split.holdout_meta, n_negatives=6)
    probability_cols = [col for col in examples.columns if col.endswith("_probability")]
    examples[probability_cols] = examples[probability_cols].round(6)
    examples.to_csv(OUTPUTS_DIR / "calibration_probability_examples.csv", index=False)

    save_reliability_curve(split.y_holdout, probabilities, OUTPUTS_DIR / "calibration_curve.png")

    print("\nMetrics on the untouched 78-company holdout:")
    print(metrics_rounded.to_string(index=False))
    print("\nChange vs uncalibrated:")
    _print_interpretation(metrics)
    print("\nPositive holdout companies and high-scored negatives:")
    print(examples.to_string(index=False))
    print("\nSaved artifacts:")
    print(f"- {OUTPUTS_DIR / 'calibration_curve.png'}")
    print(f"- {OUTPUTS_DIR / 'calibration_metrics.csv'}")
    print(f"- {OUTPUTS_DIR / 'calibration_probability_examples.csv'}")
    return metrics_rounded, examples


if __name__ == "__main__":
    run_experiment()
