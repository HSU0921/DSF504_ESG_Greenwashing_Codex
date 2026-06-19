"""
Leakage-free LightGBM comparison experiment.

This is an additive experiment. It does not overwrite the formal model or any
existing output artifact. It compares the current 14-feature matrix against a
cleaned feature set that removes direct label-axis features.
"""

from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / "outputs" / ".mplconfig"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = PROJECT_ROOT / "models"
BEST_PARAMS_PATH = OUTPUTS_DIR / "best_params.json"
OUTPUT_CSV = OUTPUTS_DIR / "leakage_free_comparison.csv"
CLEAN_MODEL_PATH = MODELS_DIR / "lgbm_clean_comparison.pkl"

RANDOM_STATE = 42
N_OUTER_FOLDS = 5
N_INNER_FOLDS = 3
HOLDOUT_TEST_SIZE = 0.30
MODEL_KEY = "lgbm"

LABEL_AXIS_FEATURES = {
    # Direct Talk-side label component and its additive components.
    "talk_total_density",
    "E_density",
    "S_density",
    "G_density",
    # Walk-side proxy from Sustainalytics E/S/G sub-score source.
    "sector_rank",
}


def _ensure_project_imports() -> None:
    sys.path.insert(0, str(PROJECT_ROOT))
    sys.path.insert(0, str(PROJECT_ROOT / "src"))


def build_feature_sets(full_features: list[str]) -> dict[str, list[str]]:
    """Return FULL and CLEAN feature lists in original column order."""
    clean_features = [feature for feature in full_features if feature not in LABEL_AXIS_FEATURES]
    return {"FULL": list(full_features), "CLEAN": clean_features}


def load_formal_matrix() -> tuple[pd.DataFrame, pd.Series]:
    """Load the formal 257-company baseline feature matrix."""
    _ensure_project_imports()
    from src.data_loader import load_merged_dataset
    from src.feature_engineering import build_feature_matrix

    df = load_merged_dataset(verbose=False)
    X, y, _ = build_feature_matrix(df, feature_set="baseline", verbose=False)
    return X.reset_index(drop=True), pd.Series(y, name="greenwashing_label").astype(int).reset_index(drop=True)


def load_lgbm_params() -> dict:
    """Load the formal LightGBM hyperparameters from outputs/best_params.json."""
    with open(BEST_PARAMS_PATH, "r", encoding="utf-8") as f:
        params = json.load(f)
    if MODEL_KEY not in params or not params[MODEL_KEY]:
        raise ValueError(f"Expected LightGBM params under key '{MODEL_KEY}' in {BEST_PARAMS_PATH}")
    return params[MODEL_KEY]


def build_lgbm_pipeline(params: dict):
    """Use the same project pipeline factory as formal model training."""
    _ensure_project_imports()
    from src.model_training import build_pipeline

    return build_pipeline(MODEL_KEY, params)


def _probability_spread(y_score: np.ndarray) -> dict:
    quantiles = np.quantile(y_score, [0.01, 0.10, 0.50, 0.90, 0.99])
    return {
        "holdout_probability_min": float(np.min(y_score)),
        "holdout_probability_p01": float(quantiles[0]),
        "holdout_probability_p10": float(quantiles[1]),
        "holdout_probability_median": float(quantiles[2]),
        "holdout_probability_p90": float(quantiles[3]),
        "holdout_probability_p99": float(quantiles[4]),
        "holdout_probability_max": float(np.max(y_score)),
        "holdout_n_prob_lt_001": int(np.sum(y_score < 0.01)),
        "holdout_n_prob_gt_099": int(np.sum(y_score > 0.99)),
    }


def evaluate_cv_fixed_params(X: pd.DataFrame, y: pd.Series, params: dict) -> dict:
    """
    Evaluate with the formal 5x3 stratified CV structure and fixed formal params.

    The inner 3-fold CV is run on each outer-training split as a protocol check;
    the formal parameters are not retuned, so the outer fold metrics are the
    reported comparison metrics.
    """
    X_arr = X.to_numpy(dtype=float)
    y_arr = y.to_numpy(dtype=int)
    outer_cv = StratifiedKFold(n_splits=N_OUTER_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    inner_cv = StratifiedKFold(n_splits=N_INNER_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    roc_aucs: list[float] = []
    pr_aucs: list[float] = []
    f1s: list[float] = []
    inner_roc_aucs: list[float] = []

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X_arr, y_arr), start=1):
        X_train, X_test = X_arr[train_idx], X_arr[test_idx]
        y_train, y_test = y_arr[train_idx], y_arr[test_idx]

        inner_pipeline = build_lgbm_pipeline(params)
        inner_scores = cross_val_score(
            inner_pipeline,
            X_train,
            y_train,
            cv=inner_cv,
            scoring="roc_auc",
            error_score=0.5,
        )
        inner_roc_aucs.append(float(np.mean(inner_scores)))

        pipeline = build_lgbm_pipeline(params)
        pipeline.fit(X_train, y_train)
        y_score = pipeline.predict_proba(X_test)[:, 1]
        y_pred = pipeline.predict(X_test)

        roc_aucs.append(float(roc_auc_score(y_test, y_score)))
        pr_aucs.append(float(average_precision_score(y_test, y_score)))
        f1s.append(float(f1_score(y_test, y_pred, zero_division=0)))

        print(
            f"  Fold {fold}: inner ROC-AUC={np.mean(inner_scores):.4f} | "
            f"outer ROC-AUC={roc_aucs[-1]:.4f} PR-AUC={pr_aucs[-1]:.4f} F1={f1s[-1]:.4f}"
        )

    return {
        "cv_mean_roc_auc": float(np.mean(roc_aucs)),
        "cv_std_roc_auc": float(np.std(roc_aucs)),
        "cv_mean_pr_auc": float(np.mean(pr_aucs)),
        "cv_std_pr_auc": float(np.std(pr_aucs)),
        "cv_mean_f1": float(np.mean(f1s)),
        "cv_std_f1": float(np.std(f1s)),
        "inner_cv_mean_roc_auc": float(np.mean(inner_roc_aucs)),
        "inner_cv_std_roc_auc": float(np.std(inner_roc_aucs)),
    }


def evaluate_fixed_holdout(X: pd.DataFrame, y: pd.Series, params: dict) -> dict:
    """Train on the fixed 179-company training slice and score the 78-company holdout."""
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X,
        y,
        test_size=HOLDOUT_TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    pipeline = build_lgbm_pipeline(params)
    pipeline.fit(X_train.to_numpy(dtype=float), y_train.to_numpy(dtype=int))
    y_score = pipeline.predict_proba(X_holdout.to_numpy(dtype=float))[:, 1]
    y_pred = pipeline.predict(X_holdout.to_numpy(dtype=float))

    metrics = {
        "holdout_roc_auc": float(roc_auc_score(y_holdout, y_score)),
        "holdout_pr_auc": float(average_precision_score(y_holdout, y_score)),
        "holdout_f1": float(f1_score(y_holdout, y_pred, zero_division=0)),
        "holdout_n": int(len(y_holdout)),
        "holdout_positives": int(np.sum(y_holdout.to_numpy(dtype=int) == 1)),
    }
    metrics.update(_probability_spread(y_score))
    return metrics


def fit_and_save_clean_model(X_clean: pd.DataFrame, y: pd.Series, params: dict) -> None:
    """Save only the new clean comparison model artifact."""
    MODELS_DIR.mkdir(exist_ok=True)
    pipeline = build_lgbm_pipeline(params)
    pipeline.fit(X_clean.to_numpy(dtype=float), y.to_numpy(dtype=int))
    with open(CLEAN_MODEL_PATH, "wb") as f:
        pickle.dump(pipeline, f)


def run_experiment() -> pd.DataFrame:
    OUTPUTS_DIR.mkdir(exist_ok=True)
    (OUTPUTS_DIR / ".mplconfig").mkdir(exist_ok=True)

    X, y = load_formal_matrix()
    params = load_lgbm_params()
    feature_sets = build_feature_sets(list(X.columns))

    print("=" * 78)
    print("DSF504 ESG Greenwashing — leakage-free LightGBM comparison")
    print("=" * 78)
    print(f"Dataset: {X.shape[0]} companies x {X.shape[1]} formal features")
    print(f"Labels: {int(y.sum())} positives / {int((y == 0).sum())} negatives")
    print(f"LightGBM params: {params}")
    print("\nCLEAN feature list:")
    for idx, feature in enumerate(feature_sets["CLEAN"], start=1):
        print(f"  {idx:02d}. {feature}")
    print(f"\nRemoved label-axis features: {sorted(LABEL_AXIS_FEATURES)}")

    rows: list[dict] = []
    for feature_set_name, feature_list in feature_sets.items():
        print("\n" + "-" * 78)
        print(f"Feature set: {feature_set_name} ({len(feature_list)} features)")
        print("-" * 78)
        X_subset = X[feature_list]
        cv_metrics = evaluate_cv_fixed_params(X_subset, y, params)
        holdout_metrics = evaluate_fixed_holdout(X_subset, y, params)
        rows.append(
            {
                "feature_set": feature_set_name,
                "n_features": len(feature_list),
                "features": ", ".join(feature_list),
                **cv_metrics,
                **holdout_metrics,
            }
        )

    comparison = pd.DataFrame(rows)
    numeric_cols = comparison.select_dtypes(include=[np.number]).columns
    comparison[numeric_cols] = comparison[numeric_cols].round(6)
    comparison.to_csv(OUTPUT_CSV, index=False)
    fit_and_save_clean_model(X[feature_sets["CLEAN"]], y, params)

    print("\nComparison table:")
    display_cols = [
        "feature_set",
        "n_features",
        "cv_mean_roc_auc",
        "cv_std_roc_auc",
        "cv_mean_pr_auc",
        "cv_std_pr_auc",
        "cv_mean_f1",
        "cv_std_f1",
        "holdout_roc_auc",
        "holdout_pr_auc",
        "holdout_f1",
        "holdout_probability_min",
        "holdout_probability_median",
        "holdout_probability_max",
        "holdout_n_prob_lt_001",
        "holdout_n_prob_gt_099",
    ]
    print(comparison[display_cols].to_string(index=False))
    print("\nSaved artifacts:")
    print(f"- {OUTPUT_CSV}")
    print(f"- {CLEAN_MODEL_PATH}")
    return comparison


if __name__ == "__main__":
    run_experiment()
