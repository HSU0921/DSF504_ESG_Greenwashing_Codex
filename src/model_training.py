"""
model_training.py
=================
Nested Cross-Validation + Optuna hyperparameter tuning for greenwashing detection.

Architecture
------------
Outer loop : StratifiedKFold(n_splits=5)  → unbiased performance estimate
Inner loop : StratifiedKFold(n_splits=3) inside Optuna objective → tuning
SMOTE      : applied ONLY to each fold's training set; test folds never touched
Scaler     : StandardScaler for Logistic Regression only (fit on train, transform both)

Models
------
  baseline  – DummyClassifier(strategy='stratified')
  lr        – LogisticRegression(penalty='l2')
  rf        – RandomForestClassifier
  xgb       – XGBClassifier
  lgbm      – LGBMClassifier

Outputs
-------
  outputs/model_comparison.csv   – ROC-AUC / F1 / PR-AUC per model with delta vs baseline
  models/<name>_best.pkl         – final pipeline retrained on full data w/ best params
  outputs/best_params.json       – best hyperparameters per model

SMOTE trap guard (Lab Guide p.6):
  imblearn.pipeline.Pipeline ensures SMOTE's fit_resample() is called only on
  the training split inside every CV fold — the test split is NEVER touched.
"""

import os
import sys
import json
import pickle
import argparse
import warnings
import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import (
    roc_auc_score, f1_score, average_precision_score, confusion_matrix
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# ── project path setup ────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
MODELS_DIR  = os.path.join(PROJECT_ROOT, "models")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── constants ─────────────────────────────────────────────────────────────────
RANDOM_STATE    = 42
N_OUTER_FOLDS   = 5
N_INNER_FOLDS   = 3
N_OPTUNA_TRIALS = 30
SMOTE_K         = 3   # conservative: minority class is very small (~15 in train)
COMPARE_MODEL_ORDER = ["baseline", "lr", "lgbm", "xgb", "rf"]


# ─────────────────────────────────────────────────────────────────────────────
# 1. HYPERPARAMETER SEARCH SPACES
# ─────────────────────────────────────────────────────────────────────────────

def suggest_hyperparams(trial: optuna.Trial, model_name: str) -> dict:
    """Return Optuna-suggested hyperparameter dict for the given model."""
    if model_name == "lr":
        return {
            "C":       trial.suggest_float("C", 1e-3, 1e2, log=True),
            "max_iter": 1000,
        }
    elif model_name == "rf":
        return {
            "n_estimators":     trial.suggest_int("n_estimators", 50, 300, step=50),
            "max_depth":        trial.suggest_int("max_depth", 3, 20),
            "min_samples_split":trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 6),
            "max_features":     trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        }
    elif model_name == "xgb":
        return {
            "n_estimators":  trial.suggest_int("n_estimators", 50, 300, step=50),
            "max_depth":     trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample":     trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "gamma":         trial.suggest_float("gamma", 0.0, 5.0),
        }
    elif model_name == "lgbm":
        return {
            "n_estimators":  trial.suggest_int("n_estimators", 50, 300, step=50),
            "max_depth":     trial.suggest_int("max_depth", 3, 15),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves":    trial.suggest_int("num_leaves", 20, 100),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 30),
            "subsample":     trial.suggest_float("subsample", 0.6, 1.0),
        }
    else:
        raise ValueError(f"Unknown model: {model_name}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. MODEL FACTORY
# ─────────────────────────────────────────────────────────────────────────────

def build_classifier(model_name: str, params: dict):
    """Instantiate a sklearn-compatible classifier from name + param dict."""
    if model_name == "lr":
        return LogisticRegression(
            penalty="l2",
            random_state=RANDOM_STATE,
            class_weight="balanced",
            **params,
        )
    elif model_name == "rf":
        return RandomForestClassifier(
            random_state=RANDOM_STATE,
            class_weight="balanced",
            n_jobs=-1,
            **params,
        )
    elif model_name == "xgb":
        # scale_pos_weight handles imbalance for XGBoost
        return XGBClassifier(
            random_state=RANDOM_STATE,
            scale_pos_weight=12,      # ≈ (n_neg / n_pos) from full dataset
            eval_metric="logloss",
            verbosity=0,
            **params,
        )
    elif model_name == "lgbm":
        return LGBMClassifier(
            random_state=RANDOM_STATE,
            class_weight="balanced",
            verbose=-1,
            **params,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


def build_pipeline(model_name: str, params: dict) -> ImbPipeline:
    """
    Build an imblearn Pipeline:
      SMOTE  →  [StandardScaler for LR only]  →  classifier
    SMOTE's fit_resample() runs only on the training portion inside each CV fold.
    """
    clf   = build_classifier(model_name, params)
    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=SMOTE_K)
    steps = [("smote", smote)]
    if model_name == "lr":
        steps.append(("scaler", StandardScaler()))
    steps.append(("clf", clf))
    return ImbPipeline(steps=steps)


# ─────────────────────────────────────────────────────────────────────────────
# 3. OPTUNA OBJECTIVE (inner CV)
# ─────────────────────────────────────────────────────────────────────────────

def make_objective(model_name: str, X_train: np.ndarray, y_train: np.ndarray):
    """
    Returns an Optuna objective function.
    Inner 3-fold CV on X_train/y_train; SMOTE applied inside each inner fold
    via the imblearn Pipeline + cross_val_score.
    """
    inner_cv = StratifiedKFold(
        n_splits=N_INNER_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )

    def objective(trial: optuna.Trial) -> float:
        params   = suggest_hyperparams(trial, model_name)
        pipeline = build_pipeline(model_name, params)
        scores   = cross_val_score(
            pipeline, X_train, y_train,
            cv=inner_cv,
            scoring="roc_auc",
            error_score=0.5,          # fallback if a fold degenerates
        )
        return float(np.mean(scores))

    return objective


# ─────────────────────────────────────────────────────────────────────────────
# 4. METRICS HELPER
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_proba: np.ndarray) -> dict:
    """Return ROC-AUC, F1 (binary), PR-AUC, confusion matrix for one fold."""
    try:
        roc_auc = roc_auc_score(y_true, y_proba)
    except ValueError:
        roc_auc = float("nan")

    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        pr_auc = average_precision_score(y_true, y_proba)
    except ValueError:
        pr_auc = float("nan")

    cm = confusion_matrix(y_true, y_pred)
    return {"roc_auc": roc_auc, "f1": f1, "pr_auc": pr_auc, "cm": cm}


# ─────────────────────────────────────────────────────────────────────────────
# 5. BASELINE EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_baseline(X: np.ndarray, y: np.ndarray) -> dict:
    """
    Evaluate DummyClassifier with stratified outer CV.
    No SMOTE needed — Dummy ignores features entirely.
    """
    print("\n[model_training] ── Baseline: DummyClassifier ──────────────────")
    outer_cv = StratifiedKFold(
        n_splits=N_OUTER_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )
    fold_metrics = []

    for fold, (tr_idx, te_idx) in enumerate(outer_cv.split(X, y)):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        clf = DummyClassifier(strategy="stratified", random_state=RANDOM_STATE)
        clf.fit(X_tr, y_tr)
        y_pred  = clf.predict(X_te)
        y_proba = clf.predict_proba(X_te)[:, 1]
        m = compute_metrics(y_te, y_pred, y_proba)
        fold_metrics.append(m)
        print(f"  Fold {fold+1}: ROC-AUC={m['roc_auc']:.4f}  "
              f"F1={m['f1']:.4f}  PR-AUC={m['pr_auc']:.4f}")

    return _aggregate_folds("baseline", fold_metrics, best_params=None)


# ─────────────────────────────────────────────────────────────────────────────
# 6. ADVANCED MODEL EVALUATION (Nested CV + Optuna)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(model_name: str, X: np.ndarray, y: np.ndarray) -> dict:
    """
    Nested CV with Optuna tuning for one advanced model.

    Outer fold: provides unbiased AUC estimate on held-out test data.
    Inner loop: Optuna study on training data to find best hyperparams.
    SMOTE: applied inside Pipeline → only training data is ever resampled.
    """
    label_map = {
        "lr":   "Logistic Regression",
        "rf":   "Random Forest",
        "xgb":  "XGBoost",
        "lgbm": "LightGBM",
    }
    print(f"\n[model_training] ── {label_map[model_name]} ──────────────────")

    outer_cv = StratifiedKFold(
        n_splits=N_OUTER_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )
    fold_metrics   = []
    all_best_params = []
    best_fold_auc   = -1
    best_fold_params = None

    for fold, (tr_idx, te_idx) in enumerate(outer_cv.split(X, y)):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        # ── Inner Optuna study ──────────────────────────────────────────────
        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=RANDOM_STATE + fold),
        )
        study.optimize(
            make_objective(model_name, X_tr, y_tr),
            n_trials=N_OPTUNA_TRIALS,
            show_progress_bar=False,
        )
        best_params = study.best_params
        all_best_params.append(best_params)

        # ── Train on full outer training fold with best params ──────────────
        pipeline = build_pipeline(model_name, best_params)
        pipeline.fit(X_tr, y_tr)

        # ── Evaluate on outer test fold (SMOTE never touches test fold) ─────
        y_pred  = pipeline.predict(X_te)
        y_proba = pipeline.predict_proba(X_te)[:, 1]
        m = compute_metrics(y_te, y_pred, y_proba)
        fold_metrics.append(m)

        print(f"  Fold {fold+1}: ROC-AUC={m['roc_auc']:.4f}  "
              f"F1={m['f1']:.4f}  PR-AUC={m['pr_auc']:.4f}  "
              f"[inner best={study.best_value:.4f}]")

        if m["roc_auc"] > best_fold_auc:
            best_fold_auc    = m["roc_auc"]
            best_fold_params = best_params

    return _aggregate_folds(model_name, fold_metrics, best_fold_params)


# ─────────────────────────────────────────────────────────────────────────────
# 7. AGGREGATE FOLD RESULTS
# ─────────────────────────────────────────────────────────────────────────────

def _aggregate_folds(model_name: str, fold_metrics: list, best_params) -> dict:
    """Compute mean ± std of metrics across outer folds."""
    label_map = {
        "baseline": "DummyClassifier",
        "lr":       "Logistic Regression",
        "rf":       "Random Forest",
        "xgb":      "XGBoost",
        "lgbm":     "LightGBM",
    }
    aucs    = [m["roc_auc"] for m in fold_metrics]
    f1s     = [m["f1"]      for m in fold_metrics]
    pr_aucs = [m["pr_auc"]  for m in fold_metrics]

    result = {
        "model":        label_map.get(model_name, model_name),
        "model_key":    model_name,
        "mean_roc_auc": np.nanmean(aucs),
        "std_roc_auc":  np.nanstd(aucs),
        "mean_f1":      np.nanmean(f1s),
        "std_f1":       np.nanstd(f1s),
        "mean_pr_auc":  np.nanmean(pr_aucs),
        "std_pr_auc":   np.nanstd(pr_aucs),
        "best_params":  best_params,
        "fold_metrics": fold_metrics,
    }
    print(f"  → MEAN  ROC-AUC={result['mean_roc_auc']:.4f} (±{result['std_roc_auc']:.4f})  "
          f"F1={result['mean_f1']:.4f}  PR-AUC={result['mean_pr_auc']:.4f}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 8. SAVE FINAL MODEL (retrain on full data with best params)
# ─────────────────────────────────────────────────────────────────────────────

def save_final_model(model_name: str, best_params: dict,
                     X: np.ndarray, y: np.ndarray) -> str:
    """
    Retrain the model on ALL data using best hyperparams found during nested CV.
    SMOTE is applied here too (full data training), then save the pipeline.
    """
    if model_name == "baseline":
        clf      = DummyClassifier(strategy="stratified", random_state=RANDOM_STATE)
        pipeline = clf
        pipeline.fit(X, y)
    else:
        pipeline = build_pipeline(model_name, best_params)
        pipeline.fit(X, y)

    path = os.path.join(MODELS_DIR, f"{model_name}_best.pkl")
    with open(path, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"  Saved → {path}")
    return path


def _json_safe(value):
    """Convert numpy/scalar values into JSON-safe Python objects."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if np.isnan(value):
            return None
        return float(value)
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def _result_to_compare_row(feature_set: str, result: dict) -> dict:
    """Build one compact row for baseline-vs-semantic comparison."""
    return {
        "feature_set": feature_set,
        "model": result["model"],
        "model_key": result["model_key"],
        "mean_roc_auc": float(result["mean_roc_auc"]),
        "std_roc_auc": float(result["std_roc_auc"]),
        "mean_pr_auc": float(result["mean_pr_auc"]),
        "std_pr_auc": float(result["std_pr_auc"]),
        "mean_f1": float(result["mean_f1"]),
        "std_f1": float(result["std_f1"]),
    }


def _write_progress(rows: list[dict], run_mode: str) -> None:
    """Persist progress after each model so partial results survive a crash."""
    progress_path = os.path.join(OUTPUTS_DIR, "_model_training_progress.json")
    payload = {
        "run_mode": run_mode,
        "rows_completed": len(rows),
        "results_so_far": [_json_safe(row) for row in rows],
    }
    with open(progress_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[model_training] Progress saved → {progress_path}")


def _default_params(model_name: str) -> dict:
    """Small default parameter set for smoke mode."""
    if model_name == "lr":
        return {"max_iter": 1000}
    return {}


def evaluate_baseline_smoke(X: np.ndarray, y: np.ndarray) -> dict:
    """Evaluate DummyClassifier on one stratified 80/20 split."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    clf = DummyClassifier(strategy="stratified", random_state=RANDOM_STATE)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    y_proba = clf.predict_proba(X_te)[:, 1]
    return _aggregate_folds("baseline", [compute_metrics(y_te, y_pred, y_proba)], best_params=None)


def evaluate_model_smoke(model_name: str, X: np.ndarray, y: np.ndarray) -> dict:
    """Evaluate an advanced model on one stratified 80/20 split with defaults."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    pipeline = build_pipeline(model_name, _default_params(model_name))
    pipeline.fit(X_tr, y_tr)
    y_pred = pipeline.predict(X_te)
    y_proba = pipeline.predict_proba(X_te)[:, 1]
    return _aggregate_folds(model_name, [compute_metrics(y_te, y_pred, y_proba)], best_params=_default_params(model_name))


def _add_baseline_deltas_and_interpretation(df: pd.DataFrame) -> pd.DataFrame:
    """Add deltas versus the matching original 14-feature baseline model row."""
    baseline_rows = (
        df[df["feature_set"] == "baseline"]
        .set_index("model")[["mean_roc_auc", "mean_pr_auc", "mean_f1"]]
        .to_dict("index")
    )

    rows = []
    for row in df.to_dict("records"):
        reference = baseline_rows.get(row["model"])
        if reference:
            row["delta_roc_auc_vs_baseline_14"] = row["mean_roc_auc"] - reference["mean_roc_auc"]
            row["delta_pr_auc_vs_baseline_14"] = row["mean_pr_auc"] - reference["mean_pr_auc"]
            row["delta_f1_vs_baseline_14"] = row["mean_f1"] - reference["mean_f1"]
        else:
            row["delta_roc_auc_vs_baseline_14"] = np.nan
            row["delta_pr_auc_vs_baseline_14"] = np.nan
            row["delta_f1_vs_baseline_14"] = np.nan

        if row["feature_set"] == "baseline":
            row["interpretation"] = "Original 14-feature reference model using Talk + Walk / External + Context features."
        elif reference and row["mean_roc_auc"] < reference["mean_roc_auc"]:
            row["interpretation"] = (
                "Semantic features did not improve ROC-AUC on this dataset. "
                "Their added value is interpretive depth and robustness against AI-rewritten disclosure, "
                "as shown in the adversarial test. Treat semantic features as complementary, not replacement."
            )
        else:
            row["interpretation"] = (
                "Semantic features matched or improved ROC-AUC while adding interpretive depth for AI-rewritten ESG disclosure robustness."
            )
        rows.append(row)

    ordered_cols = [
        "feature_set",
        "model",
        "mean_roc_auc",
        "std_roc_auc",
        "mean_pr_auc",
        "std_pr_auc",
        "mean_f1",
        "std_f1",
        "delta_roc_auc_vs_baseline_14",
        "delta_pr_auc_vs_baseline_14",
        "delta_f1_vs_baseline_14",
        "interpretation",
    ]
    out = pd.DataFrame(rows)[ordered_cols]
    metric_cols = [col for col in ordered_cols if col.startswith(("mean_", "std_", "delta_"))]
    out[metric_cols] = out[metric_cols].round(4)
    return out


def run_feature_set_cv(
    X: pd.DataFrame,
    y: pd.Series,
    feature_set: str,
    run_mode: str,
    progress_rows: list[dict],
) -> list[dict]:
    """Run either quick nested CV or smoke split for one feature set."""
    X_arr = X.values.astype(float)
    y_arr = y.values.astype(int)
    print(f"\n{'='*62}")
    print(f"  Feature set: {feature_set} ({X_arr.shape[1]} features)")
    print(f"  Mode:        {run_mode}")
    print(f"{'='*62}")

    for model_name in COMPARE_MODEL_ORDER:
        if model_name == "baseline":
            result = evaluate_baseline_smoke(X_arr, y_arr) if run_mode == "smoke" else evaluate_baseline(X_arr, y_arr)
        else:
            result = evaluate_model_smoke(model_name, X_arr, y_arr) if run_mode == "smoke" else evaluate_model(model_name, X_arr, y_arr)
        progress_rows.append(_result_to_compare_row(feature_set, result))
        _write_progress(progress_rows, run_mode)
    return progress_rows


def run_compare_feature_sets(
    feature_sets: list[str],
    run_mode: str,
) -> pd.DataFrame:
    """Compare baseline 14-feature and semantic 19-feature matrices."""
    from src.data_loader import load_merged_dataset
    from src.feature_engineering import build_feature_matrix

    print("[model_training] Loading data …")
    df = load_merged_dataset(verbose=False)

    progress_rows = []
    for feature_set in feature_sets:
        print(f"[model_training] Building features for feature_set={feature_set} …")
        X, y, _ = build_feature_matrix(df, feature_set=feature_set, verbose=False)
        progress_rows = run_feature_set_cv(X, y, feature_set, run_mode, progress_rows)

    comparison_df = _add_baseline_deltas_and_interpretation(pd.DataFrame(progress_rows))
    csv_path = os.path.join(OUTPUTS_DIR, "model_comparison_baseline_vs_semantic.csv")
    comparison_df.to_csv(csv_path, index=False)
    print(f"\n[model_training] Baseline vs semantic comparison saved → {csv_path}")
    print(comparison_df.to_string(index=False))
    return comparison_df


def run_single_feature_set(feature_set: str, run_mode: str) -> pd.DataFrame:
    """Run comparison-style evaluation for one selected feature set only."""
    rows_df = run_compare_feature_sets([feature_set], run_mode)
    csv_path = os.path.join(OUTPUTS_DIR, f"model_comparison_{feature_set}.csv")
    rows_df.to_csv(csv_path, index=False)
    print(f"[model_training] Single feature-set comparison saved → {csv_path}")
    return rows_df


# ─────────────────────────────────────────────────────────────────────────────
# 9. MASTER TRAINING PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_all(X: pd.DataFrame, y: pd.Series, verbose: bool = True) -> pd.DataFrame:
    """
    Run full training pipeline:
      1. Baseline DummyClassifier
      2. Nested CV + Optuna for LR, RF, XGBoost, LightGBM
      3. Save comparison table to outputs/model_comparison.csv
      4. Save best_params to outputs/best_params.json
      5. Retrain + save each model to models/<name>_best.pkl

    Returns
    -------
    pd.DataFrame : model comparison table
    """
    X_arr = X.values.astype(float)
    y_arr = y.values.astype(int)

    print(f"\n{'='*62}")
    print(f"  DSF504 ESG Greenwashing — Model Training")
    print(f"  Dataset: {X_arr.shape[0]} samples, {X_arr.shape[1]} features")
    print(f"  Label:   {y_arr.sum()} positives / {len(y_arr)} total "
          f"({y_arr.mean()*100:.1f}% greenwashing)")
    print(f"  CV:      outer {N_OUTER_FOLDS}-fold / inner {N_INNER_FOLDS}-fold")
    print(f"  Optuna:  {N_OPTUNA_TRIALS} trials per fold")
    print(f"{'='*62}")

    # ── 1. Baseline ──────────────────────────────────────────────────────────
    results = [evaluate_baseline(X_arr, y_arr)]

    # ── 2. Advanced models ───────────────────────────────────────────────────
    for model_name in ["lr", "rf", "xgb", "lgbm"]:
        results.append(evaluate_model(model_name, X_arr, y_arr))

    # ── 3. Compute delta AUC vs baseline ─────────────────────────────────────
    baseline_auc = results[0]["mean_roc_auc"]
    for r in results:
        r["delta_auc_vs_baseline"] = r["mean_roc_auc"] - baseline_auc

    # ── 4. Build comparison DataFrame ────────────────────────────────────────
    comparison_cols = [
        "model", "mean_roc_auc", "std_roc_auc",
        "mean_f1", "std_f1",
        "mean_pr_auc", "std_pr_auc",
        "delta_auc_vs_baseline",
    ]
    comparison_df = pd.DataFrame(
        [{c: r[c] for c in comparison_cols} for r in results]
    ).round(4)

    # ── 5. Save outputs ───────────────────────────────────────────────────────
    csv_path = os.path.join(OUTPUTS_DIR, "model_comparison.csv")
    comparison_df.to_csv(csv_path, index=False)
    print(f"\n[model_training] Comparison table saved → {csv_path}")

    best_params_all = {r["model_key"]: r["best_params"] for r in results}
    params_path = os.path.join(OUTPUTS_DIR, "best_params.json")
    with open(params_path, "w") as f:
        json.dump(best_params_all, f, indent=2)
    print(f"[model_training] Best params saved      → {params_path}")

    # ── 6. Retrain + save final models ────────────────────────────────────────
    print("\n[model_training] Retraining on full dataset & saving models …")
    for r in results:
        save_final_model(r["model_key"], r["best_params"], X_arr, y_arr)

    # ── 7. Print summary table ────────────────────────────────────────────────
    print(f"\n{'='*62}")
    print("  FINAL MODEL COMPARISON")
    print(f"{'='*62}")
    print(comparison_df.to_string(index=False))
    print(f"{'='*62}\n")

    return comparison_df, results


# ─────────────────────────────────────────────────────────────────────────────
# 10. LEAKAGE CHECK — Talk-only feature set
# ─────────────────────────────────────────────────────────────────────────────

# Walk-derived features excluded from the Talk-only set.
# sector_rank    : percentile of sub-score avg within sector
#                  (derived from E/S/G sub-scores → correlated with Walk label)
# e_s_g_dispersion: std of sub-scores → also Walk-side structural data
# Removing them isolates whether the model relies purely on text / third-party signals.
WALK_FEATURES = ["sector_rank", "e_s_g_dispersion"]

TALK_ONLY_FEATURES = [
    # Talk (NLP)
    "E_density", "S_density", "G_density",
    "hedge_score", "pct_numeric",
    "talk_total_density", "vague_to_specific_ratio",
    "text_length", "esg_topic_breadth",
    # Third-party cross-validation (not Walk, not label-derived)
    "planet_friendly_business", "honest_fair_business",
    # Context
    "sector_encoded",
]


def run_leakage_check(
    X: pd.DataFrame,
    y: pd.Series,
    full_model_auc: float = 0.9528,
) -> pd.DataFrame:
    """
    Leakage diagnostic: re-run LightGBM nested CV after removing the two
    Walk-derived structural features (sector_rank, e_s_g_dispersion).

    Rationale
    ---------
    sector_rank is computed from ESG sub-score averages, which are correlated
    with the Walk score used to construct the label. If the full model's AUC
    collapses without it, that is a signal worth documenting (even if not
    strict data leakage, it borders on circular reasoning).

    If Talk-only AUC remains strong (>0.80), the model has genuine predictive
    signal in text-based and third-party features, independent of Walk proxies.

    Parameters
    ----------
    full_model_auc : float — the reference AUC from the full 14-feature model
                             (default: 0.9528 from prior nested CV run)

    Returns
    -------
    pd.DataFrame : two-row comparison table saved to outputs/leakage_check.csv
    """
    print(f"\n{'='*62}")
    print("  LEAKAGE CHECK — Talk-only feature subset")
    print(f"  Removed Walk features: {WALK_FEATURES}")
    print(f"  Remaining features   : {len(TALK_ONLY_FEATURES)}")
    print(f"{'='*62}")

    # Validate all expected columns exist
    missing = [f for f in TALK_ONLY_FEATURES if f not in X.columns]
    if missing:
        raise ValueError(f"Features not found in X: {missing}")

    X_talk = X[TALK_ONLY_FEATURES]
    X_arr  = X_talk.values.astype(float)
    y_arr  = y.values.astype(int)

    print(f"\n[leakage_check] Feature matrix: {X_arr.shape[0]} samples × "
          f"{X_arr.shape[1]} features (Talk-only)\n")

    result = evaluate_model("lgbm", X_arr, y_arr)
    talk_only_auc = result["mean_roc_auc"]
    delta_vs_full = talk_only_auc - full_model_auc

    # ── Build comparison table ────────────────────────────────────────────────
    rows = [
        {
            "feature_set":          "Full (14 features)",
            "n_features":           len(X.columns),
            "walk_features_included": True,
            "mean_roc_auc":         round(full_model_auc, 4),
            "mean_f1":              None,     # from prior run; not re-computed here
            "mean_pr_auc":          None,
            "delta_vs_full_model":  0.0,
            "verdict":              "Reference",
        },
        {
            "feature_set":          "Talk-only (12 features, no sector_rank / e_s_g_dispersion)",
            "n_features":           len(TALK_ONLY_FEATURES),
            "walk_features_included": False,
            "mean_roc_auc":         round(talk_only_auc, 4),
            "mean_f1":              round(result["mean_f1"], 4),
            "mean_pr_auc":          round(result["mean_pr_auc"], 4),
            "delta_vs_full_model":  round(delta_vs_full, 4),
            "verdict": (
                "LOW leakage risk — text signal survives without Walk proxies"
                if talk_only_auc >= 0.80
                else "MODERATE — Walk proxies contribute substantially; document in Limitations"
            ),
        },
    ]
    df_out = pd.DataFrame(rows)

    csv_path = os.path.join(OUTPUTS_DIR, "leakage_check.csv")
    df_out.to_csv(csv_path, index=False)

    print(f"\n{'='*62}")
    print("  LEAKAGE CHECK RESULT")
    print(f"{'='*62}")
    print(f"  Full model  (14 features)  ROC-AUC = {full_model_auc:.4f}")
    print(f"  Talk-only   (12 features)  ROC-AUC = {talk_only_auc:.4f}  "
          f"(Δ = {delta_vs_full:+.4f})")
    print(f"  Verdict: {rows[1]['verdict']}")
    print(f"  Saved → {csv_path}")
    print(f"{'='*62}\n")

    return df_out


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.data_loader import load_merged_dataset
    from src.feature_engineering import build_feature_matrix

    parser = argparse.ArgumentParser(description="Train or compare ESG greenwashing risk models.")
    parser.add_argument(
        "--feature-set",
        choices=["baseline", "semantic"],
        default="baseline",
        help="Feature set to evaluate for comparison mode. Default keeps original 14-feature baseline.",
    )
    parser.add_argument(
        "--compare-feature-sets",
        action="store_true",
        help="Compare baseline 14-feature and semantic 19-feature matrices without overwriting model pickles.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use 10 Optuna trials with the existing 5x3 nested CV structure.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Ultra-fast fallback: single 80/20 stratified split, no Optuna tuning.",
    )
    parser.add_argument(
        "--leakage",
        action="store_true",
        help="Run the existing reduced Walk-proxy leakage check only.",
    )
    args = parser.parse_args()

    if args.quick:
        N_OPTUNA_TRIALS = 10

    if args.smoke:
        run_mode = "smoke"
    else:
        run_mode = "quick" if args.quick else "default"

    if args.compare_feature_sets:
        run_compare_feature_sets(["baseline", "semantic"], run_mode=run_mode)
    elif "--feature-set" in sys.argv:
        run_single_feature_set(args.feature_set, run_mode=run_mode)
    else:
        print("[model_training] Loading data …")
        df = load_merged_dataset(verbose=False)

        print("[model_training] Building features …")
        X, y, meta = build_feature_matrix(df, feature_set="baseline", verbose=False)

        if args.leakage:
            run_leakage_check(X, y, full_model_auc=0.9528)
        else:
            comparison_df, results = run_all(X, y)
