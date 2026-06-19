"""
calibration_patch.py
====================
OPTIONAL probability calibration tool for the DSF504 ESG Greenwashing
pipeline. NOT APPLIED to the submitted dashboard — supplied as a future
validation / robustness-improvement extension.

Status
------
The submitted dashboard (app_annual_report_style_v2_safe_patch.py) loads
the original models/lgbm_best.pkl and displays raw LightGBM output as a
relative model-estimated risk score (clearly labelled as ranking output,
not literal probability). This script is provided so that the team can
study probability calibration as a follow-up; it is intentionally NOT
wired into the submitted demo to keep the dashboard output consistent
with the cross-validation metrics reported in §7 of the Final Report.

Problem this WOULD address (future work)
----------------------------------------
The LightGBM pipeline (SMOTE + LGBMClassifier, class_weight='balanced',
trained on ~257 samples with ~7.4% positive rate) produces over-confident
probabilities for upper-tail cases. In the live Streamlit demo this shows
up as "100.0%" risk scores (e.g., CVX). Calibration could compress these
to a more interpretable upper band, at the cost of needing to re-run on
real data, swap the dashboard pickle, retest, and re-capture screenshots.
That re-test cycle is intentionally deferred to a follow-up iteration.

Reference
---------
Niculescu-Mizil, A. & Caruana, R. (2005). Predicting good probabilities
with supervised learning. ICML 2005.

What this script does when run
------------------------------
1. Re-instantiates the SAME imblearn Pipeline (SMOTE + LGBMClassifier) with
   the SAME best hyperparameters that produced models/lgbm_best.pkl.
2. Wraps it in sklearn.calibration.CalibratedClassifierCV with cv=5, so that
   inside each calibration fold SMOTE is only applied to the training split
   (the imblearn Pipeline guarantees this — the calibration split is never
   resampled).
3. Produces two calibrated pickles AS BUILD ARTIFACTS for future evaluation:
      models/lgbm_best_calibrated_isotonic.pkl
      models/lgbm_best_calibrated_sigmoid.pkl    (Platt scaling)
4. Prints a before/after comparison on the upper-tail demo cases so a
   future evaluator can decide whether calibration meaningfully improves
   the demo without distorting the cross-validation ranking.

Which method to pick (if/when calibration is adopted)
-----------------------------------------------------
For this dataset (n=257, only 19 positives, gradient-boosted base learner),
the empirical rule of thumb from Niculescu-Mizil & Caruana (2005) is:

  1. Try ISOTONIC first. It is non-parametric and usually best for boosted
     trees once probabilities deviate noticeably from a sigmoid shape.
  2. If isotonic still produces values >= 0.999 on upper-tail demo cases
     (which happens when the model is genuinely confident on those rows
     in every calibration fold), use SIGMOID (Platt) instead.

The diagnostic printout at the end tells the operator which to pick.

Important: any switch to a calibrated pickle requires updating Streamlit
caching, the demo screenshots, and the §7/§8 numbers in the Final Report.
Do NOT swap the dashboard pickle without doing the full re-test cycle.

How to use (FUTURE WORK ONLY — not required for v3 submission)
---------------------------------------------------------------
From the project root:

    python src/calibration_patch.py

This will produce the two calibrated pickles in models/. The submitted
dashboard does NOT load them; they exist for offline study. If a future
iteration decides to adopt calibration, the dashboard load line:

    model = joblib.load("models/lgbm_best.pkl")

would change to:

    model = joblib.load("models/lgbm_best_calibrated_<method>.pkl")

— and the §7/§8 report numbers + demo screenshots must be regenerated
together to keep the deliverables consistent.

Notes
-----
- Run on the FULL feature matrix X (same one used in run_all). The script
  does its own internal cross-validation; you do NOT need a manual split.
- Isotonic is preferred when n_pos >= 15 (we have 19) because it is
  non-parametric and handles the small-sample boosting case better.
  Sigmoid (Platt) is provided as a backup in case isotonic over-fits.
- This does NOT retrain on test data the original nested-CV used; the
  performance metrics in outputs/model_comparison.csv remain unbiased.
"""

import os
import sys
import json
import pickle
from typing import Tuple

import numpy as np
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE


# ── paths ────────────────────────────────────────────────────────────────────
HERE         = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(HERE) if os.path.basename(HERE) == "src" else HERE
MODELS_DIR   = os.path.join(PROJECT_ROOT, "models")
OUTPUTS_DIR  = os.path.join(PROJECT_ROOT, "outputs")
os.makedirs(MODELS_DIR, exist_ok=True)

ORIGINAL_PKL = os.path.join(MODELS_DIR, "lgbm_best.pkl")
BEST_PARAMS_JSON = os.path.join(OUTPUTS_DIR, "best_params.json")

RANDOM_STATE  = 42
SMOTE_K       = 3
CALIB_CV      = 5     # internal folds for CalibratedClassifierCV
DEMO_TICKERS  = ["CVX", "MTB", "WYNN"]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Load best LightGBM hyperparameters
# ─────────────────────────────────────────────────────────────────────────────

def load_lgbm_best_params() -> dict:
    """
    Read best LightGBM hyperparameters from outputs/best_params.json
    (written by model_training.run_all). Falls back to inspecting the
    existing lgbm_best.pkl if the JSON is missing.
    """
    if os.path.exists(BEST_PARAMS_JSON):
        with open(BEST_PARAMS_JSON, "r") as f:
            all_params = json.load(f)
        if "lgbm" in all_params:
            print(f"[calibration_patch] Loaded LightGBM best params from {BEST_PARAMS_JSON}")
            return all_params["lgbm"]

    # Fallback: read from the existing pickle
    print(f"[calibration_patch] best_params.json not found; "
          f"reading hyperparameters from {ORIGINAL_PKL}")
    with open(ORIGINAL_PKL, "rb") as f:
        pipe = pickle.load(f)
    clf = pipe.named_steps["clf"]
    # extract only the hyperparameters that were tuned by Optuna
    keys = ["n_estimators", "max_depth", "learning_rate",
            "num_leaves", "min_child_samples", "subsample"]
    return {k: getattr(clf, k) for k in keys if hasattr(clf, k)}


# ─────────────────────────────────────────────────────────────────────────────
# 2. Build base pipeline (matches model_training.build_pipeline for lgbm)
# ─────────────────────────────────────────────────────────────────────────────

def build_base_lgbm_pipeline(params: dict) -> ImbPipeline:
    """
    Mirrors model_training.build_pipeline('lgbm', params) so the calibration
    layer wraps an identical estimator.
    """
    return ImbPipeline(steps=[
        ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=SMOTE_K)),
        ("clf",   LGBMClassifier(
            random_state=RANDOM_STATE,
            class_weight="balanced",
            verbose=-1,
            **params,
        )),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# 3. Fit calibrated wrappers
# ─────────────────────────────────────────────────────────────────────────────

def fit_calibrated(
    X: np.ndarray,
    y: np.ndarray,
    params: dict,
    method: str,
) -> CalibratedClassifierCV:
    """
    Wrap the imblearn Pipeline in CalibratedClassifierCV with internal
    StratifiedKFold(cv=5). SMOTE inside the pipeline is applied only to
    each fold's training split; the held-out calibration split is never
    resampled, which preserves the SMOTE-trap guarantee from model_training.

    method: 'isotonic' (recommended) or 'sigmoid' (Platt, backup)
    """
    base = build_base_lgbm_pipeline(params)
    cv = StratifiedKFold(
        n_splits=CALIB_CV, shuffle=True, random_state=RANDOM_STATE
    )
    calibrated = CalibratedClassifierCV(
        estimator=base,
        method=method,
        cv=cv,
        n_jobs=1,         # keep single-threaded for reproducibility
    )
    calibrated.fit(X, y)
    return calibrated


# ─────────────────────────────────────────────────────────────────────────────
# 4. Diagnostic: before vs after on the full dataset
# ─────────────────────────────────────────────────────────────────────────────

def diagnose_calibration(
    X: np.ndarray,
    y: np.ndarray,
    original_pipe,
    calibrated_iso: CalibratedClassifierCV,
    calibrated_sig: CalibratedClassifierCV,
    ticker_col: pd.Series = None,
) -> pd.DataFrame:
    """
    Print and return a small comparison table:
      - upper-tail probability statistics
      - per-ticker scores for the demo cases (if ticker_col supplied)
    """
    p_orig = original_pipe.predict_proba(X)[:, 1]
    p_iso  = calibrated_iso.predict_proba(X)[:, 1]
    p_sig  = calibrated_sig.predict_proba(X)[:, 1]

    # ── Distribution summary ────────────────────────────────────────────────
    print("\n" + "="*62)
    print("  CALIBRATION DIAGNOSTIC  —  upper-tail summary (all 257 cos.)")
    print("="*62)
    dist = pd.DataFrame({
        "raw_lgbm":           pd.Series(p_orig).describe(percentiles=[.5, .9, .95, .99]),
        "calibrated_isotonic":pd.Series(p_iso).describe(percentiles=[.5, .9, .95, .99]),
        "calibrated_sigmoid": pd.Series(p_sig).describe(percentiles=[.5, .9, .95, .99]),
    }).round(4)
    print(dist.to_string())

    n_at_one_raw = int(np.sum(p_orig >= 0.999))
    n_at_one_iso = int(np.sum(p_iso  >= 0.999))
    n_at_one_sig = int(np.sum(p_sig  >= 0.999))
    print(f"\nCompanies with prob >= 0.999:")
    print(f"  raw_lgbm           = {n_at_one_raw:3d}")
    print(f"  calibrated_isotonic= {n_at_one_iso:3d}")
    print(f"  calibrated_sigmoid = {n_at_one_sig:3d}")

    # ── Per-ticker demo cases ───────────────────────────────────────────────
    if ticker_col is not None:
        mask = ticker_col.isin(DEMO_TICKERS).values
        if mask.any():
            print("\n" + "="*62)
            print("  DEMO CASES  —  CVX / MTB / WYNN")
            print("="*62)
            df_demo = pd.DataFrame({
                "ticker":          ticker_col[mask].values,
                "label":           y[mask],
                "raw_lgbm":        np.round(p_orig[mask], 4),
                "calibrated_iso":  np.round(p_iso[mask],  4),
                "calibrated_sig":  np.round(p_sig[mask],  4),
            })
            print(df_demo.to_string(index=False))
            return df_demo

    return None


# ─────────────────────────────────────────────────────────────────────────────
# 5. Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_calibration(X: pd.DataFrame, y: pd.Series, ticker_col: pd.Series = None):
    """
    End-to-end: load best params, fit both calibrated models, save them,
    print the before/after diagnostic.
    """
    print("\n" + "="*62)
    print("  DSF504 ESG Greenwashing — Probability Calibration Patch")
    print("="*62)

    X_arr = X.values.astype(float) if hasattr(X, "values") else np.asarray(X, dtype=float)
    y_arr = y.values.astype(int)   if hasattr(y, "values") else np.asarray(y, dtype=int)
    print(f"  Dataset: {X_arr.shape[0]} samples, {X_arr.shape[1]} features")
    print(f"  Label:   {y_arr.sum()} positives ({y_arr.mean()*100:.1f}%)")
    print(f"  Internal CV: {CALIB_CV}-fold StratifiedKFold")

    # ── Load best params ────────────────────────────────────────────────────
    params = load_lgbm_best_params()
    print(f"  LightGBM params: {params}")

    # ── Fit isotonic and sigmoid calibrators ────────────────────────────────
    print("\n[calibration_patch] Fitting isotonic calibration ...")
    calibrated_iso = fit_calibrated(X_arr, y_arr, params, method="isotonic")

    print("[calibration_patch] Fitting sigmoid (Platt) calibration ...")
    calibrated_sig = fit_calibrated(X_arr, y_arr, params, method="sigmoid")

    # ── Save ────────────────────────────────────────────────────────────────
    iso_path = os.path.join(MODELS_DIR, "lgbm_best_calibrated_isotonic.pkl")
    sig_path = os.path.join(MODELS_DIR, "lgbm_best_calibrated_sigmoid.pkl")
    with open(iso_path, "wb") as f:
        pickle.dump(calibrated_iso, f)
    with open(sig_path, "wb") as f:
        pickle.dump(calibrated_sig, f)
    print(f"\n[calibration_patch] Saved → {iso_path}")
    print(f"[calibration_patch] Saved → {sig_path}")

    # ── Diagnostic ──────────────────────────────────────────────────────────
    with open(ORIGINAL_PKL, "rb") as f:
        original_pipe = pickle.load(f)
    demo_df = diagnose_calibration(
        X_arr, y_arr, original_pipe, calibrated_iso, calibrated_sig, ticker_col
    )

    if demo_df is not None:
        demo_csv = os.path.join(OUTPUTS_DIR, "calibration_demo_cases.csv")
        demo_df.to_csv(demo_csv, index=False)
        print(f"\n[calibration_patch] Demo cases CSV → {demo_csv}")

    print("\n" + "="*62)
    print("  NEXT STEP — pick the calibration method")
    print("="*62)
    n_iso_saturated = int(np.sum(calibrated_iso.predict_proba(X_arr)[:, 1] >= 0.999))
    if n_iso_saturated == 0:
        chosen = "isotonic"
    else:
        chosen = "sigmoid"
        print(f"  Isotonic still has {n_iso_saturated} cases at p>=0.999")
        print(f"  → using SIGMOID (Platt) calibration instead")
    print(f"\n  Recommended pickle: models/lgbm_best_calibrated_{chosen}.pkl")
    print("\n  In the Streamlit app, replace:")
    print('      model = joblib.load("models/lgbm_best.pkl")')
    print("  with:")
    print(f'      model = joblib.load("models/lgbm_best_calibrated_{chosen}.pkl")')
    print("="*62 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Import the same data + feature pipeline the rest of the project uses
    sys.path.insert(0, PROJECT_ROOT)
    from src.data_loader        import load_merged_dataset
    from src.feature_engineering import build_feature_matrix

    print("[calibration_patch] Loading data ...")
    df = load_merged_dataset(verbose=False)
    print("[calibration_patch] Building feature matrix ...")
    X, y, meta = build_feature_matrix(df, verbose=False)

    # ticker column for demo-case printout (best-effort; safe if missing)
    ticker_col = None
    if "ticker" in df.columns:
        # align ticker order with X (assumes build_feature_matrix preserves row order)
        ticker_col = df.loc[X.index, "ticker"] if hasattr(X, "index") else df["ticker"]

    run_calibration(X, y, ticker_col=ticker_col)
