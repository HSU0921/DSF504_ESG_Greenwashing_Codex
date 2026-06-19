"""
v2_clean_label_experiment.py
============================
Day 1 professor-feedback response for DSF504 Team 5.

Adds a cleaner v2 label and evaluates small, traceable models without deleting
or overwriting the original v1 supervised model or model pickle files.

Governance disclaimer:
AI Decision Support Only: This system is designed for ESG risk triage and
review prioritization. It does not legally determine greenwashing, credit
approval, or investment exclusion.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
RANDOM_STATE = 42
N_SPLITS = 5
GOVERNANCE_DISCLAIMER = (
    "AI Decision Support Only: This system is designed for ESG risk triage and "
    "review prioritization. It does not legally determine greenwashing, credit "
    "approval, or investment exclusion."
)

TALK_LABEL_FEATURE = "talk_total_density"
TEXT_MODEL_FEATURES = [
    "hedge_score",
    "pct_numeric",
    "vague_to_specific_ratio",
    "esg_topic_breadth",
    "text_length",
]
EXPLAINABLE_TEXT_FEATURES = [
    "E_density",
    "S_density",
    "G_density",
    "hedge_score",
    "pct_numeric",
    "vague_to_specific_ratio",
    "esg_topic_breadth",
    "text_length",
]
LABEL_ONLY_FIELDS = {
    "Total ESG Risk score",
    "Controversy Score",
    "ESG Risk Percentile",
    "ESG Risk Level",
    "greenwashing_label",
    "high_risk_v2",
    "label",
    "y",
    "target",
    "Controversy Level",
}


def percentile_rank(series: pd.Series, higher_is_weaker: bool = True) -> pd.Series:
    """Return 0-1 percentile rank, with higher values meaning more of the target concept."""
    values = pd.to_numeric(series, errors="coerce")
    if higher_is_weaker:
        return values.rank(pct=True, method="average").fillna(0.5)
    return (-values).rank(pct=True, method="average").fillna(0.5)


def sector_percentile_rank(df: pd.DataFrame, column: str, higher_is_weaker: bool = True) -> pd.Series:
    values = pd.to_numeric(df[column], errors="coerce")
    work = df[["Sector"]].copy()
    work["_value"] = values
    if higher_is_weaker:
        ranked = work.groupby("Sector")["_value"].rank(pct=True, method="average")
    else:
        ranked = (-work["_value"]).groupby(work["Sector"]).rank(pct=True, method="average")
    return ranked.fillna(0.5)


def build_v2_label_dataset(df: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    """
    high_risk_v2 = High Talk AND Multi-source Weak Evidence.

    High Talk is top-third talk_total_density.
    Multi-source weak evidence requires both:
      1. Sustainalytics weakness from Total ESG Risk or sector_rank.
      2. Integrity weakness from planet_friendly_business / honest_fair_business.
    """
    out = pd.DataFrame(index=X.index)
    out["ticker"] = df["ticker"].astype(str).values
    out["company_name"] = df["Name"].astype(str).values if "Name" in df.columns else out["ticker"]
    out["sector"] = df["Sector"].astype(str).values
    out["talk_total_density"] = pd.to_numeric(X["talk_total_density"], errors="coerce")
    out["talk_percentile"] = percentile_rank(out["talk_total_density"], higher_is_weaker=True)
    out["high_talk"] = out["talk_percentile"] >= (2 / 3)

    out["total_esg_risk_sector_percentile"] = sector_percentile_rank(
        df, "Total ESG Risk score", higher_is_weaker=True
    )
    out["sector_rank"] = pd.to_numeric(X["sector_rank"], errors="coerce").fillna(0.5)
    out["sustainalytics_total_weak"] = out["total_esg_risk_sector_percentile"] >= (2 / 3)
    out["sustainalytics_sector_rank_weak"] = out["sector_rank"] >= (2 / 3)
    out["sustainalytics_weak"] = (
        out["sustainalytics_total_weak"] | out["sustainalytics_sector_rank_weak"]
    )

    integrity_mean = pd.concat(
        [
            pd.to_numeric(X["planet_friendly_business"], errors="coerce"),
            pd.to_numeric(X["honest_fair_business"], errors="coerce"),
        ],
        axis=1,
    ).mean(axis=1)
    out["integrity_mean"] = integrity_mean
    out["integrity_weakness_percentile"] = percentile_rank(integrity_mean, higher_is_weaker=False)
    out["integrity_weak"] = out["integrity_weakness_percentile"] >= (2 / 3)
    out["weak_evidence_source_count"] = out[["sustainalytics_weak", "integrity_weak"]].sum(axis=1)
    out["multi_source_weak_evidence"] = out["weak_evidence_source_count"] >= 2
    out["high_risk_v2"] = (out["high_talk"] & out["multi_source_weak_evidence"]).astype(int)
    return out


def _fold_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "positive_support": int(np.sum(y_true == 1)),
        "negative_support": int(np.sum(y_true == 0)),
    }


def _make_pipeline(model_name: str, y_train: np.ndarray):
    if model_name == "v2_governance_baseline":
        return DummyClassifier(strategy="stratified", random_state=RANDOM_STATE)
    minority = int(np.sum(y_train == 1))
    if minority < 2:
        return ImbPipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )
    k_neighbors = max(1, min(3, minority - 1))
    return ImbPipeline(
        steps=[
            ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=k_neighbors)),
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


@dataclass(frozen=True)
class ModelSpec:
    model_name: str
    feature_group: str
    features: list[str]
    note: str


def evaluate_model_spec(X: pd.DataFrame, y: pd.Series, spec: ModelSpec) -> dict:
    X_arr = X[spec.features].values.astype(float) if spec.features else np.zeros((len(y), 1))
    y_arr = y.values.astype(int)
    n_splits = min(N_SPLITS, int(np.sum(y_arr == 1)), int(np.sum(y_arr == 0)))
    if n_splits < 2:
        raise ValueError("Not enough positive/negative v2 labels for stratified CV.")

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    fold_rows = []
    for train_idx, test_idx in cv.split(X_arr, y_arr):
        model = _make_pipeline(spec.model_name, y_arr[train_idx])
        model.fit(X_arr[train_idx], y_arr[train_idx])
        y_pred = model.predict(X_arr[test_idx])
        y_proba = model.predict_proba(X_arr[test_idx])[:, 1]
        fold_rows.append(_fold_metrics(y_arr[test_idx], y_pred, y_proba))

    folds = pd.DataFrame(fold_rows)
    cm_total = folds[["tn", "fp", "fn", "tp"]].sum().astype(int).to_dict()
    return {
        "model_name": spec.model_name,
        "feature_group": spec.feature_group,
        "n_features": len(spec.features),
        "features_used": ", ".join(spec.features) if spec.features else "No features (DummyClassifier)",
        "roc_auc_mean": folds["roc_auc"].mean(),
        "roc_auc_std": folds["roc_auc"].std(ddof=0),
        "pr_auc_mean": folds["pr_auc"].mean(),
        "pr_auc_std": folds["pr_auc"].std(ddof=0),
        "f1_mean": folds["f1"].mean(),
        "f1_std": folds["f1"].std(ddof=0),
        "precision_mean": folds["precision"].mean(),
        "recall_mean": folds["recall"].mean(),
        "accuracy_mean": folds["accuracy"].mean(),
        "confusion_matrix": json.dumps([[cm_total["tn"], cm_total["fp"]], [cm_total["fn"], cm_total["tp"]]]),
        "positive_class_support": int(folds["positive_support"].sum()),
        "negative_class_support": int(folds["negative_support"].sum()),
        "interpretation": spec.note,
        "governance_disclaimer": GOVERNANCE_DISCLAIMER,
    }


def write_definition(v2_df: pd.DataFrame) -> None:
    positive = int(v2_df["high_risk_v2"].sum())
    total = len(v2_df)
    sector_dist = (
        v2_df[v2_df["high_risk_v2"] == 1]["sector"]
        .value_counts()
        .rename_axis("sector")
        .reset_index(name="v2_positive_count")
    )
    top_cases = v2_df[v2_df["high_risk_v2"] == 1].sort_values(
        ["talk_percentile", "integrity_weakness_percentile"], ascending=False
    ).head(12)
    path = os.path.join(OUTPUTS_DIR, "v2_clean_label_definition.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            f"""# V2 Clean Label Definition

## Definition

`high_risk_v2 = High Talk AND Multi-source Weak Evidence`

High Talk uses sustainability-report text-derived disclosure intensity:

- `talk_total_density` in the top 33% of all companies.

Multi-source Weak Evidence requires both:

- Sustainalytics weakness: weak sector-relative ESG risk evidence from `Total ESG Risk score` or `sector_rank`.
- Integrity weakness: weak integrity signal from `planet_friendly_business` and `honest_fair_business`.

This label is cleaner than v1 because it does not rely on a single external rating signal alone. It requires weak evidence from more than one source.

## Label Counts

- Positive count: {positive}
- Negative count: {total - positive}
- Positive rate: {positive / total:.1%}

## Sector Distribution of V2 Positives

{sector_dist.to_markdown(index=False) if not sector_dist.empty else "No v2 positives found."}

## Top V2 Positive Cases

{top_cases[["ticker", "company_name", "sector", "talk_percentile", "total_esg_risk_sector_percentile", "integrity_weakness_percentile"]].to_markdown(index=False) if not top_cases.empty else "No v2 positives found."}

## Governance Disclaimer

{GOVERNANCE_DISCLAIMER}
"""
        )


def write_metrics_markdown(metrics: pd.DataFrame) -> None:
    path = os.path.join(OUTPUTS_DIR, "v2_clean_model_metrics.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            f"""# V2 Clean Model Metrics

These models test whether a cleaner v2 label can be predicted without simply copying external ESG rating outputs.

{metrics[["model_name", "feature_group", "n_features", "roc_auc_mean", "pr_auc_mean", "f1_mean", "precision_mean", "recall_mean", "accuracy_mean", "interpretation"]].to_markdown(index=False)}

## Interpretation

The v2 experiment is intentionally more conservative than the original v1 model. If performance is lower, that is not a failure. It means the cleaner label is less tied to processed external ESG rating structure and provides a more honest governance response.

{GOVERNANCE_DISCLAIMER}
"""
        )


def run_experiment() -> pd.DataFrame:
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
    sys.path.insert(0, PROJECT_ROOT)
    try:
        from src.data_loader import load_merged_dataset
        from src.feature_engineering import build_feature_matrix
    except ModuleNotFoundError as exc:
        try:
            from data_loader import load_merged_dataset
            from feature_engineering import build_feature_matrix
        except ModuleNotFoundError as fallback_exc:
            raise RuntimeError(
                "Could not import the existing project data loader or feature engineering module. "
                "Run this script from the project environment with access to the existing src/ "
                "folder; on macOS, grant the terminal/Codex app permission to read the Desktop "
                "project folder if imports are blocked."
            ) from fallback_exc

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    df = load_merged_dataset(verbose=False)
    X, _, _ = build_feature_matrix(df, feature_set="baseline", verbose=False)
    leaked = LABEL_ONLY_FIELDS.intersection(X.columns)
    if leaked:
        raise ValueError(f"Label-only fields found in feature matrix: {sorted(leaked)}")

    v2_df = build_v2_label_dataset(df, X)
    summary_cols = [
        "ticker",
        "company_name",
        "sector",
        "talk_percentile",
        "high_talk",
        "total_esg_risk_sector_percentile",
        "sector_rank",
        "sustainalytics_weak",
        "integrity_weakness_percentile",
        "integrity_weak",
        "weak_evidence_source_count",
        "multi_source_weak_evidence",
        "high_risk_v2",
    ]
    v2_df[summary_cols].to_csv(
        os.path.join(OUTPUTS_DIR, "v2_clean_label_dataset_summary.csv"),
        index=False,
    )
    write_definition(v2_df)

    y_v2 = v2_df["high_risk_v2"].astype(int)
    specs = [
        ModelSpec(
            "v2_governance_baseline",
            "DummyClassifier baseline",
            [],
            "Governance baseline. Any useful v2 model should perform better than this naive reference.",
        ),
        ModelSpec(
            "v2_text_model",
            "Primary clean model: indirect text features only",
            [feature for feature in TEXT_MODEL_FEATURES if feature in X.columns],
            "Primary clean v2 model. Uses text features not directly used as the High Talk threshold, making it the cleanest text-only v2 test.",
        ),
        ModelSpec(
            "v2_explainable_model",
            "Sensitivity analysis: explainable text model with potentially label-related Talk features",
            [feature for feature in EXPLAINABLE_TEXT_FEATURES if feature in X.columns],
            "Sensitivity analysis only. Includes E/S/G text features that may be related to label construction; do not present it as the primary clean model.",
        ),
    ]
    rows = [evaluate_model_spec(X, y_v2, spec) for spec in specs]
    metrics = pd.DataFrame(rows)
    numeric_cols = [
        "roc_auc_mean",
        "roc_auc_std",
        "pr_auc_mean",
        "pr_auc_std",
        "f1_mean",
        "f1_std",
        "precision_mean",
        "recall_mean",
        "accuracy_mean",
    ]
    metrics[numeric_cols] = metrics[numeric_cols].round(4)
    metrics.to_csv(os.path.join(OUTPUTS_DIR, "v2_clean_model_metrics.csv"), index=False)
    write_metrics_markdown(metrics)

    positive = int(y_v2.sum())
    print("\n=== V2 Clean Label Experiment ===")
    print(f"V2 positives: {positive} / {len(y_v2)} ({positive / len(y_v2):.1%})")
    print(metrics[["model_name", "roc_auc_mean", "pr_auc_mean", "f1_mean"]].to_string(index=False))
    print(f"\n{GOVERNANCE_DISCLAIMER}")
    return metrics


if __name__ == "__main__":
    run_experiment()
