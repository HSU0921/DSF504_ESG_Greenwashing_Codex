"""
enhanced_text_features.py
=========================
Additive professor-feedback text feature validation.

This script creates one primary greenwashing text feature plus minimal
supporting candidates. It does not modify the original data loader, feature
engineering module, models, dashboards, notebooks, reports, or existing model
artifacts.

Governance disclaimer:
These features support ESG risk screening and due diligence prioritization.
They do not legally determine greenwashing, credit approval, or investment
exclusion.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
RANDOM_STATE = 42
EPS = 1e-6

GOVERNANCE_DISCLAIMER = (
    "These features support ESG risk screening and due diligence prioritization. "
    "They do not legally determine greenwashing, credit approval, or investment exclusion."
)

EXISTING_TEXT_FEATURES = [
    "E_density",
    "S_density",
    "G_density",
    "hedge_score",
    "pct_numeric",
    "talk_total_density",
    "vague_to_specific_ratio",
    "text_length",
    "esg_topic_breadth",
]

PROMO_PATTERNS = [
    r"\bgreen\b",
    r"\bclean\b",
    r"\beco[-\s]?friendly\b",
    r"\bsustainable future\b",
    r"\bnatural gas\b",
    r"\blow[-\s]?carbon\b",
    r"\benvironmentally friendly\b",
    r"\bclimate solutions?\b",
]

SPECIFIC_PATTERNS = [
    r"\bgri\b",
    r"\bsasb\b",
    r"\btcfd\b",
    r"\bscope\s*[123]\b",
    r"\bverified\b",
    r"\bassurance\b",
    r"%",
    r"\bpercent\b",
    r"\bbaseline\b",
    r"\btarget(s)?\b",
]

TARGET_PATTERNS = [
    r"\btarget(s)?\b",
    r"\bgoal(s)?\b",
    r"\bnet[-\s]?zero\b",
    r"\bscience[-\s]?based target(s)?\b",
    r"\breduction target(s)?\b",
]

QUANTIFIED_PATTERNS = [
    r"\d+\s*%",
    r"\bpercent\b",
    r"\b20\d{2}\b",
    r"\d+(\.\d+)?\s*(mtco2e|tco2e|tons?|tonnes?|mw|mwh|kwh)\b",
]

FRAMEWORK_PATTERNS = [
    r"\bgri\b",
    r"\bsasb\b",
    r"\btcfd\b",
    r"\bsbti\b",
    r"\bscope\s*[123]\b",
    r"\bassurance\b",
    r"\bverified\b",
    r"\bbaseline\b",
]

VAGUE_PATTERNS = [
    r"\bcommitted to\b",
    r"\bmeaningful progress\b",
    r"\bcontinue to improve\b",
    r"\baspire to\b",
    r"\baim to\b",
    r"\bsupport a better future\b",
    r"\bsustainable future\b",
    r"\brobust framework\b",
    r"\bresponsible business\b",
]


@dataclass(frozen=True)
class CandidateFeature:
    name: str
    role: str
    description: str


CANDIDATES = [
    CandidateFeature(
        "promotional_to_specific_ratio",
        "PRIMARY",
        "Promotional language density divided by specific evidence density.",
    ),
    CandidateFeature(
        "sector_adj_promotional",
        "SUPPORTING",
        "Promotional density adjusted by train-fold sector median.",
    ),
    CandidateFeature(
        "promo_density",
        "SUPPORTING",
        "Promotional ESG claim density.",
    ),
    CandidateFeature(
        "specific_density",
        "SUPPORTING",
        "Specific ESG evidence density.",
    ),
    CandidateFeature(
        "concreteness_ratio",
        "SUPPORTING",
        "Target, quantified, and framework density divided by vague density.",
    ),
]


def _load_project_modules():
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
    sys.path.insert(0, PROJECT_ROOT)
    try:
        from src.data_loader import load_merged_dataset
        from src.feature_engineering import build_feature_matrix
    except ModuleNotFoundError:
        try:
            from data_loader import load_merged_dataset
            from feature_engineering import build_feature_matrix
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Could not import the existing project data loader or feature engineering module. "
                "Run this script from a Terminal/session with permission to read the Desktop project "
                "folder, or grant Codex/Terminal access to the project folder."
            ) from exc
    return load_merged_dataset, build_feature_matrix


def _compiled(patterns: list[str]) -> list[re.Pattern]:
    return [re.compile(pattern, flags=re.IGNORECASE) for pattern in patterns]


def _count_patterns(text: str, patterns: list[re.Pattern]) -> int:
    return int(sum(len(pattern.findall(text)) for pattern in patterns))


def _word_count(text: str) -> int:
    return max(len(str(text).split()), 1)


def build_candidate_features(df: pd.DataFrame) -> pd.DataFrame:
    if "preprocessed_content" not in df.columns:
        raise ValueError("Required text column not found: preprocessed_content")

    promo = _compiled(PROMO_PATTERNS)
    specific = _compiled(SPECIFIC_PATTERNS)
    target = _compiled(TARGET_PATTERNS)
    quantified = _compiled(QUANTIFIED_PATTERNS)
    framework = _compiled(FRAMEWORK_PATTERNS)
    vague = _compiled(VAGUE_PATTERNS)

    texts = df["preprocessed_content"].fillna("").astype(str).str.lower()
    wc = texts.map(_word_count)
    out = pd.DataFrame(index=df.index)
    out["promo_density"] = texts.map(lambda text: _count_patterns(text, promo)) / wc
    out["specific_density"] = texts.map(lambda text: _count_patterns(text, specific)) / wc
    out["promotional_to_specific_ratio"] = out["promo_density"] / (out["specific_density"] + EPS)

    target_density = texts.map(lambda text: _count_patterns(text, target)) / wc
    quantified_density = texts.map(lambda text: _count_patterns(text, quantified)) / wc
    framework_density = texts.map(lambda text: _count_patterns(text, framework)) / wc
    vague_density = texts.map(lambda text: _count_patterns(text, vague)) / wc
    out["concreteness_ratio"] = (
        target_density + quantified_density + framework_density
    ) / (vague_density + EPS)

    sector = df["Sector"] if "Sector" in df.columns else pd.Series("Unknown", index=df.index)
    full_sector_median = out["promo_density"].groupby(sector).transform("median")
    out["sector_adj_promotional"] = out["promo_density"] - full_sector_median
    return out.fillna(0.0)


class TextFeatureSelector(BaseEstimator, TransformerMixin):
    """Fold-safe selector for existing text features plus one candidate."""

    def __init__(self, base_features: list[str], candidate: str | None = None):
        self.base_features = base_features
        self.candidate = candidate
        self.global_promo_median_ = 0.0
        self.sector_promo_medians_ = {}

    def fit(self, X: pd.DataFrame, y=None):
        if self.candidate == "sector_adj_promotional":
            promo = pd.to_numeric(X["promo_density"], errors="coerce").fillna(0.0)
            sector = X["Sector"].fillna("Unknown").astype(str)
            self.global_promo_median_ = float(promo.median())
            self.sector_promo_medians_ = promo.groupby(sector).median().to_dict()
        return self

    def transform(self, X: pd.DataFrame):
        parts = [
            X[self.base_features].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
        ]
        if self.candidate:
            if self.candidate == "sector_adj_promotional":
                sector = X["Sector"].fillna("Unknown").astype(str)
                medians = sector.map(self.sector_promo_medians_).fillna(self.global_promo_median_)
                candidate = pd.to_numeric(X["promo_density"], errors="coerce").fillna(0.0) - medians
            else:
                candidate = pd.to_numeric(X[self.candidate], errors="coerce").fillna(0.0)
            parts.append(candidate.to_numpy(dtype=float).reshape(-1, 1))
        return np.hstack(parts)


def _spearman(series: pd.Series, y: pd.Series) -> float:
    corr = pd.to_numeric(series, errors="coerce").corr(pd.Series(y).astype(int), method="spearman")
    return 0.0 if pd.isna(corr) else float(corr)


def _sign(value: float) -> str:
    if value > 0:
        return "+"
    if value < 0:
        return "-"
    return "0"


def _fold_sign_stability(
    validation_df: pd.DataFrame,
    y: pd.Series,
    candidate: str,
    cv: StratifiedKFold,
) -> tuple[bool, str]:
    signs = []
    y_arr = y.astype(int).to_numpy()
    for train_idx, _ in cv.split(validation_df, y_arr):
        train = validation_df.iloc[train_idx].copy()
        if candidate == "sector_adj_promotional":
            medians = train["promo_density"].groupby(train["Sector"].fillna("Unknown").astype(str)).transform("median")
            feature_values = train["promo_density"] - medians
        else:
            feature_values = train[candidate]
        signs.append(_sign(_spearman(feature_values, pd.Series(y_arr[train_idx]))))
    nonzero = [sign for sign in signs if sign != "0"]
    stable = bool(nonzero) and len(set(nonzero)) == 1 and len(nonzero) == len(signs)
    return stable, ",".join(signs)


def _prediction_pipeline(base_features: list[str], candidate: str | None) -> ImbPipeline:
    return ImbPipeline(
        steps=[
            ("select", TextFeatureSelector(base_features=base_features, candidate=candidate)),
            ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=3)),
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


def _oof_scores(validation_df: pd.DataFrame, y: pd.Series, base_features: list[str], candidate: str | None):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    proba = cross_val_predict(
        _prediction_pipeline(base_features, candidate),
        validation_df,
        y.astype(int),
        cv=cv,
        method="predict_proba",
    )[:, 1]
    return {
        "roc_auc": roc_auc_score(y, proba),
        "pr_auc": average_precision_score(y, proba),
    }


def _company_name(df: pd.DataFrame) -> pd.Series:
    if "Name" in df.columns:
        return df["Name"].astype(str)
    if "Company Name" in df.columns:
        return df["Company Name"].astype(str)
    if "ticker" in df.columns:
        return df["ticker"].astype(str)
    return pd.Series("", index=df.index)


def write_outputs(
    df: pd.DataFrame,
    features: pd.DataFrame,
    validation: pd.DataFrame,
    baseline_scores: dict,
) -> None:
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    id_cols = pd.DataFrame(index=df.index)
    if "ticker" in df.columns:
        id_cols["ticker"] = df["ticker"].astype(str)
    id_cols["company_name"] = _company_name(df)
    id_cols["sector"] = df["Sector"].astype(str) if "Sector" in df.columns else "Unknown"
    pd.concat([id_cols, features], axis=1).to_csv(
        os.path.join(OUTPUTS_DIR, "enhanced_text_features.csv"),
        index=False,
    )

    kept = validation[validation["decision"] == "KEEP"]["feature"].tolist()
    dropped = validation[validation["decision"] == "DROP"]["feature"].tolist()
    with open(os.path.join(OUTPUTS_DIR, "enhanced_text_feature_validation.md"), "w", encoding="utf-8") as f:
        f.write(
            f"""# Enhanced Text Feature Validation

## Purpose

This additive validation tests one primary greenwashing text feature and minimal supporting features. It does not use sector ranking, ESG rating scores, integrity scores, or Sustainalytics-style circular features.

## Baseline Text Model

- Existing text features only ROC-AUC: {baseline_scores["roc_auc"]:.4f}
- Existing text features only PR-AUC: {baseline_scores["pr_auc"]:.4f}

## Validation Table

{validation.to_markdown(index=False)}

## Kept Features

{", ".join(kept) if kept else "No candidate features passed the keep rule."}

## Dropped / Flagged Features

{", ".join(dropped) if dropped else "No candidate features were dropped."}

## Governance Disclaimer

{GOVERNANCE_DISCLAIMER}
"""
        )


def run_validation() -> pd.DataFrame:
    load_merged_dataset, build_feature_matrix = _load_project_modules()
    df = load_merged_dataset(verbose=False)
    X, y, _ = build_feature_matrix(df, feature_set="baseline", verbose=False)
    base_features = [feature for feature in EXISTING_TEXT_FEATURES if feature in X.columns]
    missing_base = sorted(set(EXISTING_TEXT_FEATURES) - set(base_features))
    if missing_base:
        print(f"Warning: missing existing text features skipped: {missing_base}")

    enhanced = build_candidate_features(df)
    validation_df = pd.concat(
        [
            X[base_features].reset_index(drop=True),
            enhanced.reset_index(drop=True),
            df[["Sector"]].reset_index(drop=True) if "Sector" in df.columns else pd.DataFrame({"Sector": "Unknown"}, index=df.index).reset_index(drop=True),
        ],
        axis=1,
    )
    y = pd.Series(y).astype(int).reset_index(drop=True)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    baseline_scores = _oof_scores(validation_df, y, base_features, candidate=None)

    rows = []
    for candidate in CANDIDATES:
        full_corr = _spearman(validation_df[candidate.name], y)
        stable, fold_signs = _fold_sign_stability(validation_df, y, candidate.name, cv)
        scores = _oof_scores(validation_df, y, base_features, candidate=candidate.name)
        delta_pr = scores["pr_auc"] - baseline_scores["pr_auc"]
        keep = stable and delta_pr >= -1e-12
        rows.append(
            {
                "feature": candidate.name,
                "role": candidate.role,
                "spearman_corr": round(full_corr, 4),
                "corr_sign": _sign(full_corr),
                "fold_signs": fold_signs,
                "sign_stable_all_5_folds": stable,
                "baseline_roc_auc": round(baseline_scores["roc_auc"], 4),
                "candidate_roc_auc": round(scores["roc_auc"], 4),
                "delta_roc_auc": round(scores["roc_auc"] - baseline_scores["roc_auc"], 4),
                "baseline_pr_auc": round(baseline_scores["pr_auc"], 4),
                "candidate_pr_auc": round(scores["pr_auc"], 4),
                "delta_pr_auc": round(delta_pr, 4),
                "decision": "KEEP" if keep else "DROP",
                "rationale": (
                    "Sign-stable and did not reduce OOF PR-AUC."
                    if keep
                    else "Dropped because sign was unstable or OOF PR-AUC decreased."
                ),
                "description": candidate.description,
            }
        )

    validation = pd.DataFrame(rows)
    write_outputs(df, enhanced, validation, baseline_scores)
    print("\n=== Enhanced Text Feature Validation ===")
    print(validation.to_string(index=False))
    print(f"\n{GOVERNANCE_DISCLAIMER}")
    return validation


if __name__ == "__main__":
    run_validation()
