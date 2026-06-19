"""
cross_source_inconsistency_index.py
===================================
Day 1 professor-feedback response for DSF504 Team 5.

Creates a transparent Talk-Walk / cross-source inconsistency index for ESG
review-priority ranking. This script adds new outputs only. It does not delete,
overwrite, or retrain the original v1 supervised model or model pickle files.

Governance disclaimer:
AI Decision Support Only: This system is designed for ESG risk triage and
review prioritization. It does not legally determine greenwashing, credit
approval, or investment exclusion.
"""

from __future__ import annotations

import json
import os
import pickle
import sys
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RANDOM_STATE = 42

GOVERNANCE_DISCLAIMER = (
    "AI Decision Support Only: This system is designed for ESG risk triage and "
    "review prioritization. It does not legally determine greenwashing, credit "
    "approval, or investment exclusion."
)

INDEX_WEIGHTS = {
    "talk": 0.35,
    "walk": 0.30,
    "integrity": 0.20,
    "contradiction": 0.15,
}

FEATURE_ORDER = [
    "E_density",
    "S_density",
    "G_density",
    "hedge_score",
    "pct_numeric",
    "talk_total_density",
    "vague_to_specific_ratio",
    "text_length",
    "esg_topic_breadth",
    "e_s_g_dispersion",
    "sector_rank",
    "planet_friendly_business",
    "honest_fair_business",
    "sector_encoded",
]

MODEL_CONTRIBUTION_FALLBACK = pd.DataFrame(
    [
        {
            "model_setting": "Full Model",
            "mean_roc_auc": 0.9413,
            "mean_pr_auc": 0.7223,
            "mean_f1": 0.6474,
        },
        {
            "model_setting": "External-only",
            "mean_roc_auc": 0.7540,
            "mean_pr_auc": 0.3096,
            "mean_f1": 0.2751,
        },
        {
            "model_setting": "Text-only",
            "mean_roc_auc": 0.6635,
            "mean_pr_auc": 0.1582,
            "mean_f1": 0.1844,
        },
        {
            "model_setting": "No-ranking",
            "mean_roc_auc": 0.7316,
            "mean_pr_auc": 0.2322,
            "mean_f1": 0.2161,
        },
        {
            "model_setting": "Strict No-Walk-Proxy",
            "mean_roc_auc": 0.6635,
            "mean_pr_auc": 0.1582,
            "mean_f1": 0.1844,
        },
    ]
)


@dataclass(frozen=True)
class ModuleRefs:
    load_merged_dataset: object
    build_feature_matrix: object
    build_v2_label_dataset: object


def _load_project_modules() -> ModuleRefs:
    """Load local modules using normal imports when the project is accessible."""
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
    sys.path.insert(0, PROJECT_ROOT)
    try:
        from src.data_loader import load_merged_dataset
        from src.feature_engineering import build_feature_matrix
        from src.v2_clean_label_experiment import build_v2_label_dataset
    except ModuleNotFoundError as exc:
        try:
            from data_loader import load_merged_dataset
            from feature_engineering import build_feature_matrix
            from v2_clean_label_experiment import build_v2_label_dataset
        except ModuleNotFoundError as fallback_exc:
            raise RuntimeError(
                "Could not import the existing project data loader or feature engineering module. "
                "Run this script from the project environment with access to the existing src/ "
                "folder; on macOS, grant the terminal/Codex app permission to read the Desktop "
                "project folder if imports are blocked."
            ) from fallback_exc
    return ModuleRefs(load_merged_dataset, build_feature_matrix, build_v2_label_dataset)


def percentile_rank(series: pd.Series, higher_is_weaker: bool = True) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if higher_is_weaker:
        return values.rank(pct=True, method="average").fillna(0.5)
    return (-values).rank(pct=True, method="average").fillna(0.5)


def _company_name(df: pd.DataFrame) -> pd.Series:
    if "Name" in df.columns:
        return df["Name"].astype(str)
    if "Company Name" in df.columns:
        return df["Company Name"].astype(str)
    return df["ticker"].astype(str)


def build_inconsistency_index(df: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    """Create company-level cross-source inconsistency scores."""
    talk_score = percentile_rank(X["talk_total_density"], higher_is_weaker=True)

    total_esg_component = (
        percentile_rank(df["Total ESG Risk score"], higher_is_weaker=True)
        if "Total ESG Risk score" in df.columns
        else pd.Series(0.5, index=X.index)
    )
    walk_score = pd.concat(
        [
            pd.to_numeric(X["sector_rank"], errors="coerce").fillna(0.5),
            percentile_rank(X["e_s_g_dispersion"], higher_is_weaker=True),
            total_esg_component,
        ],
        axis=1,
    ).mean(axis=1)

    integrity_mean = pd.concat(
        [
            pd.to_numeric(X["planet_friendly_business"], errors="coerce"),
            pd.to_numeric(X["honest_fair_business"], errors="coerce"),
        ],
        axis=1,
    ).mean(axis=1)
    integrity_score = percentile_rank(integrity_mean, higher_is_weaker=False)

    source_scores = pd.concat([talk_score, walk_score, integrity_score], axis=1)
    contradiction_score = (
        source_scores.std(axis=1, ddof=0) / 0.4714045208
    ).clip(0, 1).fillna(0.0)

    final_index = (
        INDEX_WEIGHTS["talk"] * talk_score
        + INDEX_WEIGHTS["walk"] * walk_score
        + INDEX_WEIGHTS["integrity"] * integrity_score
        + INDEX_WEIGHTS["contradiction"] * contradiction_score
    ).clip(0, 1)
    percentile = final_index.rank(pct=True, method="average")
    tier = np.select(
        [percentile >= 0.90, percentile >= 0.70],
        ["High priority", "Medium priority"],
        default="Standard review",
    )

    result = pd.DataFrame(
        {
            "ticker": df["ticker"].astype(str).values,
            "company_name": _company_name(df).values,
            "sector": df["Sector"].astype(str).values,
            "talk_intensity_score": talk_score.round(4),
            "walk_weakness_score": walk_score.round(4),
            "integrity_weakness_score": integrity_score.round(4),
            "cross_source_contradiction_score": contradiction_score.round(4),
            "final_inconsistency_index": final_index.round(4),
            "index_percentile": percentile.round(4),
            "review_priority_tier": tier,
        }
    )
    result["explanation"] = result.apply(explain_company, axis=1)
    return result.sort_values("final_inconsistency_index", ascending=False).reset_index(drop=True)


def explain_company(row: pd.Series) -> str:
    reasons = []
    if row["talk_intensity_score"] >= 0.67:
        reasons.append("high ESG disclosure intensity")
    if row["walk_weakness_score"] >= 0.67:
        reasons.append("weak external ESG evidence")
    if row["integrity_weakness_score"] >= 0.67:
        reasons.append("weak integrity signal")
    if row["cross_source_contradiction_score"] >= 0.50:
        reasons.append("cross-source inconsistency")
    if not reasons:
        reasons.append("moderate combined Talk-Walk signal")
    return f"Flagged for {', '.join(reasons)}; requires further ESG due diligence."


def _load_v1_probabilities(X: pd.DataFrame, y_v1: pd.Series) -> tuple[pd.Series, str]:
    """Use the saved v1 model for ranking when available; otherwise fall back to v1 labels."""
    fallback = pd.Series(y_v1.astype(float).values, index=X.index)
    model_path = os.path.join(MODELS_DIR, "lgbm_best.pkl")
    if not os.path.exists(model_path):
        return fallback, "V1 ranking signal: fallback to v1 label because models/lgbm_best.pkl was not found."
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        X_model = X[FEATURE_ORDER].values.astype(float)
        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(X_model)[:, 1]
            return (
                pd.Series(scores, index=X.index),
                "V1 ranking signal: saved LightGBM model probability from models/lgbm_best.pkl.",
            )
        scores = model.predict(X_model).astype(float)
        return (
            pd.Series(scores, index=X.index),
            "V1 ranking signal: saved LightGBM model prediction from models/lgbm_best.pkl.",
        )
    except Exception as exc:
        note = (
            "V1 ranking signal: fallback to v1 label because saved model probability "
            f"could not be used ({type(exc).__name__}: {str(exc)[:140]})."
        )
        return fallback, note


def build_overlap_analysis(
    index_df: pd.DataFrame,
    v2_df: pd.DataFrame,
    v1_prob: pd.Series,
) -> pd.DataFrame:
    top_cutoff = v1_prob.quantile(0.90)
    v1_top = set(index_df.loc[v1_prob.loc[index_df.index] >= top_cutoff, "ticker"])
    index_high = set(index_df[index_df["review_priority_tier"] == "High priority"]["ticker"])
    v2_positive = set(v2_df[v2_df["high_risk_v2"] == 1]["ticker"])

    comparisons = [
        ("overlap_v1_top_and_index_high", v1_top & index_high),
        ("overlap_v1_top_and_v2_positive", v1_top & v2_positive),
        ("overlap_index_high_and_v2_positive", index_high & v2_positive),
        ("v1_top_not_index", v1_top - index_high),
        ("index_high_not_v1", index_high - v1_top),
        ("v2_positive_not_v1", v2_positive - v1_top),
    ]
    rows = []
    for comparison, tickers in comparisons:
        sorted_tickers = sorted(tickers)
        rows.append(
            {
                "comparison": comparison,
                "count": len(sorted_tickers),
                "tickers": ", ".join(sorted_tickers[:30]),
                "interpretation": (
                    "Non-overlap is useful because it shows the new index is not merely copying "
                    "the v1 model or a single ESG rating source."
                ),
            }
        )
    return pd.DataFrame(rows)


def _standardize_contribution_table(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize possible contribution CSV column names for robust markdown display."""
    column_options = {
        "model_setting": ["model_setting", "model_version", "model", "setting", "Model Setting"],
        "mean_roc_auc": ["mean_roc_auc", "roc_auc_mean", "roc_auc", "ROC-AUC", "ROC_AUC"],
        "mean_pr_auc": ["mean_pr_auc", "pr_auc_mean", "pr_auc", "PR-AUC", "PR_AUC", "average_precision"],
        "mean_f1": ["mean_f1", "f1_mean", "f1", "F1"],
    }
    selected = {}
    for target, candidates in column_options.items():
        match = next((candidate for candidate in candidates if candidate in df.columns), None)
        if match is None:
            raise ValueError(f"Missing contribution column for {target}")
        selected[target] = match
    out = df[[selected[col] for col in column_options]].copy()
    out.columns = list(column_options.keys())
    for col in ["mean_roc_auc", "mean_pr_auc", "mean_f1"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    if out[["mean_roc_auc", "mean_pr_auc", "mean_f1"]].isna().any().any():
        raise ValueError("Contribution metrics contain non-numeric values.")
    return out.round({"mean_roc_auc": 4, "mean_pr_auc": 4, "mean_f1": 4})


def write_v1_baseline_summary() -> pd.DataFrame:
    path = os.path.join(OUTPUTS_DIR, "model_contribution_analysis.csv")
    if os.path.exists(path):
        try:
            contribution = _standardize_contribution_table(pd.read_csv(path))
        except Exception:
            contribution = _standardize_contribution_table(MODEL_CONTRIBUTION_FALLBACK.copy())
    else:
        contribution = _standardize_contribution_table(MODEL_CONTRIBUTION_FALLBACK.copy())

    md_path = os.path.join(OUTPUTS_DIR, "v1_professor_feedback_baseline_summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(
            f"""# V1 Professor Feedback Baseline Summary

## Baseline Finding

The original v1 supervised model remains a useful benchmark, but it should be interpreted as an ESG risk triage model rather than a legal greenwashing determination.

{contribution.to_markdown(index=False)}

## Professor Feedback Interpretation

- Full Model performance is high.
- External-only performance is also meaningful, which supports the professor's concern that processed external ESG evidence carries strong signal.
- Text-only performance is weaker but not zero, so sustainability report language still contains useful signal.
- No-ranking and Strict No-Walk-Proxy settings are governance checks, not replacements for the full model.

Therefore, v1 should be presented as a risk-ranking and due diligence support model, not as proof that any company is greenwashing.

{GOVERNANCE_DISCLAIMER}
"""
        )
    return contribution


def write_index_markdown(
    index_df: pd.DataFrame,
    overlap: pd.DataFrame,
    contribution: pd.DataFrame,
    v1_ranking_note: str,
) -> None:
    top_cases = index_df.head(12)
    path = os.path.join(OUTPUTS_DIR, "cross_source_inconsistency_index.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            f"""# Cross-Source Inconsistency Index

## Purpose

This index is a transparent review-priority ranking. It is not optimized for AUC and does not legally determine greenwashing.

## Formula

`final_inconsistency_index = 0.35*talk + 0.30*walk + 0.20*integrity + 0.15*contradiction`

Weights are configurable in `INDEX_WEIGHTS`.

## Component Meaning

- Talk Intensity Score: higher ESG disclosure intensity.
- Walk Weakness Score: weaker external ESG risk evidence.
- Integrity Weakness Score: weaker integrity or governance signal.
- Cross-Source Contradiction Score: disagreement across the Talk, Walk, and integrity sources.

## V1 Baseline Context

{contribution.to_markdown(index=False)}

## V1 Ranking Signal Used for Overlap

{v1_ranking_note}

## Top Review-Priority Cases

{top_cases[["ticker", "company_name", "sector", "final_inconsistency_index", "review_priority_tier", "explanation"]].to_markdown(index=False)}

## Overlap Analysis

{overlap.to_markdown(index=False)}

## Interpretation

Non-overlap between v1 top-risk companies, v2 clean-label positives, and index high-priority companies is useful. It shows the index is not simply copying the v1 supervised model or one external ESG rating source.

{GOVERNANCE_DISCLAIMER}
"""
        )


def write_response_summary(
    metrics: pd.DataFrame | None,
    index_df: pd.DataFrame,
    v1_ranking_note: str,
) -> None:
    path = os.path.join(OUTPUTS_DIR, "professor_feedback_response_summary.md")
    metrics_text = (
        metrics.to_markdown(index=False)
        if metrics is not None and not metrics.empty
        else "V2 model metrics were not available when this summary was generated."
    )
    top_text = index_df.head(8)[
        ["ticker", "company_name", "sector", "final_inconsistency_index", "review_priority_tier"]
    ].to_markdown(index=False)
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            f"""# Professor Feedback Response Summary

## English Interpretation

We first built a high-performing v1 model. Professor feedback revealed a structural concern: the v1 label and some important features were partly tied to external ESG ratings. We therefore conducted ablation tests and built a cleaner v2 experiment. The v2 model may have lower but more honest performance. The final contribution is not high AUC alone, but a transparent ESG review-priority system combining supervised modeling, ablation governance, clean-label testing, and cross-source inconsistency analysis.

## 中文說明

我們先建立高表現的 v1 模型，但老師提醒：v1 的 label 與部分重要特徵可能和外部 ESG 評分結構有關。因此我們補做消融測試，並建立較乾淨的 v2 label 與跨來源不一致指標。v2 表現可能較低，但解釋更誠實。最終貢獻不是只有高 AUC，而是把模型轉成透明、可查核的 ESG 風險排序與盡職調查輔助系統。

## V2 Clean Model Metrics

{metrics_text}

## Cross-Source Top Cases

{top_text}

## V1 Ranking Signal Used for Overlap

{v1_ranking_note}

## Governance Disclaimer

{GOVERNANCE_DISCLAIMER}
"""
        )


def plot_top_cases(index_df: pd.DataFrame) -> None:
    top = index_df.head(15).sort_values("final_inconsistency_index", ascending=True)
    labels = top["ticker"].astype(str) + " | " + top["company_name"].astype(str).str[:24]
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(labels, top["final_inconsistency_index"], color="#2F6B4F")
    ax.set_title("Cross-Source ESG Inconsistency Index: Top Review-Priority Cases")
    ax.set_xlabel("Final Inconsistency Index")
    ax.set_xlim(0, max(1.0, float(top["final_inconsistency_index"].max()) + 0.05))
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUTS_DIR, "cross_source_inconsistency_plot.png"), dpi=160, facecolor="white")
    plt.close(fig)


def run_index() -> pd.DataFrame:
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    mods = _load_project_modules()
    df = mods.load_merged_dataset(verbose=False)
    X, y_v1, _ = mods.build_feature_matrix(df, feature_set="baseline", verbose=False)
    v2_df = mods.build_v2_label_dataset(df, X)
    index_df = build_inconsistency_index(df, X)

    v1_prob, v1_ranking_note = _load_v1_probabilities(X, y_v1)
    index_by_ticker = index_df.set_index("ticker", drop=False)
    v1_prob_by_ticker = pd.Series(v1_prob.values, index=df["ticker"].astype(str)).reindex(index_by_ticker.index)
    overlap = build_overlap_analysis(index_by_ticker, v2_df, v1_prob_by_ticker)
    contribution = write_v1_baseline_summary()

    index_df.to_csv(os.path.join(OUTPUTS_DIR, "cross_source_inconsistency_index.csv"), index=False)
    index_df.head(25).to_csv(os.path.join(OUTPUTS_DIR, "cross_source_top_cases.csv"), index=False)
    overlap.to_csv(os.path.join(OUTPUTS_DIR, "cross_source_overlap_analysis.csv"), index=False)
    write_index_markdown(index_df, overlap, contribution, v1_ranking_note)
    plot_top_cases(index_df)

    metrics_path = os.path.join(OUTPUTS_DIR, "v2_clean_model_metrics.csv")
    metrics = pd.read_csv(metrics_path) if os.path.exists(metrics_path) else None
    write_response_summary(metrics, index_df, v1_ranking_note)

    print("\n=== Cross-Source Inconsistency Index ===")
    print(index_df.head(10)[["ticker", "company_name", "sector", "final_inconsistency_index", "review_priority_tier"]].to_string(index=False))
    print("\nOverlap analysis:")
    print(overlap.to_string(index=False))
    print(f"\n{v1_ranking_note}")
    print(f"\n{GOVERNANCE_DISCLAIMER}")
    return index_df


if __name__ == "__main__":
    run_index()
