"""
model_contribution_analysis.py
================================
Professor-feedback ablation analysis for the DSF504 Team 5 ESG dashboard.

This module does NOT retrain or overwrite the production model pickle files.
It evaluates the same LightGBM scorer configuration across several feature
subsets to separate external ESG ranking signal from sustainability-report
text signal.

Governance framing:
AI Decision Support Only: This model is designed for ESG risk triage and
review prioritization. It does not legally determine greenwashing, credit
approval, or investment exclusion.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
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


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
BEST_PARAMS_PATH = os.path.join(OUTPUTS_DIR, "best_params.json")
RANDOM_STATE = 42
N_SPLITS = 5
SMOTE_K = 3

GOVERNANCE_DISCLAIMER = (
    "AI Decision Support Only: This model is designed for ESG risk triage "
    "and review prioritization. It does not legally determine greenwashing, "
    "credit approval, or investment exclusion."
)

TALK_FEATURES = [
    "E_density",
    "S_density",
    "G_density",
    "talk_total_density",
    "hedge_score",
    "pct_numeric",
    "vague_to_specific_ratio",
    "esg_topic_breadth",
    "text_length",
]

EXTERNAL_FEATURES = [
    "sector_rank",
    "planet_friendly_business",
    "honest_fair_business",
    "e_s_g_dispersion",
    "sector_encoded",
]

QUESTIONABLE_RANKING_FEATURES = [
    "sector_rank",
    "planet_friendly_business",
    "honest_fair_business",
]

STRICT_WALK_PROXY_FEATURES = [
    "sector_rank",
    "e_s_g_dispersion",
    "planet_friendly_business",
    "honest_fair_business",
    "sector_encoded",
]

LABEL_ONLY_FIELDS = {
    "Total ESG Risk score",
    "Controversy Score",
    "ESG Risk Percentile",
    "ESG Risk Level",
}


@dataclass(frozen=True)
class FeatureVersion:
    model_version: str
    feature_group: str
    features: list[str]
    purpose: str


def _load_lgbm_params() -> dict:
    """Load final LightGBM best params if available; otherwise use stable defaults."""
    defaults = {
        "n_estimators": 250,
        "max_depth": 14,
        "learning_rate": 0.11536840196915875,
        "num_leaves": 54,
        "min_child_samples": 14,
        "subsample": 0.9687291700377159,
    }
    if not os.path.exists(BEST_PARAMS_PATH):
        return defaults
    try:
        with open(BEST_PARAMS_PATH, "r") as f:
            payload = json.load(f)
        params = payload.get("lgbm") or {}
        return {**defaults, **params}
    except Exception:
        return defaults


def _build_pipeline(params: dict) -> ImbPipeline:
    """SMOTE stays inside the training fold through imblearn Pipeline."""
    clf = LGBMClassifier(
        random_state=RANDOM_STATE,
        class_weight="balanced",
        verbose=-1,
        **params,
    )
    return ImbPipeline(
        steps=[
            ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=SMOTE_K)),
            ("clf", clf),
        ]
    )


def _safe_features(features: Iterable[str], available: set[str]) -> list[str]:
    return [feature for feature in features if feature in available]


def build_feature_versions(X: pd.DataFrame) -> list[FeatureVersion]:
    """Build the five professor-feedback feature versions from actual columns."""
    available = set(X.columns)
    full_features = list(X.columns)
    external_features = _safe_features(EXTERNAL_FEATURES, available)
    text_features = _safe_features(TALK_FEATURES, available)
    no_ranking_features = [
        feature for feature in full_features if feature not in QUESTIONABLE_RANKING_FEATURES
    ]
    strict_no_walk_features = [
        feature for feature in full_features if feature not in STRICT_WALK_PROXY_FEATURES
    ]

    versions = [
        FeatureVersion(
            "Full Model",
            "Talk + Walk / External + Context",
            full_features,
            "Current best ESG risk triage model using all original 14 features.",
        ),
        FeatureVersion(
            "External-only Model",
            "Walk / External ranking features",
            external_features,
            "Tests how much performance comes from external ESG ranking signals alone.",
        ),
        FeatureVersion(
            "Text-only Model",
            "Sustainability report Talk features",
            text_features,
            "Tests whether sustainability report language contains independent signal.",
        ),
        FeatureVersion(
            "No-ranking Model",
            "No sector_rank / integrity ranking outputs",
            no_ranking_features,
            "Removes the most questionable external ranking features while retaining text, E/S/G dispersion, and sector context.",
        ),
        FeatureVersion(
            "Strict No-Walk-Proxy Model",
            "Talk-only strict ablation",
            strict_no_walk_features,
            "Removes all Walk-proxy and context features. In this project it is identical to the Text-only Model if the feature list matches.",
        ),
    ]

    missing_versions = [version.model_version for version in versions if not version.features]
    if missing_versions:
        raise ValueError(f"No usable features found for: {missing_versions}")

    return versions


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


def evaluate_feature_version(
    X: pd.DataFrame,
    y: pd.Series,
    version: FeatureVersion,
    params: dict,
) -> tuple[dict, list[dict]]:
    """Evaluate one feature version with identical stratified folds."""
    X_arr = X[version.features].values.astype(float)
    y_arr = y.values.astype(int)
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    fold_rows = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_arr, y_arr), start=1):
        pipeline = _build_pipeline(params)
        pipeline.fit(X_arr[train_idx], y_arr[train_idx])
        y_pred = pipeline.predict(X_arr[test_idx])
        y_proba = pipeline.predict_proba(X_arr[test_idx])[:, 1]
        row = _fold_metrics(y_arr[test_idx], y_pred, y_proba)
        row["fold"] = fold_idx
        fold_rows.append(row)

    metrics = pd.DataFrame(fold_rows)
    cm_total = metrics[["tn", "fp", "fn", "tp"]].sum().astype(int).to_dict()
    result = {
        "model_version": version.model_version,
        "feature_group": version.feature_group,
        "n_features": len(version.features),
        "features_used": ", ".join(version.features),
        "purpose": version.purpose,
        "roc_auc_mean": metrics["roc_auc"].mean(),
        "roc_auc_std": metrics["roc_auc"].std(ddof=0),
        "pr_auc_mean": metrics["pr_auc"].mean(),
        "pr_auc_std": metrics["pr_auc"].std(ddof=0),
        "f1_mean": metrics["f1"].mean(),
        "f1_std": metrics["f1"].std(ddof=0),
        "precision_mean": metrics["precision"].mean(),
        "precision_std": metrics["precision"].std(ddof=0),
        "recall_mean": metrics["recall"].mean(),
        "recall_std": metrics["recall"].std(ddof=0),
        "accuracy_mean": metrics["accuracy"].mean(),
        "accuracy_std": metrics["accuracy"].std(ddof=0),
        "confusion_matrix": json.dumps(
            [[cm_total["tn"], cm_total["fp"]], [cm_total["fn"], cm_total["tp"]]]
        ),
        "positive_class_support": int(metrics["positive_support"].sum()),
        "negative_class_support": int(metrics["negative_support"].sum()),
    }
    return result, fold_rows


def _add_interpretations(df: pd.DataFrame) -> pd.DataFrame:
    """Add professor-feedback interpretations to each row."""
    out = df.copy()
    lookup = out.set_index("model_version")
    full_auc = float(lookup.loc["Full Model", "roc_auc_mean"])
    external_auc = float(lookup.loc["External-only Model", "roc_auc_mean"])
    text_auc = float(lookup.loc["Text-only Model", "roc_auc_mean"])
    no_ranking_auc = float(lookup.loc["No-ranking Model", "roc_auc_mean"])
    full_margin = max(full_auc - 0.5, 1e-9)
    external_share = max(0.0, (external_auc - 0.5) / full_margin)
    text_share = max(0.0, (text_auc - 0.5) / full_margin)
    no_ranking_drop = full_auc - no_ranking_auc

    interpretations = {}
    for row in out.to_dict("records"):
        version = row["model_version"]
        if version == "Full Model":
            interpretations[version] = (
                "Reference risk triage model. Strong performance should be interpreted cautiously because it combines "
                "text features with external ESG ranking signals."
            )
        elif version == "External-only Model":
            if external_share >= 0.75:
                msg = "External ESG ranking features explain a substantial share of the Full Model performance."
            elif external_share >= 0.50:
                msg = "External ESG ranking features explain a meaningful share of performance."
            else:
                msg = "External ESG ranking features alone are weaker than the Full Model."
            interpretations[version] = (
                f"{msg} This supports repositioning the system as ESG risk triage and rating inconsistency consolidation."
            )
        elif version == "Text-only Model":
            if text_auc >= 0.70:
                msg = "Text-derived sustainability report features retain meaningful independent signal."
            elif text_auc >= 0.60:
                msg = "Text-derived sustainability report features retain modest independent signal."
            else:
                msg = "Text-derived features are weak on their own in this dataset."
            interpretations[version] = (
                f"{msg} The text model captures about {text_share:.0%} of the Full Model ROC-AUC lift over random ranking."
            )
        elif version == "No-ranking Model":
            interpretations[version] = (
                f"Removing sector_rank, planet_friendly_business, and honest_fair_business changes ROC-AUC by "
                f"{-no_ranking_drop:+.3f} versus the Full Model. This shows how much performance remains after removing "
                "the most rating-like features."
            )
        else:
            strict_features = set(str(row["features_used"]).split(", "))
            text_features = set(str(lookup.loc["Text-only Model", "features_used"]).split(", "))
            same_as_text = strict_features == text_features
            if same_as_text:
                interpretations[version] = (
                    "This feature set is identical to the Text-only Model in the current 14-feature design, because all "
                    "Walk-proxy and context features are removed."
                )
            else:
                interpretations[version] = (
                    "Strict removal of Walk-proxy and context features isolates sustainability report text signal."
                )

    out["interpretation"] = out["model_version"].map(interpretations)
    numeric_cols = [
        "roc_auc_mean",
        "roc_auc_std",
        "pr_auc_mean",
        "pr_auc_std",
        "f1_mean",
        "f1_std",
        "precision_mean",
        "precision_std",
        "recall_mean",
        "recall_std",
        "accuracy_mean",
        "accuracy_std",
    ]
    out[numeric_cols] = out[numeric_cols].round(4)
    return out


def _summary_text(df: pd.DataFrame) -> dict:
    lookup = df.set_index("model_version")
    full_auc = float(lookup.loc["Full Model", "roc_auc_mean"])
    external_auc = float(lookup.loc["External-only Model", "roc_auc_mean"])
    text_auc = float(lookup.loc["Text-only Model", "roc_auc_mean"])
    no_ranking_auc = float(lookup.loc["No-ranking Model", "roc_auc_mean"])
    full_margin = max(full_auc - 0.5, 1e-9)
    external_share = max(0.0, (external_auc - 0.5) / full_margin)
    text_share = max(0.0, (text_auc - 0.5) / full_margin)
    no_ranking_drop = full_auc - no_ranking_auc

    if external_share >= 0.75:
        external_sentence = (
            "External ESG ranking features account for a substantial share of the Full Model's ranking performance."
        )
    elif external_share >= 0.50:
        external_sentence = (
            "External ESG ranking features account for a meaningful, but not complete, share of performance."
        )
    else:
        external_sentence = (
            "External ESG ranking features alone do not fully explain the Full Model's performance."
        )

    if text_auc >= 0.70:
        text_sentence = "The Text-only Model still shows meaningful independent signal from sustainability report language."
    elif text_auc >= 0.60:
        text_sentence = "The Text-only Model shows modest independent signal from sustainability report language."
    else:
        text_sentence = "The Text-only Model is weak, so the project should avoid claiming pure text-based greenwashing detection."

    if text_auc < 0.70 or external_share >= 0.50:
        positioning = (
            "The most defensible positioning is ESG risk triage and rating inconsistency consolidation, not legal greenwashing detection."
        )
    else:
        positioning = (
            "The system can discuss text-based risk signal, but should still be framed as due diligence support rather than legal determination."
        )

    return {
        "external_contribution": external_sentence,
        "text_signal": text_sentence,
        "no_ranking_drop": (
            f"The No-ranking Model ROC-AUC is {no_ranking_auc:.3f}, a drop of {no_ranking_drop:.3f} from the Full Model."
        ),
        "positioning": positioning,
        "external_share_of_auc_lift": round(external_share, 4),
        "text_share_of_auc_lift": round(text_share, 4),
    }


def _write_markdown(df: pd.DataFrame, summary: dict) -> str:
    md_path = os.path.join(OUTPUTS_DIR, "model_contribution_analysis.md")
    table_md = df[
        [
            "model_version",
            "feature_group",
            "n_features",
            "roc_auc_mean",
            "pr_auc_mean",
            "f1_mean",
            "precision_mean",
            "recall_mean",
            "accuracy_mean",
            "interpretation",
        ]
    ].to_markdown(index=False)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(
            f"""# Model Contribution Analysis and Response to Professor Feedback

## Purpose

This analysis responds to professor feedback that the current Full Model may rely strongly on already-processed external ESG ranking features, such as `sector_rank`, `planet_friendly_business`, and `honest_fair_business`.

The goal is not to replace the production model. The goal is to separate the contribution of external ESG ranking signals from sustainability report text-derived signals.

## Governance Disclaimer

{GOVERNANCE_DISCLAIMER}

## Model Versions

- **Full Model:** all original 14 features.
- **External-only Model:** Walk / external ranking features only.
- **Text-only Model:** sustainability report Talk features only.
- **No-ranking Model:** removes `sector_rank`, `planet_friendly_business`, and `honest_fair_business`.
- **Strict No-Walk-Proxy Model:** removes all Walk-proxy and context features. In this project, it is identical to the Text-only Model if the feature list matches.

## Results Table

{table_md}

## Interpretation

- {summary["external_contribution"]}
- {summary["text_signal"]}
- {summary["no_ranking_drop"]}
- {summary["positioning"]}

## Final Wording for Report / Slides

The Full Model may partially reflect the structure of external ESG rating systems. Therefore, we reposition the dashboard as an ESG risk triage and rating inconsistency consolidation tool. The system helps analysts prioritize companies for further ESG due diligence, but it does not legally determine greenwashing, credit approval, or investment exclusion.
"""
        )
    return md_path


def _write_json(df: pd.DataFrame, fold_payload: dict, summary: dict) -> str:
    json_path = os.path.join(OUTPUTS_DIR, "model_contribution_analysis.json")
    payload = {
        "governance_disclaimer": GOVERNANCE_DISCLAIMER,
        "validation_design": f"Stratified {N_SPLITS}-fold CV; SMOTE inside imblearn Pipeline training folds only; fixed random_state={RANDOM_STATE}.",
        "model_family": "LightGBM final scorer configuration",
        "summary": summary,
        "results": df.to_dict("records"),
        "fold_metrics": fold_payload,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return json_path


def _write_chart(df: pd.DataFrame) -> str:
    chart_path = os.path.join(OUTPUTS_DIR, "ablation_metrics_summary.png")
    plot_df = df.copy()
    x = np.arange(len(plot_df))
    width = 0.24

    fig, ax = plt.subplots(figsize=(12, 6), facecolor="white")
    ax.bar(x - width, plot_df["roc_auc_mean"], width, label="ROC-AUC", color="#1F5C3A")
    ax.bar(x, plot_df["pr_auc_mean"], width, label="PR-AUC", color="#D99A2B")
    ax.bar(x + width, plot_df["f1_mean"], width, label="F1", color="#5F7165")
    ax.set_title("Model Contribution / Ablation Analysis", fontsize=15, weight="bold")
    ax.set_ylabel("Mean CV score")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["model_version"], rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(frameon=False)
    ax.text(
        0.01,
        -0.26,
        "Purpose: separate external ESG ranking contribution from sustainability-report text signal.\n"
        "AI Decision Support Only: ESG risk triage and review prioritization, not legal determination.",
        transform=ax.transAxes,
        fontsize=10,
        color="#4D5D52",
    )
    fig.tight_layout()
    fig.savefig(chart_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return chart_path


def run_analysis() -> pd.DataFrame:
    """Run the contribution analysis and write CSV, Markdown, JSON, and chart outputs."""
    sys.path.insert(0, PROJECT_ROOT)
    from src.data_loader import load_merged_dataset
    from src.feature_engineering import build_feature_matrix

    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    df = load_merged_dataset(verbose=False)
    X, y, _ = build_feature_matrix(df, feature_set="baseline", verbose=False)

    leaked = LABEL_ONLY_FIELDS.intersection(X.columns)
    if leaked:
        raise ValueError(f"Label-only fields found in X: {sorted(leaked)}")

    params = _load_lgbm_params()
    versions = build_feature_versions(X)
    results: list[dict] = []
    fold_payload: dict[str, list[dict]] = {}
    cached_by_features: dict[tuple[str, ...], tuple[dict, list[dict]]] = {}

    print("[model_contribution] Running LightGBM ablation analysis")
    for version in versions:
        feature_key = tuple(version.features)
        if feature_key in cached_by_features:
            result, folds = cached_by_features[feature_key]
            result = {**result, "model_version": version.model_version, "feature_group": version.feature_group, "purpose": version.purpose}
            print(f"  {version.model_version}: reused identical feature set ({len(version.features)} features)")
        else:
            result, folds = evaluate_feature_version(X, y, version, params)
            cached_by_features[feature_key] = (result, folds)
            print(
                f"  {version.model_version}: ROC-AUC={result['roc_auc_mean']:.4f}, "
                f"PR-AUC={result['pr_auc_mean']:.4f}, F1={result['f1_mean']:.4f}"
            )
        results.append(result)
        fold_payload[version.model_version] = folds

    out = _add_interpretations(pd.DataFrame(results))
    ordered_cols = [
        "model_version",
        "feature_group",
        "n_features",
        "features_used",
        "roc_auc_mean",
        "roc_auc_std",
        "pr_auc_mean",
        "pr_auc_std",
        "f1_mean",
        "f1_std",
        "precision_mean",
        "precision_std",
        "recall_mean",
        "recall_std",
        "accuracy_mean",
        "accuracy_std",
        "confusion_matrix",
        "positive_class_support",
        "negative_class_support",
        "interpretation",
    ]
    out = out[ordered_cols]
    out["governance_disclaimer"] = GOVERNANCE_DISCLAIMER
    csv_path = os.path.join(OUTPUTS_DIR, "model_contribution_analysis.csv")
    out.to_csv(csv_path, index=False)

    summary = _summary_text(out)
    md_path = _write_markdown(out, summary)
    json_path = _write_json(out, fold_payload, summary)
    chart_path = _write_chart(out)

    print("\n=== Model Contribution Analysis Summary ===")
    print(out[["model_version", "n_features", "roc_auc_mean", "pr_auc_mean", "f1_mean"]].to_string(index=False))
    print(f"\nSaved CSV:  {csv_path}")
    print(f"Saved MD:   {md_path}")
    print(f"Saved JSON: {json_path}")
    print(f"Saved PNG:  {chart_path}")
    print(f"\n{GOVERNANCE_DISCLAIMER}")
    return out


if __name__ == "__main__":
    run_analysis()
