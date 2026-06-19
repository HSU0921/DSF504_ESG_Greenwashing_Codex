"""
explainability.py
=================
SHAP-based explainability for the greenwashing detection model.

Outputs
-------
outputs/shap_beeswarm.png        – global feature importance (beeswarm)
outputs/shap_waterfall_1/2/3.png – company-level waterfall plots for the
                                   3 highest-risk confirmed greenwashing firms
outputs/feature_importance.csv   – mean |SHAP| ranked feature table

Design
------
Uses shap.TreeExplainer on the extracted LightGBM classifier.
The imblearn Pipeline's SMOTE step is NOT called during inference
(SMOTE is a sampler, not a transformer; it does not participate in predict_proba).
SHAP values are computed on the original feature space (pre-SMOTE), which is
the correct input representation for model explanation.

Binary LightGBM note:
  explainer.shap_values(X) → ndarray (n_samples, n_features) for the positive class
  explainer.expected_value  → ndarray([base_log_odds])   → take [0]
"""

import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless backend — no display required
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shap

warnings.filterwarnings("ignore")

# ── project paths ─────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
MODELS_DIR  = os.path.join(PROJECT_ROOT, "models")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# ── colour palette (green/red ESG theme) ──────────────────────────────────────
CLR_POS  = "#d73027"   # red  → pushes toward greenwashing
CLR_NEG  = "#1a9850"   # green → reduces greenwashing risk
CLR_BASE = "#4393c3"   # blue  → base value bar


# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD PIPELINE + DATA
# ─────────────────────────────────────────────────────────────────────────────

def load_artifacts():
    """Load the best LightGBM pipeline, rebuild feature matrix, return all."""
    from src.data_loader import load_merged_dataset
    from src.feature_engineering import build_feature_matrix

    print("[explainability] Loading data …")
    df = load_merged_dataset(verbose=False)
    X, y, meta = build_feature_matrix(df, verbose=False)

    print("[explainability] Loading model pipeline …")
    with open(os.path.join(MODELS_DIR, "lgbm_best.pkl"), "rb") as f:
        pipeline = pickle.load(f)

    return pipeline, X, y, meta, df


# ─────────────────────────────────────────────────────────────────────────────
# 2. COMPUTE SHAP VALUES
# ─────────────────────────────────────────────────────────────────────────────

def compute_shap(pipeline, X: pd.DataFrame):
    """
    Extract LightGBM from pipeline and compute SHAP values.

    Returns
    -------
    shap_explanation : shap.Explanation   (n_samples × n_features, class-1 values)
    explainer        : shap.TreeExplainer
    """
    print("[explainability] Computing SHAP values …")
    clf       = pipeline.named_steps["clf"]
    X_arr     = X.values.astype(float)
    explainer = shap.TreeExplainer(clf)

    # shap_values: (n_samples, n_features) for positive class (binary LightGBM)
    shap_vals  = explainer.shap_values(X_arr)           # ndarray
    # expected_value may be a scalar or a 1-element array depending on SHAP version
    ev = explainer.expected_value
    base_value = float(ev[0]) if hasattr(ev, "__len__") else float(ev)

    explanation = shap.Explanation(
        values        = shap_vals,
        base_values   = np.full(len(X_arr), base_value),
        data          = X_arr,
        feature_names = list(X.columns),
    )
    print(f"[explainability] SHAP values shape: {shap_vals.shape}")
    return explanation, explainer


# ─────────────────────────────────────────────────────────────────────────────
# 3. GLOBAL BEESWARM PLOT
# ─────────────────────────────────────────────────────────────────────────────

def plot_beeswarm(explanation: shap.Explanation, save_path: str):
    """
    Global beeswarm: each dot = one company, colour = feature value (low→blue, high→red).
    Features ranked by mean |SHAP|.
    """
    print("[explainability] Generating beeswarm plot …")
    fig, ax = plt.subplots(figsize=(10, 7))
    plt.sca(ax)

    shap.plots.beeswarm(
        explanation,
        max_display=14,
        show=False,
        color_bar=True,
    )

    ax.set_title(
        "Global Feature Impact on Greenwashing Risk\n"
        "(SHAP values — LightGBM, all 257 companies)",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.set_xlabel("SHAP value  (positive = higher greenwashing risk)", fontsize=10)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[explainability] Saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. SELECT TOP-3 HIGH-RISK GREENWASHING COMPANIES
# ─────────────────────────────────────────────────────────────────────────────

def select_top3(pipeline, X: pd.DataFrame, y: pd.Series, meta: dict):
    """
    Among confirmed greenwashing companies (label=1), pick the 3 with the
    highest predicted probability from the LightGBM pipeline.

    Returns
    -------
    list of (row_index, ticker, company_name, proba) tuples
    """
    X_arr   = X.values.astype(float)
    y_proba = pipeline.predict_proba(X_arr)[:, 1]

    pos_idx = np.where(y.values == 1)[0]
    ranked  = pos_idx[np.argsort(y_proba[pos_idx])[::-1]][:3]

    tickers   = meta["ticker"]
    sectors   = meta["sector"]
    companies = []
    for rank, idx in enumerate(ranked, start=1):
        companies.append({
            "rank":   rank,
            "row":    idx,
            "ticker": tickers[idx],
            "sector": sectors[idx],
            "proba":  y_proba[idx],
        })
        print(f"  #{rank}  {tickers[idx]:8s}  [{sectors[idx]}]  "
              f"greenwashing prob = {y_proba[idx]:.4f}")

    return companies


# ─────────────────────────────────────────────────────────────────────────────
# 5. WATERFALL PLOT (company-level)
# ─────────────────────────────────────────────────────────────────────────────

def plot_waterfall(explanation: shap.Explanation, company: dict, save_path: str):
    """
    Waterfall plot for a single company — presented as "detective evidence":
    each bar = one red flag or mitigating factor for that company's score.
    """
    ticker = company["ticker"]
    sector = company["sector"]
    proba  = company["proba"]
    idx    = company["row"]

    print(f"[explainability] Generating waterfall for {ticker} …")
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.sca(ax)

    shap.plots.waterfall(
        explanation[idx],
        max_display=14,
        show=False,
    )

    ax.set_title(
        f"🔍 Greenwashing Evidence — {ticker}  [{sector}]\n"
        f"Predicted risk probability: {proba:.1%}",
        fontsize=12, fontweight="bold", pad=12,
    )

    # Detective-style annotation
    fig.text(
        0.01, 0.01,
        "↑ Red bars = factors that INCREASE greenwashing risk  │  "
        "↓ Green bars = factors that REDUCE risk",
        fontsize=8, color="gray", style="italic",
    )

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[explainability] Saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. FEATURE IMPORTANCE CSV
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_LABELS = {
    "E_density":              "E_density (climate keyword density)",
    "S_density":              "S_density (social keyword density)",
    "G_density":              "G_density (governance keyword density)",
    "hedge_score":            "hedge_score (vague aspiration phrases)",
    "pct_numeric":            "pct_numeric (quantified sentences)",
    "talk_total_density":     "talk_total_density (E+S+G density)",
    "vague_to_specific_ratio":"vague_to_specific_ratio (core greenwashing signal)",
    "text_length":            "text_length (report word count)",
    "esg_topic_breadth":      "esg_topic_breadth (# ESG pillars covered)",
    "e_s_g_dispersion":       "e_s_g_dispersion (sub-score std — selective disclosure)",
    "sector_rank":            "sector_rank (percentile within sector)",
    "planet_friendly_business":"planet_friendly_business (3rd-party ethics)",
    "honest_fair_business":   "honest_fair_business (3rd-party ethics)",
    "sector_encoded":         "sector_encoded (industry context)",
}

FEATURE_GROUPS = {
    "E_density":               "Talk",
    "S_density":               "Talk",
    "G_density":               "Talk",
    "hedge_score":             "Talk",
    "pct_numeric":             "Talk",
    "talk_total_density":      "Talk",
    "vague_to_specific_ratio": "Talk",
    "text_length":             "Talk",
    "esg_topic_breadth":       "Talk",
    "e_s_g_dispersion":        "Walk",
    "sector_rank":             "Walk",
    "planet_friendly_business":"Third-party",
    "honest_fair_business":    "Third-party",
    "sector_encoded":          "Context",
}


def save_feature_importance(explanation: shap.Explanation, save_path: str) -> pd.DataFrame:
    """
    Compute mean |SHAP| per feature, annotate with group and description,
    save to CSV.
    """
    mean_abs_shap = np.abs(explanation.values).mean(axis=0)
    feature_names = explanation.feature_names

    df = pd.DataFrame({
        "feature":       feature_names,
        "mean_abs_shap": mean_abs_shap,
        "group":         [FEATURE_GROUPS.get(f, "Other") for f in feature_names],
        "description":   [FEATURE_LABELS.get(f, f) for f in feature_names],
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    df["rank"] = df.index + 1
    df = df[["rank", "feature", "group", "mean_abs_shap", "description"]]
    df.to_csv(save_path, index=False)

    print(f"\n[explainability] Feature importance (top 5):")
    for _, row in df.head(5).iterrows():
        print(f"  #{int(row['rank']):2d}  {row['feature']:28s}  "
              f"mean|SHAP|={row['mean_abs_shap']:.4f}  [{row['group']}]")
    print(f"[explainability] Saved → {save_path}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 7. MASTER PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_explainability():
    """
    Full explainability pipeline.
    Outputs: beeswarm.png, waterfall_1/2/3.png, feature_importance.csv
    """
    print(f"\n{'='*60}")
    print("  DSF504 ESG Greenwashing — SHAP Explainability")
    print(f"{'='*60}")

    # 1. Load
    pipeline, X, y, meta, df = load_artifacts()

    # 2. SHAP
    explanation, explainer = compute_shap(pipeline, X)

    # 3. Beeswarm
    beeswarm_path = os.path.join(OUTPUTS_DIR, "shap_beeswarm.png")
    plot_beeswarm(explanation, beeswarm_path)

    # 4. Top-3 companies
    print(f"\n[explainability] Top-3 highest-risk greenwashing companies:")
    companies = select_top3(pipeline, X, y, meta)

    # 5. Waterfall plots
    for company in companies:
        wf_path = os.path.join(OUTPUTS_DIR, f"shap_waterfall_{company['rank']}.png")
        plot_waterfall(explanation, company, wf_path)

    # 6. Feature importance CSV
    fi_path = os.path.join(OUTPUTS_DIR, "feature_importance.csv")
    fi_df   = save_feature_importance(explanation, fi_path)

    # 7. Summary
    print(f"\n{'='*60}")
    print("  SHAP OUTPUTS COMPLETE")
    print(f"{'='*60}")
    print(f"  outputs/shap_beeswarm.png")
    for c in companies:
        print(f"  outputs/shap_waterfall_{c['rank']}.png  "
              f"→ {c['ticker']} ({c['sector']})  prob={c['proba']:.4f}")
    print(f"  outputs/feature_importance.csv  "
          f"(top feature: {fi_df.iloc[0]['feature']})")
    print(f"{'='*60}\n")

    return explanation, companies, fi_df


if __name__ == "__main__":
    run_explainability()
