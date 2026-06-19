"""
feature_engineering.py
=======================
Builds Talk features, Walk features, context features, and the greenwashing label.

CIRCULARITY GUARD
-----------------
The label y uses: Total ESG Risk score + Controversy Score (sector-relative z-scores).
Therefore these two columns are NEVER included as model features.
All other columns (sub-scores E/S/G, integrity scores, text-derived Talk features)
are safe to use as predictors.

Feature Groups
--------------
Talk (text-derived, from preprocessed_content):
  E_density         – climate/emission/carbon/transition/"net zero" per 1000 words
  S_density         – diversity/safety/"human rights"/community per 1000 words
  G_density         – governance/board/audit/compliance/transparency per 1000 words
  hedge_score       – vague-aspiration phrases per 1000 words
  pct_numeric       – fraction of sentences containing a digit or "%"
  talk_total_density– E_density + S_density + G_density
  vague_to_specific_ratio – hedge_score / (pct_numeric + 1e-6)  [core greenwashing signal]
  text_length       – word count of the report
  esg_topic_breadth – number of ESG pillars with density > 0 (0-3)

Walk (structural, no leakage):
  e_s_g_dispersion  – std(Environment, Social, Governance sub-scores)  [selective disclosure]
  sector_rank       – percentile of avg sub-score within sector (NOT Total ESG Risk score)
  planet_friendly_business – third-party ethics score (-30 to +30)
  honest_fair_business     – third-party ethics score (-30 to +30)

Context:
  Sector            – label-encoded

Target:
  greenwashing_label – 1 if Talk in top-1/3 AND Walk in bottom-1/3 within sector
"""

import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# ---------------------------------------------------------------------------
# Keyword dictionaries
# ---------------------------------------------------------------------------
E_KEYWORDS = [
    "climate", "emission", "carbon", "transition", "net zero", "renewable",
    "greenhouse", "biodiversity", "pollution", "fossil", "decarboniz",
    "scope 1", "scope 2", "scope 3", "energy efficiency", "clean energy",
]

S_KEYWORDS = [
    "diversity", "safety", "human rights", "community", "inclusion",
    "labour", "labor", "employee", "wellbeing", "equality", "supply chain",
    "social impact", "indigenous",
]

G_KEYWORDS = [
    "governance", "board", "audit", "compliance", "transparency",
    "accountability", "ethics", "integrity", "risk management",
    "anti-corruption", "bribery", "whistleblower",
]

HEDGE_PHRASES = [
    r"\bmay\b", r"\bmight\b", r"\bcould\b", r"\baspire\b",
    r"\bintend to\b", r"\baim to\b", r"\bseek to\b",
    r"\bcommit to\b", r"\bplan to\b", r"\bhope to\b",
    r"\bstrive\b", r"\bwork towards\b",
]


# ---------------------------------------------------------------------------
# Semantic robustness patterns
# ---------------------------------------------------------------------------
CLIMATE_TRANSITION_PATTERNS = [
    re.compile(r"\btransition\s+plans?\b", re.IGNORECASE),
    re.compile(r"\blow[-\s]?carbon\s+transitions?\b", re.IGNORECASE),
    re.compile(r"\bclimate\s+strateg(y|ies)\b", re.IGNORECASE),
    re.compile(r"\bdecarboni[sz]ation\s+pathways?\b", re.IGNORECASE),
    re.compile(r"\benergy\s+transitions?\b", re.IGNORECASE),
    re.compile(r"\bsustainable\s+transitions?\b", re.IGNORECASE),
    re.compile(r"\bclimate\s+resilien(ce|t)\b", re.IGNORECASE),
    re.compile(r"\bjust\s+transitions?\b", re.IGNORECASE),
    re.compile(r"\bnet[-\s]?zero\s+pathways?\b", re.IGNORECASE),
]

MEASURABLE_COMMITMENT_PATTERNS = [
    re.compile(r"\d+\s*%", re.IGNORECASE),
    re.compile(r"\b(20\d{2})\b", re.IGNORECASE),
    re.compile(r"\bbaseline\s+years?\b", re.IGNORECASE),
    re.compile(r"\bscience[-\s]?based\s+targets?\b", re.IGNORECASE),
    re.compile(r"\bSBTi\b", re.IGNORECASE),
    re.compile(r"\bthird[-\s]?party\s+assurance\b", re.IGNORECASE),
    re.compile(r"\bexternally\s+verified\b", re.IGNORECASE),
    re.compile(r"\baudited\s+by\b", re.IGNORECASE),
    re.compile(r"\breductions?\s+of\s+\d+", re.IGNORECASE),
    re.compile(r"\bby\s+20(30|40|50)\b", re.IGNORECASE),
]

VAGUE_COMMITMENT_PATTERNS = [
    re.compile(r"\bcommitted\s+to\s+sustainability\b", re.IGNORECASE),
    re.compile(r"\bmeaningful\s+progress\b", re.IGNORECASE),
    re.compile(r"\blong[-\s]?term\s+value\b", re.IGNORECASE),
    re.compile(r"\bresponsible\s+business(es)?\b", re.IGNORECASE),
    re.compile(r"\bcontinues?\s+to\s+improve\b", re.IGNORECASE),
    re.compile(r"\baspir(e|es|ed|ing)\s+to\b", re.IGNORECASE),
    re.compile(r"\baim(s|ed|ing)?\s+to\b", re.IGNORECASE),
    re.compile(r"\bsupport(s|ed|ing)?\s+a\s+better\s+future\b", re.IGNORECASE),
    re.compile(r"\bappropriate\s+governance\s+process(es)?\b", re.IGNORECASE),
    re.compile(r"\brobust\s+frameworks?\b", re.IGNORECASE),
]

ASSURANCE_ACCOUNTABILITY_PATTERNS = [
    re.compile(r"\bboard\s+oversight\b", re.IGNORECASE),
    re.compile(r"\bindependent\s+assurance\b", re.IGNORECASE),
    re.compile(r"\bexternally\s+verified\b", re.IGNORECASE),
    re.compile(r"\baudited\b", re.IGNORECASE),
    re.compile(r"\bexternal\s+reviews?\b", re.IGNORECASE),
    re.compile(r"\bgovernance\s+committees?\b", re.IGNORECASE),
    re.compile(r"\baudit\s+committees?\b", re.IGNORECASE),
    re.compile(r"\baccountabilit(y|ies)\b", re.IGNORECASE),
    re.compile(r"\btransparent\s+reporting\b", re.IGNORECASE),
    re.compile(r"\bmateriality\s+assessments?\b", re.IGNORECASE),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _word_count(text: str) -> int:
    return len(text.split())


def _keyword_density(text: str, keywords: list, word_count: int) -> float:
    """Count keyword occurrences per 1000 words. Case-insensitive."""
    if word_count == 0:
        return 0.0
    text_lower = text.lower()
    count = sum(text_lower.count(kw.lower()) for kw in keywords)
    return count / word_count * 1000


def _hedge_density(text: str, word_count: int) -> float:
    """Count hedge-phrase occurrences per 1000 words."""
    if word_count == 0:
        return 0.0
    text_lower = text.lower()
    count = sum(len(re.findall(p, text_lower)) for p in HEDGE_PHRASES)
    return count / word_count * 1000


def _pct_numeric(text: str) -> float:
    """Fraction of sentences that contain a digit or '%'."""
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return 0.0
    numeric_count = sum(1 for s in sentences if re.search(r"\d|%", s))
    return numeric_count / len(sentences)


def _esg_topic_breadth(e_den: float, s_den: float, g_den: float) -> int:
    """Count how many ESG pillars have any coverage (density > 0)."""
    return int(e_den > 0) + int(s_den > 0) + int(g_den > 0)


def _semantic_density(text: str, patterns: list[re.Pattern], word_count: int) -> float:
    """Count semantic regex matches per 1000 words."""
    if word_count == 0:
        return 0.0
    count = sum(len(pattern.findall(text)) for pattern in patterns)
    return count / word_count * 1000


# ---------------------------------------------------------------------------
# Talk feature extraction
# ---------------------------------------------------------------------------

def extract_talk_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute NLP-based Talk features from the 'preprocessed_content' column.
    Returns a DataFrame with the same index as df, with new Talk feature columns.
    """
    records = []
    for _, row in df.iterrows():
        text = str(row["preprocessed_content"])
        wc = _word_count(text)

        e_den = _keyword_density(text, E_KEYWORDS, wc)
        s_den = _keyword_density(text, S_KEYWORDS, wc)
        g_den = _keyword_density(text, G_KEYWORDS, wc)
        hedge = _hedge_density(text, wc)
        pct_num = _pct_numeric(text)
        total_den = e_den + s_den + g_den
        vague_ratio = hedge / (pct_num + 1e-6)
        breadth = _esg_topic_breadth(e_den, s_den, g_den)

        records.append({
            "E_density": e_den,
            "S_density": s_den,
            "G_density": g_den,
            "hedge_score": hedge,
            "pct_numeric": pct_num,
            "talk_total_density": total_den,
            "vague_to_specific_ratio": vague_ratio,
            "text_length": wc,
            "esg_topic_breadth": breadth,
        })

    return pd.DataFrame(records, index=df.index)


def extract_talk_features_from_text(text: str) -> dict[str, float]:
    """Extract Talk features for one text using the formal dashboard definitions.

    This wrapper is the single source of truth for external inference: it
    calls extract_talk_features(), so densities stay per 1000 words,
    pct_numeric stays sentence-based, and vague_to_specific_ratio stays
    hedge_score / (pct_numeric + 1e-6).
    """
    row = pd.DataFrame({"preprocessed_content": [str(text)]})
    return extract_talk_features(row).iloc[0].astype(float).to_dict()


# ---------------------------------------------------------------------------
# Semantic robustness feature extraction
# ---------------------------------------------------------------------------

def extract_semantic_robustness_features(
    df: pd.DataFrame,
    text_col: str = "preprocessed_content",
) -> pd.DataFrame:
    """
    Returns DataFrame with same index as df and these columns
    (all per-1000-words normalized, except gap which is a difference):
      - climate_transition_semantic_score
      - measurable_commitment_score
      - vague_commitment_score
      - assurance_and_accountability_score
      - semantic_specificity_gap   (= vague - measurable)
    """
    records = []
    for _, row in df.iterrows():
        text = "" if pd.isna(row.get(text_col, "")) else str(row.get(text_col, ""))
        wc = _word_count(text)

        climate = _semantic_density(text, CLIMATE_TRANSITION_PATTERNS, wc)
        measurable = _semantic_density(text, MEASURABLE_COMMITMENT_PATTERNS, wc)
        vague = _semantic_density(text, VAGUE_COMMITMENT_PATTERNS, wc)
        assurance = _semantic_density(text, ASSURANCE_ACCOUNTABILITY_PATTERNS, wc)

        records.append({
            "climate_transition_semantic_score": climate,
            "measurable_commitment_score": measurable,
            "vague_commitment_score": vague,
            "assurance_and_accountability_score": assurance,
            "semantic_specificity_gap": vague - measurable,
        })

    return pd.DataFrame(records, index=df.index)


# ---------------------------------------------------------------------------
# Walk feature extraction (no leakage of Total ESG Risk / Controversy Score)
# ---------------------------------------------------------------------------

def extract_walk_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute structural Walk features.
    Uses sub-scores (E/S/G) and integrity scores — NOT Total ESG Risk or Controversy Score.
    """
    sub = df[["Environment Risk Score", "Social Risk Score", "Governance Risk Score"]].copy()

    # Dispersion of sub-scores: high dispersion = selective ESG disclosure
    e_s_g_dispersion = sub.std(axis=1)

    # Sector rank: percentile of avg sub-score within sector
    # (proxy for "how bad is this company's ESG profile vs peers" — avoids leakage
    #  because we rank by sub-score average, not by Total ESG Risk Score itself)
    avg_sub = sub.mean(axis=1)
    df2 = df.copy()
    df2["_avg_sub"] = avg_sub
    sector_rank = (
        df2.groupby("Sector")["_avg_sub"]
        .rank(pct=True, ascending=True)
    )

    walk = pd.DataFrame(index=df.index)
    walk["e_s_g_dispersion"] = e_s_g_dispersion.values
    walk["sector_rank"] = sector_rank.values
    walk["planet_friendly_business"] = df["planet_friendly_business"].values
    walk["honest_fair_business"] = df["honest_fair_business"].values

    return walk


# ---------------------------------------------------------------------------
# Label construction
# ---------------------------------------------------------------------------

def build_label(df: pd.DataFrame, talk_total_density: pd.Series) -> pd.Series:
    """
    Greenwashing label (binary):
      1 = company is in the TOP 1/3 of Talk AND BOTTOM 1/3 of Walk, within its sector.

    Walk Score = sector z-score of (Total ESG Risk score + Controversy Score).
    Note: Higher ESG Risk score = worse performance; higher Controversy Score = worse.
    We add them (both are 'bad' when large), then z-score within sector.

    Talk Score = talk_total_density (E+S+G keyword density from text).

    We rank both within sector and flag companies that talk a lot but perform poorly.
    Controversy Score NaN → filled with sector median before z-scoring.
    """
    df2 = df.copy()
    df2["_talk"] = talk_total_density.values

    # Fill missing Controversy Score with sector median (missing ≠ no controversy)
    df2["_controversy"] = df2["Controversy Score"].copy()
    df2["_controversy"] = (
        df2.groupby("Sector")["_controversy"]
        .transform(lambda x: x.fillna(x.median()))
    )

    # Walk composite: sum of risk scores (both directionally "bad when high")
    df2["_walk_raw"] = df2["Total ESG Risk score"] + df2["_controversy"]

    # Sector-relative percentile ranks (ascending = lower rank means worse performer)
    df2["_talk_pct"] = (
        df2.groupby("Sector")["_talk"]
        .rank(pct=True, ascending=True)
    )
    # For walk, ascending=False so that HIGH risk score = LOW percentile (bad)
    df2["_walk_pct"] = (
        df2.groupby("Sector")["_walk_raw"]
        .rank(pct=True, ascending=False)  # low pct = high risk = bad performance
    )

    top_third_talk = df2["_talk_pct"] >= 2 / 3   # top 1/3 talkers
    bottom_third_walk = df2["_walk_pct"] <= 1 / 3  # worst 1/3 performers

    label = (top_third_talk & bottom_third_walk).astype(int)
    return label


# ---------------------------------------------------------------------------
# Context features
# ---------------------------------------------------------------------------

def encode_sector(df: pd.DataFrame) -> tuple[pd.Series, LabelEncoder]:
    """Label-encode the Sector column."""
    le = LabelEncoder()
    sector_encoded = le.fit_transform(df["Sector"].fillna("Unknown"))
    return pd.Series(sector_encoded, index=df.index, name="sector_encoded"), le


# ---------------------------------------------------------------------------
# Master pipeline
# ---------------------------------------------------------------------------

def build_feature_matrix(
    df: pd.DataFrame,
    feature_set: str = "baseline",
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.Series, dict]:
    """
    Full feature engineering pipeline.

    Parameters
    ----------
    df : merged dataset from data_loader.load_merged_dataset()
    feature_set : "baseline" returns original 14 columns only; "semantic"
        returns the original 14 columns plus 5 semantic robustness columns.
        The default remains "baseline" to preserve existing dashboard behavior.

    Returns
    -------
    X : pd.DataFrame   — feature matrix (no leakage columns)
    y : pd.Series      — greenwashing label (0/1)
    meta : dict        — ticker list, sector encoder, and interim columns for EDA
    """
    if isinstance(feature_set, bool):
        verbose = feature_set
        feature_set = "baseline"
    if feature_set not in {"baseline", "semantic"}:
        raise ValueError("feature_set must be 'baseline' or 'semantic'")

    if verbose:
        print("[feature_engineering] Extracting Talk features …")
    talk = extract_talk_features(df)

    if verbose:
        print("[feature_engineering] Extracting Walk features …")
    walk = extract_walk_features(df)

    if verbose:
        print("[feature_engineering] Building greenwashing label …")
    y = build_label(df, talk["talk_total_density"])

    sector_encoded, le = encode_sector(df)

    X = pd.concat(
        [
            talk,
            walk,
            sector_encoded,
        ],
        axis=1,
    )
    X.index = df.index

    semantic = None
    if feature_set == "semantic":
        if verbose:
            print("[feature_engineering] Extracting semantic robustness features …")
        semantic = extract_semantic_robustness_features(df)
        X = pd.concat([X, semantic], axis=1)

    meta = {
        "ticker": df["ticker"].values,
        "sector": df["Sector"].values,
        "industry": df["Industry"].values,
        "year": df["year"].values,
        "sector_encoder": le,
        "talk_df": talk,
        "walk_df": walk,
        "semantic_df": semantic,
        # Raw scores kept only for EDA / label diagnostics — never go into X
        "total_esg_risk_score": df["Total ESG Risk score"].values,
        "controversy_score": df["Controversy Score"].values,
    }

    if verbose:
        print(f"[feature_engineering] Feature matrix shape: {X.shape}")
        print(f"[feature_engineering] Label distribution:\n{y.value_counts()}")
        pos_rate = y.mean()
        print(f"[feature_engineering] Positive rate (greenwashing=1): {pos_rate:.1%}")
        print(f"[feature_engineering] Features: {list(X.columns)}")

    return X, y, meta


# ---------------------------------------------------------------------------
# Entry point for quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.data_loader import load_merged_dataset

    df = load_merged_dataset(verbose=True)
    X, y, meta = build_feature_matrix(df, feature_set="baseline", verbose=True)
    X_semantic, y_semantic, meta_semantic = build_feature_matrix(df, feature_set="semantic", verbose=True)

    print(f"\nBaseline feature matrix: {X.shape[0]} x {X.shape[1]}")
    print(X.head(3))
    print("\nBaseline NaN counts:")
    print(X.isna().sum())

    print(f"\nSemantic feature matrix: {X_semantic.shape[0]} x {X_semantic.shape[1]}")
    print(X_semantic.head(3))
    print("\nSemantic NaN counts:")
    print(X_semantic.isna().sum())

    baseline_nan_total = int(X.isna().sum().sum())
    semantic_nan_total = int(X_semantic.isna().sum().sum())
    print(f"\nBaseline total NaN count: {baseline_nan_total}")
    print(f"Semantic total NaN count: {semantic_nan_total}")
    if baseline_nan_total or semantic_nan_total:
        print("\nNaN details:")
        print("Baseline columns with NaNs:")
        print(X.isna().sum()[X.isna().sum() > 0])
        print("Semantic columns with NaNs:")
        print(X_semantic.isna().sum()[X_semantic.isna().sum() > 0])

    print("\nLabel counts:")
    print(y.value_counts())
    if not y.equals(y_semantic):
        raise ValueError("Semantic feature_set changed label values; labels must remain unchanged.")

    # Quick sanity check: confirm forbidden columns are absent from X
    forbidden = {"Total ESG Risk score", "Controversy Score", "ESG Risk Percentile"}
    leaked = forbidden.intersection(set(X.columns)).union(forbidden.intersection(set(X_semantic.columns)))
    if leaked:
        raise ValueError(f"LEAKAGE DETECTED — forbidden columns in X: {leaked}")
    print("\nLeakage check PASSED: no forbidden columns in feature matrix.")
