"""
Approximate preprocessing comparison for external sustainability-report scoring.

This experiment is intentionally separate from the production external scoring
path. The project training text in data/preprocessed_content.csv.gz was already
preprocessed upstream, but the original Kaggle preprocessing code is not in this
repository. The function below APPROXIMATES the observed training steps; it is
not an identical recreation of the original Kaggle preprocessing.
"""

from __future__ import annotations

import os
import pickle
import re
import string
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "approx_preprocess_comparison.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "lgbm_best.pkl"
SUSTAINABILITY_INFERENCE_PATH = PROJECT_ROOT / "outputs" / "external_sustainability_report_inference.csv"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

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
TEXT_FEATURES = FEATURE_ORDER[:9]


def _load_stopwords() -> tuple[set[str], str]:
    """Return English stopwords without downloading external corpora."""
    try:
        from nltk.corpus import stopwords

        return set(stopwords.words("english")), "nltk"
    except Exception:
        return set(ENGLISH_STOP_WORDS), "sklearn"


def _wordnet_lemmatizer():
    """Return a WordNet lemmatizer when the local corpus exists."""
    try:
        from nltk.stem import WordNetLemmatizer

        lemmatizer = WordNetLemmatizer()
        lemmatizer.lemmatize("emissions")
        return lemmatizer, True
    except Exception:
        return None, False


def _fallback_lemma(token: str) -> str:
    """Small deterministic fallback used only when local WordNet data is absent."""
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("sses"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def approximate_preprocess_text(text: str) -> tuple[str, dict[str, object]]:
    """
    Approximate the observed Kaggle preprocessing order.

    Order:
      1. lowercase
      2. remove digits and %
      3. remove punctuation
      4. whitespace tokenize
      5. remove English stopwords
      6. remove single-character tokens
      7. lemmatize with WordNet when the local WordNet corpus is available

    This approximates, but does not exactly reproduce, the original Kaggle
    preprocessing that produced data/preprocessed_content.csv.gz.
    """
    stopwords, stopword_source = _load_stopwords()
    lemmatizer, wordnet_available = _wordnet_lemmatizer()

    lowered = str(text or "").lower()
    no_digits = re.sub(r"[\d%]+", " ", lowered)
    punctuation_map = str.maketrans({ch: " " for ch in string.punctuation})
    no_punctuation = no_digits.translate(punctuation_map)
    tokens = no_punctuation.split()
    tokens = [token for token in tokens if token not in stopwords]
    tokens = [token for token in tokens if len(token) > 1]
    if lemmatizer is not None:
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
    else:
        tokens = [_fallback_lemma(token) for token in tokens]

    return " ".join(tokens), {
        "approximate_kaggle_preprocessing": True,
        "stopword_source": stopword_source,
        "wordnet_available": wordnet_available,
    }


def _load_formal_feature_matrix() -> pd.DataFrame:
    from src.data_loader import load_merged_dataset
    from src.feature_engineering import build_feature_matrix

    df = load_merged_dataset(verbose=False)
    X, _, _ = build_feature_matrix(df, feature_set="baseline", verbose=False)
    return X[FEATURE_ORDER].copy()


def _extract_nvda_text() -> tuple[str, str]:
    from src.score_sustainability_reports import extract_pdf_text

    if not SUSTAINABILITY_INFERENCE_PATH.exists():
        raise FileNotFoundError(f"Missing external inference file: {SUSTAINABILITY_INFERENCE_PATH}")
    inference = pd.read_csv(SUSTAINABILITY_INFERENCE_PATH)
    nvda = inference[inference["ticker"].astype(str).str.upper().eq("NVDA")]
    if nvda.empty:
        raise ValueError("NVDA row not found in external_sustainability_report_inference.csv")
    source_file = str(nvda.iloc[0].get("source_file", "")).strip()
    if not source_file or not os.path.exists(source_file):
        raise FileNotFoundError(f"NVDA source PDF not found: {source_file}")
    return extract_pdf_text(source_file)


def _load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def _formal_talk_features(text: str) -> dict[str, float]:
    from src.feature_engineering import extract_talk_features_from_text

    return extract_talk_features_from_text(text)


def _feature_vector(talk_features: dict[str, float], medians: pd.Series) -> dict[str, float]:
    vector = {feature: float(medians.get(feature, 0.0)) for feature in FEATURE_ORDER}
    for feature in TEXT_FEATURES:
        vector[feature] = float(talk_features.get(feature, vector[feature]))
    return vector


def _score(model, feature_vector: dict[str, float]) -> float:
    row = pd.DataFrame([[feature_vector[feature] for feature in FEATURE_ORDER]], columns=FEATURE_ORDER)
    return float(model.predict_proba(row)[0, 1])


def _density_vs_training_median(density: float, median: float) -> str:
    delta = density - median
    pct = (density / median - 1.0) * 100.0 if median else np.nan
    return f"{delta:+.3f} ({pct:+.1f}%)"


def build_comparison() -> pd.DataFrame:
    formal_X = _load_formal_feature_matrix()
    training_median = float(formal_X["talk_total_density"].median())
    medians = formal_X.median(numeric_only=True).reindex(FEATURE_ORDER).fillna(0.0)
    model = _load_model()
    raw_text, extraction_note = _extract_nvda_text()
    approx_text, metadata = approximate_preprocess_text(raw_text)

    variants = [
        ("raw extracted text", raw_text),
        ("approximate preprocessing", approx_text),
    ]
    rows = []
    for variant, text in variants:
        talk_features = _formal_talk_features(text)
        vector = _feature_vector(talk_features, medians)
        score = _score(model, vector)
        density = float(talk_features["talk_total_density"])
        rows.append(
            {
                "variant": variant,
                "post-processing word count": int(len(str(text).split())),
                "talk_total_density": round(density, 6),
                "density vs training median": _density_vs_training_median(density, training_median),
                "model_risk_score": round(score, 10),
            }
        )

    output = pd.DataFrame(rows)
    output.attrs["extraction_note"] = extraction_note
    output.attrs["preprocessing_metadata"] = metadata
    return output


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    comparison = build_comparison()
    comparison.to_csv(OUTPUT_PATH, index=False)

    print("Approximate preprocessing comparison")
    print(f"Output: {OUTPUT_PATH}")
    print(f"Extraction: {comparison.attrs.get('extraction_note', '')}")
    metadata = comparison.attrs.get("preprocessing_metadata", {})
    if metadata:
        print(
            "Approximate preprocessing metadata: "
            f"stopwords={metadata.get('stopword_source')}; "
            f"wordnet_available={metadata.get('wordnet_available')}"
        )
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
