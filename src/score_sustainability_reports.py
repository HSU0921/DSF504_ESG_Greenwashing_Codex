"""
Score downloaded sustainability report PDFs with the formal LightGBM scorer.

This script performs inference only. It extracts text from downloaded
sustainability-report PDFs, computes Talk features through
src.feature_engineering.extract_talk_features_from_text, fills the remaining
formal model features with project medians, and scores with models/lgbm_best.pkl.
"""

from __future__ import annotations

import os
import pickle
import re
import shutil
import subprocess
import sys
from typing import Callable

import numpy as np
import pandas as pd


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT_DIR = os.path.join(PROJECT_ROOT, "data_external", "sustainability_reports")
MANIFEST_PATH = os.path.join(REPORT_DIR, "download_manifest.csv")
OUTPUT_PATH = os.path.join(PROJECT_ROOT, "outputs", "external_sustainability_report_inference.csv")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "lgbm_best.pkl")
CASE_STUDY_MODEL = "lgbm_best (formal final scorer)"

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

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
REQUIRED_COLUMNS = [
    "company",
    "ticker",
    "document_type",
    "document_year",
    "source_file",
    "source_url",
    "case_study_model",
    "model_flagged_greenwashing_risk_score",
    "extraction_note",
]


def _load_formal_feature_matrix() -> pd.DataFrame:
    from src.data_loader import load_merged_dataset
    from src.feature_engineering import build_feature_matrix

    df = load_merged_dataset(verbose=False)
    try:
        X, _, _ = build_feature_matrix(df, feature_set="baseline", verbose=False)
    except TypeError:
        X, _, _ = build_feature_matrix(df, verbose=False)
    return X[FEATURE_ORDER].copy()


def _formal_talk_features(text: str) -> dict[str, float]:
    from src.feature_engineering import extract_talk_features_from_text

    return extract_talk_features_from_text(text)


def _extract_with_pdfplumber(path: str) -> str:
    import pdfplumber  # type: ignore

    chunks = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            chunks.append(page.extract_text() or "")
    return "\n".join(chunks)


def _extract_with_pypdf2(path: str) -> str:
    from PyPDF2 import PdfReader  # type: ignore

    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _extract_with_pypdf(path: str) -> str:
    from pypdf import PdfReader  # type: ignore

    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _extract_with_textutil(path: str) -> str:
    if not shutil.which("textutil"):
        raise RuntimeError("textutil is unavailable")
    result = subprocess.run(
        ["textutil", "-convert", "txt", "-stdout", path],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "textutil failed")
    return result.stdout


def _extract_with_strings(path: str) -> str:
    if not shutil.which("strings"):
        raise RuntimeError("strings is unavailable")
    result = subprocess.run(
        ["strings", path],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "strings failed")
    return result.stdout


def extract_pdf_text(path: str) -> tuple[str, str]:
    extractors: list[tuple[str, Callable[[str], str]]] = [
        ("pdfplumber", _extract_with_pdfplumber),
        ("PyPDF2", _extract_with_pypdf2),
        ("pypdf", _extract_with_pypdf),
        ("textutil", _extract_with_textutil),
        ("strings", _extract_with_strings),
    ]
    errors = []
    for name, extractor in extractors:
        try:
            text = extractor(path)
            text = re.sub(r"\s+", " ", text or "").strip()
            if len(text) >= 1000:
                return text, f"Extracted text with {name}."
            errors.append(f"{name}: extracted text too short ({len(text)} chars)")
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    raise RuntimeError("; ".join(errors))


def _infer_document_year(*values: object) -> str:
    joined = " ".join(str(value or "") for value in values)
    years = [int(match) for match in re.findall(r"\b(20(?:1[8-9]|2[0-9]))\b", joined)]
    return str(max(years)) if years else ""


def _feature_vector(talk_features: dict[str, float], medians: pd.Series) -> dict[str, float]:
    vector = {feature: float(medians.get(feature, 0.0)) for feature in FEATURE_ORDER}
    for feature in TEXT_FEATURES:
        vector[feature] = float(talk_features.get(feature, vector[feature]))
    return vector


def _score(model, feature_vector: dict[str, float]) -> float:
    row = pd.DataFrame([[feature_vector[feature] for feature in FEATURE_ORDER]], columns=FEATURE_ORDER)
    return float(model.predict_proba(row)[0, 1])


def _range_flag(value: float, values: pd.Series) -> str:
    clean = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return "range unavailable"
    low = float(clean.min())
    high = float(clean.max())
    q25 = float(clean.quantile(0.25))
    q75 = float(clean.quantile(0.75))
    if value < low or value > high:
        return f"outside training range [{low:.2f}, {high:.2f}]"
    if value < q25 or value > q75:
        return f"outside interquartile range [{q25:.2f}, {q75:.2f}]"
    return "inside interquartile range"


def score_downloaded_reports() -> pd.DataFrame:
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    if not os.path.exists(MANIFEST_PATH):
        raise FileNotFoundError(f"Download manifest not found: {MANIFEST_PATH}")

    manifest = pd.read_csv(MANIFEST_PATH)
    formal_X = _load_formal_feature_matrix()
    medians = formal_X.median(numeric_only=True).reindex(FEATURE_ORDER).fillna(0.0)
    talk_stats = formal_X["talk_total_density"].quantile([0.25, 0.5, 0.75])
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    rows = []
    table_rows = []
    if "status" not in manifest.columns:
        success_manifest = pd.DataFrame(columns=manifest.columns)
    else:
        success_manifest = manifest[manifest["status"].astype(str).str.lower().eq("downloaded")]
    for _, item in success_manifest.iterrows():
        ticker = str(item.get("ticker", "")).upper().strip()
        company = str(item.get("company", ticker)).strip() or ticker
        pdf_path = str(item.get("local_file", "")).strip()
        source_url = str(item.get("source_url", "")).strip()
        if not pdf_path or not os.path.exists(pdf_path):
            table_rows.append((ticker, company, "failed", "", "", "", "PDF file missing"))
            continue
        try:
            text, method_note = extract_pdf_text(pdf_path)
            char_count = len(text)
            word_count = len(text.split())
            talk = _formal_talk_features(text)
            vector = _feature_vector(talk, medians)
            score = _score(model, vector)
            talk_total = float(talk.get("talk_total_density", np.nan))
            range_note = _range_flag(talk_total, formal_X["talk_total_density"])
            extraction_note = (
                f"{method_note} chars={char_count}; words={word_count}; "
                f"talk_total_density={talk_total:.3f}; {range_note}. "
                "Walk / External / Context features filled with formal project medians; inference only."
            )
            row = {
                "company": company,
                "ticker": ticker,
                "document_type": "Corporate sustainability report",
                "document_year": _infer_document_year(source_url, pdf_path, text[:3000]),
                "source_file": pdf_path,
                "source_url": source_url,
                "case_study_model": CASE_STUDY_MODEL,
                "model_flagged_greenwashing_risk_score": round(score, 10),
                "extraction_note": extraction_note,
            }
            row.update({feature: float(vector.get(feature, 0.0)) for feature in FEATURE_ORDER})
            rows.append(row)
            table_rows.append((ticker, company, "scored", f"{char_count:,}", f"{word_count:,}", f"{score:.10f}", range_note))
        except Exception as exc:
            table_rows.append((ticker, company, "failed", "", "", "", f"{type(exc).__name__}: {exc}"))

    out = pd.DataFrame(rows, columns=REQUIRED_COLUMNS + FEATURE_ORDER)
    out.to_csv(OUTPUT_PATH, index=False)

    print("Sustainability report text extraction and scoring")
    if table_rows:
        printable = pd.DataFrame(
            table_rows,
            columns=["ticker", "company", "status", "characters", "words", "score", "note"],
        )
        print(printable.to_string(index=False))
    else:
        print("No downloaded sustainability-report PDFs were available to score.")
    print("\nTalk total density reference:")
    print(
        f"q25={float(talk_stats.loc[0.25]):.3f}, "
        f"median={float(talk_stats.loc[0.5]):.3f}, "
        f"q75={float(talk_stats.loc[0.75]):.3f}"
    )
    print(f"\nWrote {OUTPUT_PATH}")
    return out


def main() -> None:
    score_downloaded_reports()


if __name__ == "__main__":
    main()
