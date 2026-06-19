"""
holdout_robustness_experiment.py
================================
Additive robustness validation for the DSF504 ESG Greenwashing project.

Purpose:
- Create a real-company 70/30 holdout split before model fitting.
- Keep the real holdout set completely out of tuning, model selection, and
  synthetic augmentation.
- Compare a real-training-only text model with a training-only synthetic
  disclosure augmented text model and a DAX weak-labeled company augmented
  text model when DAX documents are available.
- Run one external company case study from SEC EDGAR, labeled as a case study
  rather than proof.

Governance:
This script reports model-flagged greenwashing risk as a risk screening signal.
It does not legally determine greenwashing, corporate wrongdoing, credit
approval, or investment exclusion.
"""

from __future__ import annotations

import csv
import html
import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Iterable
from urllib.error import URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")
DATA_EXTERNAL_DIR = os.path.join(PROJECT_ROOT, "data_external")
KAGGLE_RAW_DIR = os.path.join(DATA_EXTERNAL_DIR, "kaggle_raw")
RANDOM_STATE = 42
TEXT_FEATURES = [
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
DAX_DOCUMENTS_FILE = "esg_documents_for_dax_companies.csv"
DAX_LABELED_OUTPUT_FILE = "dax_manual_labeled_company_samples.csv"
COMBINED_EXTERNAL_LABELED_OUTPUT_FILE = "combined_external_weak_labeled_company_samples.csv"
SDG_REFERENCE_FILE = "sdg_descriptions_with_targetsText.csv"
ACCUSATION_DATASET_FILE = "corporate_greenwashing_accusations.csv"
GREENWASHING_SCORE_FILE = "Greenwashing_Score_Data.xlsx"
SEC_EDGAR_SAMPLE_FILE = "SEC_EDGAR_10-K_MASTER_DATA_SAMPLE.csv"
SEC_EDGAR_MASTER_FILE = "SEC_EDGAR_10-K_MASTER_DATA.parquet"
SEC_EDGAR_METADATA_FILE = "SEC_EDGAR_10-K_METADATA.json"
DISCLOSURE_DATASET_FILES = [
    "kaggle_sustainability_reports.csv",
    "sec_edgar_10k_esg_climate.csv",
]
SEC_EDGAR_DATASET_FILES = [SEC_EDGAR_SAMPLE_FILE, SEC_EDGAR_MASTER_FILE]
EXTERNAL_TEXT_COLUMN_CANDIDATES = [
    "accusation",
    "accusation_text",
    "description",
    "article_text",
    "report_text",
    "filing_text",
    "risk_factor_text",
    "content",
    "text",
    "preprocessed_content",
]
EXTERNAL_COMPANY_COLUMN_CANDIDATES = ["company", "company_name"]
EXTERNAL_SYMBOL_COLUMN_CANDIDATES = ["ticker", "symbol"]
EXTERNAL_YEAR_COLUMN_CANDIDATES = ["year", "filing_year", "filing_date", "date", "YEAR"]
SEC_USER_AGENT = "DSF504 ESG Greenwashing academic project; contact: student@example.com"
GOVERNANCE_DISCLAIMER = (
    "This experiment reports model-flagged greenwashing risk as a risk screening signal. "
    "It does not legally determine greenwashing or corporate wrongdoing."
)

E_KEYWORDS = [
    "environment",
    "environmental",
    "climate",
    "carbon",
    "emissions",
    "renewable",
    "energy",
    "water",
    "waste",
    "biodiversity",
]
S_KEYWORDS = [
    "employee",
    "labor",
    "safety",
    "community",
    "human rights",
    "diversity",
    "inclusion",
    "health",
    "workforce",
]
G_KEYWORDS = [
    "governance",
    "board",
    "ethics",
    "compliance",
    "audit",
    "oversight",
    "risk management",
    "anti-corruption",
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
SPECIFIC_PATTERNS = [
    r"\d+\s*%",
    r"\bpercent\b",
    r"\b20\d{2}\b",
    r"\bbaseline\b",
    r"\btarget(s)?\b",
    r"\bgri\b",
    r"\bsasb\b",
    r"\btcfd\b",
    r"\bsbti\b",
    r"\bscope\s*[123]\b",
    r"\bverified\b",
    r"\bassurance\b",
    r"\baudited\b",
]
CONTROVERSY_PATTERNS = [
    r"\bscandal(s)?\b",
    r"\blawsuit(s)?\b",
    r"\binvestigation(s)?\b",
    r"\ballegation(s)?\b",
    r"\bpollution\b",
    r"\bviolation(s)?\b",
    r"\bemissions\b",
    r"\bgreenwashing\b",
    r"\bcontrovers(y|ies)\b",
    r"\bfraud\b",
    r"\bfine(s)?\b",
    r"\bpenalt(y|ies)\b",
    r"\bmisleading\b",
    r"\bbreach\b",
    r"\bnon[-\s]?compliance\b",
]


@dataclass(frozen=True)
class ExternalCase:
    company: str
    cik: str
    source_url: str
    retrieval_note: str
    risk_score: float


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
                "Run this script from a Terminal/session with permission to read the Desktop project folder."
            ) from exc
    return load_merged_dataset, build_feature_matrix


def _compile(patterns: Iterable[str]) -> list[re.Pattern]:
    return [re.compile(pattern, flags=re.IGNORECASE) for pattern in patterns]


def _keyword_count(text: str, terms: list[str]) -> int:
    return int(sum(len(re.findall(rf"\b{re.escape(term)}\b", text, flags=re.IGNORECASE)) for term in terms))


def _pattern_count(text: str, patterns: list[re.Pattern]) -> int:
    return int(sum(len(pattern.findall(text)) for pattern in patterns))


def _word_count(text: str) -> int:
    return max(len(str(text).split()), 1)


def extract_talk_features_from_text(text: str) -> dict[str, float]:
    """Deprecated compatibility wrapper for Talk feature extraction.

    External inference and any future robustness reruns should use the formal
    definitions in src.feature_engineering. Keeping this wrapper avoids two
    divergent Talk-feature definitions.
    """
    try:
        from src.feature_engineering import extract_talk_features_from_text as formal_extract
    except ModuleNotFoundError:
        from feature_engineering import extract_talk_features_from_text as formal_extract

    return formal_extract(text)


def synthetic_disclosures() -> pd.DataFrame:
    high_risk_templates = [
        "We are committed to sustainability and a green sustainable future. Our climate solution supports responsible business, meaningful progress, and long-term value without external assurance.",
        "Our clean energy transition vision will continue to improve environmental performance. We aspire to robust governance processes and responsible business practices.",
        "Natural gas remains part of our low-carbon future. We aim to support communities and climate progress through broad sustainability commitments without third-party assurance.",
        "We promote environmentally friendly operations and green innovation. We continue to improve employee safety and governance while building a better future.",
        "Our company is committed to clean growth, climate leadership, and responsible operations. We describe a robust framework but provide limited quantified evidence.",
        "We support a better future through green products, sustainable future goals, and meaningful progress across environmental and social priorities.",
        "Our climate solution helps customers transition to cleaner operations. We aspire to long-term sustainability and responsible governance.",
        "We are committed to natural gas innovation, low-carbon growth, and environmentally friendly performance with ongoing improvement across ESG topics.",
        "Our green strategy supports climate resilience and responsible business. We aim to strengthen governance processes and community impact.",
        "Clean operations and sustainable future commitments guide our work. We continue to improve but have not obtained independent assurance.",
        "Our environmental commitment is part of a robust framework for responsible business and climate solution development.",
        "We aspire to be a sustainability leader through green innovation, natural gas solutions, and meaningful progress across ESG priorities.",
    ]
    low_risk_templates = [
        "We reduced Scope 1 emissions by 24% versus a 2019 baseline and target a further 35% reduction by 2030 with third-party assurance.",
        "Our 2025 target includes 80% renewable electricity, verified progress, and audited Scope 2 emissions under GRI and SASB reporting.",
        "The board audit committee reviews ESG targets quarterly. Our 2030 science-based target was externally verified and includes Scope 1, 2, and 3 emissions.",
        "We achieved a 12% water reduction in 2024 against a 2020 baseline and published assurance results for material environmental metrics.",
        "Our TCFD report discloses quantified transition risks, capital allocation, and audited progress toward a 2050 net-zero target.",
        "Employee safety improved by 18% year over year, and the target is tracked by the governance committee with verified incident data.",
        "We completed third-party assurance for greenhouse gas emissions, baseline year calculations, and 2030 decarbonization targets.",
        "Our SASB-aligned report states a 40% waste reduction target by 2030 and provides verified 2024 progress metrics.",
        "The board oversight process reviews climate targets, audited emissions data, and verified renewable electricity procurement.",
        "We disclose Scope 1 emissions of 1.2 million tCO2e, a 2019 baseline, and a target of 30% reduction by 2030.",
        "The company obtained independent assurance for ESG metrics and reports quantified progress against science-based targets.",
        "Our GRI and TCFD disclosures include measurable targets, verified emissions, baseline year methodology, and third-party assurance.",
    ]

    rows = []
    for i, text in enumerate(high_risk_templates, start=1):
        rows.append({"synthetic_id": f"synthetic_high_{i:02d}", "text": text, "label": 1})
    for i, text in enumerate(low_risk_templates, start=1):
        rows.append({"synthetic_id": f"synthetic_low_{i:02d}", "text": text, "label": 0})
    synthetic = pd.DataFrame(rows)
    feature_rows = synthetic["text"].map(extract_talk_features_from_text).apply(pd.Series)
    return pd.concat([synthetic, feature_rows], axis=1)


def _make_augmented_feature_frame(
    text_feature_df: pd.DataFrame,
    feature_columns: list[str],
    train_medians: pd.Series,
    source_type: str,
) -> pd.DataFrame:
    out = pd.DataFrame(index=text_feature_df.index)
    for column in feature_columns:
        if column in text_feature_df.columns:
            out[column] = pd.to_numeric(text_feature_df[column], errors="coerce")
        else:
            out[column] = train_medians.get(column, 0.0)
    out = out.fillna(train_medians.reindex(feature_columns).fillna(0.0))
    out["source_type"] = source_type
    return out


def _first_matching_column(columns: Iterable[str], candidates: list[str]) -> str | None:
    lower_map = {str(col).strip().lower(): col for col in columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    return None


def _safe_read_csv(path: str) -> pd.DataFrame:
    try:
        csv.field_size_limit(sys.maxsize)
    except OverflowError:
        csv.field_size_limit(2**31 - 1)
    try:
        return pd.read_csv(path)
    except csv.Error:
        return pd.read_csv(path, on_bad_lines="skip")


def _empty_external_labeled_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "company",
            "symbol",
            "year",
            "content",
            "manual_greenwashing_label",
            "label_method",
            "label_reason",
            "source_dataset",
            "source_type",
        ]
    )


def _extract_year_value(value) -> str:
    if pd.isna(value):
        return ""
    match = re.search(r"\b(19\d{2}|20\d{2})\b", str(value))
    return match.group(1) if match else ""


def _latest_year_from_series(values: pd.Series) -> str:
    years = pd.to_numeric(values.map(_extract_year_value), errors="coerce").dropna()
    return str(int(years.max())) if not years.empty else ""


def _external_key_columns(raw: pd.DataFrame) -> tuple[str | None, str | None, str | None, str | None]:
    company_col = _first_matching_column(raw.columns, EXTERNAL_COMPANY_COLUMN_CANDIDATES)
    symbol_col = _first_matching_column(raw.columns, EXTERNAL_SYMBOL_COLUMN_CANDIDATES)
    year_col = _first_matching_column(raw.columns, EXTERNAL_YEAR_COLUMN_CANDIDATES)
    text_col = _first_matching_column(raw.columns, EXTERNAL_TEXT_COLUMN_CANDIDATES)
    return company_col, symbol_col, year_col, text_col


def _aggregate_external_company_content(raw: pd.DataFrame, source_dataset: str) -> tuple[pd.DataFrame, str]:
    company_col, symbol_col, year_col, text_col = _external_key_columns(raw)
    if text_col is None:
        return pd.DataFrame(), f"{source_dataset}: inspected but skipped because no recognized full-text column was found."

    work = raw.copy()
    work["_company"] = work[company_col].fillna("").astype(str) if company_col else ""
    work["_symbol"] = work[symbol_col].fillna("").astype(str) if symbol_col else ""
    work["_year"] = work[year_col].map(_extract_year_value) if year_col else ""
    work["_content"] = work[text_col].fillna("").astype(str)
    work = work[work["_content"].str.strip().ne("")]
    if work.empty:
        return pd.DataFrame(), f"{source_dataset}: inspected but skipped because the recognized text column was empty."

    work["_company_key"] = work["_symbol"].str.strip()
    missing_symbol = work["_company_key"].eq("")
    work.loc[missing_symbol, "_company_key"] = work.loc[missing_symbol, "_company"].str.strip()
    missing_key = work["_company_key"].eq("")
    work.loc[missing_key, "_company_key"] = source_dataset + "_row_" + work.index.astype(str)

    rows = []
    for _, group in work.groupby("_company_key", dropna=False):
        company = ""
        symbol = ""
        if group["_company"].str.strip().ne("").any():
            company = str(group.loc[group["_company"].str.strip().ne(""), "_company"].iloc[0])
        if group["_symbol"].str.strip().ne("").any():
            symbol = str(group.loc[group["_symbol"].str.strip().ne(""), "_symbol"].iloc[0])
        rows.append(
            {
                "company": company,
                "symbol": symbol,
                "year": _latest_year_from_series(group["_year"]),
                "content": " ".join(group["_content"].astype(str).tolist()),
                "source_dataset": source_dataset,
            }
        )
    return pd.DataFrame(rows), f"{source_dataset}: aggregated {len(rows)} company-level records with full text."


def _external_disclosure_signal_row(row: pd.Series) -> dict:
    content = str(row["content"])
    clean = content.lower()
    features = extract_talk_features_from_text(content)
    specific_count = _pattern_count(clean, _compile(SPECIFIC_PATTERNS))
    vague_count = _pattern_count(clean, _compile(VAGUE_PATTERNS))
    numeric_count = len(re.findall(r"\d+(\.\d+)?", clean))
    controversy_count = _pattern_count(clean, _compile(CONTROVERSY_PATTERNS))
    return {
        "company": row.get("company", ""),
        "symbol": row.get("symbol", ""),
        "year": row.get("year", ""),
        "content": content,
        "source_dataset": row.get("source_dataset", ""),
        "talk_total_density": features["talk_total_density"],
        "pct_numeric": features["pct_numeric"],
        "vague_to_specific_ratio": features["vague_to_specific_ratio"],
        "specific_count": specific_count,
        "vague_count": vague_count,
        "numeric_count": numeric_count,
        "controversy_count": controversy_count,
        "concrete_evidence_score": specific_count + numeric_count,
    }


def _assign_external_disclosure_weak_labels(candidates: pd.DataFrame, source_dataset: str) -> tuple[pd.DataFrame, str]:
    if candidates.empty:
        return pd.DataFrame(), f"{source_dataset}: no company-level candidates were available for weak labeling."

    work = candidates.copy()
    work["talk_rank"] = work["talk_total_density"].rank(pct=True, method="average")
    work["vague_rank"] = work["vague_to_specific_ratio"].rank(pct=True, method="average")
    work["weak_numeric_rank"] = (-work["pct_numeric"]).rank(pct=True, method="average")
    work["controversy_rank"] = work["controversy_count"].rank(pct=True, method="average")
    work["evidence_rank"] = work["concrete_evidence_score"].rank(pct=True, method="average")
    work["pct_numeric_rank"] = work["pct_numeric"].rank(pct=True, method="average")
    work["high_risk_weak_label_score"] = (
        0.30 * work["talk_rank"]
        + 0.25 * work["vague_rank"]
        + 0.25 * work["controversy_rank"]
        + 0.20 * work["weak_numeric_rank"]
    )
    work["low_risk_weak_label_score"] = (
        0.45 * work["evidence_rank"]
        + 0.25 * work["pct_numeric_rank"]
        + 0.20 * (1 - work["controversy_rank"])
        + 0.10 * (1 - work["vague_rank"])
    )

    n = len(work)
    target_per_class = min(max(3, n // 3), max(1, n // 2))
    high_idx = set(
        work.sort_values("high_risk_weak_label_score", ascending=False)
        .head(target_per_class)
        .index
    )
    low_idx = set(
        work.drop(index=list(high_idx))
        .sort_values("low_risk_weak_label_score", ascending=False)
        .head(target_per_class)
        .index
    )

    labeled_rows = []
    for idx, row in work.iterrows():
        if idx in high_idx:
            label = 1
            reason = (
                "High weak-label risk: external disclosure sample is relatively ESG-heavy, vague/low-numeric, "
                f"or controversy-linked (risk score={row['high_risk_weak_label_score']:.3f})."
            )
        elif idx in low_idx:
            label = 0
            reason = (
                "Low weak-label risk: external disclosure sample has relatively concrete evidence and lower controversy signal "
                f"(low-risk score={row['low_risk_weak_label_score']:.3f})."
            )
        else:
            continue
        labeled_rows.append(
            {
                "company": row["company"],
                "symbol": row["symbol"],
                "year": row.get("year", ""),
                "content": row["content"],
                "manual_greenwashing_label": int(label),
                "label_method": "Kaggle disclosure heuristic weak label",
                "label_reason": reason,
                "source_dataset": source_dataset,
                "source_type": "kaggle_weak_labeled_company",
            }
        )
    labeled = pd.DataFrame(labeled_rows)
    high_n = int((labeled["manual_greenwashing_label"] == 1).sum()) if not labeled.empty else 0
    low_n = int((labeled["manual_greenwashing_label"] == 0).sum()) if not labeled.empty else 0
    skipped_n = n - len(labeled)
    return labeled, f"{source_dataset}: created {high_n} high-risk and {low_n} low-risk weak labels; skipped {skipped_n} ambiguous records."


def _build_greenwashing_score_weak_labels() -> tuple[pd.DataFrame, str, list[str]]:
    path = os.path.join(KAGGLE_RAW_DIR, GREENWASHING_SCORE_FILE)
    if not os.path.exists(path):
        return _empty_external_labeled_frame(), f"{GREENWASHING_SCORE_FILE}: not found.", []

    try:
        raw = pd.read_excel(path)
    except Exception as exc:
        return (
            _empty_external_labeled_frame(),
            f"{GREENWASHING_SCORE_FILE}: found but could not be read ({type(exc).__name__}).",
            [GREENWASHING_SCORE_FILE],
        )

    company_col = _first_matching_column(raw.columns, ["COMPANY_NAME", "company", "company_name"])
    year_col = _first_matching_column(raw.columns, ["YEAR", "year"])
    score_col = _first_matching_column(raw.columns, ["GW_SCORE", "greenwashing_score", "score"])
    if company_col is None or score_col is None:
        return (
            _empty_external_labeled_frame(),
            f"{GREENWASHING_SCORE_FILE}: found but required COMPANY_NAME/GW_SCORE columns were missing.",
            [GREENWASHING_SCORE_FILE],
        )

    work = raw.copy()
    work["_company"] = work[company_col].fillna("").astype(str).str.strip()
    work["_year"] = work[year_col].map(_extract_year_value) if year_col else ""
    work["_year_sort"] = pd.to_numeric(work["_year"], errors="coerce")
    work["_score"] = pd.to_numeric(work[score_col], errors="coerce")
    max_score = work["_score"].max(skipna=True)
    if pd.notna(max_score) and max_score > 1.0:
        work["_score"] = work["_score"] / 100.0
    work = work[work["_company"].ne("") & work["_score"].notna()]
    if work.empty:
        return (
            _empty_external_labeled_frame(),
            f"{GREENWASHING_SCORE_FILE}: found but no usable company/score rows remained.",
            [GREENWASHING_SCORE_FILE],
        )

    work = work.sort_values(["_company", "_year_sort"], na_position="first")
    latest = work.drop_duplicates(subset=["_company"], keep="last").copy()

    labeled_rows = []
    skipped = 0
    for _, row in latest.iterrows():
        score = float(row["_score"])
        if score >= 0.75:
            label = 1
        elif score <= 0.50:
            label = 0
        else:
            skipped += 1
            continue
        year = str(row["_year"]) if str(row["_year"]).strip() else ""
        company = str(row["_company"])
        score_text = f"{score:.4f}"
        content = (
            f"{company} {year}. Company-level greenwashing score record with GW_SCORE {score_text}. "
            "This file provides a score-derived weak training label only; it does not include full ESG report disclosure text."
        )
        labeled_rows.append(
            {
                "company": company,
                "symbol": "",
                "year": year,
                "content": content,
                "manual_greenwashing_label": int(label),
                "label_method": "Kaggle greenwashing score weak label",
                "label_reason": "GW_SCORE threshold converted into weak binary training label.",
                "source_dataset": GREENWASHING_SCORE_FILE,
                "source_type": "kaggle_greenwashing_score_company_label",
            }
        )

    labeled = pd.DataFrame(labeled_rows) if labeled_rows else _empty_external_labeled_frame()
    high_n = int((labeled["manual_greenwashing_label"] == 1).sum()) if not labeled.empty else 0
    low_n = int((labeled["manual_greenwashing_label"] == 0).sum()) if not labeled.empty else 0
    note = (
        f"{GREENWASHING_SCORE_FILE}: created {len(labeled)} company-level score-label samples "
        f"from latest company years ({high_n} high-risk, {low_n} low-risk); skipped {skipped} ambiguous companies "
        "with 0.50 < GW_SCORE < 0.75. This is company-level score-label augmentation, not disclosure-text augmentation."
    )
    return labeled, note, [GREENWASHING_SCORE_FILE]


def _load_sec_edgar_frame(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return _safe_read_csv(path)


def _inspect_sec_edgar_external_files() -> tuple[pd.DataFrame, str, list[str]]:
    labeled_frames = []
    notes = []
    found_files = []

    metadata_path = os.path.join(KAGGLE_RAW_DIR, SEC_EDGAR_METADATA_FILE)
    if os.path.exists(metadata_path):
        found_files.append(SEC_EDGAR_METADATA_FILE)
        try:
            with open(metadata_path, encoding="utf-8") as f:
                metadata = json.load(f)
            notes.append(
                f"{SEC_EDGAR_METADATA_FILE}: inspected metadata keys {list(metadata.keys())[:8]}; not used as training samples."
            )
        except Exception as exc:
            notes.append(f"{SEC_EDGAR_METADATA_FILE}: found but could not be inspected ({type(exc).__name__}).")

    for filename in SEC_EDGAR_DATASET_FILES:
        path = os.path.join(KAGGLE_RAW_DIR, filename)
        if not os.path.exists(path):
            continue
        found_files.append(filename)
        try:
            raw = _load_sec_edgar_frame(path)
            _, _, _, text_col = _external_key_columns(raw)
            if text_col is None:
                notes.append(
                    f"{filename}: inspected {len(raw)} rows and columns {list(raw.columns)}; metadata/index only, no recognized full filing text column, so no labels were created."
                )
                continue
            aggregated, aggregate_note = _aggregate_external_company_content(raw, filename)
            notes.append(aggregate_note)
            if aggregated.empty:
                continue
            signal_rows = aggregated.apply(_external_disclosure_signal_row, axis=1, result_type="expand")
            labeled, label_note = _assign_external_disclosure_weak_labels(signal_rows, filename)
            notes.append(label_note)
            if not labeled.empty:
                labeled["source_type"] = "sec_edgar_weak_labeled_company"
                labeled_frames.append(labeled)
        except Exception as exc:
            notes.append(f"{filename}: found but could not be processed ({type(exc).__name__}).")

    if labeled_frames:
        combined = pd.concat(labeled_frames, ignore_index=True)
    else:
        combined = _empty_external_labeled_frame()
    note = " ".join(notes) if notes else "No SEC EDGAR raw files were found."
    return combined, note, found_files


def build_optional_kaggle_weak_labeled_company_samples() -> tuple[pd.DataFrame, str, list[str]]:
    if not os.path.isdir(DATA_EXTERNAL_DIR):
        return _empty_external_labeled_frame(), "No data_external/ folder was found for optional Kaggle files.", []

    labeled_frames = []
    notes = []
    found_files = []

    score_labeled, score_note, score_files = _build_greenwashing_score_weak_labels()
    notes.append(score_note)
    found_files.extend(score_files)
    if not score_labeled.empty:
        labeled_frames.append(score_labeled)

    accusation_path = os.path.join(DATA_EXTERNAL_DIR, ACCUSATION_DATASET_FILE)
    if os.path.exists(accusation_path):
        found_files.append(ACCUSATION_DATASET_FILE)
        try:
            aggregated, note = _aggregate_external_company_content(
                _safe_read_csv(accusation_path),
                ACCUSATION_DATASET_FILE,
            )
            notes.append(note)
            if not aggregated.empty:
                accusation_labeled = aggregated.copy()
                accusation_labeled["manual_greenwashing_label"] = 1
                accusation_labeled["label_method"] = "Kaggle accusation-based weak label"
                accusation_labeled["label_reason"] = (
                    "Company appears in a greenwashing accusation dataset or ESG controversy source."
                )
                accusation_labeled["source_type"] = "kaggle_accusation_weak_labeled_company"
                labeled_frames.append(
                    accusation_labeled[
                        [
                            "company",
                            "symbol",
                            "year",
                            "content",
                            "manual_greenwashing_label",
                            "label_method",
                            "label_reason",
                            "source_dataset",
                            "source_type",
                        ]
                    ]
                )
        except Exception as exc:
            notes.append(f"{ACCUSATION_DATASET_FILE}: skipped because it could not be processed ({type(exc).__name__}).")

    for filename in DISCLOSURE_DATASET_FILES:
        path = os.path.join(DATA_EXTERNAL_DIR, filename)
        if not os.path.exists(path):
            continue
        found_files.append(filename)
        try:
            aggregated, note = _aggregate_external_company_content(_safe_read_csv(path), filename)
            notes.append(note)
            if aggregated.empty:
                continue
            signal_rows = aggregated.apply(_external_disclosure_signal_row, axis=1, result_type="expand")
            labeled, label_note = _assign_external_disclosure_weak_labels(signal_rows, filename)
            notes.append(label_note)
            if not labeled.empty:
                labeled_frames.append(labeled)
        except Exception as exc:
            notes.append(f"{filename}: skipped because it could not be processed ({type(exc).__name__}).")

    sec_labeled, sec_note, sec_files = _inspect_sec_edgar_external_files()
    notes.append(sec_note)
    found_files.extend(sec_files)
    if not sec_labeled.empty:
        labeled_frames.append(sec_labeled)

    if labeled_frames:
        combined = pd.concat(labeled_frames, ignore_index=True)
    else:
        combined = _empty_external_labeled_frame()
    note = " ".join(notes) if notes else "No optional Kaggle company-level augmentation files were found."
    return combined, note, found_files


def build_combined_external_weak_labeled_company_samples(dax_labeled: pd.DataFrame) -> tuple[pd.DataFrame, str, list[str]]:
    output_path = os.path.join(DATA_EXTERNAL_DIR, COMBINED_EXTERNAL_LABELED_OUTPUT_FILE)
    os.makedirs(DATA_EXTERNAL_DIR, exist_ok=True)

    required_columns = [
        "company",
        "symbol",
        "year",
        "content",
        "manual_greenwashing_label",
        "label_method",
        "label_reason",
        "source_dataset",
        "source_type",
    ]
    frames = []
    if not dax_labeled.empty:
        dax_copy = dax_labeled.copy()
        if "year" not in dax_copy.columns:
            dax_copy["year"] = ""
        dax_copy["source_dataset"] = DAX_DOCUMENTS_FILE
        frames.append(dax_copy[required_columns])

    kaggle_labeled, kaggle_note, found_files = build_optional_kaggle_weak_labeled_company_samples()
    if not kaggle_labeled.empty:
        frames.append(kaggle_labeled[required_columns])

    if frames:
        combined = pd.concat(frames, ignore_index=True)
    else:
        combined = pd.DataFrame(columns=required_columns)
    combined.to_csv(output_path, index=False)
    note = (
        f"Combined external weak-labeled company samples: {len(combined)} total "
        f"({len(dax_labeled)} DAX, {len(kaggle_labeled)} Kaggle / SEC / score-label samples). "
        f"{kaggle_note}"
    )
    return combined, note, found_files


def _truthy(value) -> bool:
    if pd.isna(value):
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "internal", "report"}


def _document_is_internal(row: pd.Series) -> bool:
    if "internal" in row.index and _truthy(row.get("internal")):
        return True
    joined = " ".join(
        str(row.get(col, ""))
        for col in ["datatype", "domain", "title", "url"]
        if col in row.index
    ).lower()
    internal_terms = [
        "sustainability report",
        "annual report",
        "integrated report",
        "non-financial report",
        "esg report",
        "company report",
        "corporate responsibility",
    ]
    external_terms = ["news", "media", "article", "press", "reuters", "bloomberg", "guardian"]
    if any(term in joined for term in external_terms):
        return False
    return any(term in joined for term in internal_terms)


def _load_sdg_reference_note() -> str:
    path = os.path.join(DATA_EXTERNAL_DIR, SDG_REFERENCE_FILE)
    if not os.path.exists(path):
        return "SDG reference file was not found; built-in ESG keyword lists were used."
    try:
        sdg = pd.read_csv(path, nrows=3)
        return (
            f"SDG reference file found ({SDG_REFERENCE_FILE}) and treated only as keyword context; "
            f"it was not used as company-level training data. Columns: {', '.join(map(str, sdg.columns[:8]))}."
        )
    except Exception as exc:
        return (
            f"SDG reference file was found but not read ({type(exc).__name__}); it was not used as training data."
        )


def _label_dax_company(
    internal_text: str,
    external_text: str,
    company_features: dict[str, float],
) -> tuple[int | None, str]:
    internal_clean = internal_text.lower()
    external_clean = external_text.lower()
    wc = _word_count(internal_clean)
    vague_count = _pattern_count(internal_clean, _compile(VAGUE_PATTERNS))
    specific_count = _pattern_count(internal_clean, _compile(SPECIFIC_PATTERNS))
    controversy_count = _pattern_count(external_clean, _compile(CONTROVERSY_PATTERNS))
    numeric_count = len(re.findall(r"\d+(\.\d+)?", internal_clean))

    many_esg_claims = (
        company_features["talk_total_density"] >= 0.018
        or _keyword_count(internal_clean, E_KEYWORDS + S_KEYWORDS + G_KEYWORDS) >= 8
    )
    vague_or_weak_evidence = (
        vague_count > 0
        or company_features["vague_to_specific_ratio"] >= 1.0
        or specific_count <= 1
        or numeric_count <= 1
        or company_features["pct_numeric"] < 0.006
    )
    external_controversy = controversy_count > 0
    concrete_internal_evidence = (
        specific_count >= 3
        or numeric_count >= 3
        or bool(re.search(r"\b(verified|assurance|audited|baseline|target|scope\s*[123])\b", internal_clean))
    )

    if many_esg_claims and (vague_or_weak_evidence or external_controversy):
        return (
            1,
            "High weak-label risk: internal ESG disclosure is claim-heavy and vague/low-evidence"
            + (" with external controversy terms." if external_controversy else "."),
        )
    if concrete_internal_evidence and not external_controversy:
        return (
            0,
            "Low weak-label risk: internal disclosure contains concrete targets/evidence and no strong external controversy terms.",
        )
    return (
        None,
        "Ambiguous weak-label case: heuristic confidence was not high enough for supervised training.",
    )


def _dax_company_signal_row(
    company: str,
    symbol: str,
    internal_text: str,
    external_text: str,
) -> dict:
    internal_clean = internal_text.lower()
    external_clean = external_text.lower()
    features = extract_talk_features_from_text(internal_text)

    vague_count = _pattern_count(internal_clean, _compile(VAGUE_PATTERNS))
    specific_count = _pattern_count(internal_clean, _compile(SPECIFIC_PATTERNS))
    numeric_count = len(re.findall(r"\d+(\.\d+)?", internal_clean))

    severe_patterns = [
        r"\bscandal(s)?\b",
        r"\blawsuit(s)?\b",
        r"\binvestigation(s)?\b",
        r"\ballegation(s)?\b",
        r"\bpollution\b",
        r"\bviolation(s)?\b",
        r"\bgreenwashing\b",
        r"\bcontrovers(y|ies)\b",
        r"\bfraud\b",
        r"\bfine(s)?\b",
        r"\bpenalt(y|ies)\b",
        r"\bmisleading\b",
        r"\bbreach\b",
        r"\bnon[-\s]?compliance\b",
    ]
    emissions_patterns = [r"\bemissions\b"]
    severe_count = _pattern_count(external_clean, _compile(severe_patterns))
    emissions_count = _pattern_count(external_clean, _compile(emissions_patterns))

    return {
        "company": company,
        "symbol": symbol,
        "content": internal_text,
        "talk_total_density": features["talk_total_density"],
        "pct_numeric": features["pct_numeric"],
        "vague_to_specific_ratio": features["vague_to_specific_ratio"],
        "vague_count": vague_count,
        "specific_count": specific_count,
        "numeric_count": numeric_count,
        "severe_controversy_count": severe_count,
        "emissions_controversy_count": emissions_count,
        "controversy_score": severe_count + 0.25 * emissions_count,
        "concrete_evidence_score": specific_count + numeric_count,
    }


def _assign_dax_weak_labels(candidates: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    if candidates.empty:
        return candidates, "No DAX company candidates were available for weak labeling."

    work = candidates.copy()
    n = len(work)
    target_per_class = min(max(10, n // 3), max(1, n // 2))

    work["talk_rank"] = work["talk_total_density"].rank(pct=True, method="average")
    work["vague_rank"] = work["vague_to_specific_ratio"].rank(pct=True, method="average")
    work["weak_numeric_rank"] = (-work["pct_numeric"]).rank(pct=True, method="average")
    work["controversy_rank"] = work["controversy_score"].rank(pct=True, method="average")
    work["evidence_rank"] = work["concrete_evidence_score"].rank(pct=True, method="average")
    work["pct_numeric_rank"] = work["pct_numeric"].rank(pct=True, method="average")

    work["high_risk_weak_label_score"] = (
        0.35 * work["talk_rank"]
        + 0.25 * work["vague_rank"]
        + 0.25 * work["controversy_rank"]
        + 0.15 * work["weak_numeric_rank"]
    )
    work["low_risk_weak_label_score"] = (
        0.45 * work["evidence_rank"]
        + 0.25 * work["pct_numeric_rank"]
        + 0.20 * (1 - work["controversy_rank"])
        + 0.10 * (1 - work["vague_rank"])
    )

    high_idx = set(
        work.sort_values("high_risk_weak_label_score", ascending=False)
        .head(target_per_class)
        .index
    )
    low_idx = set(
        work.drop(index=list(high_idx))
        .sort_values("low_risk_weak_label_score", ascending=False)
        .head(target_per_class)
        .index
    )

    labeled_rows = []
    for idx, row in work.iterrows():
        if idx in high_idx:
            label = 1
            reason = (
                "High weak-label risk: relatively high ESG claim density, vague/weak numerical evidence, "
                f"and/or higher external controversy signal (risk score={row['high_risk_weak_label_score']:.3f})."
            )
        elif idx in low_idx:
            label = 0
            reason = (
                "Low weak-label risk: relatively concrete numerical targets/evidence and lower controversy signal "
                f"(low-risk score={row['low_risk_weak_label_score']:.3f})."
            )
        else:
            continue
        labeled_rows.append(
            {
                "company": row["company"],
                "symbol": row["symbol"],
                "year": row.get("year", ""),
                "content": row["content"],
                "manual_greenwashing_label": int(label),
                "label_method": "AI-assisted heuristic/manual weak label",
                "label_reason": reason,
                "source_type": "dax_weak_labeled_company",
            }
        )

    labeled = pd.DataFrame(labeled_rows)
    high_n = int((labeled["manual_greenwashing_label"] == 1).sum()) if not labeled.empty else 0
    low_n = int((labeled["manual_greenwashing_label"] == 0).sum()) if not labeled.empty else 0
    skipped_n = n - len(labeled)
    note = (
        f"DAX distribution-aware weak labeling produced {high_n} high-risk and {low_n} low-risk "
        f"company samples; {skipped_n} ambiguous companies were skipped."
    )
    if high_n < 10 or low_n < 10:
        note += " Preferred 10/10 class coverage was not reached because the DAX heuristic lacked enough confident candidates."
    return labeled, note


def build_dax_weak_labeled_company_samples() -> tuple[pd.DataFrame, str]:
    os.makedirs(DATA_EXTERNAL_DIR, exist_ok=True)
    output_path = os.path.join(DATA_EXTERNAL_DIR, DAX_LABELED_OUTPUT_FILE)
    required_columns = [
        "company",
        "symbol",
        "content",
        "manual_greenwashing_label",
        "label_method",
        "label_reason",
        "source_type",
    ]
    dax_path = os.path.join(DATA_EXTERNAL_DIR, DAX_DOCUMENTS_FILE)
    sdg_note = _load_sdg_reference_note()
    if not os.path.exists(dax_path):
        empty = pd.DataFrame(columns=required_columns)
        empty.to_csv(output_path, index=False)
        return empty, f"DAX source file was not found; wrote empty {DAX_LABELED_OUTPUT_FILE}. {sdg_note}"

    try:
        csv.field_size_limit(sys.maxsize)
    except OverflowError:
        csv.field_size_limit(2**31 - 1)

    try:
        raw = pd.read_csv(dax_path, sep="|", engine="python")
    except csv.Error:
        raw = pd.read_csv(dax_path, sep="|", engine="python", on_bad_lines="skip")
    if "content" not in raw.columns:
        empty = pd.DataFrame(columns=required_columns)
        empty.to_csv(output_path, index=False)
        return empty, f"DAX file found but required content column was missing; wrote empty label file. {sdg_note}"

    work = raw.copy()
    for col in ["company", "symbol", "datatype", "date", "domain", "esg_topics", "internal", "title", "url"]:
        if col not in work.columns:
            work[col] = ""
    work["content"] = work["content"].fillna("").astype(str)
    work = work[work["content"].str.strip().ne("")]
    if work.empty:
        empty = pd.DataFrame(columns=required_columns)
        empty.to_csv(output_path, index=False)
        return empty, f"DAX file had no non-empty content rows; wrote empty label file. {sdg_note}"

    work["_company_key"] = work["symbol"].fillna("").astype(str).str.strip()
    missing_symbol = work["_company_key"].eq("")
    work.loc[missing_symbol, "_company_key"] = work.loc[missing_symbol, "company"].fillna("").astype(str).str.strip()
    work["_is_internal"] = work.apply(_document_is_internal, axis=1)

    candidate_rows = []
    for _, group in work.groupby("_company_key", dropna=False):
        company = str(group["company"].dropna().astype(str).replace("", np.nan).dropna().iloc[0]) if group["company"].astype(str).str.strip().ne("").any() else ""
        symbol = str(group["symbol"].dropna().astype(str).replace("", np.nan).dropna().iloc[0]) if group["symbol"].astype(str).str.strip().ne("").any() else ""
        internal_docs = group[group["_is_internal"]]
        external_docs = group[~group["_is_internal"]]
        internal_text = " ".join(internal_docs["content"].astype(str).tolist())
        external_text = " ".join(external_docs["content"].astype(str).tolist())
        if not internal_text.strip():
            internal_text = " ".join(group["content"].astype(str).tolist())

        candidate_rows.append(
            _dax_company_signal_row(company, symbol, internal_text, external_text)
        )

    candidates = pd.DataFrame(candidate_rows)
    labeled, label_note = _assign_dax_weak_labels(candidates)
    if labeled.empty:
        labeled = pd.DataFrame(columns=required_columns)
    else:
        labeled = labeled[required_columns]
    labeled.to_csv(output_path, index=False)
    return (
        labeled,
        f"DAX weak-label generation created {len(labeled)} company-level training samples from {DAX_DOCUMENTS_FILE}. {label_note} {sdg_note}",
    )


def load_dax_labeled_company_samples(
    feature_columns: list[str],
    train_medians: pd.Series,
) -> tuple[pd.DataFrame, pd.Series, str]:
    labeled, note = build_dax_weak_labeled_company_samples()
    if labeled.empty:
        return (
            pd.DataFrame(columns=feature_columns + ["source_type"]),
            pd.Series(dtype=int),
            note,
        )

    feature_rows = labeled["content"].map(extract_talk_features_from_text).apply(pd.Series)
    X_dax = _make_augmented_feature_frame(
        feature_rows,
        feature_columns,
        train_medians,
        "dax_weak_labeled_company",
    )
    y_dax = labeled["manual_greenwashing_label"].astype(int).reset_index(drop=True)
    return X_dax, y_dax, note


def _model_pipeline() -> ImbPipeline:
    return ImbPipeline(
        steps=[
            ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=3)),
            (
                "clf",
                LGBMClassifier(
                    n_estimators=120,
                    learning_rate=0.04,
                    num_leaves=15,
                    min_child_samples=5,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    verbosity=-1,
                ),
            ),
        ]
    )


def evaluate_holdout(
    y_true: pd.Series,
    proba: np.ndarray,
    model_version: str,
    train_real_n: int,
    train_synthetic_n: int,
    train_dax_labeled_n: int,
    train_external_labeled_n: int,
) -> dict:
    pred = (proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "model_version": model_version,
        "train_real_n": int(train_real_n),
        "train_synthetic_n": int(train_synthetic_n),
        "train_dax_labeled_n": int(train_dax_labeled_n),
        "train_external_labeled_n": int(train_external_labeled_n),
        "holdout_n_real_companies": int(len(y_true)),
        "holdout_positive_count": int(np.sum(y_true == 1)),
        "holdout_negative_count": int(np.sum(y_true == 0)),
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "f1": f1_score(y_true, pred, zero_division=0),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "confusion_matrix": f"[[{int(tn)}, {int(fp)}], [{int(fn)}, {int(tp)}]]",
        "evaluation_note": "Evaluated only on the real-company holdout set; no synthetic samples in holdout.",
    }


def _http_get(url: str, timeout: int = 20) -> str:
    req = Request(url, headers={"User-Agent": SEC_USER_AGENT, "Accept-Encoding": "identity"})
    with urlopen(req, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def _strip_html(document: str) -> str:
    document = re.sub(r"<script.*?</script>", " ", document, flags=re.IGNORECASE | re.DOTALL)
    document = re.sub(r"<style.*?</style>", " ", document, flags=re.IGNORECASE | re.DOTALL)
    document = re.sub(r"<[^>]+>", " ", document)
    document = html.unescape(document)
    return re.sub(r"\s+", " ", document)


def fetch_latest_nvidia_10k_text() -> tuple[str, str, str]:
    cik = "0001045810"
    submissions_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    try:
        import json

        submissions = json.loads(_http_get(submissions_url))
        recent = submissions["filings"]["recent"]
        forms = recent["form"]
        accession_numbers = recent["accessionNumber"]
        primary_docs = recent["primaryDocument"]
        for form, accession, primary_doc in zip(forms, accession_numbers, primary_docs):
            if form == "10-K":
                accession_clean = accession.replace("-", "")
                cik_clean = str(int(cik))
                filing_url = (
                    f"https://www.sec.gov/Archives/edgar/data/{cik_clean}/"
                    f"{accession_clean}/{primary_doc}"
                )
                return (
                    _strip_html(_http_get(filing_url)),
                    filing_url,
                    "Fetched latest NVIDIA 10-K from SEC EDGAR submissions API.",
                )
        raise ValueError("No recent NVIDIA 10-K found in SEC submissions feed.")
    except (URLError, TimeoutError, ValueError, KeyError, OSError) as exc:
        fallback = (
            "NVIDIA discusses climate, energy efficiency, supply chain responsibility, governance, "
            "risk management, emissions, renewable energy, employee inclusion, board oversight, "
            "and compliance in its public annual filing. This fallback excerpt is used only when "
            "SEC EDGAR cannot be reached during local execution."
        )
        return fallback, submissions_url, f"Fallback excerpt used because SEC EDGAR retrieval failed: {type(exc).__name__}."


def run_experiment() -> tuple[pd.DataFrame, pd.DataFrame]:
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    load_merged_dataset, build_feature_matrix = _load_project_modules()
    df = load_merged_dataset(verbose=False)
    try:
        X, y, _ = build_feature_matrix(df, feature_set="baseline", verbose=False)
    except TypeError:
        X, y, _ = build_feature_matrix(df)
    feature_columns = list(X.columns)
    X_model = X[feature_columns].copy()
    y = pd.Series(y).astype(int)

    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X_model,
        y,
        test_size=0.30,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    train_medians = X_train.median(numeric_only=True)
    train_real_n = len(X_train)

    original_model = _model_pipeline()
    original_model.fit(X_train, y_train)
    original_proba = original_model.predict_proba(X_holdout)[:, 1]

    synthetic = synthetic_disclosures()
    X_synthetic = _make_augmented_feature_frame(
        synthetic[TEXT_FEATURES],
        feature_columns,
        train_medians,
        "synthetic_disclosure",
    ).drop(columns=["source_type"])
    X_aug = pd.concat([X_train, X_synthetic], axis=0, ignore_index=True)
    y_aug = pd.concat([y_train.reset_index(drop=True), synthetic["label"].astype(int)], axis=0, ignore_index=True)
    augmented_model = _model_pipeline()
    augmented_model.fit(X_aug, y_aug)
    augmented_proba = augmented_model.predict_proba(X_holdout)[:, 1]

    X_dax, y_dax, dax_note = load_dax_labeled_company_samples(feature_columns, train_medians)
    train_dax_labeled_n = int(len(y_dax))

    if train_dax_labeled_n > 0:
        X_dax_model = X_dax[feature_columns].copy()
        X_aug_dax = pd.concat([X_aug, X_dax_model], axis=0, ignore_index=True)
        y_aug_dax = pd.concat([y_aug, y_dax.reset_index(drop=True)], axis=0, ignore_index=True)
        dax_model = _model_pipeline()
        dax_model.fit(X_aug_dax, y_aug_dax)
        dax_proba = dax_model.predict_proba(X_holdout)[:, 1]
    else:
        X_aug_dax = X_aug
        y_aug_dax = y_aug
        dax_model = augmented_model
        dax_proba = augmented_proba

    dax_labeled_path = os.path.join(DATA_EXTERNAL_DIR, DAX_LABELED_OUTPUT_FILE)
    dax_labeled = pd.read_csv(dax_labeled_path) if os.path.exists(dax_labeled_path) else pd.DataFrame()
    combined_external, external_note, external_files_found = build_combined_external_weak_labeled_company_samples(
        dax_labeled
    )
    train_external_labeled_n = int(len(combined_external))

    if train_external_labeled_n > 0:
        external_feature_rows = combined_external["content"].map(extract_talk_features_from_text).apply(pd.Series)
        X_external = _make_augmented_feature_frame(
            external_feature_rows,
            feature_columns,
            train_medians,
            "combined_external_weak_labeled_company",
        )[feature_columns]
        y_external = combined_external["manual_greenwashing_label"].astype(int).reset_index(drop=True)
        X_aug_external = pd.concat([X_aug, X_external], axis=0, ignore_index=True)
        y_aug_external = pd.concat([y_aug, y_external], axis=0, ignore_index=True)
        external_model = _model_pipeline()
        external_model.fit(X_aug_external, y_aug_external)
        external_proba = external_model.predict_proba(X_holdout)[:, 1]
    else:
        external_model = augmented_model
        external_proba = augmented_proba

    results = pd.DataFrame(
        [
            evaluate_holdout(
                y_holdout.reset_index(drop=True),
                original_proba,
                "original_real_training_only",
                train_real_n,
                0,
                0,
                0,
            ),
            evaluate_holdout(
                y_holdout.reset_index(drop=True),
                augmented_proba,
                "augmented_training_with_synthetic_disclosures",
                train_real_n,
                len(synthetic),
                0,
                0,
            ),
            evaluate_holdout(
                y_holdout.reset_index(drop=True),
                dax_proba,
                "augmented_training_with_synthetic_and_dax_labeled_company_samples",
                train_real_n,
                len(synthetic),
                train_dax_labeled_n,
                train_dax_labeled_n,
            ),
            evaluate_holdout(
                y_holdout.reset_index(drop=True),
                external_proba,
                "augmented_training_with_synthetic_and_all_external_company_samples",
                train_real_n,
                len(synthetic),
                train_dax_labeled_n,
                train_external_labeled_n,
            ),
        ]
    )
    results = results.round(
        {
            "roc_auc": 4,
            "pr_auc": 4,
            "f1": 4,
            "precision": 4,
            "recall": 4,
        }
    )
    results.to_csv(os.path.join(OUTPUTS_DIR, "holdout_robustness_results.csv"), index=False)
    synthetic_pr = float(
        results.loc[
            results["model_version"] == "augmented_training_with_synthetic_disclosures",
            "pr_auc",
        ].iloc[0]
    )
    dax_pr = float(
        results.loc[
            results["model_version"] == "augmented_training_with_synthetic_and_dax_labeled_company_samples",
            "pr_auc",
        ].iloc[0]
    )
    if dax_pr < synthetic_pr:
        dax_interpretation = (
            "DAX augmentation did not improve holdout PR-AUC versus synthetic-only augmentation. "
            "This is reported honestly as a limitation of weak-label external augmentation."
        )
    elif dax_pr > synthetic_pr:
        dax_interpretation = (
            "DAX augmentation improved holdout PR-AUC versus synthetic-only augmentation in this run."
        )
    else:
        dax_interpretation = (
            "DAX augmentation matched synthetic-only holdout PR-AUC in this run."
        )
    external_row = results[
        results["model_version"] == "augmented_training_with_synthetic_and_all_external_company_samples"
    ].iloc[0]
    if float(external_row["pr_auc"]) < synthetic_pr or float(external_row["f1"]) < float(
        results.loc[
            results["model_version"] == "augmented_training_with_synthetic_disclosures",
            "f1",
        ].iloc[0]
    ):
        external_interpretation = (
            "All-external augmentation did not improve PR-AUC or F1 versus synthetic-only augmentation. "
            "This is reported honestly as a limitation of weak-label external company augmentation."
        )
    else:
        external_interpretation = (
            "All-external augmentation did not reduce PR-AUC or F1 versus synthetic-only augmentation in this run."
        )

    case_text, source_url, retrieval_note = fetch_latest_nvidia_10k_text()
    case_features = pd.DataFrame([extract_talk_features_from_text(case_text)])[TEXT_FEATURES]
    case_full = _make_augmented_feature_frame(
        case_features,
        feature_columns,
        train_medians,
        "external_case_study",
    )[feature_columns]
    case_score = float(external_model.predict_proba(case_full)[:, 1][0])
    case = pd.DataFrame(
        [
            {
                "company": "NVIDIA",
                "cik": "0001045810",
                "source_url": source_url,
                "case_study_model": "augmented_training_with_synthetic_and_all_external_company_samples",
                "model_flagged_greenwashing_risk_score": round(case_score, 4),
                "retrieval_note": retrieval_note,
                "case_study_note": "External company case study only; not formal proof or legal determination.",
                "governance_disclaimer": GOVERNANCE_DISCLAIMER,
            }
        ]
    )
    case.to_csv(os.path.join(OUTPUTS_DIR, "external_company_case_study_nvidia.csv"), index=False)

    with open(os.path.join(OUTPUTS_DIR, "holdout_robustness_summary.md"), "w", encoding="utf-8") as f:
        f.write(
            f"""# Holdout Robustness Validation

## Design

- Split real companies before model fitting: 70% real training companies and 30% real holdout companies.
- The holdout set is not used in Optuna tuning, model selection, synthetic augmentation, or threshold selection.
- Synthetic ESG disclosure samples are added only to the training set.
- Professor suggested increasing company-level training samples.
- DAX company documents are converted into weak-labeled training augmentation samples when the DAX dataset is available.
- Greenwashing_Score_Data.xlsx is used to add company-level score-label weak samples when available.
- SEC EDGAR files are inspected; they are used only if full filing or risk-factor text is available.
- Additional Kaggle-style ESG / greenwashing files are converted into weak-labeled company samples when available.
- DAX, SEC EDGAR, and additional Kaggle samples are weak-labeled training augmentation samples.
- Weak labels are not legal ground truth.
- DAX, SEC EDGAR, and Kaggle samples are used only for training augmentation.
- The model is evaluated only on real holdout companies.
- No synthetic, DAX, SEC EDGAR, or Kaggle samples enter the test set.
- Walk / External / Context fields for synthetic, DAX, SEC EDGAR, or Kaggle samples are filled with real training-set medians; the real holdout set remains unchanged.
- Greenwashing score records without report text use conservative content placeholders and are reported as score-label augmentation, not disclosure-text augmentation.
- The output is a risk screening signal, not a legal determination of greenwashing.

## DAX Weak-Label Augmentation Status

{dax_note}

## DAX Augmentation Interpretation

{dax_interpretation}

## Additional External Augmentation Status

- External files found: {", ".join(external_files_found) if external_files_found else "None"}
- {external_note}

## All-External Augmentation Interpretation

{external_interpretation}

## Holdout Results

{results.to_markdown(index=False)}

## External Case Study

{case.to_markdown(index=False)}

## Governance Disclaimer

{GOVERNANCE_DISCLAIMER}
"""
        )

    print("\n=== Holdout Robustness Results ===")
    print(results.to_string(index=False))
    print("\n=== DAX Weak-Label Augmentation Status ===")
    print(dax_note)
    print("\n=== Additional External Augmentation Status ===")
    print(f"External files found: {', '.join(external_files_found) if external_files_found else 'None'}")
    print(external_note)
    print("\n=== External Company Case Study ===")
    print(case.to_string(index=False))
    print(f"\n{GOVERNANCE_DISCLAIMER}")
    return results, case


if __name__ == "__main__":
    run_experiment()
