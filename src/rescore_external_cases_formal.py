"""
Precompute external SEC 10-K inference examples with the formal final scorer.

No model retraining is performed. The script fetches latest 10-K filings via
SEC submissions API, extracts Talk features through src.feature_engineering as
the single source of truth, fills Walk / External / Context fields with formal
project medians, and scores with models/lgbm_best.pkl.
"""

from __future__ import annotations

import html
import json
import os
import pickle
import re
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS = os.path.join(PROJECT_ROOT, "outputs")
MODELS = os.path.join(PROJECT_ROOT, "models")
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
SEC_USER_AGENT = "DSF504 ESG Greenwashing academic project; contact: student@example.com"
CASE_STUDY_MODEL = "lgbm_best (formal final scorer)"
BATCH_TICKERS = [
    "NVDA",
    "AAPL",
    "IBM",
    "MSFT",
    "GOOGL",
    "AMZN",
    "META",
    "TSLA",
    "XOM",
    "JPM",
    "WMT",
    "KO",
    "PFE",
    "NKE",
    "DIS",
]
CIK_BY_TICKER = {
    "NVDA": "0001045810",
    "AAPL": "0000320193",
    "IBM": "0000051143",
    "MSFT": "0000789019",
    "GOOGL": "0001652044",
    "AMZN": "0001018724",
    "META": "0001326801",
    "TSLA": "0001318605",
    "XOM": "0000034088",
    "JPM": "0000019617",
    "WMT": "0000104169",
    "KO": "0000021344",
    "PFE": "0000078003",
    "NKE": "0000320187",
    "DIS": "0001744489",
}
COMPANY_BY_TICKER = {
    "NVDA": "NVIDIA",
    "AAPL": "Apple",
    "IBM": "IBM",
    "MSFT": "Microsoft",
    "GOOGL": "Alphabet",
    "AMZN": "Amazon",
    "META": "Meta Platforms",
    "TSLA": "Tesla",
    "XOM": "Exxon Mobil",
    "JPM": "JPMorgan Chase",
    "WMT": "Walmart",
    "KO": "Coca-Cola",
    "PFE": "Pfizer",
    "NKE": "Nike",
    "DIS": "Disney",
}


def _strip_html(document: str) -> str:
    document = re.sub(r"<script.*?</script>", " ", document, flags=re.IGNORECASE | re.DOTALL)
    document = re.sub(r"<style.*?</style>", " ", document, flags=re.IGNORECASE | re.DOTALL)
    document = re.sub(r"<[^>]+>", " ", document)
    document = html.unescape(document)
    return re.sub(r"\s+", " ", document)


def _http_get(url: str, timeout: int = 45) -> str:
    request = Request(url, headers={"User-Agent": SEC_USER_AGENT, "Accept-Encoding": "identity"})
    with urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def _format_cik(value) -> str:
    raw = "" if pd.isna(value) else str(value).strip()
    if raw.endswith(".0"):
        raw = raw[:-2]
    digits = "".join(ch for ch in raw if ch.isdigit())
    return digits.zfill(10) if digits else raw


def _fetch_latest_10k(ticker: str, cik: str) -> tuple[str, str, str, str]:
    cik = _format_cik(cik)
    submissions_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    submissions = json.loads(_http_get(submissions_url))
    recent = submissions.get("filings", {}).get("recent", {})
    names = submissions.get("name", COMPANY_BY_TICKER.get(ticker, ticker))
    for form, accession, primary_doc in zip(
        recent.get("form", []),
        recent.get("accessionNumber", []),
        recent.get("primaryDocument", []),
    ):
        if str(form).upper() == "10-K":
            accession_clean = str(accession).replace("-", "")
            cik_clean = str(int(cik))
            filing_url = f"https://www.sec.gov/Archives/edgar/data/{cik_clean}/{accession_clean}/{primary_doc}"
            filing_text = _strip_html(_http_get(filing_url))
            return str(names), filing_url, filing_text, f"Fetched latest {ticker} 10-K from SEC EDGAR submissions API."
    raise ValueError(f"No 10-K found in SEC submissions feed for {ticker}.")


def _load_project_data() -> tuple[pd.DataFrame, pd.Series]:
    from src.data_loader import load_merged_dataset
    from src.feature_engineering import build_feature_matrix

    df = load_merged_dataset(verbose=False)
    try:
        X, y, meta = build_feature_matrix(df, feature_set="baseline", verbose=False)
    except TypeError:
        X, y, meta = build_feature_matrix(df, verbose=False)
    return X[FEATURE_ORDER].copy(), pd.Series(y).astype(int)


def _formal_talk_features(text: str) -> dict[str, float]:
    from src.feature_engineering import extract_talk_features_from_text

    return extract_talk_features_from_text(text)


def _model_score(model, feature_vector: dict[str, float]) -> float:
    row = pd.DataFrame([[feature_vector[feature] for feature in FEATURE_ORDER]], columns=FEATURE_ORDER)
    return float(model.predict_proba(row)[0, 1])


def _feature_vector(talk_features: dict[str, float], medians: pd.Series) -> dict[str, float]:
    vector = {feature: float(medians.get(feature, 0.0)) for feature in FEATURE_ORDER}
    for feature in TEXT_FEATURES:
        vector[feature] = float(talk_features.get(feature, vector[feature]))
    return vector


def _print_talk_sanity(talk_rows: dict[str, dict[str, float]], formal_X: pd.DataFrame) -> None:
    features = ["E_density", "S_density", "G_density", "hedge_score", "pct_numeric", "talk_total_density", "vague_to_specific_ratio", "text_length", "esg_topic_breadth"]
    stats = formal_X[features].quantile([0.25, 0.5, 0.75]).T.rename(columns={0.25: "q25", 0.5: "median", 0.75: "q75"})
    print("\nTalk-feature sanity check vs formal dataset quartiles")
    for ticker in ["NVDA", "AAPL", "IBM"]:
        if ticker not in talk_rows:
            continue
        print(f"\n{ticker}")
        for feature in features:
            value = float(talk_rows[ticker].get(feature, np.nan))
            q25 = float(stats.loc[feature, "q25"])
            median = float(stats.loc[feature, "median"])
            q75 = float(stats.loc[feature, "q75"])
            print(f"  {feature:28s} value={value:14.6f}  q25={q25:14.6f}  median={median:14.6f}  q75={q75:14.6f}")


def main() -> None:
    os.makedirs(OUTPUTS, exist_ok=True)
    formal_X, y = _load_project_data()
    medians = formal_X.median(numeric_only=True).reindex(FEATURE_ORDER).fillna(0.0)
    with open(os.path.join(MODELS, "lgbm_best.pkl"), "rb") as f:
        model = pickle.load(f)

    rows = []
    failures = []
    talk_rows = {}
    for ticker in BATCH_TICKERS:
        try:
            cik = CIK_BY_TICKER[ticker]
            company, filing_url, filing_text, retrieval_note = _fetch_latest_10k(ticker, cik)
            talk_features = _formal_talk_features(filing_text)
            talk_rows[ticker] = talk_features
            vector = _feature_vector(talk_features, medians)
            score = _model_score(model, vector)
            rows.append(
                {
                    "company": company or COMPANY_BY_TICKER.get(ticker, ticker),
                    "ticker": ticker,
                    "cik": _format_cik(cik),
                    "source": "SEC EDGAR 10-K",
                    "filing_type": "10-K",
                    "filing_url": filing_url,
                    "case_study_model": CASE_STUDY_MODEL,
                    "model_flagged_greenwashing_risk_score": round(score, 10),
                    "retrieval_note": retrieval_note + " Scored with formal lgbm_best.pkl; no model retraining performed.",
                    "case_note": "External SEC 10-K inference example only; not formal model evaluation and not a legal greenwashing determination.",
                }
            )
        except (KeyError, ValueError, HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
            failures.append((ticker, type(exc).__name__, str(exc)))

    out = pd.DataFrame(rows)
    if out.empty:
        raise RuntimeError("No SEC 10-K external cases were successfully scored.")
    out = out.sort_values(["company", "ticker"]).reset_index(drop=True)
    output_path = os.path.join(OUTPUTS, "external_company_case_studies.csv")
    out.to_csv(output_path, index=False)

    print("Formal external SEC 10-K batch scores")
    for _, row in out.sort_values("ticker").iterrows():
        print(f"{row['ticker']:5s} {row['company']}: {float(row['model_flagged_greenwashing_risk_score']):.10f}")
    _print_talk_sanity(talk_rows, formal_X)
    if failures:
        print("\nSkipped / failed tickers")
        for ticker, err_type, msg in failures:
            print(f"{ticker}: {err_type}: {msg}")
    else:
        print("\nSkipped / failed tickers: none")
    print(f"\nWrote {output_path}")


if __name__ == "__main__":
    main()
