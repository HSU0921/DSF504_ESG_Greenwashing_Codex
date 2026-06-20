"""
app_annual_report_style.py — Annual Report Style ESG Dashboard
==============================================================
Run:
    streamlit run app_annual_report_style.py --server.port 8503

This is a presentation-ready Streamlit version of the DSF504 ESG
Greenwashing Risk Scoring dashboard. It reuses existing project data,
model artifacts, prediction feature order, and SHAP outputs without
modifying the stable app.py.
"""

import base64
import html
import os
import pickle
import sys
import tempfile
import textwrap
import warnings
from io import BytesIO
from datetime import date, datetime
from functools import lru_cache

ROOT = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "dsf504_matplotlib"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import confusion_matrix, precision_score, recall_score

warnings.filterwarnings("ignore")

OUTPUTS = os.path.join(ROOT, "outputs")
MODELS = os.path.join(ROOT, "models")
REPORTS = os.path.join(ROOT, "reports")
DATA_EXTERNAL = os.path.join(ROOT, "data_external")
ASSETS = os.path.join(ROOT, "assets")
PDF_FONT_DIR = os.path.join(ASSETS, "fonts")
sys.path.insert(0, ROOT)

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

TALK_FEATURES = [
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
WALK_FEATURES = [
    "e_s_g_dispersion",
    "sector_rank",
    "planet_friendly_business",
    "honest_fair_business",
]
CONTEXT_FEATURES = ["sector_encoded"]

FEATURE_DISPLAY = {
    "sector_rank": "Sector-relative ESG Risk",
    "talk_total_density": "ESG Disclosure Intensity",
    "e_s_g_dispersion": "ESG Pillar Imbalance",
    "pct_numeric": "Quantitative Disclosure Ratio",
    "vague_to_specific_ratio": "Disclosure Specificity Ratio",
    "planet_friendly_business": "External Environmental Signal",
    "honest_fair_business": "External Governance / Ethics Signal",
    "E_density": "Environmental Disclosure Density",
    "S_density": "Social Disclosure Density",
    "G_density": "Governance Disclosure Density",
    "hedge_score": "Hedging Language Score",
    "text_length": "Report Length",
    "sector_encoded": "Industry Sector",
    "esg_topic_breadth": "ESG Topic Coverage",
}

FEATURE_BUSINESS_MEANING = {
    "sector_rank": "Compares ESG risk within the same sector.",
    "talk_total_density": "Measures the intensity of ESG disclosure language.",
    "e_s_g_dispersion": "Captures imbalance across E, S, and G risk signals.",
    "pct_numeric": "Shows whether claims include measurable evidence.",
    "vague_to_specific_ratio": "Compares vague claims with specific, measurable disclosure.",
    "planet_friendly_business": "External signal related to environmental credibility.",
    "honest_fair_business": "External signal related to governance and ethics.",
    "E_density": "Environmental disclosure emphasis in company reports.",
    "S_density": "Social disclosure emphasis in company reports.",
    "G_density": "Governance disclosure emphasis in company reports.",
    "hedge_score": "Hedging language that may require supporting evidence.",
    "text_length": "Report length and disclosure volume.",
    "sector_encoded": "Industry sector context used by the model.",
    "esg_topic_breadth": "Coverage across ESG disclosure topics.",
}

SECTORS = [
    "Basic Materials",
    "Communication Services",
    "Consumer Cyclical",
    "Consumer Defensive",
    "Energy",
    "Financial Services",
    "Healthcare",
    "Industrials",
    "Real Estate",
    "Technology",
    "Utilities",
]
SECTOR_ENC = {sector: i for i, sector in enumerate(SECTORS)}

CONTROVERSY_ORDER = [
    "None Controversy Level",
    "Low Controversy Level",
    "Moderate Controversy Level",
    "Significant Controversy Level",
    "High Controversy Level",
    "Severe Controversy Level",
]
CONTROVERSY_SHORT = {
    "None Controversy Level": "None",
    "Low Controversy Level": "Low",
    "Moderate Controversy Level": "Moderate",
    "Significant Controversy Level": "Significant",
    "High Controversy Level": "High",
    "Severe Controversy Level": "Severe",
}

CVX_DEFAULTS = {
    "E_density": 32.0687,
    "S_density": 20.5188,
    "G_density": 11.1627,
    "hedge_score": 0.3226,
    "pct_numeric": 0.0,
    "talk_total_density": 63.7502,
    "vague_to_specific_ratio": 322622.27,
    "text_length": 15498.0,
    "esg_topic_breadth": 3.0,
    "e_s_g_dispersion": 4.20,
    "sector_rank": 0.8000,
    "planet_friendly_business": -40.0,
    "honest_fair_business": -40.0,
    "sector_encoded": float(SECTOR_ENC["Energy"]),
    "sector": "Energy",
    "company": "Chevron Corporation",
    "ticker": "CVX",
}

EXTERNAL_VALIDITY_EVIDENCE = {
    "CVX": {
        "en": "FTC greenwashing complaint (Global Witness/Earthworks/Greenpeace) — first against an oil major.",
        "zh": "Global Witness、Earthworks、Greenpeace 向 FTC 提出漂綠申訴，為首件針對石油巨頭的案例。",
    },
    "WFC": {
        "en": "NGO greenwashing accusations over fossil-fuel financing; shareholder litigation over diversity claims.",
        "zh": "NGO 指控化石燃料融資與永續形象不一致；另有股東針對多元化聲明提起訴訟。",
    },
    "TSLA": {
        "en": "Removed from S&P 500 ESG Index (2022); widely accused of exaggerated environmental claims.",
        "zh": "2022 年遭 S&P 500 ESG Index 移除，並被廣泛質疑環境主張過度誇大。",
    },
    "AMZN": {
        "en": "Climate Pledge criticized as greenwashing; emissions rose ~40%; Shipment Zero dropped.",
        "zh": "Climate Pledge 被批評為漂綠；排放量上升約 40%，Shipment Zero 目標也被撤下。",
    },
    "AAPL": {
        "en": "Sued over 'carbon neutral' Apple Watch claims; German court barred the CO2-neutral ad.",
        "zh": "因 Apple Watch「碳中和」主張遭訴；德國法院禁止 CO2-neutral 廣告。",
    },
    "CB": {
        "en": "Broke its own coal-underwriting pledge by reinsuring a Vietnam coal plant.",
        "zh": "因再保越南燃煤電廠，被指違反自身煤炭承保承諾。",
    },
    "PNC": {
        "en": "'Sustainable' image vs ~$108B fossil-fuel financing; shareholder climate-risk proposals.",
        "zh": "永續形象與約 1,080 億美元化石燃料融資形成落差；股東提出氣候風險相關提案。",
    },
}

FALLBACK_MEDIANS = {
    "E_density": 14.51,
    "S_density": 27.54,
    "G_density": 11.71,
    "hedge_score": 0.46,
    "pct_numeric": 0.0,
    "talk_total_density": 55.37,
    "vague_to_specific_ratio": 461947.11,
    "text_length": 12471.0,
    "esg_topic_breadth": 3.0,
    "e_s_g_dispersion": 3.52,
    "sector_rank": 0.526,
    "planet_friendly_business": -30.0,
    "honest_fair_business": -10.0,
    "sector_encoded": 0.0,
}

MODEL_ROLE = {
    "DummyClassifier": "Baseline",
    "Logistic Regression": "Interpretable Benchmark",
    "Random Forest": "Tree-based Benchmark",
    "XGBoost": "F1 / PR-AUC Challenger",
    "LightGBM": "Final Scorer",
}

CONTRIBUTION_GOVERNANCE_DISCLAIMER = (
    "AI Decision Support Only: This model is designed for ESG risk triage and "
    "review prioritization. It does not legally determine greenwashing, credit "
    "approval, or investment exclusion."
)

ESG_SUMMARY_REPORT = """# ESG Summary Preview

## Project Title
AI-Powered ESG Greenwashing Risk Scoring System

## Team
Team 5  
M14B020807 Stanley Cheng  
M14B020803 Chester Wu  
M14B020809 Sophia Hsu  
M14B020812 Alice Hsu  

## Business Problem
Financial institutions face reputational, credit, and sustainable investment mispricing risks when companies make strong ESG claims without consistent external ESG evidence.

中文：
當企業 ESG 說得很漂亮，但外部 ESG 證據不一致時，金融機構可能面臨聲譽風險、授信風險與永續投資誤判風險。

## Core Idea
High Talk + Weak Walk = Greenwashing Risk Screening Signal

Talk means what companies say in ESG reports.  
Walk means external ESG evidence, including ESG risk indicators and controversy information.

中文：
Talk 是企業在 ESG 報告中怎麼說；Walk 是外部 ESG 資料顯示企業實際表現如何。

## Data Sources
The project uses three public datasets:
1. S&P 500 ESG Risk Ratings
2. Sustainability report text dataset
3. S&P 500 integrity score dataset

## Key Results
- Final sample: 257 companies
- Feature count: 14 features
- Positive rate: 7.4%
- Baseline ROC-AUC: 0.475
- LightGBM ROC-AUC: 0.953
- PR-AUC: 0.745
- F1 Score: 0.658

## Business Use Case
The system helps ESG analysts, credit officers, and sustainable investment managers prioritize companies requiring deeper ESG due diligence before sustainable lending or ESG investment decisions.

## Dashboard Pages
1. Home — problem overview and Talk vs Walk logic
2. Data Explorer — dataset summary and sector distribution
3. Model Performance — model comparison, PR-AUC, F1, confusion matrix
4. Prediction Demo — company-level risk scoring
5. Explainability — SHAP beeswarm and company-level waterfall analysis
6. Business Recommendation — ESG credit memo and governance action

## Responsible AI Note
The model does not accuse a company of greenwashing. It flags potential risk for further ESG due diligence.

中文：
模型不直接認定企業漂綠，而是提供進一步 ESG 查核的風險提示。
"""

MODEL_REPORT = """# Model Report Preview

## Machine Learning Objective
Build a classification model to flag companies with potential ESG greenwashing risk based on the gap between ESG disclosure language and external ESG evidence.

## Dataset
The final merged dataset contains 257 companies after joining:
1. S&P 500 ESG Risk Ratings
2. Sustainability report text dataset
3. S&P 500 integrity score dataset

## Feature Engineering
The model uses 14 features:

### Talk Features — 9
- E_density
- S_density
- G_density
- hedge_score
- pct_numeric
- talk_total_density
- vague_to_specific_ratio
- text_length
- esg_topic_breadth

### Walk / External Features — 4
- e_s_g_dispersion
- sector_rank
- planet_friendly_business
- honest_fair_business

### Context Feature — 1
- sector_encoded

## Target Label
High Talk + Weak Walk within the same sector is labeled as a higher greenwashing risk screening signal.

中文：
同產業內，如果企業 ESG 揭露很多，但外部 ESG 證據偏弱，會被標記為高風險訊號。

## Models
The project compares:
- DummyClassifier baseline
- Logistic Regression
- Random Forest
- XGBoost
- LightGBM

## Validation Strategy
- Outer loop: Stratified 5-fold cross-validation
- Inner loop: Stratified 3-fold cross-validation for Optuna tuning
- SMOTE applied only inside training folds
- Test folds were not resampled
- Metrics: ROC-AUC, PR-AUC, F1 Score

## Model Performance

| Model | ROC-AUC | PR-AUC | F1 |
|---|---:|---:|---:|
| DummyClassifier | 0.475 | 0.080 | 0.044 |
| Logistic Regression | 0.922 | 0.521 | 0.458 |
| Random Forest | 0.927 | 0.676 | 0.483 |
| XGBoost | 0.926 | 0.755 | 0.712 |
| LightGBM | 0.953 | 0.745 | 0.658 |

## Final Model
LightGBM was selected as the final scorer because it achieved the highest ROC-AUC and stable cross-validation performance.

Baseline AUC 0.475 → LightGBM AUC 0.953, showing a +0.478 improvement over the naive benchmark.

中文：
從基準模型到最佳模型，AUC 明顯提升，代表模型不是只靠猜測。

## Ablation / Leakage Check
Reduced Walk-Proxy Ablation AUC = 0.741.

This suggests that disclosure-side features still carry directional signal after removing key Walk-proxy variables; because talk_total_density remains in this ablation, it is a robustness check, not a pure independent text model.

中文：
0.741 是保守穩健性檢查，不代表完全純文字模型；模型仍需搭配 SHAP 與人工查核解讀。

## Explainability
The dashboard uses:
- SHAP beeswarm plot for global explanation
- Company-level SHAP waterfall plots for CVX, MTB, and WYNN
- Feature importance table using mean absolute SHAP values

## Responsible AI Limitation
The model is a decision-support tool. It does not legally determine greenwashing or automatically approve / reject credit decisions.

中文：
本模型是決策輔助工具，不是法律認定工具，也不是自動授信核准工具。
"""

DATA_DICTIONARY_REPORT = """# Data Dictionary Preview

## Feature Groups
The final model uses 14 features:

14 = 9 Talk + 4 Walk / External + 1 Context

## Talk Features

| Feature | Business Meaning |
|---|---|
| E_density | Environmental disclosure density |
| S_density | Social disclosure density |
| G_density | Governance disclosure density |
| hedge_score | Vague commitment language |
| pct_numeric | Quantitative disclosure ratio |
| talk_total_density | Overall ESG disclosure intensity |
| vague_to_specific_ratio | Vague vs specific language ratio |
| text_length | Sustainability report length |
| esg_topic_breadth | Breadth of ESG topics covered |

## Walk / External Features

| Feature | Business Meaning |
|---|---|
| e_s_g_dispersion | ESG pillar imbalance |
| sector_rank | Sector-relative ESG risk |
| planet_friendly_business | External environmental signal |
| honest_fair_business | External governance / ethics signal |

## Context Feature

| Feature | Business Meaning |
|---|---|
| sector_encoded | Industry sector control |

## Business-Friendly Feature Names

| Original Feature | Dashboard Display Name |
|---|---|
| sector_rank | Sector-relative ESG Risk |
| talk_total_density | ESG Disclosure Intensity |
| e_s_g_dispersion | ESG Pillar Imbalance |
| pct_numeric | Quantitative Disclosure Ratio |
| vague_to_specific_ratio | Vague vs Specific Language Ratio |
| planet_friendly_business | External Environmental Signal |
| honest_fair_business | External Governance / Ethics Signal |
| E_density | Environmental Disclosure Density |
| S_density | Social Disclosure Density |
| G_density | Governance Disclosure Density |
| hedge_score | Vague Commitment Language |
| text_length | Report Length |
| sector_encoded | Sector Context |

## Label Rule
A company is flagged as higher greenwashing risk when it shows:

High ESG disclosure intensity + weak external ESG evidence within the same sector.

中文：
同產業內，如果企業 ESG 揭露很多，但外部 ESG 證據偏弱，就會被標記為高風險訊號。

## Leakage Control
The following columns are used only for label construction or reference and are not used directly as model input features:
- Total ESG Risk score
- ESG Risk Percentile
- Controversy Score

## Risk Tier Definition
- Low Risk: 0–30%
- Medium Risk: 30–70%
- High Risk: 70–100%

## Recommended Action Logic
- Low Risk: Standard ESG review
- Medium Risk: Enhanced due diligence and supporting ESG evidence
- High Risk: Escalate to enhanced ESG due diligence, require third-party assurance and governance review

## Responsible Use
The model output should be interpreted as a risk signal for further due diligence, not as a legal conclusion.

中文：
模型輸出是查核優先順序，不是法律定論。
"""

PROJECT_PDF_REPORTS = {
    "esg_summary": {
        "title": "ESG Summary Report",
        "sections": [
            ("Project", ["AI-Powered ESG Greenwashing Risk Scoring System"]),
            (
                "Team",
                [
                    "Team 5",
                    "M14B020807 Stanley Cheng",
                    "M14B020803 Chester Wu",
                    "M14B020809 Sophia Hsu",
                    "M14B020812 Alice Hsu",
                ],
            ),
            (
                "Business Problem",
                [
                    "Financial institutions face reputational, credit, and sustainable investment mispricing risks when companies make strong ESG claims without consistent external ESG evidence.",
                ],
            ),
            (
                "Core Idea",
                [
                    "High Talk + Weak Walk = Greenwashing Risk Screening Signal",
                    "Talk means what companies say in ESG reports.",
                    "Walk means external ESG evidence, including ESG risk indicators and controversy information.",
                ],
            ),
            (
                "Key Results",
                [
                    "- Final sample: 257 companies",
                    "- Feature count: 14 features",
                    "- Positive rate: 7.4%",
                    "- Baseline ROC-AUC: 0.475",
                    "- LightGBM ROC-AUC: 0.953",
                    "- PR-AUC: 0.745",
                    "- F1 Score: 0.658",
                ],
            ),
            (
                "Business Use Case",
                [
                    "The system helps ESG analysts, credit officers, and sustainable investment managers prioritize companies requiring deeper ESG due diligence.",
                ],
            ),
            (
                "Responsible AI Note",
                [
                    "The model does not accuse a company of greenwashing. It flags potential risk for further ESG due diligence.",
                ],
            ),
        ],
    },
    "model_report": {
        "title": "Model Report",
        "sections": [
            (
                "Machine Learning Objective",
                [
                    "Build a classification model to flag companies with potential ESG greenwashing risk based on the gap between ESG disclosure language and external ESG evidence.",
                ],
            ),
            (
                "Dataset",
                [
                    "The final merged dataset contains 257 companies after joining:",
                    "1. S&P 500 ESG Risk Ratings",
                    "2. Sustainability report text dataset",
                    "3. S&P 500 integrity score dataset",
                ],
            ),
            ("Feature Engineering", ["14 = 9 Talk + 4 Walk / External + 1 Context"]),
            (
                "Talk Features",
                [
                    "E_density, S_density, G_density, hedge_score, pct_numeric, talk_total_density, vague_to_specific_ratio, text_length, esg_topic_breadth",
                ],
            ),
            (
                "Walk / External Features",
                ["e_s_g_dispersion, sector_rank, planet_friendly_business, honest_fair_business"],
            ),
            ("Context Feature", ["sector_encoded"]),
            (
                "Target Label",
                ["High Talk + Weak Walk within the same sector is labeled as a higher greenwashing risk screening signal."],
            ),
            ("Models", ["DummyClassifier baseline, Logistic Regression, Random Forest, XGBoost, LightGBM"]),
            (
                "Validation Strategy",
                [
                    "- Outer loop: Stratified 5-fold cross-validation",
                    "- Inner loop: Stratified 3-fold cross-validation for Optuna tuning",
                    "- SMOTE applied only inside training folds",
                    "- Test folds were not resampled",
                    "- Metrics: ROC-AUC, PR-AUC, F1 Score",
                ],
            ),
            (
                "Model Performance",
                [
                    "DummyClassifier: ROC-AUC 0.475, PR-AUC 0.080, F1 0.044",
                    "Logistic Regression: ROC-AUC 0.922, PR-AUC 0.521, F1 0.458",
                    "Random Forest: ROC-AUC 0.927, PR-AUC 0.676, F1 0.483",
                    "XGBoost: ROC-AUC 0.926, PR-AUC 0.755, F1 0.712",
                    "LightGBM: ROC-AUC 0.953, PR-AUC 0.745, F1 0.658",
                ],
            ),
            (
                "Final Model",
                [
                    "LightGBM was selected as the final scorer because it achieved the highest ROC-AUC and stable cross-validation performance.",
                ],
            ),
            (
                "Ablation / Leakage Check",
                [
                    "Reduced Walk-Proxy Ablation AUC = 0.741.",
                    "This suggests that disclosure-side features still carry directional signal after removing key Walk-proxy variables; because talk_total_density remains in this ablation, it is a robustness check, not a pure independent text model.",
                ],
            ),
            (
                "Explainability",
                ["The dashboard uses SHAP beeswarm and company-level waterfall plots to explain model decisions."],
            ),
            (
                "Responsible AI Limitation",
                [
                    "The model is a decision-support tool. It does not legally determine greenwashing or automatically approve / reject credit decisions.",
                ],
            ),
        ],
    },
    "data_dictionary": {
        "title": "Data Dictionary",
        "sections": [
            ("Feature Groups", ["14 = 9 Talk + 4 Walk / External + 1 Context"]),
            (
                "Talk Features",
                [
                    "- E_density: Environmental disclosure density",
                    "- S_density: Social disclosure density",
                    "- G_density: Governance disclosure density",
                    "- hedge_score: Vague commitment language",
                    "- pct_numeric: Quantitative disclosure ratio",
                    "- talk_total_density: Overall ESG disclosure intensity",
                    "- vague_to_specific_ratio: Vague vs specific language ratio",
                    "- text_length: Sustainability report length",
                    "- esg_topic_breadth: Breadth of ESG topics covered",
                ],
            ),
            (
                "Walk / External Features",
                [
                    "- e_s_g_dispersion: ESG pillar imbalance",
                    "- sector_rank: Sector-relative ESG risk",
                    "- planet_friendly_business: External environmental signal",
                    "- honest_fair_business: External governance / ethics signal",
                ],
            ),
            ("Context Feature", ["- sector_encoded: Industry sector control"]),
            (
                "Business-Friendly Feature Names",
                [
                    "- sector_rank → Sector-relative ESG Risk",
                    "- talk_total_density → ESG Disclosure Intensity",
                    "- e_s_g_dispersion → ESG Pillar Imbalance",
                    "- pct_numeric → Quantitative Disclosure Ratio",
                    "- vague_to_specific_ratio → Vague vs Specific Language Ratio",
                    "- planet_friendly_business → External Environmental Signal",
                    "- honest_fair_business → External Governance / Ethics Signal",
                ],
            ),
            (
                "Label Rule",
                [
                    "A company is flagged as higher greenwashing risk when it shows high ESG disclosure intensity and weak external ESG evidence within the same sector.",
                ],
            ),
            (
                "Leakage Control",
                [
                    "The following columns are used only for label construction or reference and are not used directly as model input features:",
                    "- Total ESG Risk score",
                    "- ESG Risk Percentile",
                    "- Controversy Score",
                ],
            ),
            ("Risk Tier Definition", ["- Low Risk: 0–30%", "- Medium Risk: 30–70%", "- High Risk: 70–100%"]),
            (
                "Recommended Action Logic",
                [
                    "- Low Risk: Standard ESG review",
                    "- Medium Risk: Enhanced due diligence and supporting ESG evidence",
                    "- High Risk: Escalate to enhanced ESG due diligence, require third-party assurance and governance review",
                ],
            ),
            (
                "Responsible Use",
                [
                    "The model output should be interpreted as a risk signal for further due diligence, not as a legal conclusion.",
                ],
            ),
        ],
    },
}


PDF_UNICODE_FONT_CANDIDATES = [
    ("Noto Sans CJK TC", os.path.join(PDF_FONT_DIR, "NotoSansCJKtc-Regular.otf")),
    ("Source Han Sans TC", os.path.join(PDF_FONT_DIR, "SourceHanSansTC-Regular.otf")),
]


@lru_cache(maxsize=1)
def _pdf_unicode_font_candidates() -> tuple[tuple[str, str], ...]:
    return tuple((name, path) for name, path in PDF_UNICODE_FONT_CANDIDATES if os.path.exists(path))


@lru_cache(maxsize=1)
def _reportlab_unicode_font_names() -> tuple[str, str]:
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    for index, (font_label, font_path) in enumerate(_pdf_unicode_font_candidates()):
        regular_name = f"ESGUnicode-{index}"
        bold_name = f"ESGUnicodeBold-{index}"
        try:
            pdfmetrics.registerFont(TTFont(regular_name, font_path))
            pdfmetrics.registerFont(TTFont(bold_name, font_path))
            return regular_name, bold_name
        except Exception:
            continue
    raise RuntimeError("No embeddable Unicode PDF font found for bilingual memo output.")


def _build_pdf_with_reportlab(report: dict) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

    body_font, heading_font = _reportlab_unicode_font_names()
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=48, leftMargin=48, topMargin=46, bottomMargin=42)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ESGTitle",
        parent=styles["Title"],
        fontName=heading_font,
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#173D2A"),
        spaceAfter=14,
    )
    heading_style = ParagraphStyle(
        "ESGHeading",
        parent=styles["Heading2"],
        fontName=heading_font,
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#1F5C3A"),
        spaceBefore=8,
        spaceAfter=5,
    )
    body_style = ParagraphStyle(
        "ESGBody",
        parent=styles["BodyText"],
        fontName=body_font,
        fontSize=9.8,
        leading=13.2,
        textColor=colors.HexColor("#34453A"),
        spaceAfter=4,
    )

    story = [Paragraph(html.escape(report["title"]), title_style)]
    for heading, lines in report["sections"]:
        story.append(Paragraph(html.escape(heading), heading_style))
        for line in lines:
            story.append(Paragraph(html.escape(line), body_style))
        story.append(Spacer(1, 5))
    doc.build(story)
    return buffer.getvalue()


def _matplotlib_font_properties():
    from matplotlib import font_manager

    font_candidates = _pdf_unicode_font_candidates()
    if font_candidates:
        return font_manager.FontProperties(fname=font_candidates[0][1])
    raise RuntimeError("No bundled Unicode PDF font found under assets/fonts.")


def _build_pdf_with_matplotlib(report: dict) -> bytes:
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"] = 42
    matplotlib.rcParams["pdf.use14corefonts"] = False
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    buffer = BytesIO()
    font_prop = _matplotlib_font_properties()
    line_height = 0.023
    heading_gap = 0.028

    def wrapped_lines(text: str, width: int = 88):
        if not text:
            return [""]
        prefix = ""
        body = text
        if text.startswith("- "):
            prefix, body = "- ", text[2:]
        lines = textwrap.wrap(body, width=width - len(prefix), break_long_words=False, replace_whitespace=False) or [body]
        if prefix:
            return [prefix + lines[0]] + ["  " + line for line in lines[1:]]
        return lines

    with PdfPages(buffer) as pdf:
        fig = plt.figure(figsize=(8.5, 11))
        fig.patch.set_facecolor("#F7FAF5")
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")
        y = 0.955

        def draw(text, x, y_pos, size=9.7, weight="normal", color="#34453A"):
            fig.text(
                x,
                y_pos,
                text,
                ha="left",
                va="top",
                fontsize=size,
                fontweight=weight,
                color=color,
                fontproperties=font_prop,
            )

        def new_page():
            nonlocal fig, ax, y
            pdf.savefig(fig)
            plt.close(fig)
            fig = plt.figure(figsize=(8.5, 11))
            fig.patch.set_facecolor("#F7FAF5")
            ax = fig.add_axes([0, 0, 1, 1])
            ax.axis("off")
            y = 0.955

        draw(report["title"], 0.08, y, size=20, weight="bold", color="#173D2A")
        y -= 0.048
        draw("Project-level reference report", 0.08, y, size=9.2, color="#6F7F72")
        y -= 0.045
        for heading, lines in report["sections"]:
            if y < 0.12:
                new_page()
            draw(heading, 0.08, y, size=13, weight="bold", color="#1F5C3A")
            y -= heading_gap
            for line in lines:
                for wrapped in wrapped_lines(line):
                    if y < 0.08:
                        new_page()
                    draw(wrapped, 0.1, y, size=9.7, color="#34453A")
                    y -= line_height
            y -= 0.012
        pdf.savefig(fig)
        plt.close(fig)
    return buffer.getvalue()


@st.cache_data(show_spinner=False)
def project_report_pdf_bytes(report_key: str) -> bytes:
    report = PROJECT_PDF_REPORTS[report_key]
    try:
        pdf_bytes = _build_pdf_with_reportlab(report)
    except Exception:
        pdf_bytes = _build_pdf_with_matplotlib(report)
    return pdf_bytes


def project_report_pdf_engine() -> str:
    try:
        import reportlab  # noqa: F401

        return "reportlab"
    except Exception:
        return "matplotlib PdfPages fallback"


@st.cache_data(show_spinner=False)
def project_report_file_bytes(filename: str, fallback_key: str | None = None) -> bytes:
    path = os.path.join(REPORTS, filename)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return f.read()
    if fallback_key:
        return project_report_pdf_bytes(fallback_key)
    return b""


st.set_page_config(
    page_title="DSF504 ESG Greenwashing Risk System",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
<style>
:root{
  --bg:#F7FAF5;
  --sidebar:#1F5C3A;
  --sidebar-dark:#173D2A;
  --card:#FFFFFF;
  --border:#DDEBDD;
  --primary:#173D2A;
  --secondary:#5F7165;
  --green:#2E7D32;
  --soft:#EAF4EC;
  --amber:#D99A2B;
  --amber-bg:#FFF7E5;
  --red:#C94C3A;
  --red-bg:#FDECEC;
  --shadow:0 10px 24px rgba(10,41,26,.08);
  --serif:Georgia,'Times New Roman',serif;
  --sans:Inter,'Segoe UI',Arial,sans-serif;
}
html,body,[data-testid="stAppViewContainer"]{
  background:
    radial-gradient(circle at 92% 0%,rgba(222,239,226,.9) 0,transparent 25%),
    linear-gradient(rgba(31,92,58,.028) 1px,transparent 1px),
    linear-gradient(90deg,rgba(31,92,58,.024) 1px,transparent 1px),
    var(--bg);
  background-size:auto,42px 42px,42px 42px,auto;
  color:var(--primary);
  font-family:var(--sans);
  font-size:17px;
}
[data-testid="stAppViewBlockContainer"]{
  max-width:1280px;
  padding:28px 34px 56px;
}
[data-testid="stHeader"]{background:rgba(247,250,245,.72);backdrop-filter:blur(12px);}
[data-testid="stSidebar"]{
  background:
    radial-gradient(circle at 24px 42px,rgba(184,216,189,.30) 0 70px,transparent 71px),
    radial-gradient(circle at 92% 18%,rgba(137,186,202,.18) 0 82px,transparent 83px),
    linear-gradient(180deg,#1F5C3A 0%,#173D2A 72%,#143322 100%);
  border-right:0;
}
[data-testid="stSidebar"] *{color:#F7FAF5 !important;}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"]{
  gap:.72rem;
}
[data-testid="stSidebar"] div[role="radiogroup"]{
  padding:10px;
  margin:0 0 14px;
  border:1px solid rgba(255,255,255,.16);
  border-radius:22px;
  background:rgba(255,255,255,.075);
  box-shadow:0 12px 26px rgba(6,28,17,.16);
}
[data-testid="stSidebar"] div[role="radiogroup"] label{
  display:flex;
  border-radius:16px;
  padding:11px 12px;
  margin:5px 0;
  min-height:46px;
  align-items:center;
  gap:10px;
  border:1px solid transparent;
}
[data-testid="stSidebar"] div[role="radiogroup"] label:before{
  content:"";
  width:28px;
  height:28px;
  flex:0 0 28px;
  display:grid;
  place-items:center;
  border-radius:11px;
  background:rgba(255,255,255,.10);
  border:1px solid rgba(255,255,255,.16);
  box-shadow:inset 0 1px 0 rgba(255,255,255,.14);
  color:#EAF4EC !important;
  font-size:20px;
  line-height:1;
}
[data-testid="stSidebar"] div[role="radiogroup"] label:nth-of-type(1):before{content:"🏠";}
[data-testid="stSidebar"] div[role="radiogroup"] label:nth-of-type(2):before{content:"📊";}
[data-testid="stSidebar"] div[role="radiogroup"] label:nth-of-type(3):before{content:"🤖";}
[data-testid="stSidebar"] div[role="radiogroup"] label:nth-of-type(4):before{content:"🧩";}
[data-testid="stSidebar"] div[role="radiogroup"] label:nth-of-type(5):before{content:"🔎";}
[data-testid="stSidebar"] div[role="radiogroup"] label:nth-of-type(6):before{content:"🔬";}
[data-testid="stSidebar"] div[role="radiogroup"] label:nth-of-type(7):before{content:"💼";}
[data-testid="stSidebar"] div[role="radiogroup"] label p{
  font-size:1rem !important;
  font-weight:650 !important;
  line-height:1.25 !important;
  margin:0 !important;
}
[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked):before{
  background:rgba(234,244,236,.92);
  border-color:rgba(234,244,236,.72);
  color:#173D2A !important;
  box-shadow:0 7px 16px rgba(6,28,17,.12);
}
[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child{
  position:absolute !important;
  opacity:0 !important;
  width:1px !important;
  height:1px !important;
  margin:0 !important;
  overflow:hidden !important;
}
[data-testid="stSidebar"] div[role="radiogroup"] label:hover{
  background:rgba(255,255,255,.10);
}
[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked){
  background:rgba(255,255,255,.20);
  border:1px solid rgba(255,255,255,.26);
  box-shadow:inset 4px 0 0 #B8D8BD;
  font-weight:800;
}
.sidebar-brand{
  position:relative;
  overflow:hidden;
  padding:18px 14px 16px;
  border:1px solid rgba(255,255,255,.28);
  border-radius:24px;
  background:
    radial-gradient(circle at 92% 14%,rgba(137,186,202,.24) 0 52px,transparent 53px),
    linear-gradient(145deg,rgba(255,255,255,.96) 0%,rgba(239,248,237,.94) 62%,rgba(237,247,248,.92) 100%);
  box-shadow:0 14px 28px rgba(6,28,17,.20);
  margin:8px 0 16px;
}
.sidebar-brand:after{
  content:"";
  position:absolute;
  right:-28px;
  bottom:-36px;
  width:120px;
  height:92px;
  border-radius:70% 30% 0 0;
  background:linear-gradient(135deg,rgba(46,125,50,.12),rgba(137,186,202,.18));
  transform:rotate(-14deg);
  pointer-events:none;
}
.sidebar-brand > *{position:relative;z-index:1;}
.sidebar-brand-top{
  display:flex;
  align-items:center;
  justify-content:space-between;
  gap:10px;
  margin-bottom:13px;
}
.sidebar-mark{
  width:auto;min-width:104px;height:42px;border-radius:16px;
  background:linear-gradient(145deg,#173D2A,#1F5C3A);
  border:1px solid rgba(31,92,58,.20);
  display:flex;align-items:center;justify-content:center;gap:8px;color:#F7FAF5 !important;
  font-family:var(--serif);font-weight:900;font-size:21px;line-height:1;letter-spacing:.04em;margin-bottom:0;
  box-sizing:border-box;overflow:visible;
  box-shadow:0 9px 18px rgba(23,61,42,.18);
}
.sidebar-leaf{
  width:13px;height:18px;border-radius:12px 12px 12px 3px;
  background:#B8D8BD;display:inline-block;transform:rotate(35deg);
  flex:0 0 auto;
}
.sidebar-shield{
  width:21px;
  height:21px;
  border-radius:50%;
  display:grid;
  place-items:center;
  background:#EAF4EC;
  color:#1F5C3A !important;
  font-size:.82rem;
  font-family:var(--sans);
  font-weight:900;
}
.sidebar-title{font-family:var(--serif);line-height:1.18;font-weight:800;color:#173D2A !important;}
.sidebar-title-main{display:block;font-size:1.42rem;font-weight:900;color:#173D2A !important;}
.sidebar-title-subline{display:block;font-size:1.14rem;font-weight:800;margin-top:4px;color:#1F5C3A !important;}
.sidebar-sub{font-size:.9rem;color:#5F7165 !important;margin-top:11px;line-height:1.38;font-weight:750;}
.sidebar-team{
  margin-top:14px;
  padding:14px 13px 13px;
  border:1px solid rgba(31,92,58,.16);
  border-radius:18px;
  background:rgba(255,255,255,.88);
  box-shadow:0 9px 18px rgba(23,61,42,.10),inset 0 0 0 1px rgba(255,255,255,.65);
}
.sidebar-team-eyebrow{
  display:flex;
  justify-content:space-between;
  gap:8px;
  color:#1F5C3A !important;
  font-size:.78rem;
  line-height:1.15;
  font-weight:900;
  letter-spacing:.04em;
  margin-bottom:8px;
}
.sidebar-team-eyebrow span{color:#4F6F5A !important;font-weight:900;}
.sidebar-identity{
  display:flex;
  align-items:center;
  gap:8px;
  margin-top:12px;
  padding:11px 12px;
  border-radius:16px;
  background:linear-gradient(145deg,#F7FAF5 0%,#EAF4EC 100%);
  border:1px solid rgba(31,92,58,.16);
  color:#173D2A !important;
  font-size:.84rem;
  line-height:1.3;
  font-weight:900;
  box-shadow:0 8px 16px rgba(23,61,42,.08);
}
.sidebar-identity .identity-icon{
  width:24px;height:24px;border-radius:10px;
  display:grid;place-items:center;
  background:#1F5C3A;
  color:#FFFFFF !important;
  font-size:.82rem;
  font-weight:900;
  flex:0 0 auto;
}
.sidebar-identity span:not(.identity-icon){
  color:#173D2A !important;
}
.sidebar-identity .zh{
  margin-top:2px;
  color:#4F6F5A !important;
  font-size:.8rem;
  line-height:1.25;
  font-weight:850;
}
.sidebar-team-title{
  font-size:.94rem;
  letter-spacing:.08em;
  text-transform:uppercase;
  color:#173D2A !important;
  font-weight:900;
  margin-bottom:11px;
}
.sidebar-member{
  display:grid;
  grid-template-columns:104px minmax(0,1fr);
  align-items:baseline;
  gap:10px;
  font-size:.9rem;
  line-height:1.22;
  color:#173D2A !important;
  margin:6px 0;
  padding:5px 6px;
  border-radius:10px;
  background:rgba(234,244,236,.72);
  white-space:nowrap;
}
.sidebar-member .member-id{
  color:#4F6F5A !important;
  font-size:.82rem;
  font-weight:850;
  font-variant-numeric:tabular-nums;
}
.sidebar-member .member-name{
  color:#173D2A !important;
  font-size:1.04rem;
  font-weight:950;
}
.sidebar-box{
  border:1px solid rgba(255,255,255,.18);
  background:rgba(255,255,255,.08);
  border-radius:20px;
  padding:14px 14px;
  margin:18px 0;
}
.sidebar-box h4{font-size:.9rem;text-transform:uppercase;letter-spacing:.08em;margin:0 0 9px;color:#EAF4EC !important;}
.sidebar-link{font-size:.88rem;line-height:1.45;color:#F7FAF5 !important;}
.sidebar-nav-heading{
  display:flex;
  align-items:center;
  justify-content:space-between;
  gap:10px;
  padding:12px 14px 9px;
  margin:4px 0 0;
  border-radius:18px 18px 8px 8px;
  background:rgba(255,255,255,.10);
  border:1px solid rgba(255,255,255,.14);
  border-bottom:0;
  box-shadow:0 8px 18px rgba(6,28,17,.12);
}
.sidebar-nav-heading .nav-title{
  font-size:.88rem;
  text-transform:uppercase;
  letter-spacing:.08em;
  font-weight:900;
  color:#F7FAF5 !important;
}
.sidebar-nav-heading .nav-zh{
  font-size:.78rem;
  font-weight:850;
  color:#DCEBDD !important;
}
.sidebar-status-card{
  display:grid;
  grid-template-columns:34px minmax(0,1fr);
  gap:10px;
  align-items:center;
  margin:6px 0 14px;
  padding:12px 13px;
  border-radius:18px;
  background:rgba(255,255,255,.12);
  border:1px solid rgba(255,255,255,.18);
  box-shadow:0 10px 22px rgba(6,28,17,.14);
}
.sidebar-status-icon{
  width:34px;height:34px;border-radius:13px;
  display:grid;place-items:center;
  background:#EAF4EC;
  color:#1F5C3A !important;
  font-weight:900;
}
.sidebar-status-label{
  font-size:.78rem;
  text-transform:uppercase;
  letter-spacing:.08em;
  color:#CFE4D2 !important;
  font-weight:900;
}
.sidebar-status-date{
  margin-top:3px;
  color:#FFFFFF !important;
  font-size:1rem;
  line-height:1.1;
  font-weight:900;
}
.sidebar-note-card{
  margin:4px 0 16px;
  padding:14px 14px 15px;
  border-radius:20px;
  background:linear-gradient(145deg,rgba(234,244,236,.18),rgba(255,255,255,.08));
  border:1px solid rgba(207,228,210,.32);
  box-shadow:0 12px 24px rgba(6,28,17,.16);
}
.sidebar-note-card h4{
  display:flex;
  justify-content:space-between;
  gap:8px;
  margin:0 0 11px;
  font-size:.88rem;
  text-transform:uppercase;
  letter-spacing:.08em;
  color:#F7FAF5 !important;
  font-weight:900;
}
.sidebar-note-card h4 span{
  color:#DCEBDD !important;
  font-size:.78rem;
  letter-spacing:0;
  text-transform:none;
  font-weight:850;
}
.sidebar-note-card .sidebar-link{
  color:#F1FAF2 !important;
  font-size:.86rem;
  line-height:1.44;
}
.sidebar-warning-row{
  display:grid;
  grid-template-columns:28px minmax(0,1fr);
  gap:10px;
  align-items:start;
  margin-top:10px;
  padding:11px 12px;
  border-radius:16px;
  background:rgba(255,255,255,.10);
  border:1px solid rgba(255,255,255,.18);
  box-shadow:inset 0 1px 0 rgba(255,255,255,.10);
}
.sidebar-warning-icon{
  width:28px;
  height:28px;
  border-radius:11px;
  display:grid;
  place-items:center;
  background:#FFF4D6;
  color:#8A5E11 !important;
  font-size:1rem;
  line-height:1;
  box-shadow:0 6px 14px rgba(6,28,17,.13);
}
.sidebar-warning-main{
  color:#F7FAF5 !important;
  font-size:.88rem;
  line-height:1.35;
  font-weight:850;
}
.sidebar-warning-main .danger{
  color:#FFB3A8 !important;
  font-weight:950;
}
.sidebar-warning-zh{
  margin-top:5px;
  color:#DCEBDD !important;
  font-size:.8rem;
  line-height:1.35;
  font-weight:750;
}
.page-head{
  display:grid;
  grid-template-columns:92px minmax(0,1fr) auto;
  gap:24px;
  align-items:start;
  margin:8px 0 24px;
}
.page-num{
  font-family:var(--serif);font-size:82px;line-height:.9;font-weight:800;color:#D3E5D7;
}
.page-title h1{
  margin:0;
  font-family:var(--serif);
  color:var(--primary) !important;
  font-size:46px;
  line-height:1.12;
  font-weight:800;
}
.page-title .subtitle{color:var(--secondary);font-size:1.08rem;font-weight:700;margin-top:8px;line-height:1.35;}
.page-title .zh{font-size:1.06rem;line-height:1.45;margin-top:8px;}
.zh{color:#6F7F72;font-size:.96rem;line-height:1.5;margin-top:7px;}
.card{
  background:var(--card);
  border:1px solid var(--border);
  border-radius:24px;
  box-shadow:var(--shadow);
  padding:24px 26px;
  box-sizing:border-box;
  transition:transform .15s ease,box-shadow .15s ease;
}
.card:hover{transform:translateY(-2px);box-shadow:0 14px 30px rgba(10,41,26,.11);}
.card.compact{padding:18px 20px;border-radius:20px;}
.card.soft{background:#F9FCF8;}
.card.warn{background:#FFF7E5;border-color:#F2D6A2;}
.card.danger{background:#FDECEC;border-color:#E7B5AA;}
.card h2,.card h3{
  margin:0 0 14px;
  font-family:var(--serif);
  color:var(--primary) !important;
}
.card h2{font-size:26px;line-height:1.1;}
.card h3{font-size:24px;line-height:1.18;}
.card p{color:#34453A;font-size:1.02rem;line-height:1.58;margin:0 0 10px;}
.report-panel{
  border-top:4px solid #1B6B4C;
}
.report-panel .kicker,.report-panel .label-emphasis{
  color:#B8860B !important;
}
.report-caption{
  border-left:4px solid #1B6B4C;
  padding:10px 14px;
  margin:0 0 12px;
  background:#F9FCF8;
  border-radius:0 14px 14px 0;
  box-shadow:0 8px 20px rgba(10,41,26,.06);
}
.report-caption-zh{
  color:#173D2A;
  font-size:1.08rem;
  line-height:1.42;
  font-weight:900;
}
.report-caption-en{
  color:#6F7F72;
  font-size:.9rem;
  line-height:1.4;
  font-weight:700;
  margin-top:3px;
}
.kicker{
  text-transform:uppercase;letter-spacing:.09em;color:var(--green);
  font-size:.88rem;font-weight:900;margin-bottom:8px;
}
.grid{display:grid;gap:22px;}
.grid-2{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:22px;}
.grid-3{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:22px;}
.grid-4{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:18px;}
.grid-6{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:14px;}
.metric{
  min-height:116px;background:#FFFFFF;border:1px solid var(--border);
  border-radius:20px;box-shadow:var(--shadow);padding:18px 18px;
  transition:transform .15s ease,box-shadow .15s ease;
}
.metric:hover{transform:translateY(-2px);box-shadow:0 14px 30px rgba(10,41,26,.11);}
.metric .label{color:var(--secondary);font-size:.98rem;font-weight:800;text-transform:uppercase;letter-spacing:.04em;}
.metric .value{font-family:var(--serif);font-size:36px;line-height:1;color:var(--green);font-weight:800;margin:13px 0 8px;}
.metric .zh{font-size:.9rem;margin:0;}
.home-metrics .metric{
  position:relative;
  overflow:hidden;
  min-height:132px;
  border-radius:22px;
  border:1px solid rgba(31,92,58,.13);
  background:
    radial-gradient(circle at 92% 12%,rgba(137,186,202,.22) 0 36px,transparent 37px),
    linear-gradient(145deg,#FFFFFF 0%,#F4FBF3 54%,#EEF7F8 100%);
  box-shadow:0 13px 28px rgba(23,61,42,.09);
  padding:18px 18px 17px;
}
.home-metrics .metric:before{
  content:"";
  position:absolute;
  top:16px;
  right:16px;
  width:34px;
  height:34px;
  border-radius:12px;
  background:rgba(31,92,58,.10);
  border:1px solid rgba(31,92,58,.12);
}
.home-metrics .metric:after{
  content:"";
  position:absolute;
  top:26px;
  right:26px;
  width:12px;
  height:12px;
  border:2px solid rgba(31,92,58,.70);
  border-top:0;
  border-left:0;
  transform:rotate(45deg);
}
.home-metrics .metric:nth-child(2n){
  background:
    radial-gradient(circle at 92% 12%,rgba(184,216,189,.34) 0 36px,transparent 37px),
    linear-gradient(145deg,#FFFFFF 0%,#F6FAFD 50%,#EDF6F2 100%);
}
.home-metrics .metric:nth-child(6) .value{color:#B8860B !important;}
.home-metrics .metric .label{
  max-width:calc(100% - 48px);
  font-size:.86rem;
  letter-spacing:.06em;
  color:#5B7162;
}
.home-metrics .metric .value{
  font-size:34px;
  margin:18px 0 7px;
}
.home-metrics .metric .zh{
  max-width:calc(100% - 24px);
  font-size:.9rem;
  font-weight:700;
  color:#607467;
}
.hero{
  background:
    radial-gradient(circle at 90% 12%,rgba(137,186,202,.22) 0 74px,transparent 75px),
    radial-gradient(circle at 74% 88%,rgba(217,154,43,.16) 0 88px,transparent 89px),
    linear-gradient(140deg,rgba(255,255,255,.98) 0%,rgba(246,251,243,.98) 56%,rgba(237,247,248,.98) 100%);
  min-height:355px;
  display:grid;
  grid-template-columns:minmax(0,1.15fr) minmax(320px,.85fr);
  gap:30px;
  position:relative;
  overflow:hidden;
}
.hero:before{
  content:"";
  position:absolute;
  inset:auto -80px -120px auto;
  width:360px;
  height:260px;
  border-radius:52% 48% 0 0;
  background:linear-gradient(135deg,rgba(46,125,50,.13),rgba(137,186,202,.16));
  transform:rotate(-12deg);
  pointer-events:none;
}
.hero:after{
  content:"";
  position:absolute;
  left:42%;
  top:28px;
  width:120px;
  height:54px;
  border-radius:100px 100px 100px 8px;
  border:1px solid rgba(31,92,58,.12);
  background:rgba(234,244,236,.52);
  transform:rotate(-18deg);
  pointer-events:none;
}
.hero > *{position:relative;z-index:1;}
.hero .mission{font-size:1.08rem;line-height:1.6;color:#34453A;}
.home-hero-panel{
  border-color:#CFE4D2;
  box-shadow:0 18px 38px rgba(23,61,42,.10);
}
.home-hero-copy{
  padding-right:10px;
}
.botanical{
  min-height:250px;border-radius:24px;border:1px solid #CFE4D2;
  position:relative;
  overflow:hidden;
  background:
    radial-gradient(circle at 74% 18%,#F8D989 0 30px,transparent 31px),
    radial-gradient(circle at 22% 24%,rgba(137,186,202,.50) 0 52px,transparent 53px),
    linear-gradient(165deg,transparent 40%,rgba(184,216,189,.85) 41% 60%,transparent 61%),
    linear-gradient(205deg,transparent 30%,rgba(140,185,153,.80) 31% 63%,transparent 64%),
    repeating-linear-gradient(90deg,transparent 0 56px,rgba(31,92,58,.14) 57px 58px,transparent 59px),
    linear-gradient(180deg,#F8FCF6,#EAF5F6);
  box-shadow:inset 0 0 0 1px rgba(255,255,255,.46);
}
.botanical:before{
  content:"";
  position:absolute;
  left:20px;
  bottom:20px;
  width:118px;
  height:118px;
  border-radius:64% 36% 62% 38%;
  background:linear-gradient(145deg,rgba(46,125,50,.38),rgba(184,216,189,.62));
  transform:rotate(-18deg);
}
.botanical:after{
  content:"";
  position:absolute;
  right:24px;
  bottom:26px;
  width:150px;
  height:58px;
  border-radius:999px;
  border:1px solid rgba(31,92,58,.16);
  background:rgba(255,255,255,.42);
  box-shadow:0 -16px 0 rgba(137,186,202,.18);
}
.flow-row{
  display:flex;gap:9px;align-items:center;flex-wrap:wrap;
}
.flow-step{
  background:var(--soft);border:1px solid #D3E6D7;border-radius:999px;
  padding:9px 13px;font-size:.9rem;font-weight:800;color:#1F5C3A;
}
.arrow{color:#9AAC9E;font-weight:900;}
.concept{
  display:grid;grid-template-columns:1fr auto 1fr;gap:16px;align-items:stretch;
}
.concept .block{background:#F9FCF8;border:1px solid var(--border);border-radius:20px;padding:18px;}
.formula{
  display:grid;place-items:center;padding:12px 18px;border-radius:20px;
  background:var(--soft);font-weight:900;color:var(--primary);text-align:center;
}
.home-process{
  border-top:4px solid #1B6B4C;
  background:
    linear-gradient(135deg,#FFFFFF 0%,#F6FBF4 55%,#EEF7F8 100%);
}
.home-process-grid{
  display:grid;
  grid-template-columns:repeat(4,minmax(0,1fr));
  gap:14px;
  margin-top:16px;
}
.home-process-step{
  position:relative;
  min-height:120px;
  border:1px solid rgba(31,92,58,.13);
  border-radius:20px;
  background:rgba(255,255,255,.72);
  padding:16px 14px 14px;
  box-shadow:0 10px 22px rgba(23,61,42,.07);
}
.home-process-step:after{
  content:"";
  position:absolute;
  right:-12px;
  top:50%;
  width:20px;
  height:1px;
  background:#B8D8BD;
}
.home-process-step:last-child:after{display:none;}
.process-icon{
  width:34px;
  height:34px;
  border-radius:14px;
  display:grid;
  place-items:center;
  background:#EAF4EC;
  border:1px solid #CFE4D2;
  color:#1F5C3A;
  font-weight:900;
  font-size:.95rem;
  margin-bottom:12px;
}
.process-title{
  font-family:var(--serif);
  font-size:1.18rem;
  line-height:1.1;
  color:#173D2A;
  font-weight:900;
}
.process-zh{
  margin-top:5px;
  color:#5F7165;
  font-size:.95rem;
  font-weight:800;
}
.home-final-message{
  position:relative;
  overflow:hidden;
  border-left:5px solid #1B6B4C;
  background:
    radial-gradient(circle at 95% 15%,rgba(137,186,202,.24) 0 52px,transparent 53px),
    linear-gradient(135deg,#FFFFFF 0%,#F8FCF6 56%,#F2F8F9 100%);
}
.home-final-message:after{
  content:"";
  position:absolute;
  right:22px;
  bottom:18px;
  width:68px;
  height:68px;
  border-radius:50%;
  border:1px solid rgba(31,92,58,.12);
  background:rgba(234,244,236,.50);
}
.home-final-message p{
  position:relative;
  z-index:1;
  max-width:880px;
  font-size:1.13rem;
  line-height:1.55;
  font-weight:850;
  color:#173D2A;
}
.home-final-message .zh{
  position:relative;
  z-index:1;
  max-width:900px;
  font-size:1.02rem;
  color:#5F7165;
  font-weight:750;
}
.risk-badge{
  display:inline-flex;align-items:center;justify-content:center;min-height:30px;
  border-radius:999px;padding:0 13px;font-size:.9rem;font-weight:900;
}
.risk-low{background:#EAF4EC;color:#1F5C3A;}
.risk-medium{background:#FFF7E5;color:#8A5E11;}
.risk-high{background:#FDECEC;color:#C94C3A;}
.risk-tier-badge{
  display:inline-flex;align-items:center;justify-content:center;gap:6px;
  border-radius:999px;padding:7px 12px;font-size:.9rem;font-weight:900;line-height:1.15;
  border:1px solid transparent;white-space:normal;text-align:center;
}
.risk-tier-low{background:#EAF4EC;color:#1B6B4C;border-color:#CFE4D2;}
.risk-tier-watch{background:#FFF7E5;color:#8A5E11;border-color:#F2D6A2;}
.risk-tier-high{background:#FDECEC;color:#A73627;border-color:#E7B5AA;}
.talk-walk-bars{
  margin-top:14px;padding:14px 14px;border:1px solid var(--border);border-radius:18px;background:#F9FCF8;
}
.tw-row{margin:9px 0;}
.tw-top{display:flex;justify-content:space-between;gap:12px;align-items:baseline;font-size:.9rem;font-weight:900;color:var(--primary);}
.tw-raw{font-size:.84rem;color:#6F7F72;font-weight:800;}
.tw-track{height:10px;border-radius:999px;background:#E4EEE5;overflow:hidden;margin-top:6px;}
.tw-fill{height:100%;border-radius:999px;}
.tw-talk{background:linear-gradient(90deg,#7FB685,#1B6B4C);}
.tw-walk{background:linear-gradient(90deg,#F8D989,#C94C3A);}
.tw-caption{margin-top:9px;font-size:.88rem;line-height:1.4;color:#5F7165;font-weight:750;}
.plot-card [data-testid="stPlotlyChart"]{
  border:1px solid var(--border);border-radius:20px;box-shadow:none;background:#fff;
}
div[data-testid="stVerticalBlockBorderWrapper"]{
  background:#FFFFFF !important;
  border:1px solid var(--border) !important;
  border-radius:24px !important;
  box-shadow:var(--shadow) !important;
  padding:18px 20px !important;
}
.chart-title{
  margin:0 0 10px;
  font-family:var(--serif);
  color:var(--primary);
  font-size:24px;
  line-height:1.15;
  font-weight:800;
}
[data-testid="stDataFrame"],[data-testid="stTable"]{
  border:1px solid var(--border);border-radius:18px;overflow:hidden;
}
[data-testid="stDataFrame"] *,[data-testid="stTable"] *{
  font-size:.92rem !important;
}
[data-testid="stExpander"]{
  background:#FFFFFF;border:1px solid var(--border);border-radius:18px;box-shadow:var(--shadow);
}
.note{
  background:#F9FCF8;border-left:4px solid var(--green);border-radius:16px;
  padding:13px 16px;color:#34453A;font-size:1rem;line-height:1.55;
}
.warning-note{background:#FFF7E5;border-left-color:var(--amber);}
.danger-note{background:#FDECEC;border-left-color:var(--red);}
.table-wrap{overflow-x:auto;padding:2px 0 4px;}
.report-table,.dashboard-table{
  width:100%;border-collapse:separate;border-spacing:0 8px;font-size:.96rem !important;line-height:1.42;
}
.report-table thead tr,.dashboard-table thead tr{
  background:linear-gradient(90deg,#123D2B,#1F6A43);
}
.report-table th,.dashboard-table th{
  text-align:left;color:#F7FAF5 !important;font-size:.86rem;text-transform:uppercase;letter-spacing:.07em;
  padding:12px 14px;font-weight:950;border-top:1px solid rgba(255,255,255,.12);border-bottom:1px solid rgba(0,0,0,.06);
}
.report-table th:first-child,.dashboard-table th:first-child{border-radius:14px 0 0 14px;}
.report-table th:last-child,.dashboard-table th:last-child{border-radius:0 14px 14px 0;}
.report-table td,.dashboard-table td{
  background:linear-gradient(180deg,#FFFFFF,#F8FCF7);border-top:1px solid #DCE9DD;border-bottom:1px solid #DCE9DD;
  padding:12px 14px;vertical-align:middle;color:#173D2A !important;font-weight:720;
}
.report-table tr:nth-child(even) td,.dashboard-table tr:nth-child(even) td{background:linear-gradient(180deg,#FBFDFB,#F1F8F0);}
.report-table tbody tr:hover td,.dashboard-table tbody tr:hover td{background:#EAF4EC;}
.report-table td:first-child,.dashboard-table td:first-child{border-left:1px solid #DCE9DD;border-radius:14px 0 0 14px;font-weight:900;color:var(--primary) !important;}
.report-table td:last-child,.dashboard-table td:last-child{border-right:1px solid #DCE9DD;border-radius:0 14px 14px 0;}
.report-table td.num,.dashboard-table td.num{text-align:right;font-variant-numeric:tabular-nums;}
.table-panel{
  background:#fff;border:1px solid var(--border);border-radius:22px;padding:16px 18px 18px;box-shadow:var(--shadow);margin:8px 0 16px;
}
.table-panel-title{display:flex;align-items:center;gap:10px;margin-bottom:4px;font-family:var(--serif);font-size:1.3rem;font-weight:900;color:var(--primary);}
.table-panel-title:before{content:"";width:9px;height:28px;border-radius:99px;background:linear-gradient(180deg,#2E7D32,#D99A2B);}
.table-panel-subtitle{color:#5F7165;font-size:.94rem;line-height:1.42;font-weight:750;margin:0 0 10px;}
.badge-pill{display:inline-flex;align-items:center;justify-content:center;padding:5px 10px;border-radius:999px;font-size:.78rem;font-weight:950;letter-spacing:.02em;white-space:nowrap;}
.badge-talk{background:#EAF4EC;color:#1F5C3A;border:1px solid #CFE4D2;}
.badge-walk{background:#E7F1EB;color:#0B5D42;border:1px solid #C2DCCC;}
.badge-context{background:#EEF3EF;color:#52655B;border:1px solid #D7E2D8;}
.badge-final{background:#DFF3E0;color:#0D5D37;border:1px solid #B8DCBE;}
.badge-challenger{background:#FFF4D6;color:#8A5E11;border:1px solid #E9CB7B;}
.badge-benchmark{background:#EEF3EF;color:#4D6255;border:1px solid #D7E2D8;}
.badge-low{background:#EAF4EC;color:#17613F;border:1px solid #CFE4D2;}
.badge-medium{background:#FFF4D6;color:#8A5E11;border:1px solid #E9CB7B;}
.badge-high{background:#FDECEC;color:#A73627;border:1px solid #E7B5AA;}
.rank-chip{display:inline-grid;place-items:center;width:25px;height:25px;border-radius:50%;background:#1F6A43;color:#fff;font-size:.8rem;font-weight:950;margin-right:8px;}
.rank-chip.secondary{background:#D99A2B;}
.rank-chip.muted{background:#91A196;}
.dashboard-highlight td{box-shadow:inset 4px 0 0 #2E7D32;background:linear-gradient(90deg,#EEF9EE,#FFFFFF) !important;}
.dashboard-warning td{box-shadow:inset 4px 0 0 #D99A2B;background:linear-gradient(90deg,#FFF8E7,#FFFFFF) !important;}
.insight-strip{display:flex;align-items:center;gap:12px;background:linear-gradient(90deg,#EAF4EC,#F9FCF8);border:1px solid var(--border);border-radius:18px;padding:12px 15px;margin:10px 0;color:#173D2A;font-weight:800;}
.insight-strip .icon{display:grid;place-items:center;width:34px;height:34px;border-radius:12px;background:#1F6A43;color:white;font-size:1.05rem;}
.gauge{
  width:230px;height:230px;border-radius:50%;margin:12px auto 16px;
  display:grid;place-items:center;position:relative;
}
.gauge:after{content:"";position:absolute;inset:54px;border-radius:50%;background:#fff;}
.gauge-inner{position:relative;z-index:1;text-align:center;}
.gauge-value{font-family:var(--serif);font-size:46px;font-weight:900;line-height:1;}
.gauge-tier{font-weight:900;margin-top:6px;}
.signal-row{display:grid;grid-template-columns:1.2fr .8fr;gap:12px;border-top:1px solid var(--border);padding:10px 0;}
.signal-row:first-child{border-top:0;}
.signal-label{font-size:.98rem;font-weight:800;color:var(--primary);}
.signal-value{text-align:right;font-size:.96rem;color:var(--secondary);}
.risk-policy-row{
  display:grid;
  grid-template-columns:1.15fr 1fr 1.35fr 1fr;
  gap:16px;
  align-items:start;
  padding:20px 22px;
  margin-bottom:14px;
  border:1px solid var(--border);
  border-radius:22px;
  box-shadow:var(--shadow);
}
.risk-policy-row.low{background:#F4FBF3;}
.risk-policy-row.medium{background:#FFF8E8;border-color:#F2D6A2;}
.risk-policy-row.high{background:#FDECEC;border-color:#E7B5AA;}
.risk-tier-title{font-family:var(--serif);font-size:24px;line-height:1.12;font-weight:900;color:var(--primary);}
.risk-policy-label{font-size:.9rem;text-transform:uppercase;letter-spacing:.07em;font-weight:900;color:var(--secondary);margin-bottom:7px;}
.risk-policy-text{font-size:.98rem;line-height:1.5;color:#34453A;font-weight:650;}
.quick-demo-row .stButton>button{
  min-height:34px !important;
  padding:6px 10px !important;
  border-radius:12px !important;
  font-size:.94rem !important;
}
.shap-img{width:100%;border-radius:18px;border:1px solid var(--border);background:#fff;}
.shap-beeswarm .shap-img{max-height:430px;object-fit:contain;}
.shap-expanded .shap-img{max-height:none;object-fit:contain;}
.shap-explain-box{
  display:flex;
  align-items:flex-start;
  gap:14px;
  margin-top:14px;
  padding:16px 18px;
  border:1px solid #CFE4D2;
  border-left:5px solid var(--green);
  border-radius:18px;
  background:linear-gradient(90deg,#EAF4EC,#F9FCF8);
  color:#173D2A;
}
.shap-explain-icon{
  display:grid;
  place-items:center;
  flex:0 0 48px;
  width:48px;
  height:38px;
  border-radius:13px;
  background:#1F6A43;
  color:#fff;
  font-size:.72rem;
  font-weight:950;
  letter-spacing:.04em;
  line-height:1;
}
.shap-explain-title{
  font-family:var(--serif);
  font-size:1.18rem;
  line-height:1.2;
  font-weight:900;
  color:var(--primary);
  margin-bottom:4px;
}
.shap-explain-text{
  font-size:1rem;
  line-height:1.55;
  font-weight:720;
}
.company-shap-matrix{
  display:grid;
  grid-template-columns:minmax(0,1fr);
  gap:2.2rem;
  width:100%;
}
.company-shap-row{
  display:block;
  width:100%;
  padding:2.15rem 2.35rem;
  margin-bottom:0;
  border-radius:28px;
  background:#FFFFFF;
  border:1px solid var(--border);
  box-shadow:var(--shadow);
  box-sizing:border-box;
}
.company-shap-header{
  display:flex;
  justify-content:space-between;
  align-items:flex-start;
  gap:1.5rem;
  margin-bottom:1.2rem;
}
.company-shap-title{
  font-family:var(--serif);
  font-size:1.35rem;
  line-height:1.18;
  font-weight:900;
  color:var(--primary);
  margin-bottom:4px;
}
.company-shap-sector{
  font-size:.92rem;
  line-height:1.25;
  color:var(--secondary);
  font-weight:750;
  margin-bottom:10px;
}
.company-shap-risk{
  min-width:190px;
  text-align:right;
}
.company-shap-image-wrap{
  width:100%;
  display:flex;
  justify-content:center;
  align-items:center;
  margin:1.05rem 0 1.65rem;
  padding:1.2rem 1.25rem;
  border:1px solid var(--border);
  border-radius:22px;
  background:#FFFFFF;
  box-sizing:border-box;
  overflow:visible;
}
.company-shap-image-wrap .shap-img{
  width:100%;
  max-width:1120px;
  height:540px;
  object-fit:contain;
  object-position:center;
  display:block;
}
.company-shap-detail-grid{
  display:grid;
  grid-template-columns:repeat(3,minmax(300px,1fr));
  gap:1.35rem;
  margin-top:1.2rem;
  align-items:stretch;
}
.company-shap-detail-card{
  display:flex;
  flex-direction:column;
  background:#F9FCF8;
  border:1px solid var(--border);
  border-radius:20px;
  padding:1.5rem 1.55rem;
  min-height:218px;
}
.company-shap-label{
  display:flex;
  align-items:center;
  gap:9px;
  font-size:.9rem;
  text-transform:uppercase;
  letter-spacing:.06em;
  font-weight:900;
  color:var(--secondary);
  margin-bottom:11px;
}
.company-shap-label .label-icon{
  display:inline-grid;
  place-items:center;
  flex:0 0 30px;
  width:30px;
  height:30px;
  border-radius:9px;
  background:#EAF4EC;
  color:var(--primary);
  font-size:.8rem;
  font-weight:950;
  line-height:1;
}
.company-shap-text{
  color:#34453A;
  font-size:1.03rem;
  line-height:1.58;
  font-weight:720;
}
.company-shap-row .zh{
  font-size:.98rem;
  line-height:1.62;
  margin-top:8px;
  word-break:keep-all;
  line-break:strict;
}
.section-title-row{
  display:flex;justify-content:space-between;gap:18px;align-items:flex-end;
  border-top:1px solid var(--border);padding-top:18px;
}
.section-title-row h2{margin:0;font-family:var(--serif);color:var(--primary);font-size:30px;line-height:1.12;}
.audit-table{font-size:.9rem !important;table-layout:fixed;min-width:980px;}
.audit-table td,.audit-table th{overflow-wrap:anywhere;}
.small-caption{color:#6F7F72;font-size:.94rem;line-height:1.5;margin-top:8px;}
.download-row{display:grid;gap:8px;}
.team-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px 18px;}
.team-member{font-size:.96rem;line-height:1.5;color:#34453A;}
.team-member .member-id{font-weight:900;color:var(--primary);}
.stButton>button,.stDownloadButton>button{
  background:#1F5C3A !important;color:#fff !important;border:0 !important;border-radius:14px !important;
  font-weight:800 !important;
  font-size:1rem !important;
  min-height:42px !important;
  padding:9px 14px !important;
}
.stButton>button{
  min-height:42px !important;
  padding:9px 14px !important;
  font-size:1rem !important;
}
.stSelectbox div[data-baseweb="select"]>div,.stTextInput input{
  border-radius:14px;border-color:var(--border);background:#fff !important;
  color:var(--primary) !important;
  font-size:1rem;
  opacity:1 !important;
}
.stSelectbox div[data-baseweb="select"],
.stSelectbox div[data-baseweb="select"] *,
.stSelectbox div[data-baseweb="select"] span,
.stSelectbox div[data-baseweb="select"] input{
  color:var(--primary) !important;
  opacity:1 !important;
}
.stSelectbox div[data-baseweb="select"] svg{
  color:var(--primary) !important;
  fill:var(--primary) !important;
}
.stSelectbox label,.stTextInput label,.stSlider label{font-size:1rem !important;font-weight:700 !important;color:var(--primary) !important;}
.filter-status{
  margin-top:10px;
  padding:10px 12px;
  border:1px solid var(--border);
  border-radius:14px;
  background:#F9FCF8;
  color:var(--primary);
  font-size:.96rem;
  font-weight:800;
  line-height:1.35;
}
.filter-status .zh{
  margin-top:4px;
  font-size:.9rem;
  font-weight:700;
  color:#6F7F72;
}
.filter-count{
  margin-top:8px;
  color:#5F7165;
  font-size:.92rem;
  font-weight:800;
  line-height:1.35;
}
.stSlider [data-baseweb="slider"]{color:var(--green);}
button[role="tab"]{border-radius:14px 14px 0 0;font-weight:800;color:var(--primary) !important;font-size:1rem !important;}
button[role="tab"][aria-selected="true"]{background:var(--soft);}
.stCaptionContainer,.stCaptionContainer p{font-size:.9rem !important;color:#6F7F72 !important;}
@media(max-width:980px){
  .page-head,.hero,.grid-2,.grid-3,.grid-4,.grid-6,.concept,.team-grid,.risk-policy-row,.home-process-grid{grid-template-columns:1fr;}
  .home-process-step:after{display:none;}
  .company-shap-header{flex-direction:column;}
  .company-shap-risk{text-align:left;}
  .company-shap-detail-grid{grid-template-columns:1fr;}
  .company-shap-image-wrap .shap-img{height:450px;}
  .page-num{font-size:52px;}
}
</style>
""",
    unsafe_allow_html=True,
)


DISPLAY_TEXT_REPLACEMENTS = {
    "3m " + "Company": "3M Company",
    "Abbvie " + "Inc.": "AbbVie Inc.",
}


def display_text(value) -> str:
    text = "" if value is None else str(value)
    for old, new in DISPLAY_TEXT_REPLACEMENTS.items():
        text = text.replace(old, new)
    return text


def esc(value) -> str:
    return html.escape(display_text(value))


def display_company_name(value) -> str:
    return display_text(value)


def zh(text: str) -> str:
    return f'<div class="zh">{esc(text)}</div>'


def warning_card(title: str, message: str):
    st.markdown(
        f"""
<div class="card warn">
  <h3>{esc(title)}</h3>
  <p>{esc(message)}</p>
  <div class="zh">缺少必要檔案時，儀表板會以提示卡顯示，不會直接中斷。</div>
</div>
""",
        unsafe_allow_html=True,
    )


def page_header(num: str, title: str, subtitle: str, zh_line: str = ""):
    icon_map = {
        "Home": "🏠",
        "AI-Powered ESG Greenwashing Risk Scoring System": "🏠",
        "Data Explorer": "🔎",
        "Model Performance": "📈",
        "Model Contribution Analysis": "🧩",
        "Company Risk Lookup": "🏢",
        "Explainability": "🔍",
        "Business Recommendation": "📝",
    }
    icon = icon_map.get(title, "✅")
    st.markdown(
        f"""
<div class="page-head">
  <div class="page-num">{esc(num)}</div>
  <div class="page-title">
    <div class="page-title-line"><span class="icon-badge">{esc(icon)}</span><h1>{esc(title)}</h1></div>
    <div class="subtitle">{esc(subtitle)}</div>
    {zh(zh_line) if zh_line else ""}
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def kpi_icon(label: str) -> str:
    text = str(label or "").lower()
    if "final scorer" in text or "final model" in text:
        return "🏆"
    if "roc" in text or "auc" in text:
        return "📈"
    if "pr-auc" in text or "high-risk detection" in text:
        return "🎯"
    if "f1" in text:
        return "⚖️"
    if "feature" in text:
        return "🧩"
    if "high-risk" in text or "positive" in text:
        return "⚠️"
    if "holdout" in text:
        return "🛡️"
    if "training" in text or "setup" in text:
        return "🔁"
    if "company" in text or "universe" in text:
        return "🏢"
    if "sector" in text:
        return "🧭"
    if "report" in text or "length" in text:
        return "📄"
    if "external" in text:
        return "🌐"
    if "text" in text or "talk" in text:
        return "💬"
    if "workload" in text:
        return "✅"
    return "📊"


def metric_card(label: str, value: str, zh_text: str = "", color: str = "#2E7D32") -> str:
    icon = kpi_icon(label)
    subtitle = f'<div class="kpi-card-subtitle">{esc(zh_text)}</div>' if zh_text else ""
    return (
        '<div class="metric kpi-card-premium">'
        '<div class="kpi-card-accent-circle"></div>'
        f'<div class="kpi-card-icon">{esc(icon)}</div>'
        f'<div class="label kpi-card-title">{esc(label)}</div>'
        f'<div class="value kpi-card-value" style="color:{color};">{esc(value)}</div>'
        f"{subtitle}"
        "</div>"
    )


def report_caption(zh_text: str, en_text: str) -> str:
    return (
        '<div class="report-caption">'
        f'<div class="report-caption-zh">{esc(zh_text)}</div>'
        f'<div class="report-caption-en">{esc(en_text)}</div>'
        '</div>'
    )


def mini_card(title: str, body: str, zh_text: str = "", css_class: str = "") -> str:
    return (
        f'<div class="card {css_class}"><h3>{esc(title)}</h3>'
        f'<p>{esc(body)}</p>{zh(zh_text) if zh_text else ""}</div>'
    )


def table_badge(value: str) -> str:
    text = str(value or "").strip()
    key = text.lower()
    if "talk" == key:
        cls = "badge-talk feature-pill-talk"
        text = "💬 " + text
    elif "walk" in key or "external" in key:
        cls = "badge-walk feature-pill-walk"
        text = "🌐 " + text
    elif "context" in key:
        cls = "badge-context feature-pill-context"
        text = "🧭 " + text
    elif "final" in key or "formal" in key or "lightgbm" in key:
        cls = "badge-final model-pill-final"
        text = "🛡️ " + text
    elif "challenger" in key or "xgboost" in key:
        cls = "badge-challenger model-pill-challenger"
        text = "📈 " + text
    elif "baseline" in key or "dummy" in key:
        cls = "badge-benchmark model-pill-baseline"
        text = "⚪ " + text
    elif "high" in key:
        cls = "badge-high risk-pill-high"
        text = "🚨 " + text
    elif "medium" in key or "watch" in key:
        cls = "badge-medium risk-pill-medium"
        text = "⚠️ " + text
    elif "low" in key:
        cls = "badge-low risk-pill-low"
        text = "✅ " + text
    elif "benchmark" in key or "forest" in key or "regression" in key:
        cls = "badge-benchmark model-pill-benchmark"
        text = "◼ " + text
    else:
        cls = "badge-benchmark dashboard-badge"
    return f'<span class="dashboard-badge badge-pill {cls}">{esc(text)}</span>'


def styled_table_html(
    df: pd.DataFrame,
    title: str | None = None,
    subtitle: str | None = None,
    badge_cols: list[str] | None = None,
    numeric_cols: list[str] | None = None,
    highlight_first: bool = False,
    challenger_second: bool = False,
    max_rows: int | None = None,
    rank: bool = False,
) -> str:
    if df is None or df.empty:
        body = '<div class="note warning-note">No rows available.</div>'
        return f'<div class="table-panel">{body}</div>'
    data = df.copy()
    if max_rows is not None:
        data = data.head(max_rows)
    badge_cols = set(badge_cols or [])
    numeric_cols = set(numeric_cols or [])
    ths = ''.join(f'<th>{esc(str(col))}</th>' for col in data.columns)
    rows = []
    for i, (_, row) in enumerate(data.iterrows(), start=1):
        row_cls = ""
        if highlight_first and i == 1:
            row_cls = ' class="dashboard-highlight"'
        elif challenger_second and i == 2:
            row_cls = ' class="dashboard-warning"'
        cells = []
        risk_key = ""
        if "Risk Level" in data.columns:
            risk_key = str(row.get("Risk Level", "")).lower()
        risk_score_cls = ""
        if risk_key.startswith("high"):
            risk_score_cls = " risk-pill-high"
        elif risk_key.startswith("medium") or risk_key.startswith("watch"):
            risk_score_cls = " risk-pill-medium"
        elif risk_key.startswith("low"):
            risk_score_cls = " risk-pill-low"
        for j, col in enumerate(data.columns):
            raw = row[col]
            value = "" if pd.isna(raw) else str(raw)
            cls = ' class="num"' if col in numeric_cols else ""
            if col in badge_cols:
                content = table_badge(value)
            elif col == "Risk Score" and risk_score_cls:
                content = f'<span class="metric-chip{risk_score_cls}">{esc(value)}</span>'
            elif col in numeric_cols:
                content = f'<span class="metric-chip">{esc(value)}</span>'
            elif rank and j == 0:
                chip_class = "" if i == 1 else " secondary" if i == 2 else " muted"
                content = f'<span class="dashboard-rank-chip rank-chip{chip_class}">{i}.</span> {esc(value)}'
            else:
                content = esc(value)
            cells.append(f'<td{cls}>{content}</td>')
        rows.append(f'<tr{row_cls}>{"".join(cells)}</tr>')
    header = ""
    if title or subtitle:
        header = '<div class="dashboard-table-header">'
        title_style = ' style="margin-bottom:10px;"' if title == "Input ESG Disclosure Profile" else ""
        header += f'<div class="dashboard-table-title styled-table-title table-panel-title"{title_style}>📊 {esc(title or "")}</div>'
        if subtitle:
            if title == "Input ESG Disclosure Profile":
                subtitle_html = (
                    "Feature values used for the selected company risk score."
                    '<br><span class="zh">本表顯示模型評分使用的 ESG 特徵值。</span>'
                )
            else:
                subtitle_html = esc(subtitle).replace("\n", "<br>")
            header += f'<div class="dashboard-table-subtitle styled-table-subtitle table-panel-subtitle">{subtitle_html}</div>'
        header += "</div>"
    return f'<div class="dashboard-table-card styled-table-card table-panel">{header}<div class="table-wrap"><table class="styled-table dashboard-table"><thead><tr>{ths}</tr></thead><tbody>{"".join(rows)}</tbody></table></div></div>'


def image_html(path: str, alt: str) -> str:
    if not os.path.exists(path):
        return (
            f'<div class="note warning-note"><strong>{esc(alt)} unavailable.</strong><br>'
            "The image file is missing, so this page keeps running with a warning card.</div>"
        )
    with open(path, "rb") as img:
        encoded = base64.b64encode(img.read()).decode("utf-8")
    return f'<img class="shap-img" src="data:image/png;base64,{encoded}" alt="{esc(alt)}">'


def holdout_image_html(path: str, alt: str) -> str:
    if not os.path.exists(path):
        return (
            f'<div class="note warning-note"><strong>{esc(alt)} unavailable.</strong><br>'
            "Run <code>python src/generate_holdout_eval_artifacts.py</code> to generate this chart.</div>"
        )
    with open(path, "rb") as img:
        encoded = base64.b64encode(img.read()).decode("utf-8")
    return f'<div class="holdout-chart-frame"><img class="holdout-img" src="data:image/png;base64,{encoded}" alt="{esc(alt)}"></div>'


def plot_layout(height=360):
    effective_height = int(round(height * 1.2))
    return dict(
        template="plotly_white",
        height=effective_height,
        margin=dict(l=50, r=50, t=60, b=50),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Segoe UI, sans-serif", color="#34453A", size=14),
        title=dict(text="", font=dict(size=17)),
        colorway=["#2E7D32", "#1F5C3A", "#D99A2B", "#C94C3A", "#5F7165"],
    )


@st.cache_data(show_spinner="Loading project dataset ...")
def load_project_data():
    try:
        from src.data_loader import load_merged_dataset
        from src.feature_engineering import build_feature_matrix

        df = load_merged_dataset(verbose=False)
        X, y, meta = build_feature_matrix(df, verbose=False)
        return df, X, y, meta, None
    except Exception as exc:
        return pd.DataFrame(), pd.DataFrame(), pd.Series(dtype=int), {}, str(exc)


@st.cache_resource(show_spinner="Loading LightGBM model ...")
def load_project_model():
    try:
        path = os.path.join(MODELS, "lgbm_best.pkl")
        with open(path, "rb") as model_file:
            return pickle.load(model_file), None
    except Exception as exc:
        return None, str(exc)


@st.cache_data
def safe_read_csv(path: str):
    if not os.path.exists(path):
        return pd.DataFrame(), f"Missing file: {path}"
    try:
        return pd.read_csv(path), None
    except Exception as exc:
        return pd.DataFrame(), str(exc)


def external_validity_summary_html() -> str:
    return f"""
<div class="card compact report-panel">
  <h3>External Validity Check</h3>
  <p>11 of 19 flagged firms (58%) have documented ESG controversies.</p>
  <div class="zh">外部驗證結果：<br>19 家高風險公司中，<br>11 家（58%）具有公開 ESG 爭議紀錄。</div>
</div>
"""


def model_comparison_df():
    df, err = safe_read_csv(os.path.join(OUTPUTS, "model_comparison.csv"))
    if err or df.empty:
        df = pd.DataFrame(
            {
                "model": [
                    "DummyClassifier",
                    "Logistic Regression",
                    "Random Forest",
                    "XGBoost",
                    "LightGBM",
                ],
                "mean_roc_auc": [0.4746, 0.9215, 0.9269, 0.9261, 0.9528],
                "mean_pr_auc": [0.0801, 0.5207, 0.6760, 0.7549, 0.7450],
                "mean_f1": [0.0444, 0.4580, 0.4833, 0.7119, 0.6576],
                "delta_auc_vs_baseline": [0.0, 0.4469, 0.4524, 0.4515, 0.4782],
            }
        )
    return df


def feature_importance_df():
    df, err = safe_read_csv(os.path.join(OUTPUTS, "feature_importance.csv"))
    if err or df.empty:
        df = pd.DataFrame(
            {
                "feature": [
                    "sector_rank",
                    "talk_total_density",
                    "honest_fair_business",
                    "planet_friendly_business",
                    "text_length",
                    "e_s_g_dispersion",
                ],
                "group": ["Walk / External", "Talk", "Walk / External", "Walk / External", "Talk", "Walk / External"],
                "mean_abs_shap": [7.941, 7.169, 1.136, 1.031, 0.812, 0.706],
            }
        )
    return df


def leakage_df():
    df, err = safe_read_csv(os.path.join(OUTPUTS, "leakage_check.csv"))
    if err or df.empty:
        df = pd.DataFrame(
            {
                "feature_set": ["Full (14 features)", "Reduced Walk-Proxy Ablation"],
                "n_features": [14, 12],
                "mean_roc_auc": [0.9528, 0.7410],
            }
        )
    return df


def holdout_robustness_df():
    return safe_read_csv(os.path.join(OUTPUTS, "holdout_robustness_results.csv"))


def combined_external_samples_df():
    return safe_read_csv(os.path.join(DATA_EXTERNAL, "combined_external_weak_labeled_company_samples.csv"))


def nvidia_case_study_df():
    return safe_read_csv(os.path.join(OUTPUTS, "external_company_case_study_nvidia.csv"))


def external_case_studies_df():
    case_df, err = safe_read_csv(os.path.join(OUTPUTS, "external_company_case_studies.csv"))
    if not err and not case_df.empty:
        case_df = case_df.copy()
        if "model_used" not in case_df.columns and "case_study_model" in case_df.columns:
            case_df["model_used"] = case_df["case_study_model"]
        if "document_type" not in case_df.columns:
            case_df["document_type"] = "SEC 10-K"
        return case_df, None

    nvidia_df, nvidia_err = nvidia_case_study_df()
    if nvidia_err or nvidia_df.empty:
        return pd.DataFrame(), err or nvidia_err or "External SEC 10-K company database file is unavailable."

    row = nvidia_df.iloc[0]
    return pd.DataFrame(
        [
            {
                "company": "NVIDIA",
                "ticker": "NVDA",
                "cik": "0001045810",
                "source": "SEC EDGAR 10-K",
                "filing_type": "10-K",
                "filing_url": row.get("source_url", ""),
                "model_used": row.get("case_study_model", "augmented_training_with_synthetic_and_all_external_company_samples"),
                "model_flagged_greenwashing_risk_score": row.get("model_flagged_greenwashing_risk_score", ""),
                "risk_interpretation": "Not flagged as high greenwashing risk under this model.",
                "retrieval_note": row.get("retrieval_note", ""),
                "case_note": "External SEC 10-K inference database row; inference only, not legal proof or formal greenwashing determination.",
            }
        ]
    ), None


def external_sustainability_report_df():
    report_df, err = safe_read_csv(os.path.join(OUTPUTS, "external_sustainability_report_inference.csv"))
    if err or report_df.empty:
        return pd.DataFrame(), err or "External sustainability report inference file is unavailable."
    report_df = report_df.copy()
    if "model_used" not in report_df.columns and "case_study_model" in report_df.columns:
        report_df["model_used"] = report_df["case_study_model"]
    report_df["source"] = "Corporate sustainability report"
    report_df["filing_type"] = "Sustainability Report"
    report_df["filing_url"] = report_df.get("source_url", "")
    report_df["retrieval_note"] = report_df.get("extraction_note", "")
    report_df["case_note"] = "External sustainability report inference example only; not training data and not formal holdout evaluation."
    report_df["document_type"] = "Corporate sustainability report"
    return report_df, None


def combined_external_inference_df():
    sec_df, sec_err = external_case_studies_df()
    sust_df, sust_err = external_sustainability_report_df()
    frames = []
    if sec_err is None and not sec_df.empty:
        sec_df = sec_df.copy()
        sec_df["document_label"] = "SEC 10-K"
        sec_df["document_sort"] = 1
        frames.append(sec_df)
    if sust_err is None and not sust_df.empty:
        sust_df = sust_df.copy()
        sust_df["document_label"] = "Sustainability Report"
        sust_df["document_sort"] = 0
        frames.append(sust_df)
    if not frames:
        return pd.DataFrame(), sec_err or sust_err or "No external inference profiles were found."
    out = pd.concat(frames, ignore_index=True, sort=False)
    if "model_used" not in out.columns and "case_study_model" in out.columns:
        out["model_used"] = out["case_study_model"]
    return out, None


ROBUSTNESS_LABELS = {
    "original_real_training_only": "Original real training only",
    "augmented_training_with_synthetic_disclosures": "Synthetic only",
    "augmented_training_with_synthetic_and_dax_labeled_company_samples": "Synthetic + DAX",
    "augmented_training_with_synthetic_and_all_external_company_samples": "Synthetic + all external",
}


def robustness_display_table() -> tuple[pd.DataFrame, dict, str | None]:
    df, err = holdout_robustness_df()
    external_df, external_err = combined_external_samples_df()
    counts = {
        "train_real": 0,
        "holdout": 0,
        "synthetic": 0,
        "dax": 0,
        "greenwashing_score": 0,
        "external": 0,
        "external_low": 0,
        "external_high": 0,
        "largest_augmented_train": 0,
    }
    if err or df.empty:
        return pd.DataFrame(), counts, err or "Holdout robustness results are unavailable."

    work = df.copy()
    counts["train_real"] = int(work["train_real_n"].max()) if "train_real_n" in work.columns else 0
    counts["holdout"] = int(work["holdout_n_real_companies"].max()) if "holdout_n_real_companies" in work.columns else 0
    counts["synthetic"] = int(work["train_synthetic_n"].max()) if "train_synthetic_n" in work.columns else 0
    counts["dax"] = int(work["train_dax_labeled_n"].max()) if "train_dax_labeled_n" in work.columns else 0
    counts["external"] = int(work["train_external_labeled_n"].max()) if "train_external_labeled_n" in work.columns else 0
    if "train_external_labeled_n" in work.columns:
        totals = work["train_real_n"].fillna(0) + work["train_synthetic_n"].fillna(0) + work["train_external_labeled_n"].fillna(0)
        counts["largest_augmented_train"] = int(totals.max())

    if not external_err and not external_df.empty:
        if "source_dataset" in external_df.columns:
            counts["greenwashing_score"] = int((external_df["source_dataset"].astype(str) == "Greenwashing_Score_Data.xlsx").sum())
        if "manual_greenwashing_label" in external_df.columns:
            labels = pd.to_numeric(external_df["manual_greenwashing_label"], errors="coerce")
            counts["external_low"] = int((labels == 0).sum())
            counts["external_high"] = int((labels == 1).sum())

    rows = []
    for _, row in work.iterrows():
        version = str(row.get("model_version", ""))
        label = ROBUSTNESS_LABELS.get(version, version.replace("_", " ").title())
        train_samples = str(int(row.get("train_real_n", 0)))
        if int(row.get("train_synthetic_n", 0)) > 0:
            train_samples += f" + {int(row.get('train_synthetic_n', 0))}"
        if version.endswith("dax_labeled_company_samples"):
            train_samples += f" + {int(row.get('train_dax_labeled_n', 0))}"
        elif version.endswith("all_external_company_samples"):
            train_samples += f" + {int(row.get('train_external_labeled_n', 0))}"
        rows.append(
            {
                "Validation Setup": label,
                "Training Samples": train_samples,
                "Holdout Test": f"{int(row.get('holdout_n_real_companies', 0))} real companies",
                "ROC-AUC": f"{float(row.get('roc_auc', 0)):.4f}",
                "PR-AUC": f"{float(row.get('pr_auc', 0)):.4f}",
                "F1": f"{float(row.get('f1', 0)):.4f}",
            }
        )
    return pd.DataFrame(rows), counts, None


def nvidia_case_html() -> str:
    case_df, err = external_case_studies_df()
    if err or case_df.empty:
        return (
            '<div class="card compact warning-note"><h3>SEC 10-K External Case Study</h3>'
            '<p>SEC EDGAR metadata was inspected, but a local external case-study output is not available.</p>'
            '<div class="zh">目前找不到本機外部案例輸出，因此不建立虛構分數。</div></div>'
        )
    case_df = case_df.copy()
    case_df["_ticker"] = case_df.get("ticker", "").astype(str).str.upper()
    match = case_df[case_df["_ticker"] == "NVDA"]
    row = match.iloc[0] if not match.empty else case_df.iloc[0]
    score = external_case_score(row)
    score_label = f"{score:.4f}" if score is not None else "N/A"
    source_url = str(row.get("filing_url", row.get("source_url", "")))
    retrieval_note = str(row.get("retrieval_note", ""))
    company = str(row.get("company", "NVIDIA"))
    ticker = str(row.get("ticker", "NVDA"))
    return f"""
<div class="card compact">
  <h3>SEC 10-K External Case Study</h3>
  <div class="zh">外部真實公司案例測試</div>
  <div class="grid-4" style="margin-top:12px;">
    {metric_card("Company", f"{company} / {ticker}", "公司")}
    {metric_card("Source", str(row.get("source", "SEC EDGAR 10-K")), "資料來源", "#5F7165")}
    {metric_card("Risk Score", score_label, "模型風險分數")}
    {metric_card("Interpretation", "Low" if (score is not None and score < 0.30) else "Review", "推論結果", "#2E7D32" if (score is not None and score < 0.30) else "#D99A2B")}
  </div>
  <div class="signal-row"><div class="signal-label">Model used</div><div class="signal-value">{esc(row.get("model_used", row.get("case_study_model", "")))}</div></div>
  <div class="signal-row"><div class="signal-label">Result</div><div class="signal-value">{esc(row.get("risk_interpretation", ""))}<br><span class="zh">此為外部公司推論案例，不是正式模型評估。</span></div></div>
  <div class="signal-row"><div class="signal-label">Retrieval note</div><div class="signal-value">{esc(retrieval_note)}</div></div>
  <div class="small-caption">External qualitative case study only; not formal model evaluation, not legal proof, and not formal greenwashing determination.</div>
  <div class="zh">外部案例僅作定性展示，不屬於正式模型評估，也不是法律認定。</div>
  {f'<div class="small-caption">Source URL: {esc(source_url)}</div>' if source_url else ''}
</div>
"""


def model_contribution_df():
    return safe_read_csv(os.path.join(OUTPUTS, "model_contribution_analysis.csv"))


def project_medians(X: pd.DataFrame) -> dict:
    if X.empty:
        return FALLBACK_MEDIANS.copy()
    med = {}
    for feature in FEATURE_ORDER:
        if feature in X.columns:
            med[feature] = float(pd.to_numeric(X[feature], errors="coerce").median())
        else:
            med[feature] = FALLBACK_MEDIANS.get(feature, 0.0)
    return med


def risk_tier(prob: float):
    if prob >= 0.70:
        return "High", "Escalate to Enhanced ESG Due Diligence", "#C94C3A", "高風險，升級盡職調查"
    if prob >= 0.30:
        return "Medium", "Request supporting ESG evidence and peer comparison.", "#D99A2B", "中風險，要求補充資料"
    return "Low", "Standard ESG Review", "#2E7D32", "低風險，標準審查"


def predict_probability(model, feature_vector: dict):
    if model is None:
        return None
    try:
        row = np.array([[feature_vector[f] for f in FEATURE_ORDER]], dtype=float)
        return float(model.predict_proba(row)[0, 1])
    except Exception:
        return None


def company_profile(ticker: str, df: pd.DataFrame, X: pd.DataFrame, y: pd.Series, meta: dict, medians: dict):
    ticker = ticker.upper().strip()
    if ticker == "CVX":
        fallback = CVX_DEFAULTS.copy()
    else:
        fallback = medians.copy()
        fallback.update({"ticker": ticker, "company": ticker, "sector": "Energy" if ticker == "CVX" else SECTORS[0]})
        fallback["sector_encoded"] = float(SECTOR_ENC.get(fallback["sector"], 0))
    if X.empty or not meta or "ticker" not in meta:
        return fallback, False, "Not available"

    tickers = pd.Series(meta["ticker"]).astype(str).str.upper()
    match = tickers[tickers == ticker]
    if match.empty:
        return fallback, False, "Not available"
    idx = int(match.index[0])
    profile = {feature: float(X.iloc[idx][feature]) for feature in FEATURE_ORDER}
    sector = str(pd.Series(meta.get("sector", [])).iloc[idx]) if "sector" in meta else fallback.get("sector", SECTORS[0])
    profile["sector"] = sector
    profile["sector_encoded"] = float(SECTOR_ENC.get(sector, profile.get("sector_encoded", 0.0)))
    profile["ticker"] = ticker
    if not df.empty and idx < len(df):
        row = df.iloc[idx]
        profile["company"] = row.get("Name", row.get("company_name", ticker))
    else:
        profile["company"] = ticker
    label = "High-risk label" if int(pd.Series(y).iloc[idx]) == 1 else "Non-high-risk label"
    return profile, True, label


def dataset_company_options(df: pd.DataFrame, meta: dict) -> list[dict]:
    if df.empty or not meta or "ticker" not in meta:
        return []
    tickers = pd.Series(meta.get("ticker", [])).astype(str)
    sectors = pd.Series(meta.get("sector", df.get("Sector", []))).astype(str)
    options = []
    for idx, ticker in tickers.items():
        if idx >= len(df) or not ticker or ticker.lower() == "nan":
            continue
        row = df.iloc[int(idx)]
        company = str(row.get("Name", row.get("company_name", ticker)))
        sector = str(sectors.iloc[int(idx)]) if int(idx) < len(sectors) else str(row.get("Sector", "Unknown"))
        options.append(
            {
                "label": f"{company} / {ticker.upper()} / {sector}",
                "ticker": ticker.upper(),
                "company": company,
                "sector": sector,
            }
        )
    return sorted(options, key=lambda item: (item["company"].lower(), item["ticker"]))


def next_due_diligence_step(tier: str) -> tuple[str, str]:
    if tier == "High":
        return (
            "Request third-party assurance, board oversight explanation, and recent controversy review before sustainable lending approval.",
            "建議要求第三方確信、董事會監督說明與近期爭議紀錄，再評估是否適用永續授信。",
        )
    if tier == "Medium":
        return (
            "Request supporting ESG evidence, peer comparison, and clarification of key ESG claims.",
            "建議要求補充 ESG 證據、同業比較與關鍵揭露說明。",
        )
    return (
        "Proceed with standard ESG review and verify loan purpose alignment.",
        "維持一般 ESG 審查，並確認資金用途是否符合永續金融條件。",
    )


def median_comparison_label(value: float, median: float) -> str:
    tolerance = max(1e-6, abs(float(median)) * 0.02)
    diff = float(value) - float(median)
    if abs(diff) <= 1e-9:
        return "at median / 等於中位數"
    if abs(diff) <= tolerance:
        return "near median / 接近中位數"
    if diff > 0:
        return "above median / 高於中位數"
    return "below median / 低於中位數"


RESPONSIBLE_MEMO_DISCLAIMER = (
    "This model does not accuse a company of greenwashing. It flags potential risk for further ESG due diligence."
)
RESPONSIBLE_MEMO_DISCLAIMER_ZH = "模型不直接認定企業漂綠，而是提供進一步 ESG 查核的風險提示。"
EXTERNAL_INFERENCE_DISCLAIMER = (
    "Final credit and investment decisions remain subject to internal review and supporting documentation."
)
EXTERNAL_INFERENCE_DISCLAIMER_ZH = "最終授信與投資決策仍須依內部審查與佐證文件辦理。"
EXTERNAL_TEXT_PREPROCESSING_MEMO_PHRASE = "(external text not identically preprocessed) / （外部文件未經完全相同預處理）"
def memo_tier_content(tier: str) -> dict:
    if tier == "High":
        return {
            "main_concern": "High ESG disclosure intensity combined with weak external ESG evidence.",
            "main_concern_zh": "ESG 揭露強度高，但外部 ESG 證據偏弱。",
            "recommended_action": "Escalate to Enhanced ESG Due Diligence.",
            "recommended_action_zh": "升級 ESG 盡職調查。",
            "required_evidence": "Third-party assurance, board oversight explanation, and recent controversy review before sustainable lending approval.",
            "required_evidence_zh": "建議要求第三方確信、董事會監督說明與近期爭議紀錄，再評估是否適用永續授信。",
            "monitoring": "Quarterly review or watchlist monitoring.",
            "monitoring_zh": "季度追蹤或列入觀察名單。",
        }
    if tier == "Medium":
        return {
            "main_concern": "Moderate Talk-Walk mismatch or uncertain external ESG support.",
            "main_concern_zh": "可能存在中度說做落差，或外部 ESG 證據仍需補強。",
            "recommended_action": "Request supporting ESG evidence and peer comparison.",
            "recommended_action_zh": "要求補充 ESG 證據，並與同業比較。",
            "required_evidence": "Supporting ESG evidence, peer comparison, and clarification of key ESG claims.",
            "required_evidence_zh": "要求 ESG 證據、同業比較與關鍵揭露說明。",
            "monitoring": "Semi-annual review.",
            "monitoring_zh": "半年追蹤。",
        }
    return {
        "main_concern": "No strong Talk-Walk mismatch detected under current assumptions.",
        "main_concern_zh": "目前未偵測到明顯的說做落差。",
        "recommended_action": "Standard ESG review.",
        "recommended_action_zh": "維持一般 ESG 審查流程。",
        "required_evidence": "Regular sustainability disclosure and loan-purpose documentation.",
        "required_evidence_zh": "確認永續揭露與資金用途文件。",
        "monitoring": "Annual ESG review.",
        "monitoring_zh": "年度追蹤。",
    }


def profile_source_label(profile_source: str) -> tuple[str, str]:
    if profile_source == "Dataset Company":
        return "Internal benchmark profile", "資料來源：內部基準公司資料"
    if profile_source == "External SEC 10-K company database":
        return "External official filing inference", "資料來源：外部正式申報推論"
    return "Custom Input", "資料來源：手動輸入"


def company_memo_details(company, ticker, sector, prob, tier, profile_source, industry="") -> dict:
    source, source_zh = profile_source_label(profile_source)
    clean_ticker = "" if str(ticker).upper() in {"", "CUSTOM", "NAN", "NONE"} else str(ticker)
    lookup_mode = "Manual Company Input" if profile_source == "Manual Input" else (
        "External Official Filing Example" if profile_source == "External SEC 10-K company database" else "Select Existing Company"
    )
    return {
        "company": str(company or "Custom Company"),
        "ticker": clean_ticker,
        "sector": str(sector or "Unknown"),
        "industry": str(industry or ""),
        "lookup_mode": lookup_mode,
        "data_source_type": source,
        "model_used": "LightGBM final scorer",
        "risk_probability": float(prob or 0.0),
        "risk_tier": str(tier or "Low"),
        "source": source,
        "source_zh": source_zh,
        "disclaimer": RESPONSIBLE_MEMO_DISCLAIMER,
        "disclaimer_zh": RESPONSIBLE_MEMO_DISCLAIMER_ZH,
        **memo_tier_content(tier),
    }


def format_cik(value) -> str:
    raw = "" if pd.isna(value) else str(value).strip()
    if not raw:
        return ""
    if raw.endswith(".0"):
        raw = raw[:-2]
    digits = "".join(ch for ch in raw if ch.isdigit())
    return digits.zfill(10) if digits else raw


def clean_optional_text(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "null"} else text


def external_document_type_label(document_label: str) -> str:
    label = str(document_label or "").strip()
    if EXTERNAL_TEXT_PREPROCESSING_MEMO_PHRASE in label:
        return label
    return f"{label} {EXTERNAL_TEXT_PREPROCESSING_MEMO_PHRASE}".strip()


def external_case_score(row) -> float | None:
    value = pd.to_numeric(row.get("model_flagged_greenwashing_risk_score", np.nan), errors="coerce")
    if pd.isna(value):
        return None
    return float(value)


def risk_tier_display_range(tier: str) -> str:
    if tier == "High":
        return "70–100%"
    if tier == "Medium":
        return "30–70%"
    return "0–30%"


def risk_review_action(tier: str) -> tuple[str, str]:
    if tier == "High":
        return "Escalate to enhanced ESG due diligence.", "升級 ESG 盡職調查。"
    if tier == "Medium":
        return "Request supporting evidence and peer comparison.", "要求補充佐證資料並進行同業比較。"
    return "Standard ESG review.", "一般 ESG 審查。"


def external_case_interpretation(company: str, ticker: str, tier: str) -> tuple[str, str]:
    company_upper = str(company or "").upper()
    ticker_upper = str(ticker or "").upper()
    if ticker_upper == "NVDA" or "NVIDIA" in company_upper:
        return (
            "NVIDIA was not used for training. We used NVIDIA's official SEC filing disclosure as an external company inference case. Under the current LightGBM final scorer and available filing-based features, NVIDIA was not flagged as high greenwashing risk.",
            "NVIDIA 沒有被模型標記為高漂綠風險；這不等於法律上判定沒有漂綠。",
        )
    if tier == "High":
        return (
            "The company is model-flagged as higher greenwashing risk under the current LightGBM final scorer and available filing-based features; further ESG due diligence is recommended.",
            "模型標記為較高風險，建議進一步 ESG 查核。",
        )
    return (
        "The company was not used for training. Under the current LightGBM final scorer and available filing-based features, the company was not flagged as high greenwashing risk.",
        "該公司沒有被模型標記為高漂綠風險；這不等於法律上判定沒有漂綠。",
    )


def display_case_note(text: str) -> str:
    note = str(text or "").strip()
    if not note:
        return ""
    replacements = {
        "not legal proof or ": "not ",
        "not legal proof": "not a formal decision",
        "formal greenwashing determination": "formal greenwashing conclusion",
    }
    for old, new in replacements.items():
        note = note.replace(old, new)
    return note


def external_case_memo_details(row) -> dict | None:
    score = external_case_score(row)
    if score is None:
        return None
    tier, _, _, _ = risk_tier(score)
    tier_content = memo_tier_content(tier)
    recommended_action, recommended_action_zh = risk_review_action(tier)
    company = str(row.get("company", "External Company"))
    ticker = str(row.get("ticker", "")).strip()
    document_type = str(row.get("document_type", row.get("filing_type", "SEC 10-K"))).strip() or "SEC 10-K"
    document_label = str(row.get("document_label", document_type)).strip() or document_type
    is_sustainability = "sustainability" in document_type.lower() or "sustainability" in document_label.lower()
    interpretation, interpretation_zh = external_case_interpretation(company, ticker, tier)
    if is_sustainability:
        source = "External sustainability report inference"
        source_zh = "資料來源：外部公司永續報告推論"
        data_source_type = "External sustainability report inference"
        filing_type = "Sustainability Report"
        key_input_signals = "Scored from the company's sustainability report — same document type as model training."
        document_note = "Scored from the company's sustainability report — same document type as model training."
        document_note_zh = "以公司永續報告書評分，與模型訓練文件類型一致。"
        domain_shift_caveat = ""
        domain_shift_caveat_zh = ""
    else:
        source = "External SEC 10-K inference"
        source_zh = "資料來源：外部 SEC 10-K 推論"
        data_source_type = "External SEC 10-K inference"
        filing_type = str(row.get("filing_type", "10-K")).strip() or "10-K"
        key_input_signals = "Precomputed SEC 10-K risk profile; final score reflects all 14 model features combined."
        document_note = "SEC 10-K inference is shown as an out-of-domain filing example."
        document_note_zh = "SEC 10-K 推論為文件類型差異案例。"
        domain_shift_caveat = (
            "Out-of-domain caveat: the model was trained on sustainability reports; full SEC 10-K filings have structurally lower ESG language density, "
            "so 10-K scores indicate document-type domain shift rather than a fine-grained company ranking."
        )
        domain_shift_caveat_zh = "文件類型提醒：模型以永續報告訓練，完整 SEC 10-K 的 ESG 語言密度較低，因此 10-K 分數反映文件類型差異，不宜作細緻公司排名。"
        document_label = f"{document_label} (official annual filing with the U.S. SEC) / （美國 SEC 正式年度申報）"
    document_label = external_document_type_label(document_label)
    return {
        "company": company,
        "ticker": ticker,
        "cik": format_cik(row.get("cik", "")),
        "sector": "Not assigned in prototype profile",
        "industry": "",
        "lookup_mode": "External Official Filing Example",
        "data_source_type": data_source_type,
        "model_used": "LightGBM final scorer",
        "document_type": document_label,
        "document_note": document_note,
        "document_note_zh": document_note_zh,
        "domain_shift_caveat": domain_shift_caveat,
        "domain_shift_caveat_zh": domain_shift_caveat_zh,
        "filing_type": filing_type,
        "filing_url": clean_optional_text(row.get("filing_url", row.get("source_url", ""))),
        "source_file": clean_optional_text(row.get("source_file", "")),
        "external_source": clean_optional_text(row.get("source", document_label)),
        "retrieval_note": clean_optional_text(row.get("retrieval_note", "")),
        "scoring_note": clean_optional_text(row.get("scoring_note", "")),
        "risk_probability": score,
        "risk_tier": tier,
        "source": source,
        "source_zh": source_zh,
        "main_concern": interpretation,
        "main_concern_zh": interpretation_zh,
        "recommended_action": recommended_action,
        "recommended_action_zh": recommended_action_zh,
        "required_evidence": "Review official filing disclosure, external ESG evidence, controversy history, and supporting documents.",
        "required_evidence_zh": "查核正式申報揭露、外部 ESG 證據、爭議紀錄與佐證文件。",
        "monitoring": tier_content["monitoring"],
        "monitoring_zh": tier_content["monitoring_zh"],
        "key_input_signals": key_input_signals,
        "disclaimer": EXTERNAL_INFERENCE_DISCLAIMER,
        "disclaimer_zh": "本 memo 供 ESG 授信、永續金融與投資審查參考，不代表法律上認定漂綠。",
    }


def memo_row(label: str, value: str, zh: str = "") -> str:
    zh_html = f'<br><span class="zh">{esc(zh)}</span>' if zh else ""
    return f'<div class="signal-row memo-summary-card"><div class="signal-label">{esc(label)}</div><div class="signal-value">{esc(value)}{zh_html}</div></div>'


def memo_html(details: dict) -> str:
    ticker_row = memo_row("Ticker / 股票代碼", details.get("ticker", "")) if details.get("ticker") else ""
    cik_row = memo_row("CIK", details.get("cik", "")) if details.get("cik") else ""
    document_type_row = memo_row("Document Type / 文件類型", details.get("document_type", "")) if details.get("document_type") else ""
    document_note_row = memo_row("Document Note / 文件說明", details.get("document_note", ""), details.get("document_note_zh", "")) if details.get("document_note") else ""
    domain_shift_row = memo_row("Domain Shift Caveat / 文件類型差異提醒", details.get("domain_shift_caveat", ""), details.get("domain_shift_caveat_zh", "")) if details.get("domain_shift_caveat") else ""
    filing_type_row = memo_row("Filing Type", details.get("filing_type", "")) if details.get("filing_type") else ""
    filing_url_row = memo_row("Filing URL", details.get("filing_url", "")) if details.get("filing_url") else ""
    source_file_row = memo_row("Source File", details.get("source_file", "")) if details.get("source_file") else ""
    industry_row = memo_row("Industry / 子產業", details.get("industry", "")) if details.get("industry") else ""
    lookup_row = memo_row("Lookup Mode / 查詢模式", details.get("lookup_mode", "")) if details.get("lookup_mode") else ""
    data_source_row = memo_row("Data Source Type / 資料類型", details.get("data_source_type", details.get("source", "")))
    model_row = memo_row("Model Used / 使用模型", details.get("model_used", "LightGBM final scorer"), "模型：正式 LightGBM 評分器")
    key_signal_row = memo_row("Key Input Signals / 主要輸入訊號", details.get("key_input_signals", "")) if details.get("key_input_signals") else ""
    return (
        memo_row("Source / 資料來源", details.get("source", ""), details.get("source_zh", ""))
        + lookup_row
        + data_source_row
        + document_type_row
        + document_note_row
        + domain_shift_row
        + model_row
        + memo_row("Company / 公司", details.get("company", ""))
        + ticker_row
        + cik_row
        + filing_type_row
        + filing_url_row
        + source_file_row
        + memo_row("Sector / 產業", details.get("sector", ""))
        + industry_row
        + memo_row("Model Risk Score / 模型風險分數", f"{details.get('risk_probability', 0)*100:.1f}%")
        + memo_row("Risk Tier / 風險等級", details.get("risk_tier", ""))
        + key_signal_row
        + memo_row("Main Concern / 主要疑慮", details.get("main_concern", ""), details.get("main_concern_zh", ""))
        + memo_row("Recommended Action / 建議行動", details.get("recommended_action", ""), details.get("recommended_action_zh", ""))
        + memo_row("Required Evidence / 需補充證據", details.get("required_evidence", ""), details.get("required_evidence_zh", ""))
        + memo_row("Monitoring / 追蹤頻率", details.get("monitoring", ""), details.get("monitoring_zh", ""))
        + memo_row("Final Note / 最終提醒", details.get("disclaimer", ""), details.get("disclaimer_zh", ""))
    )


def memo_markdown_from_details(details: dict) -> str:
    ticker_line = f"Ticker / 股票代碼: {details.get('ticker', '')}\n" if details.get("ticker") else ""
    cik_line = f"CIK: {details.get('cik', '')}\n" if details.get("cik") else ""
    document_type_line = f"Document Type / 文件類型: {details.get('document_type', '')}\n" if details.get("document_type") else ""
    document_note_line = (
        f"Document Note / 文件說明: {details.get('document_note', '')}\n{details.get('document_note_zh', '')}\n"
        if details.get("document_note")
        else ""
    )
    domain_shift_line = (
        f"Domain Shift Caveat / 文件類型差異提醒:\n{details.get('domain_shift_caveat', '')}\n{details.get('domain_shift_caveat_zh', '')}\n"
        if details.get("domain_shift_caveat")
        else ""
    )
    filing_type_line = f"Filing Type: {details.get('filing_type', '')}\n" if details.get("filing_type") else ""
    filing_url_line = f"Filing URL: {details.get('filing_url', '')}\n" if details.get("filing_url") else ""
    source_file_line = f"Source File: {details.get('source_file', '')}\n" if details.get("source_file") else ""
    industry_line = f"Industry / 子產業: {details.get('industry', '')}\n" if details.get("industry") else ""
    key_signal_line = f"\nKey Input Signals / 主要輸入訊號:\n{details.get('key_input_signals', '')}\n" if details.get("key_input_signals") else ""
    return f"""# ESG Credit Review Memo

Source / 資料來源: {details.get('source', '')}
Lookup Mode / 查詢模式: {details.get('lookup_mode', '')}
Data Source Type / 資料類型: {details.get('data_source_type', details.get('source', ''))}
{document_type_line}{document_note_line}{domain_shift_line}
Model Used / 使用模型: {details.get('model_used', 'LightGBM final scorer')}
Company / 公司: {details.get('company', '')}
{ticker_line}{cik_line}{filing_type_line}{filing_url_line}{source_file_line}Sector / 產業: {details.get('sector', '')}
{industry_line}Model Risk Score / 模型風險分數: {details.get('risk_probability', 0)*100:.1f}%
Risk Tier / 風險等級: {details.get('risk_tier', '')}
{key_signal_line}
Main Concern / 主要疑慮:
{details.get('main_concern', '')}
{details.get('main_concern_zh', '')}

Recommended Credit Action / 建議授信行動:
{details.get('recommended_action', '')}
{details.get('recommended_action_zh', '')}

Required Evidence / 需補充證據:
{details.get('required_evidence', '')}
{details.get('required_evidence_zh', '')}

Monitoring / 追蹤頻率:
{details.get('monitoring', '')}
{details.get('monitoring_zh', '')}

Final Note / 最終提醒:
{details.get('disclaimer', '')}
{details.get('disclaimer_zh', '')}
"""


def memo_filename(details: dict) -> str:
    stem = details.get("ticker") or details.get("company") or "Company"
    safe = "".join(ch if ch.isalnum() else "_" for ch in str(stem)).strip("_")
    return f"{safe or 'Company'}_ESG_Credit_Memo.md"


def memo_pdf_filename(details: dict) -> str:
    stem = details.get("ticker") or details.get("company") or "Company"
    safe = "".join(ch if ch.isalnum() else "_" for ch in str(stem)).strip("_")
    return f"{safe or 'Company'}_ESG_Credit_Memo.pdf"


def company_memo_pdf_bytes(details: dict) -> bytes:
    ticker_lines = [details.get("ticker", "")] if details.get("ticker") else ["Not available"]
    sections = [
        ("Source / 資料來源", [details.get("source", ""), details.get("source_zh", "")]),
        ("Lookup Mode / 查詢模式", [details.get("lookup_mode", "Not available") or "Not available"]),
        ("Data Source Type / 資料類型", [details.get("data_source_type", details.get("source", ""))]),
        ("Document Type / 文件類型", [details.get("document_type", "Not available") or "Not available"]),
        ("Model Used / 使用模型", [details.get("model_used", "LightGBM final scorer"), "模型：正式 LightGBM 評分器"]),
        ("Company / 公司", [details.get("company", "")]),
        ("Ticker / 股票代碼", ticker_lines),
        ("CIK", [details.get("cik", "Not available") or "Not available"]),
        ("Filing Type", [details.get("filing_type", "Not available") or "Not available"]),
        ("Filing URL", [details.get("filing_url", "Not available") or "Not available"]),
        ("Sector / 產業", [details.get("sector", "")]),
    ]
    if details.get("source_file"):
        sections.append(("Source File", [details.get("source_file", "")]))
    if details.get("document_note"):
        sections.append(("Document Note / 文件說明", [details.get("document_note", ""), details.get("document_note_zh", "")]))
    if details.get("domain_shift_caveat"):
        sections.append(("Domain Shift Caveat / 文件類型差異提醒", [details.get("domain_shift_caveat", ""), details.get("domain_shift_caveat_zh", "")]))
    if details.get("industry"):
        sections.append(("Industry / 子產業", [details.get("industry", "")]))
    sections.extend(
        [
            ("Model Risk Score / 模型風險分數", [f"{details.get('risk_probability', 0)*100:.1f}%"]),
            ("Risk Tier / 風險等級", [details.get("risk_tier", "")]),
        ]
    )
    if details.get("key_input_signals"):
        sections.append(("Key Input Signals / 主要輸入訊號", [details.get("key_input_signals", "")]))
    sections.extend(
        [
            ("Main Concern / 主要疑慮", [details.get("main_concern", ""), details.get("main_concern_zh", "")]),
            ("Recommended Credit Action / 建議授信行動", [details.get("recommended_action", ""), details.get("recommended_action_zh", "")]),
            ("Required Evidence / 需補充證據", [details.get("required_evidence", ""), details.get("required_evidence_zh", "")]),
            ("Monitoring Frequency / 追蹤頻率", [details.get("monitoring", ""), details.get("monitoring_zh", "")]),
            ("Final Note / 最終提醒", [details.get("disclaimer", RESPONSIBLE_MEMO_DISCLAIMER), details.get("disclaimer_zh", RESPONSIBLE_MEMO_DISCLAIMER_ZH)]),
        ]
    )
    report = {"title": "ESG Credit Review Memo", "sections": sections}
    try:
        return _build_pdf_with_reportlab(report)
    except Exception:
        return _build_pdf_with_matplotlib(report)


def feature_group(feature: str) -> str:
    if feature in TALK_FEATURES:
        return "Talk"
    if feature in WALK_FEATURES:
        return "Walk / External"
    return "Context"


def prep_hover_df(df, X, y, meta):
    if df.empty or X.empty:
        return pd.DataFrame()
    names = df["Name"] if "Name" in df.columns else df.get("company_name", pd.Series(meta.get("ticker", [])))
    out = pd.DataFrame(
        {
            "ticker": pd.Series(meta.get("ticker", df.get("ticker", []))).astype(str),
            "company": pd.Series(names).astype(str),
            "sector": pd.Series(meta.get("sector", df.get("Sector", []))).astype(str),
            "talk_score": pd.to_numeric(X["talk_total_density"], errors="coerce"),
            "walk_external_score": pd.to_numeric(X["sector_rank"], errors="coerce"),
            "risk_label": pd.Series(y).map({1: "High-risk Label", 0: "Non-high-risk"}),
        }
    )
    return out


def sidebar():
    with st.sidebar:
        today = datetime.now().strftime("%Y-%m-%d")
        st.markdown(
            """
<div class="sidebar-brand">
  <div class="sidebar-brand-top">
    <div class="sidebar-mark"><span class="sidebar-leaf"></span><span>ESG</span><span class="sidebar-shield">✓</span></div>
  </div>
  <div class="sidebar-title"><span class="sidebar-title-main">Greenwashing</span><span class="sidebar-title-subline">Risk System</span></div>
  <div class="sidebar-sub">DSF504 Final Project | NSYSU</div>
  <div class="sidebar-team">
    <div class="sidebar-team-eyebrow">Project Team <span>專案團隊</span></div>
    <div class="sidebar-team-title">TEAM 5</div>
    <div class="sidebar-member"><span class="member-id">M14B020803</span><span class="member-name">Chester Wu</span></div>
    <div class="sidebar-member"><span class="member-id">M14B020807</span><span class="member-name">Stanley Cheng</span></div>
    <div class="sidebar-member"><span class="member-id">M14B020809</span><span class="member-name">Sophia Hsu</span></div>
    <div class="sidebar-member"><span class="member-id">M14B020812</span><span class="member-name">Alice Hsu</span></div>
  </div>
  <div class="sidebar-identity">
    <span class="identity-icon">AI</span>
    <span>AI-assisted ESG risk screening<div class="zh">AI 輔助 ESG 風險篩選</div></span>
  </div>
  <div class="sidebar-link" style="margin-top:10px;font-size:.82rem;opacity:.82;">UI version: all tables redesigned v5</div>
</div>
""",
            unsafe_allow_html=True,
        )
        st.markdown(
            """
<div class="sidebar-nav-heading">
  <div class="nav-title">Navigation</div>
  <div class="nav-zh">頁面導覽</div>
</div>
""",
            unsafe_allow_html=True,
        )
        page = st.radio(
            "Navigation",
            [
                "Home",
                "Data Explorer",
                "Model Performance",
                "Model Contribution Analysis",
                "Company Risk Lookup",
                "Explainability",
                "Business Recommendation",
            ],
            label_visibility="collapsed",
        )
        st.markdown(
            f"""
<div class="sidebar-status-card">
  <div class="sidebar-status-icon">↻</div>
  <div>
    <div class="sidebar-status-label">Data Refresh</div>
    <div class="sidebar-status-date">{today}</div>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )
        return page


def page_home():
    page_header(
        "01",
        "AI-Powered ESG Greenwashing Risk Scoring System",
        "DSF504 Final Project | Team 5 | NSYSU",
        "永續授信與 ESG 投資的綠漂風險輔助判斷工具",
    )
    st.markdown(
        """
<div class="card hero report-panel home-hero-panel">
  <div class="home-hero-copy">
    <div class="kicker">Business Purpose</div>
    <h2>ESG Greenwashing Risk Screening for Financial Institutions</h2>
    <p class="mission">Financial institutions need a practical way to screen ESG disclosure credibility before sustainable lending, ESG investment, or credit review decisions. This dashboard uses official company filing text and ESG-related features to generate a model-flagged greenwashing risk score for further due diligence.</p>
  </div>
  <div class="botanical"></div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:22px;'></div>", unsafe_allow_html=True)

    st.markdown(
        report_caption(
            "正式模型成果一覽:LightGBM 排序 ROC-AUC 0.953",
            "Final model results at a glance.",
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div class='card report-panel'>"
        "<div class='kicker'>Model Outcome</div>"
        "<h3>Final Results Summary</h3>"
        "<div class='zh'>最終模型成果摘要</div>"
        "<div style='height:16px;'></div>"
        "<div class='grid-4 home-metrics'>"
        + metric_card("Final Scorer", "LightGBM", "Formal risk-ranking model")
        + metric_card("Formal ROC-AUC", "0.953", "Nested-CV result")
        + metric_card("Formal PR-AUC", "0.745", "High-risk detection")
        + metric_card("Formal F1", "0.658", "Precision-recall balance")
        + metric_card("Features", "14", "Talk + Walk/External + Context")
        + metric_card("High-Risk Labels", "19 / 257", "7.4% positive rate", "#B8860B")
        + metric_card("Real Holdout", "78", "Fixed real-company test set")
        + metric_card("Largest Training Setup", "407", "Robustness validation / training augmentation")
        + "</div>"
        + "<div class='note' style='margin-top:16px;'>"
        + "Largest Training Setup = 407 refers to the largest robustness validation and training-augmentation setup. It is not the formal holdout or final test set. The 78 real-company holdout set remains fixed."
        + "<br><span class='zh'>407 是最大訓練補強設定，用於穩健性驗證；不是正式 holdout 或 final test set。78 家真實公司 holdout set 維持固定。</span>"
        + "</div></div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:22px;'></div>", unsafe_allow_html=True)

    st.markdown(
        """
<div class="card compact report-panel">
  <div class="kicker">Filing Source</div>
  <h3>Data Source</h3>
  <p>Data source: official annual filings submitted by U.S. listed companies to the SEC.</p>
  <div class="zh">資料來源：美國上市公司提交給 SEC 的正式年度申報文件。</div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:22px;'></div>", unsafe_allow_html=True)

    st.markdown(
        """
<div class="card report-panel">
  <div class="kicker">Model Use Case</div>
  <h3>From Filing Text to ESG Review Memo</h3>
  <div class="table-wrap">
    <table class="report-table" style="min-width:760px;">
      <tbody>
        <tr><td><strong>1</strong></td><td>Company official filing</td><td class="zh">公司正式申報文件</td></tr>
        <tr><td><strong>2</strong></td><td>ESG disclosure feature extraction</td><td class="zh">ESG 揭露特徵擷取</td></tr>
        <tr><td><strong>3</strong></td><td>LightGBM risk scoring</td><td class="zh">LightGBM 風險評分</td></tr>
        <tr><td><strong>4</strong></td><td>ESG credit review memo</td><td class="zh">ESG 授信查核 Memo</td></tr>
        <tr><td><strong>5</strong></td><td>Human review and supporting documentation</td><td class="zh">人工審查與佐證文件</td></tr>
      </tbody>
    </table>
  </div>
  <div class="note" style="margin-top:14px;">Company official filing → ESG disclosure feature extraction → LightGBM risk scoring → ESG credit review memo → Human review and supporting documentation</div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:22px;'></div>", unsafe_allow_html=True)

    st.markdown(
        report_caption(
            "核心概念:公司「說得多」但「做得差」,就是漂綠風險訊號",
            "High disclosure + weak external evidence = greenwashing risk signal.",
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        """
<div class="card report-panel">
  <div class="kicker">Greenwashing Risk Definition</div>
  <h3>Talk + Walk Screening Logic</h3>
  <p><strong>High ESG disclosure intensity + weak external ESG signals = model-flagged greenwashing risk.</strong></p>
  <div class="zh">ESG 揭露強度高 + 外部 ESG 訊號偏弱 = 模型標記的漂綠風險。</div>
  <div class="concept" style="margin-top:16px;">
    <div class="block"><h3>💬 Talk</h3><p>ESG disclosure intensity from company text</p><div class="zh">公司文字揭露強度</div></div>
    <div class="formula">High Talk + Weak Walk<br>= Risk Screening Signal</div>
    <div class="block"><h3>🌐 Walk / External</h3><p>External ESG signals and sector-relative evidence</p><div class="zh">外部 ESG 訊號與同產業相對證據</div></div>
  </div>
  <div class="chip-row" style="margin-top:14px;">
    <span class="badge-pill risk-pill-high">🚨 High Risk / Review Required</span>
    <span class="badge-pill risk-pill-medium">⚠️ Watch / Enhanced Review</span>
    <span class="badge-pill risk-pill-low">✅ Low Risk / Standard Review</span>
    <span class="badge-pill feature-pill-context">🧭 Monitor with sector context</span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:22px;'></div>", unsafe_allow_html=True)

    st.markdown(
        """
<div class="card compact home-process">
  <div class="kicker">Review Workflow</div>
  <h3>Detect → Prioritize → Explain → Govern</h3>
  <div class="zh">偵測 → 排序 → 解釋 → 治理</div>
  <div class="home-process-grid" style="margin-top:14px;">
    <div class="decision-flow-card"><span class="flow-step-number">🔎</span><div class="flow-title">Detect</div><div class="flow-body">Screen Talk-Walk mismatch signals.<br><span class="zh">偵測揭露與外部證據落差。</span></div></div>
    <div class="decision-flow-card"><span class="flow-step-number">🎯</span><div class="flow-title">Prioritize</div><div class="flow-body">Rank companies for ESG review.<br><span class="zh">排序 ESG 查核優先順序。</span></div></div>
    <div class="decision-flow-card"><span class="flow-step-number">🧩</span><div class="flow-title">Explain</div><div class="flow-body">Use SHAP evidence and drivers.<br><span class="zh">用 SHAP 呈現判斷依據。</span></div></div>
    <div class="decision-flow-card"><span class="flow-step-number">✅</span><div class="flow-title">Govern</div><div class="flow-body">Convert score into credit memo actions.<br><span class="zh">轉換成授信查核行動。</span></div></div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:22px;'></div>", unsafe_allow_html=True)

    st.markdown(
        """
<div class="card home-final-message">
  <p>⚠️ This system helps financial institutions prioritize ESG reviews and identify companies requiring further due diligence. It does not legally determine whether a company has engaged in greenwashing.</p>
  <div class="zh">⚠️ 本系統僅協助金融機構排序 ESG 查核優先順序並識別需進一步審查之公司，不構成企業漂綠之法律認定。</div>
</div>
""",
        unsafe_allow_html=True,
    )



def page_data(df, X, y, meta, data_error):
    page_header(
        "02",
        "Data Explorer",
        "Company Universe, ESG Signals, and Talk-Walk Gap",
        "檢視公司樣本、產業分布、ESG 揭露訊號與外部風險落差。",
    )
    if data_error:
        warning_card("Dataset unavailable", data_error)
        return

    st.markdown(
        """
<div class="card compact report-panel">
  <p>This page summarizes the formal company universe used by the dashboard, including sector distribution, ESG disclosure intensity, external ESG risk indicators, and the Talk-Walk mismatch that motivates greenwashing risk screening.</p>
  <div class="zh">重點是看「公司說了多少 ESG」與「外部 ESG 風險訊號」是否一致。</div>
  <div class="small-caption" style="margin-top:10px;">This page shows the formal 257-company universe used by the final model. Training-augmentation samples (synthetic and external weak-labeled) are model-validation material and are summarized on the Model Performance page.</div>
  <div class="zh">本頁呈現正式模型使用的 257 家公司資料；訓練補強樣本（合成與外部弱標籤）屬模型驗證材料，彙整於 Model Performance 頁。</div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)

    if "Sector" in df.columns:
        sector_source = df["Sector"]
    elif "sector" in df.columns:
        sector_source = df["sector"]
    else:
        sector_source = pd.Series(meta.get("sector", []))
    if len(sector_source) != len(df):
        sector_source = pd.Series(meta.get("sector", df.get("Sector", [])))
    sector_series = pd.Series(sector_source).astype(str).replace({"nan": np.nan}).dropna()
    sector_opts = ["All"] + sorted(sector_series.unique().tolist())

    filter_col, help_col = st.columns([0.34, 0.66], gap="large")
    with filter_col:
        selected_sector = st.selectbox("Sector Filter", sector_opts, key="annual_sector_filter")
    sector_series_full = pd.Series(sector_source).astype(str)
    if selected_sector != "All":
        mask = sector_series_full == selected_sector
        df_f = df.loc[mask.to_numpy()].reset_index(drop=True)
        X_f = X.loc[mask.to_numpy()].reset_index(drop=True)
        y_f = pd.Series(y).loc[mask.to_numpy()].reset_index(drop=True)
        meta_f = {}
        for key in ["ticker", "sector", "industry", "year", "total_esg_risk_score", "controversy_score"]:
            if key in meta and hasattr(meta[key], "__len__") and len(meta[key]) == len(df):
                meta_f[key] = pd.Series(meta[key]).loc[mask.to_numpy()].reset_index(drop=True).tolist()
    else:
        df_f = df.reset_index(drop=True)
        X_f = X.reset_index(drop=True)
        y_f = pd.Series(y).reset_index(drop=True)
        meta_f = meta

    selected_display = "All sectors" if selected_sector == "All" else selected_sector
    with filter_col:
        st.markdown(
            f"""
<div class="filter-status">
  Selected sector: {esc(selected_display)}
  <div class="zh">目前篩選產業：{esc(selected_display)}</div>
</div>
<div class="filter-count">
  Number of companies shown: {len(df_f)}<br>
  <span class="zh">目前顯示公司數：{len(df_f)}</span>
</div>
""",
            unsafe_allow_html=True,
        )
    with help_col:
        st.markdown(
            """
<div class="note">
  Use the sector filter to review comparable companies within the same industry context.
  <br><span class="zh">可切換產業，觀察同產業公司資料與風險訊號。</span>
</div>
""",
            unsafe_allow_html=True,
        )

    if selected_sector != "All" and df_f.empty:
        st.markdown(
            '<div class="note warning-note">No companies available for this sector.<br>'
            '<span class="zh">此產業目前沒有可顯示資料。</span></div>',
            unsafe_allow_html=True,
        )

    full_sector_count = int(sector_series.nunique()) if len(sector_series) else 0
    full_positive = int(pd.Series(y).sum()) if len(y) else 0
    full_rate = (full_positive / len(y) * 100) if len(y) else 0.0
    feature_count = int(len(X.columns)) if not X.empty else 14
    avg_len_series = pd.to_numeric(X.get("text_length", pd.Series(dtype=float)), errors="coerce") if not X.empty else pd.Series(dtype=float)
    avg_len = float(avg_len_series.mean()) if not avg_len_series.dropna().empty else None

    kpi_html = (
        "<div class='grid-4'>"
        + metric_card("Original Company Universe", f"{len(df)}", "Formal company-level dataset")
        + metric_card("Positive High-Risk Labels", f"{full_positive}", "Model development labels", "#B8860B")
        + metric_card("Positive Rate", f"{full_rate:.1f}%", "High-risk label share", "#B8860B")
        + metric_card("Sectors", f"{full_sector_count}", "Industry groups")
        + metric_card("Features", f"{feature_count}", "Model input features")
    )
    if avg_len is not None:
        kpi_html += metric_card("Avg Report Length", f"{avg_len:,.0f}", "Average disclosure text length", "#5F7165")
    kpi_html += "</div>"
    st.markdown(kpi_html, unsafe_allow_html=True)
    st.markdown("<div style='height:22px;'></div>", unsafe_allow_html=True)

    hover = prep_hover_df(df_f, X_f, y_f, meta_f)
    with st.container(border=True):
        st.markdown(
            report_caption(
                "每個點代表一家公司；右上區域表示 ESG 揭露強度較高但外部 ESG 訊號較弱；紅點為模型標記高風險案例。",
                "Each dot represents one company. The upper-right area indicates high ESG disclosure intensity with weaker external ESG signals. Red dots are model-flagged high-risk cases.",
            ),
            unsafe_allow_html=True,
        )
        st.markdown('<div class="chart-title">Talk-Walk Gap: ESG Disclosure Intensity vs External ESG Risk</div>', unsafe_allow_html=True)
        if hover.empty:
            st.markdown('<div class="note warning-note">No companies available for this sector.<br><span class="zh">此產業目前沒有可顯示資料。</span></div>', unsafe_allow_html=True)
        else:
            fig = px.scatter(
                hover,
                x="talk_score",
                y="walk_external_score",
                color="risk_label",
                hover_data={
                    "ticker": True,
                    "company": True,
                    "sector": True,
                    "talk_score": ":.2f",
                    "walk_external_score": ":.2f",
                    "risk_label": True,
                },
                color_discrete_map={"High-risk Label": "#C94C3A", "Non-high-risk": "#2E7D32"},
                labels={
                    "talk_score": "ESG Disclosure Intensity",
                    "walk_external_score": "External ESG Risk Signal",
                },
            )
            fig.update_traces(marker=dict(size=9, opacity=0.74, line=dict(width=0.8, color="#FFFFFF")))
            fig.update_traces(
                selector=dict(name="High-risk Label"),
                marker=dict(size=12, opacity=0.94, line=dict(width=1.2, color="#FFFFFF")),
            )
            fig.update_layout(
                **plot_layout(360),
                legend_title=dict(text=""),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=12)),
            )
            fig.update_layout(
                margin=dict(l=50, r=50, t=60, b=50),
                xaxis=dict(automargin=True),
                yaxis=dict(automargin=True),
            )
            st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        st.markdown(
            """
<div class="note-strip-green">🔎 <div>Greenwashing risk is concentrated where ESG disclosure intensity is high but external ESG risk signals are also weak or elevated.<br>
<span class="zh">高風險公司通常集中在「ESG 說得多，但外部風險訊號也偏高」的區域。</span>
</div></div>
""",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:22px;'></div>", unsafe_allow_html=True)

    chart_col_1, chart_col_2 = st.columns(2, gap="large")
    with chart_col_1:
        with st.container(border=True):
            st.markdown(
                report_caption(
                    "257 家公司的產業分布",
                    "Company count by sector.",
                ),
                unsafe_allow_html=True,
            )
            st.markdown('<div class="chart-title">Company Distribution by Sector</div>', unsafe_allow_html=True)
            if "Sector" in df_f.columns:
                sectors = df_f["Sector"].value_counts().reset_index()
                sectors.columns = ["Sector", "Company Count"]
            elif "sector" in df_f.columns:
                sectors = df_f["sector"].value_counts().reset_index()
                sectors.columns = ["Sector", "Company Count"]
            else:
                sectors = pd.DataFrame({"Sector": [], "Company Count": []})
            sectors = sectors.sort_values("Company Count", ascending=False)
            if sectors.empty:
                st.markdown('<div class="note warning-note">No companies available for this sector.</div>', unsafe_allow_html=True)
            else:
                fig_sector = px.bar(sectors, x="Company Count", y="Sector", orientation="h", color_discrete_sequence=["#2E7D32"])
                fig_sector.update_layout(**plot_layout(330))
                fig_sector.update_layout(
                    yaxis=dict(categoryorder="array", categoryarray=sectors["Sector"].tolist()[::-1]),
                    margin=dict(l=50, r=50, t=60, b=50),
                )
                fig_sector.update_xaxes(automargin=True)
                fig_sector.update_yaxes(automargin=True)
                st.plotly_chart(fig_sector, width="stretch", config={"displayModeBar": False})
            st.markdown(
                '<div class="small-caption">Sector composition matters because ESG disclosure style and external risk expectations differ across industries.</div>'
                '<div class="zh">不同產業的 ESG 語言與外部風險不能直接硬比較。</div>',
                unsafe_allow_html=True,
            )
    with chart_col_2:
        with st.container(border=True):
            st.markdown(
                report_caption(
                    "公司爭議程度分布",
                    "Distribution of controversy levels.",
                ),
                unsafe_allow_html=True,
            )
            st.markdown('<div class="chart-title">External ESG Controversy Profile</div>', unsafe_allow_html=True)
            if "Controversy Level" in df_f.columns:
                controversy_existing = df_f["Controversy Level"].dropna().astype(str)
                ctrl = controversy_existing.value_counts().reindex(CONTROVERSY_ORDER).dropna().reset_index()
                ctrl.columns = ["Controversy Level", "Company Count"]
                ctrl["Company Count"] = ctrl["Company Count"].astype(int)
                ctrl["Level"] = ctrl["Controversy Level"].map(CONTROVERSY_SHORT).fillna(ctrl["Controversy Level"])
            else:
                ctrl = pd.DataFrame({"Controversy Level": [], "Company Count": [], "Level": []})
            if ctrl.empty:
                st.markdown('<div class="note warning-note">Controversy Level is unavailable for the filtered data.</div>', unsafe_allow_html=True)
            else:
                category_order = ctrl["Level"].tolist()[::-1]
                fig_ctrl = px.bar(ctrl, x="Company Count", y="Level", orientation="h", color_discrete_sequence=["#D99A2B"])
                fig_ctrl.update_layout(**plot_layout(330))
                fig_ctrl.update_layout(
                    yaxis=dict(categoryorder="array", categoryarray=category_order, title_font=dict(size=14), tickfont=dict(size=12)),
                    margin=dict(l=50, r=50, t=60, b=50),
                )
                fig_ctrl.update_xaxes(automargin=True)
                fig_ctrl.update_yaxes(automargin=True)
                st.plotly_chart(fig_ctrl, width="stretch", config={"displayModeBar": False})
            st.markdown(
                '<div class="small-caption">External controversy indicators help the dashboard compare company disclosure with outside ESG risk signals.</div>'
                '<div class="zh">外部爭議訊號可協助檢查 ESG 揭露是否需要進一步查核。</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height:22px;'></div>", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown(
            report_caption(
                "四種揭露密度特徵的分布",
                "Distribution of disclosure-density features.",
            ),
            unsafe_allow_html=True,
        )
        st.markdown('<div class="chart-title">ESG Disclosure Signal Distribution</div>', unsafe_allow_html=True)
        signal_cols = [col for col in ["E_density", "S_density", "G_density", "talk_total_density"] if col in X_f.columns]
        if not signal_cols or X_f.empty:
            st.markdown('<div class="note warning-note">ESG disclosure signal columns are unavailable for the filtered data.</div>', unsafe_allow_html=True)
        else:
            signal_display = {
                "E_density": "Environmental Disclosure Density",
                "S_density": "Social Disclosure Density",
                "G_density": "Governance Disclosure Density",
                "talk_total_density": "Overall ESG Disclosure Intensity",
            }
            signal_df = X_f[signal_cols].apply(pd.to_numeric, errors="coerce")
            signal_long = signal_df.rename(columns=signal_display).melt(var_name="ESG Signal", value_name="Value").dropna()
            fig_signal = px.box(
                signal_long,
                x="ESG Signal",
                y="Value",
                color="ESG Signal",
                color_discrete_sequence=["#2E7D32", "#1F5C3A", "#D99A2B", "#5F7165"],
                points="outliers",
            )
            fig_signal.update_layout(**plot_layout(360))
            fig_signal.update_layout(showlegend=False, margin=dict(l=50, r=50, t=60, b=60))
            fig_signal.update_xaxes(tickangle=-12)
            fig_signal.update_xaxes(automargin=True)
            fig_signal.update_yaxes(automargin=True)
            st.plotly_chart(fig_signal, width="stretch", config={"displayModeBar": False})
        st.markdown(
            '<div class="small-caption">Disclosure intensity helps measure how strongly companies communicate ESG topics in official filings.</div>'
            '<div class="zh">揭露強度代表公司在正式文件中談 ESG 的程度。</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:22px;'></div>", unsafe_allow_html=True)
    with st.expander("Raw Data Preview"):
        st.markdown(
            """
<div class="note">
  This preview is for transparency and data inspection only. It is not a model decision table.
  <br><span class="zh">此表僅供資料檢查，不代表模型決策。</span>
</div>
""",
            unsafe_allow_html=True,
        )
        cols = [
            c
            for c in ["ticker", "Name", "Sector", "year", "Total ESG Risk score", "Controversy Level", "ESG Risk Percentile", "ESG Risk Level"]
            if c in df_f.columns
        ]
        if cols and not df_f.empty:
            raw_preview = df_f[cols].head(40).copy()
            st.markdown(
                styled_table_html(
                    raw_preview,
                    title="Raw Data Preview",
                    subtitle="Compact raw-data preview for transparency. This is not a model decision table.",
                    badge_cols=["Sector", "Controversy Level", "ESG Risk Level"],
                    max_rows=40,
                ),
                unsafe_allow_html=True,
            )
        else:
            st.info("No rows are available for the current filter.")



def page_model(df, X, y, model, data_error, model_error):
    page_header(
        "03",
        "Model Performance",
        "Formal Model Comparison and Robustness Validation",
        "正式模型比較與樣本補強後的穩健性檢查。",
    )

    model_rows = [
        {
            "Model": "DummyClassifier",
            "ROC-AUC": 0.4746,
            "F1": 0.0444,
            "PR-AUC": 0.0801,
            "Role": "Baseline",
        },
        {
            "Model": "Logistic Regression",
            "ROC-AUC": 0.9215,
            "F1": 0.4580,
            "PR-AUC": 0.5207,
            "Role": "Interpretable benchmark",
        },
        {
            "Model": "Random Forest",
            "ROC-AUC": 0.9269,
            "F1": 0.4833,
            "PR-AUC": 0.6760,
            "Role": "Tree-based benchmark",
        },
        {
            "Model": "XGBoost",
            "ROC-AUC": 0.9261,
            "F1": 0.7119,
            "PR-AUC": 0.7549,
            "Role": "Challenger model for minority detection",
        },
        {
            "Model": "LightGBM",
            "ROC-AUC": 0.9528,
            "F1": 0.6576,
            "PR-AUC": 0.7450,
            "Role": "Formal final risk-ranking scorer",
        },
    ]
    model_table = pd.DataFrame(model_rows)

    st.markdown(
        """
<div class="card warn report-panel">
  <h3>Model Selection Summary</h3>
  <p>LightGBM remains the formal final risk-ranking scorer because it achieved the highest ROC-AUC and stable cross-validation performance. XGBoost is retained as a challenger model because its F1 and PR-AUC are strong for minority high-risk detection.</p>
  <div class="zh">LightGBM 適合整體風險排序；XGBoost 對少數高風險樣本偵測表現強。</div>
  <div class="small-caption" style="margin-top:10px;">This page reports model validation results for risk screening and due diligence support.</div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

    st.markdown(
        "<div class='grid-4 model-kpis'>"
        + metric_card("Final Scorer", "LightGBM", "Formal final risk-ranking scorer")
        + metric_card("Formal ROC-AUC", "0.953", "Highest nested-CV ranking performance")
        + metric_card("Formal PR-AUC", "0.745", "High-risk detection")
        + metric_card("Formal F1", "0.658", "Precision-recall balance")
        + "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        """
<div class="grid-4" style="margin-top:14px;">
  <div class="insight-card"><span class="badge-pill model-pill-final">🛡️ Nested CV</span><div class="insight-body">Outer 5-fold evaluation with inner tuning for robust model comparison.</div><div class="zh">外層 5-fold 評估，內層調參。</div></div>
  <div class="insight-card"><span class="badge-pill feature-pill-walk">📌 Holdout</span><div class="insight-body">Fixed 78-company real holdout is kept separate from training and tuning.</div><div class="zh">固定 78 家真實公司 holdout 不參與訓練。</div></div>
  <div class="insight-card"><span class="badge-pill risk-pill-low">✅ Leakage Control</span><div class="insight-body">Label-building raw scores are not used as direct prediction features.</div><div class="zh">建標籤原始分數不直接放入模型特徵。</div></div>
  <div class="insight-card"><span class="badge-pill model-pill-challenger">📈 Robustness</span><div class="insight-body">Augmentation results are sensitivity checks, not replacements for the formal scorer.</div><div class="zh">樣本補強屬穩健性檢查，不取代正式模型。</div></div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:22px;'></div>", unsafe_allow_html=True)

    st.markdown(
        report_caption(
            "五個模型比較：LightGBM 排序能力最佳，選為正式模型。",
            "Five-model comparison; LightGBM ranks best and is our final scorer.",
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        """
<div class="card report-panel">
  <h3>Formal Nested-CV Model Comparison</h3>
  <div class="zh">正式 Nested-CV 模型比較</div>
  <div class="small-caption">LightGBM remains the formal final scorer. XGBoost is a challenger model because F1 / PR-AUC is strong; it does not replace LightGBM.</div>
</div>
""",
        unsafe_allow_html=True,
    )
    display_table = model_table.copy()
    display_table["Role"] = display_table["Role"].replace({
        "Formal final risk-ranking scorer": "Final Scorer",
        "Challenger model for minority detection": "Challenger",
        "Tree-based benchmark": "Benchmark",
        "Interpretable benchmark": "Benchmark",
        "Simple baseline model": "Baseline",
    })
    role_order = {"Final Scorer": 0, "Challenger": 1, "Benchmark": 2, "Baseline": 3}
    display_table["_display_order"] = display_table["Role"].map(role_order).fillna(9)
    display_table = display_table.sort_values(
        ["_display_order", "ROC-AUC"],
        ascending=[True, False],
    ).drop(columns=["_display_order"]).reset_index(drop=True)
    for col in ["ROC-AUC", "F1", "PR-AUC"]:
        display_table[col] = display_table[col].map(lambda value: f"{value:.4f}")
    st.markdown(
        styled_table_html(
            display_table[["Model", "Role", "ROC-AUC", "F1", "PR-AUC"]],
            title="Ranked Model Validation Table",
            subtitle="LightGBM is highlighted as the formal scorer; XGBoost remains a challenger for minority high-risk detection.",
            badge_cols=["Role"],
            numeric_cols=["ROC-AUC", "F1", "PR-AUC"],
            highlight_first=True,
            challenger_second=True,
            rank=True,
        ),
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:22px;'></div>", unsafe_allow_html=True)

    st.markdown(
        report_caption(
            "三項指標逐模型比較",
            "ROC-AUC / PR-AUC / F1 by model.",
        ),
        unsafe_allow_html=True,
    )
    tab_auc, tab_pr, tab_f1, tab_notes = st.tabs(["ROC-AUC", "PR-AUC", "F1", "Validation Notes"])
    chart_df = model_table.copy().sort_values("ROC-AUC", ascending=True)
    chart_df["Highlight"] = chart_df["Model"].map(lambda name: "LightGBM" if name == "LightGBM" else "XGBoost" if name == "XGBoost" else "Other")
    with tab_auc:
        fig_auc = px.bar(
            chart_df,
            x="ROC-AUC",
            y="Model",
            orientation="h",
            color="Highlight",
            color_discrete_map={"LightGBM": "#2E7D32", "XGBoost": "#D99A2B", "Other": "#A9B9AD"},
            text=chart_df["ROC-AUC"].map(lambda value: f"{value:.4f}"),
        )
        fig_auc.update_layout(**plot_layout(340))
        fig_auc.update_layout(showlegend=False, xaxis_title="ROC-AUC", margin=dict(l=50, r=80, t=60, b=50))
        fig_auc.update_xaxes(range=[0, 1.08], automargin=True)
        fig_auc.update_yaxes(automargin=True)
        fig_auc.update_traces(textposition="outside", cliponaxis=False)
        st.plotly_chart(fig_auc, width="stretch", config={"displayModeBar": False})
        st.markdown(
            '<div class="note">LightGBM provides the strongest overall risk-ranking performance.<br><span class="zh">LightGBM 最適合整體風險排序。</span></div>',
            unsafe_allow_html=True,
        )
    with tab_pr:
        pr_df = model_table.copy().sort_values("PR-AUC", ascending=True)
        pr_df["Highlight"] = pr_df["Model"].map(lambda name: "LightGBM" if name == "LightGBM" else "XGBoost" if name == "XGBoost" else "Other")
        fig_pr = px.bar(
            pr_df,
            x="PR-AUC",
            y="Model",
            orientation="h",
            color="Highlight",
            color_discrete_map={"LightGBM": "#2E7D32", "XGBoost": "#D99A2B", "Other": "#A9B9AD"},
            text=pr_df["PR-AUC"].map(lambda value: f"{value:.4f}"),
        )
        fig_pr.update_layout(**plot_layout(340))
        fig_pr.update_layout(showlegend=False, xaxis_title="PR-AUC", margin=dict(l=50, r=80, t=60, b=50))
        fig_pr.update_xaxes(range=[0, 1.08], automargin=True)
        fig_pr.update_yaxes(automargin=True)
        fig_pr.update_traces(textposition="outside", cliponaxis=False)
        st.plotly_chart(fig_pr, width="stretch", config={"displayModeBar": False})
        st.markdown(
            '<div class="note">PR-AUC is important because only 7.4% of companies are labeled high-risk.<br><span class="zh">正例很少，因此 PR-AUC 比單看 accuracy 更重要。</span></div>',
            unsafe_allow_html=True,
        )
    with tab_f1:
        f1_df = model_table.copy().sort_values("F1", ascending=True)
        f1_df["Highlight"] = f1_df["Model"].map(lambda name: "LightGBM" if name == "LightGBM" else "XGBoost" if name == "XGBoost" else "Other")
        fig_f1 = px.bar(
            f1_df,
            x="F1",
            y="Model",
            orientation="h",
            color="Highlight",
            color_discrete_map={"LightGBM": "#2E7D32", "XGBoost": "#D99A2B", "Other": "#A9B9AD"},
            text=f1_df["F1"].map(lambda value: f"{value:.4f}"),
        )
        fig_f1.update_layout(**plot_layout(340))
        fig_f1.update_layout(showlegend=False, xaxis_title="F1", margin=dict(l=50, r=80, t=60, b=50))
        fig_f1.update_xaxes(range=[0, 1.08], automargin=True)
        fig_f1.update_yaxes(automargin=True)
        fig_f1.update_traces(textposition="outside", cliponaxis=False)
        st.plotly_chart(fig_f1, width="stretch", config={"displayModeBar": False})
        st.markdown(
            '<div class="note">XGBoost achieves the strongest F1 score, making it a useful challenger model for minority high-risk detection.<br><span class="zh">XGBoost 對少數高風險樣本偵測表現較強。</span></div>',
            unsafe_allow_html=True,
        )
    with tab_notes:
        st.markdown(
            """
<div class="card compact report-panel">
  <h3>Validation Notes</h3>
  <p>The formal comparison uses nested cross-validation and keeps LightGBM as the final risk-ranking scorer. XGBoost is retained as a challenger model for minority high-risk detection.</p>
  <div class="zh">正式模型比較使用 nested cross-validation；LightGBM 仍為最終風險排序模型，XGBoost 作為少數高風險偵測的 challenger model。</div>
</div>
""",
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:24px;'></div>", unsafe_allow_html=True)
    robust_rows = [
        {
            "Model Version": "Original real training only",
            "Training setup": "Real training only",
            "Sample setup": "179 real training samples",
            "ROC-AUC": 0.8924,
            "PR-AUC": 0.5708,
            "F1": 0.5000,
            "Interpretation": "Baseline robustness check",
        },
        {
            "Model Version": "Augmented with synthetic ESG disclosures",
            "Training setup": "Real + synthetic ESG disclosures",
            "Sample setup": "179 + 24 synthetic samples",
            "ROC-AUC": 0.9421,
            "PR-AUC": 0.7663,
            "F1": 0.8333,
            "Interpretation": "Best minority-class improvement in this robustness check",
        },
        {
            "Model Version": "Augmented with synthetic + DAX weak-labeled samples",
            "Training setup": "Real + synthetic + DAX weak labels",
            "Sample setup": "179 + 24 + 26",
            "ROC-AUC": 0.9630,
            "PR-AUC": 0.6001,
            "F1": 0.6667,
            "Interpretation": "Highest ROC-AUC, but not best PR-AUC / F1",
        },
        {
            "Model Version": "Augmented with synthetic + all external weak-labeled samples",
            "Training setup": "Real + synthetic + all external weak labels",
            "Sample setup": "Largest augmented training setup = 407",
            "ROC-AUC": 0.9514,
            "PR-AUC": 0.6319,
            "F1": 0.6000,
            "Interpretation": "High ROC-AUC retained, but not a full replacement of the formal model",
        },
    ]
    robustness_table = pd.DataFrame(robust_rows)
    st.markdown(
        report_caption(
            "回應教授建議：補樣本後的穩健性驗證，測試集固定不變。",
            "Robustness validation after augmentation; holdout kept fixed.",
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        """
<div class="card report-panel">
  <h3>Robustness Validation and Sample-Size Sensitivity Check</h3>
  <div class="zh">樣本補強後的穩健性檢查</div>
  <p>These experiments evaluate model robustness under additional high-risk training examples. They are sensitivity checks only and do not replace the formal nested cross-validation results or the final LightGBM scorer.</p>
  <div class="zh">這些實驗用於評估模型在補充高風險樣本後的穩健性，屬於敏感度分析，不取代正式 Nested-CV 結果或最終 LightGBM 模型。</div>
</div>
""",
        unsafe_allow_html=True,
    )
    robustness_display = robustness_table.copy()
    for col in ["ROC-AUC", "PR-AUC", "F1"]:
        robustness_display[col] = robustness_display[col].map(lambda value: f"{value:.4f}")
    st.markdown(
        styled_table_html(
            robustness_display,
            title="Robustness Validation Table",
            subtitle="Augmentation experiments are sensitivity checks; the formal LightGBM scorer and fixed holdout interpretation remain unchanged.",
            badge_cols=["Training setup", "Sample setup", "Interpretation"],
            numeric_cols=["ROC-AUC", "PR-AUC", "F1"],
            highlight_first=False,
        ),
        unsafe_allow_html=True,
    )


    st.markdown("<div style='height:22px;'></div>", unsafe_allow_html=True)
    st.markdown(
        report_caption(
            "保留樣本上的 ROC、PR 曲線與混淆矩陣（乾淨評估）",
            "ROC / PR curves and confusion matrix on the fixed holdout.",
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        """
<div class="card compact report-panel">
  <h3>Holdout Evaluation Curves</h3>
  <div class="zh">保留樣本評估曲線</div>
  <p>These curves are computed on the fixed 78-company real holdout set only, which was never used for training or tuning.</p>
  <div class="zh">以下曲線僅以固定 78 家真實 holdout 計算，該樣本從未參與訓練或調參。</div>
</div>
""",
        unsafe_allow_html=True,
    )
    holdout_images = [
        ("Holdout ROC Curve", os.path.join(OUTPUTS, "holdout_roc_curve.png")),
        ("Holdout PR Curve", os.path.join(OUTPUTS, "holdout_pr_curve.png")),
        ("Holdout Confusion Matrix", os.path.join(OUTPUTS, "holdout_confusion_matrix.png")),
    ]
    tab_roc, tab_pr_holdout, tab_cm = st.tabs(["ROC Curve", "PR Curve", "Confusion Matrix"])
    for tab, (title, path) in zip([tab_roc, tab_pr_holdout, tab_cm], holdout_images):
        with tab:
            st.markdown(holdout_image_html(path, title), unsafe_allow_html=True)
    st.markdown(
        """
<div class="note">
  Threshold = review-priority cutoff (top 10% of holdout ranking).
  <br><span class="zh">門檻為查核排序用途（holdout 排序前 10%），非法律判定門檻。</span>
</div>
<div class="note" style="margin-top:12px;">
  These results reflect performance on this fixed holdout sample only and should not be interpreted as guaranteed real-world performance.
  <br><span class="zh">此結果僅反映固定 holdout 樣本表現，不代表實際應用一定達到相同準確度。</span>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:22px;'></div>", unsafe_allow_html=True)
    st.markdown(
        """
<div class="card compact report-panel">
  <h3>Holdout and Leakage Control</h3>
  <p>The 78-company real holdout set remains fixed. Synthetic, DAX, Kaggle / Greenwashing Score, and SEC EDGAR external inference cases were not added to the formal holdout test set. This avoids contaminating the final evaluation.</p>
  <div class="zh">正式 78 家 holdout 不被新增資料污染。</div>
  <p>NVIDIA, Apple, and IBM are external inference examples only. They were not used for training and were not added to the formal holdout set.</p>
  <div class="zh">外部公司案例只做推論，不參與訓練或正式測試。</div>
</div>
""",
        unsafe_allow_html=True,
    )



def page_model_contribution():
    page_header(
        "03A",
        "Model Contribution Analysis",
        "Feature Contribution and Final Scorer Interpretation",
        "正式 LightGBM 模型的特徵貢獻與風險解釋。",
    )
    st.markdown(
        """
<div class="card compact report-panel">
  <p>This page explains how the formal final LightGBM scorer converts ESG disclosure, Walk / External indicators, and sector context into model-flagged greenwashing risk scores. Feature contribution analysis supports analyst review, but it does not replace internal due diligence.</p>
  <div class="zh">本頁說明模型如何使用特徵產生風險分數，供審查人員判讀。</div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)

    st.markdown(
        """
<div class="card report-panel">
  <h3>Final Scorer Summary</h3>
  <div class="grid-4">
"""
        + metric_card("Final Scorer", "LightGBM", "Formal scorer")
        + metric_card("Formal ROC-AUC", "0.953", "Nested-CV ranking")
        + metric_card("Formal PR-AUC", "0.745", "High-risk detection")
        + metric_card("Formal F1", "0.658", "Precision / recall")
        + metric_card("Features", "14", "9 Talk + 4 Walk + 1 Context")
        + """
  </div>
  <div class="note" style="margin-top:14px;">
    These are the formal final model results from nested cross-validation. Robustness experiments do not replace this formal scorer.
    <br><span class="zh">正式模型仍為 LightGBM；穩健性實驗不取代正式模型。</span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:22px;'></div>", unsafe_allow_html=True)

    st.markdown(external_validity_summary_html(), unsafe_allow_html=True)
    st.markdown("<div style='height:22px;'></div>", unsafe_allow_html=True)

    st.markdown(
        report_caption(
            "14 個特徵分三組：Talk 揭露語言、Walk 外部證據、Context 產業。",
            "14 features in three groups: Talk, Walk, Context.",
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        """
<div class="card report-panel">
  <h3>Feature Groups</h3>
  <div class="zh">特徵群組</div>
  <div class="small-caption">Unified feature classification: 14 = 9 Talk + 4 Walk / External + 1 Context.</div>
  <div class="grid-3" style="margin-top:14px;">
    <div class="card compact soft">
      <h3>💬 Talk Features</h3>
      <p>ESG disclosure intensity, ESG keyword density, hedge language, numeric disclosure, topic breadth, and vague vs specific language ratio.</p>
      <div class="zh">衡量公司在正式文件中如何談 ESG。</div>
    </div>
    <div class="card compact warn">
      <h3>🌐 Walk / External Features</h3>
      <p>Sector-relative ESG risk, ESG pillar imbalance, external environmental signal, and external governance / ethics signal.</p>
      <div class="zh">衡量外部 ESG 風險與第三方訊號。</div>
    </div>
    <div class="card compact soft">
      <h3>🧭 Context Feature</h3>
      <p>Sector context used to compare companies more fairly across industries.</p>
      <div class="zh">用產業脈絡避免跨產業硬比較。</div>
    </div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:22px;'></div>", unsafe_allow_html=True)

    feature_rows = [
        ("sector_rank", "Sector-relative ESG Risk"),
        ("talk_total_density", "ESG Disclosure Intensity"),
        ("E_density", "Environmental Disclosure Intensity"),
        ("S_density", "Social Disclosure Intensity"),
        ("G_density", "Governance Disclosure Intensity"),
        ("e_s_g_dispersion", "ESG Pillar Imbalance"),
        ("pct_numeric", "Quantitative Disclosure Ratio"),
        ("vague_to_specific_ratio", "Vague vs Specific Language Ratio"),
        ("planet_friendly_business", "External Environmental Signal"),
        ("honest_fair_business", "External Governance / Ethics Signal"),
        ("sector_encoded", "Sector Context"),
    ]
    feature_table_rows = "".join(
        f"<tr><td>{esc(raw)}</td><td>{esc(label)}</td><td>{esc(feature_group(raw))}</td></tr>"
        for raw, label in feature_rows
    )
    with st.expander("View business-friendly feature labels / 查看商業友善特徵名稱", expanded=False):
        st.markdown(
            f"""
<div class="table-wrap">
  <table class="report-table" style="min-width:760px;">
    <thead><tr><th>Original Feature</th><th>Dashboard Display Name</th><th>Feature Group</th></tr></thead>
    <tbody>{feature_table_rows}</tbody>
  </table>
</div>
""",
            unsafe_allow_html=True,
        )

    contrib, err = model_contribution_df()
    if err or contrib.empty:
        warning_card("Model contribution analysis is not available", err or "Missing model contribution output.")
        st.info("Run `python3 src/model_contribution_analysis.py` to generate outputs/model_contribution_analysis.csv.")
        return

    metric_cols = [
        "roc_auc_mean",
        "pr_auc_mean",
        "f1_mean",
        "precision_mean",
        "recall_mean",
        "accuracy_mean",
    ]
    for col in metric_cols:
        if col in contrib.columns:
            contrib[col] = pd.to_numeric(contrib[col], errors="coerce")

    lookup = contrib.set_index("model_version") if "model_version" in contrib.columns else pd.DataFrame()

    def _metric(version, col, default=np.nan):
        try:
            return float(lookup.loc[version, col])
        except Exception:
            return default

    model_display_names = {
        "Full Model": "Full Model",
        "External-only Model": "External-only",
        "Text-only Model": "Text-only",
        "No-ranking Model": "Without Sector Ranking",
        "Strict No-Walk-Proxy Model": "Without Walk Proxy Features",
    }
    feature_group_display_names = {
        "Talk + Walk / External + Context": "Talk + Walk + Context Features",
        "Walk / External ranking features": "External ESG Features",
        "Sustainability report Talk features": "Sustainability Report Features",
        "No sector_rank / integrity ranking outputs": "Without Sector Ranking Features",
        "Talk-only strict ablation": "Talk-only Features",
    }

    def _display_model_name(value: str) -> str:
        return model_display_names.get(str(value), str(value))

    def _display_feature_group(value: str) -> str:
        return feature_group_display_names.get(str(value), str(value))

    def _fmt_metric(value, digits: int = 3) -> str:
        try:
            number = float(value)
        except Exception:
            return ""
        return f"{number:.{digits}f}" if np.isfinite(number) else ""

    full_auc = _metric("Full Model", "roc_auc_mean")
    external_auc = _metric("External-only Model", "roc_auc_mean")
    text_auc = _metric("Text-only Model", "roc_auc_mean")
    no_ranking_auc = _metric("No-ranking Model", "roc_auc_mean")
    no_ranking_drop = full_auc - no_ranking_auc if np.isfinite(full_auc) and np.isfinite(no_ranking_auc) else np.nan

    st.markdown(
        report_caption(
            "消融分析比較不同特徵組合對模型辨識能力的貢獻",
            "Ablation analysis compares how each feature group contributes to discrimination performance.",
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        """
<div class="card compact report-panel">
  <h3>Feature Contribution / Ablation</h3>
  <div class="zh">特徵貢獻與消融分析</div>
  <div class="note" style="margin-top:12px;">
    Feature contribution analysis explains which signals drive the final LightGBM risk score.
    <br><span class="zh">特徵貢獻分析用於解釋正式 LightGBM 模型的風險評分來源。</span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

    st.markdown(
        "<div class='grid-4'>"
        + metric_card("Full contribution model ROC-AUC", f"{full_auc:.3f}", "Ablation result")
        + metric_card("External-only ROC-AUC", f"{external_auc:.3f}", "Walk / External signals", "#D99A2B")
        + metric_card("Text-only ROC-AUC", f"{text_auc:.3f}", "Talk disclosure signal", "#5F7165")
        + metric_card("Without Sector Ranking AUC Drop", f"{no_ranking_drop:.3f}", "Full 0.941 - Without Sector Ranking 0.732 = 0.210", "#C94C3A")
        + "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
<div class="insight-card report-panel" style="margin-top:16px;">
  <h3>Key Findings</h3>
  <p>• Full model delivers the strongest overall performance.<br>
  • External ESG signals provide substantial predictive value.<br>
  • Sustainability-report text remains informative.<br>
  • Combining Talk and Walk produces the best discrimination.</p>
  <div class="zh"><strong>主要發現</strong><br>
  • 完整模型效果最佳。<br>
  • 外部 ESG 訊號具有重要貢獻。<br>
  • 永續報告文字仍具有預測力。<br>
  • 結合 Talk 與 Walk 的效果最佳。</div>
</div>
<div class="small-caption">Detailed SHAP global and company-level explanations are shown on the Explainability page.</div>
""",
        unsafe_allow_html=True,
    )

    chart_cols = [
        ("roc_auc_mean", "ROC-AUC"),
        ("pr_auc_mean", "PR-AUC"),
        ("f1_mean", "F1"),
    ]
    chart_rows = []
    for _, row in contrib.iterrows():
        for col, label in chart_cols:
            if col in contrib.columns:
                chart_rows.append(
                    {
                        "Model Version": _display_model_name(row["model_version"]),
                        "Metric": label,
                        "Score": float(row[col]),
                    }
                )
    chart_df = pd.DataFrame(chart_rows)
    if not chart_df.empty:
        fig = px.bar(
            chart_df,
            x="Model Version",
            y="Score",
            color="Metric",
            barmode="group",
            color_discrete_map={"ROC-AUC": "#1F5C3A", "PR-AUC": "#D99A2B", "F1": "#5F7165"},
        )
        fig.update_layout(**plot_layout(450))
        fig.update_layout(
            title=dict(
                text="Model Contribution Analysis (Ablation Study)<br><sup>模型貢獻分析（消融實驗）</sup>",
                font=dict(size=20),
                x=0.02,
                xanchor="left",
            ),
            height=540,
            margin=dict(l=70, r=60, t=95, b=125),
            yaxis_range=[0, 1.05],
            xaxis_title="",
            legend_title_text="Metric",
        )
        fig.update_xaxes(tickangle=-12, tickfont=dict(size=13))
        fig.update_xaxes(automargin=True)
        fig.update_yaxes(automargin=True)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

    st.markdown(
        """
<div class="insight-card report-panel" style="margin-top:16px;">
  <h3>Why this matters</h3>
  <p>Text-derived ESG disclosure features retain meaningful independent signal, but perform best when combined with external ESG evidence.</p>
  <div class="zh">ESG 揭露文字本身具有一定辨識能力，但與外部 ESG 證據結合時效果最佳。</div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
<div class="card soft report-panel">
  <h3>Business Interpretation</h3>
  <p>For ESG analysts and credit officers, high contribution from disclosure intensity or vague language suggests the need to verify whether ESG claims are supported by measurable evidence. High contribution from Walk / External features suggests the need to compare company disclosure with external ESG risk indicators and controversy history.</p>
  <div class="zh">若模型分數偏高，重點在於要求更多 ESG 佐證資料與盡職調查，而非直接判定企業漂綠。</div>
</div>
""",
        unsafe_allow_html=True,
    )

    png_path = os.path.join(OUTPUTS, "ablation_metrics_summary.png")
    if os.path.exists(png_path):
        with st.expander("View saved ablation summary chart"):
            st.image(png_path, width="stretch")

    def _friendly_feature_text(value):
        text_value = str(value)
        for raw, label in FEATURE_DISPLAY.items():
            text_value = text_value.replace(raw, label)
        return text_value

    detail_cols = [
        "model_version",
        "features_used",
        "confusion_matrix",
        "positive_class_support",
        "negative_class_support",
        "interpretation",
    ]
    detail_cols = [col for col in detail_cols if col in contrib.columns]
    detail_display = contrib[detail_cols].copy()
    if "model_version" in detail_display.columns:
        detail_display["model_version"] = detail_display["model_version"].map(_display_model_name)
    if "feature_group" in detail_display.columns:
        detail_display["feature_group"] = detail_display["feature_group"].map(_display_feature_group)
    if "features_used" in detail_display.columns:
        detail_display["features_used"] = detail_display["features_used"].map(_friendly_feature_text)
    if {"model_version", "interpretation"}.issubset(detail_display.columns):
        text_only_mask = detail_display["model_version"].eq("Text-only")
        detail_display.loc[text_only_mask, "interpretation"] = (
            "Text-derived ESG disclosure features retain meaningful independent signal, "
            "but perform best when combined with external ESG evidence. "
            "ESG 揭露文字本身具有一定辨識能力，但與外部 ESG 證據結合時效果最佳。"
        )
    with st.expander("View feature lists and detailed interpretation"):
        st.markdown(
            '<div class="zh">技術附錄（供稽核與檢核使用）。</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            styled_table_html(
                detail_display,
                title="Detailed Feature Lists and Interpretation",
                subtitle="Technical appendix for audit transparency.",
                badge_cols=["model_version"],
            ),
            unsafe_allow_html=True,
        )



def industry_column_name(df: pd.DataFrame) -> str | None:
    for col in ["Industry", "industry", "GICS Industry", "GICS Sub-Industry", "Sub-Industry", "sub_industry"]:
        if col in df.columns:
            return col
    return None


def ticker_index_map(meta: dict) -> dict:
    if not meta or "ticker" not in meta:
        return {}
    tickers = pd.Series(meta.get("ticker", [])).astype(str)
    return {str(ticker).upper(): int(idx) for idx, ticker in tickers.items() if str(ticker).strip()}


def internal_lookup_table(df: pd.DataFrame, X: pd.DataFrame, y: pd.Series, meta: dict, model) -> pd.DataFrame:
    medians = project_medians(X)
    options = dataset_company_options(df, meta)
    idx_by_ticker = ticker_index_map(meta)
    industry_col = industry_column_name(df)
    rows = []
    for item in options:
        ticker = item["ticker"]
        profile, found, label = company_profile(ticker, df, X, y, meta, medians)
        feature_vector = {feature: float(profile.get(feature, medians.get(feature, 0.0))) for feature in FEATURE_ORDER}
        prob = predict_probability(model, feature_vector) if model is not None else None
        tier, _, _, _ = risk_tier(prob if prob is not None else 0.0)
        action, _ = risk_review_action(tier)
        row_idx = idx_by_ticker.get(ticker.upper())
        industry = ""
        if industry_col and row_idx is not None and row_idx < len(df):
            industry = str(df.iloc[row_idx].get(industry_col, "") or "")
            if industry.lower() == "nan":
                industry = ""
        rows.append(
            {
                "Company": str(profile.get("company", item["company"])),
                "Ticker": ticker,
                "Sector": str(profile.get("sector", item["sector"])),
                "Industry": industry,
                "Risk Score Value": prob,
                "Risk Score": f"{prob:.4f}" if prob is not None else "N/A",
                "Risk Level": tier if prob is not None else "N/A",
                "Data Source Type": "Internal benchmark profile",
                "Recommended Action": action if prob is not None else "Model unavailable",
                "Profile Status": label,
                "Select Label": f"{profile.get('company', item['company'])} / {ticker} / {profile.get('sector', item['sector'])}",
            }
        )
    return pd.DataFrame(rows)


def sector_coverage_table(lookup_df: pd.DataFrame) -> pd.DataFrame:
    if lookup_df.empty or "Sector" not in lookup_df.columns:
        return pd.DataFrame()
    rows = []
    for sector, group in lookup_df.groupby("Sector", dropna=False):
        tickers = [str(v) for v in group["Ticker"].dropna().astype(str).head(6).tolist()]
        row = {
            "Sector": str(sector),
            "Number of Companies": int(len(group)),
            "Example Tickers": ", ".join(tickers),
        }
        if "Industry" in group.columns and group["Industry"].astype(str).str.strip().any():
            industries = [v for v in group["Industry"].dropna().astype(str).unique().tolist() if v and v.lower() != "nan"]
            row["Example Industries"] = ", ".join(industries[:4])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("Number of Companies", ascending=False)


def comparison_company_card(details: dict, feature_vector: dict, hidden: bool, range_context: dict | None = None) -> str:
    tier = details.get("risk_tier", "Low")
    prob = float(details.get("risk_probability", 0.0) or 0.0)
    score_text = "?" if hidden else f"{prob*100:.1f}%"
    tier_html = '<span class="risk-tier-badge">?</span>' if hidden else risk_badge_html(tier)
    action_text = "?" if hidden else details.get("recommended_action", "")
    disclosure = feature_vector.get("talk_total_density", np.nan)
    governance = feature_vector.get("honest_fair_business", np.nan)
    external = feature_vector.get("sector_rank", np.nan)
    bars_html = "" if hidden else talk_walk_bars_html(feature_vector, range_context, tier)
    return f"""
<div class="card compact">
  <h3>{esc(details.get('company', ''))}</h3>
  <div class="small-caption">{esc(details.get('ticker', ''))} · {esc(details.get('sector', ''))}</div>
  <div class="metric" style="margin-top:12px;">
    <div class="label">Model Risk Score</div>
    <div class="value" style="font-size:30px;color:#1F5C3A;">{esc(score_text)}</div>
    <div class="zh">模型標記風險</div>
  </div>
  <div class="signal-row"><div class="signal-label">Risk Level</div><div class="signal-value">{tier_html}</div></div>
  <div class="signal-row"><div class="signal-label">Recommended Action</div><div class="signal-value">{esc(action_text)}</div></div>
  <div class="signal-row"><div class="signal-label">ESG Disclosure Intensity</div><div class="signal-value">{format_input_feature_value('talk_total_density', disclosure)}</div></div>
  <div class="signal-row"><div class="signal-label">Sector-relative ESG Risk</div><div class="signal-value">{format_input_feature_value('sector_rank', external)}</div></div>
  <div class="signal-row"><div class="signal-label">External Governance / Ethics Signal</div><div class="signal-value">{format_input_feature_value('honest_fair_business', governance)}</div></div>
  {bars_html}
</div>
"""


def render_company_comparison(lookup_df: pd.DataFrame, df: pd.DataFrame, X: pd.DataFrame, y: pd.Series, meta: dict, model) -> None:
    if lookup_df.empty or model is None:
        return
    st.markdown(
        """
<div class="card compact">
  <h3>Company Comparison</h3>
  <div class="zh">公司並排比較</div>
  <p>Compare two internal benchmark companies using the same LightGBM final scorer and 14-feature profile.</p>
  <div class="zh">使用相同正式 LightGBM 評分器比較兩家公司。</div>
</div>
""",
        unsafe_allow_html=True,
    )
    labels = lookup_df["Select Label"].tolist()
    cvx_label = next((label for label in labels if " / CVX / " in label or label.endswith("/ CVX")), labels[0])
    low_labels = lookup_df.loc[lookup_df["Risk Level"].astype(str) == "Low", "Select Label"].tolist()
    low_label = next((label for label in low_labels if label != cvx_label), labels[1] if len(labels) > 1 else cvx_label)
    c1, c2 = st.columns(2, gap="large")
    with c1:
        left_label = st.selectbox("Comparison Company A", labels, index=labels.index(cvx_label), key="annual_compare_left", format_func=display_text)
    with c2:
        right_label = st.selectbox("Comparison Company B", labels, index=labels.index(low_label), key="annual_compare_right", format_func=display_text)

    st.markdown(
        report_caption(
            "🤔 互動比較：先預測哪家公司具有較高 ESG 漂綠風險，再揭曉模型結果。",
            "Interactive Comparison: Predict which company has higher greenwashing risk before revealing the model results.",
        ),
        unsafe_allow_html=True,
    )
    if st.button("揭曉分數 / Reveal scores", key="annual_compare_reveal", width="stretch"):
        st.session_state["annual_compare_revealed"] = True
        st.balloons()
    hide_scores = st.checkbox(
        "隱藏分數 / Hide scores",
        value=not st.session_state.get("annual_compare_revealed", False),
        key="annual_compare_hide_scores",
    )
    hidden = bool(hide_scores)

    medians = project_medians(X)
    range_context = feature_range_context(X)
    cards = []
    for label in [left_label, right_label]:
        selected = lookup_df[lookup_df["Select Label"] == label].iloc[0]
        profile, found, status_label = company_profile(str(selected["Ticker"]), df, X, y, meta, medians)
        feature_vector = {feature: float(profile.get(feature, medians.get(feature, 0.0))) for feature in FEATURE_ORDER}
        prob = predict_probability(model, feature_vector)
        if prob is None:
            continue
        tier, _, _, _ = risk_tier(prob)
        action, action_zh = risk_review_action(tier)
        details = company_memo_details(
            selected["Company"],
            selected["Ticker"],
            selected["Sector"],
            prob,
            tier,
            "Dataset Company",
            selected.get("Industry", ""),
        )
        details.update({"recommended_action": action, "recommended_action_zh": action_zh})
        cards.append(comparison_company_card(details, feature_vector, hidden, range_context))
    if cards:
        cols = st.columns(len(cards), gap="large")
        for col, card_html in zip(cols, cards):
            with col:
                st.markdown(card_html, unsafe_allow_html=True)


def memo_final_note(details: dict) -> None:
    details["disclaimer"] = "Final credit and investment decisions remain subject to internal review and supporting documentation."
    details["disclaimer_zh"] = "本 memo 供 ESG 授信、永續金融與投資審查參考，不代表法律上認定漂綠。"


def store_lookup_memo_session(details: dict, feature_vector: dict | None = None, company_found: bool = True) -> None:
    memo_final_note(details)
    memo_text = memo_markdown_from_details(details)
    next_step, _ = next_due_diligence_step(details.get("risk_tier", "Low"))
    st.session_state["annual_selected_company"] = details.get("company", "")
    st.session_state["annual_selected_ticker"] = details.get("ticker", "")
    st.session_state["annual_selected_sector"] = details.get("sector", "")
    st.session_state["annual_selected_industry"] = details.get("industry", "")
    st.session_state["annual_risk_probability"] = details.get("risk_probability")
    st.session_state["annual_risk_tier"] = details.get("risk_tier")
    st.session_state["annual_main_concern"] = details.get("main_concern", "")
    st.session_state["annual_recommended_action"] = details.get("recommended_action", "")
    st.session_state["annual_required_evidence"] = details.get("required_evidence", "")
    st.session_state["annual_monitoring"] = details.get("monitoring", "")
    st.session_state["annual_next_due_diligence_step"] = next_step
    st.session_state["annual_feature_vector"] = feature_vector or {}
    st.session_state["annual_company_found"] = company_found
    st.session_state["annual_memo_markdown"] = memo_text
    st.session_state["annual_memo_details"] = details
    st.session_state["profile_source"] = details.get("data_source_type", details.get("source", ""))
    st.session_state["selected_company"] = details.get("company", "")
    st.session_state["selected_ticker"] = details.get("ticker", "")
    st.session_state["selected_sector"] = details.get("sector", "")
    st.session_state["selected_industry"] = details.get("industry", "")
    st.session_state["risk_probability"] = details.get("risk_probability")
    st.session_state["risk_tier"] = details.get("risk_tier")
    st.session_state["main_concern"] = details.get("main_concern", "")
    st.session_state["recommended_action"] = details.get("recommended_action", "")
    st.session_state["required_evidence"] = details.get("required_evidence", "")
    st.session_state["monitoring"] = details.get("monitoring", "")
    st.session_state["next_due_diligence_step"] = next_step
    st.session_state["feature_vector"] = feature_vector or {}
    st.session_state["memo_markdown"] = memo_text


def format_input_feature_value(feature: str, value) -> str:
    try:
        if pd.isna(value):
            return ""
        numeric = float(value)
    except Exception:
        return str(value)
    if feature == "vague_to_specific_ratio":
        return f"{numeric:,.2f}"
    return f"{numeric:,.2f}"


def risk_badge_html(tier: str) -> str:
    tier_text = str(tier or "Low").strip().lower()
    if tier_text.startswith("high"):
        return '<span class="risk-tier-badge risk-tier-high">🚨 High Review Priority / 高查核優先</span>'
    if tier_text.startswith("medium") or tier_text.startswith("watch"):
        return '<span class="risk-tier-badge risk-tier-watch">⚠️ Watch / 觀察</span>'
    return '<span class="risk-tier-badge risk-tier-low">✅ Low / 低風險</span>'


def risk_level_cell_style(value) -> str:
    tier_text = str(value or "").strip().lower()
    if tier_text.startswith("high"):
        return "background-color:#FDECEC;color:#A73627;font-weight:900;"
    if tier_text.startswith("medium") or tier_text.startswith("watch"):
        return "background-color:#FFF7E5;color:#8A5E11;font-weight:900;"
    if tier_text.startswith("low"):
        return "background-color:#EAF4EC;color:#1B6B4C;font-weight:900;"
    return ""


def feature_range_context(X: pd.DataFrame | None) -> dict:
    context = {}
    if X is None or getattr(X, "empty", True):
        return context
    for feature in ["talk_total_density", "sector_rank"]:
        if feature not in X.columns:
            continue
        values = pd.to_numeric(X[feature], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if values.empty:
            continue
        context[feature] = (float(values.min()), float(values.max()))
    return context


def normalize_feature_value(value, bounds) -> float:
    try:
        numeric = float(value)
    except Exception:
        return 0.0
    if not np.isfinite(numeric) or not bounds:
        return 0.0
    low, high = bounds
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return 0.0
    return float(np.clip((numeric - low) / (high - low), 0.0, 1.0))


def talk_walk_caption(tier: str | None) -> tuple[str, str]:
    tier_text = str(tier or "Low").strip().lower()
    if tier_text.startswith("high"):
        return (
            "A large Talk-Walk mismatch raises review priority.",
            "Talk 與 Walk 存在明顯落差，建議提高查核優先順序。",
        )
    if tier_text.startswith("medium") or tier_text.startswith("watch"):
        return (
            "Some Talk-Walk inconsistency may warrant additional review.",
            "Talk 與 Walk 可能存在部分不一致，建議補充審查。",
        )
    return (
        "No strong Talk-Walk mismatch is detected under the current profile.",
        "目前未偵測到明顯的 Talk-Walk 落差。",
    )


def talk_walk_bars_html(feature_vector: dict | None, range_context: dict | None = None, tier: str | None = None) -> str:
    if not feature_vector:
        return ""
    ranges = range_context or {}
    talk_raw = feature_vector.get("talk_total_density", np.nan)
    walk_raw = feature_vector.get("sector_rank", np.nan)
    talk_norm = normalize_feature_value(talk_raw, ranges.get("talk_total_density"))
    walk_norm = normalize_feature_value(walk_raw, ranges.get("sector_rank"))
    talk_width = max(3.0, talk_norm * 100.0)
    walk_width = max(3.0, walk_norm * 100.0)
    caption_en, caption_zh = talk_walk_caption(tier)
    return f'''
  <div class="talk-walk-bars">
    <div class="tw-row">
      <div class="tw-top"><span>Talk — ESG Disclosure Intensity</span><span class="tw-raw">{format_input_feature_value('talk_total_density', talk_raw)}</span></div>
      <div class="tw-track"><div class="tw-fill tw-talk" style="width:{talk_width:.1f}%;"></div></div>
    </div>
    <div class="tw-row">
      <div class="tw-top"><span>Walk — External ESG Evidence</span><span class="tw-raw">{format_input_feature_value('sector_rank', walk_raw)}</span></div>
      <div class="tw-track"><div class="tw-fill tw-walk" style="width:{walk_width:.1f}%;"></div></div>
    </div>
    <div class="tw-caption">{esc(caption_en)}<br><span class="zh">{esc(caption_zh)}</span></div>
  </div>
'''


def risk_gauge_figure(probability: float) -> go.Figure:
    value = float(np.clip(probability if probability is not None else 0.0, 0.0, 1.0))
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"valueformat": ".1%", "font": {"size": 34, "color": "#173D2A"}},
            title={
                "text": "Model-Flagged Greenwashing Risk<br><span style='font-size:0.78em;color:#6F7F72'>模型標記漂綠風險</span>",
                "font": {"size": 16, "color": "#173D2A"},
            },
            gauge={
                "axis": {"range": [0, 1], "tickformat": ".0%", "tickwidth": 1, "tickcolor": "#5F7165"},
                "bar": {"color": "#173D2A", "thickness": 0.22},
                "bgcolor": "white",
                "borderwidth": 1,
                "bordercolor": "#DDEBDD",
                "steps": [
                    {"range": [0, 0.35], "color": "rgba(27,107,76,0.28)"},
                    {"range": [0.35, 0.65], "color": "rgba(217,154,43,0.30)"},
                    {"range": [0.65, 1.0], "color": "rgba(201,76,58,0.30)"},
                ],
                "threshold": {"line": {"color": "#C94C3A", "width": 4}, "thickness": 0.75, "value": value},
            },
        )
    )
    fig.update_layout(height=300, margin=dict(l=50, r=50, t=60, b=50), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def render_risk_gauge(probability: float) -> None:
    st.plotly_chart(risk_gauge_figure(probability), width="stretch", config={"displayModeBar": False})


def risk_result_html(details: dict, feature_vector: dict | None = None, range_context: dict | None = None) -> str:
    tier = details.get("risk_tier", "Low")
    prob = float(details.get("risk_probability", 0.0) or 0.0)
    industry_row = ""
    if details.get("industry"):
        industry_row = f'<div class="signal-row"><div class="signal-label">Industry</div><div class="signal-value">{esc(details.get("industry", ""))}</div></div>'
    bars_html = talk_walk_bars_html(feature_vector, range_context, tier)
    talk_value = format_input_feature_value("talk_total_density", feature_vector.get("talk_total_density", np.nan)) if feature_vector else ""
    walk_value = format_input_feature_value("sector_rank", feature_vector.get("sector_rank", np.nan)) if feature_vector else ""
    required_evidence = details.get("required_evidence", "")
    return f"""
<div class="card">
  <h3>Risk Score Display</h3>
  <div class="zh">風險分數顯示</div>
  <div class="credit-summary-grid">
    <div class="insight-mini-card"><div class="memo-label">🎯 Risk Score</div><div class="memo-value">{prob*100:.1f}%</div></div>
    <div class="insight-mini-card"><div class="memo-label">🚦 Risk Tier</div><div class="memo-value">{risk_badge_html(tier)}</div></div>
    <div class="insight-mini-card"><div class="memo-label">💬 Talk Score</div><div class="memo-value">{esc(talk_value)}</div></div>
    <div class="insight-mini-card"><div class="memo-label">🌐 Walk / External</div><div class="memo-value">{esc(walk_value)}</div></div>
    <div class="insight-mini-card"><div class="memo-label">🧾 Recommended Action</div><div class="memo-value">{esc(details.get('recommended_action', ''))}</div></div>
    <div class="insight-mini-card"><div class="memo-label">📄 Required Evidence</div><div class="memo-value">{esc(required_evidence)}</div></div>
  </div>
  <div class="metric" style="min-height:auto;text-align:center;margin:10px 0 14px;">
    <div class="label">Exact Model Risk Score</div>
    <div class="value" style="color:#173D2A;">{prob*100:.1f}%</div>
    <div>{risk_badge_html(tier)}</div>
  </div>
  <div class="small-caption" style="text-align:center;">Model used: LightGBM final scorer · Risk level: {esc(tier)} Risk ({risk_tier_display_range(tier)})</div>
  <div class="zh" style="text-align:center;">模型：正式 LightGBM 評分器</div>
  {bars_html}
  <div class="signal-row"><div class="signal-label">Company</div><div class="signal-value">{esc(details.get('company', ''))}</div></div>
  <div class="signal-row"><div class="signal-label">Ticker</div><div class="signal-value">{esc(details.get('ticker', '') or 'N/A')}</div></div>
  <div class="signal-row"><div class="signal-label">Sector</div><div class="signal-value">{esc(details.get('sector', ''))}</div></div>
  {industry_row}
  <div class="signal-row"><div class="signal-label">Data Source Type</div><div class="signal-value">{esc(details.get('data_source_type', details.get('source', '')))}</div></div>
  <div class="signal-row"><div class="signal-label">Recommended Action</div><div class="signal-value">{esc(details.get('recommended_action', ''))}<br><span class="zh">{esc(details.get('recommended_action_zh', ''))}</span></div></div>
  <div class="signal-row"><div class="signal-label">Required Evidence</div><div class="signal-value">{esc(details.get('required_evidence', ''))}<br><span class="zh">{esc(details.get('required_evidence_zh', ''))}</span></div></div>
</div>
"""


def external_validity_evidence_html(ticker: str, is_model_flagged: bool) -> str:
    if not is_model_flagged:
        return ""
    ticker_key = str(ticker or "").upper().strip()
    evidence = EXTERNAL_VALIDITY_EVIDENCE.get(ticker_key)
    if evidence:
        body = (
            f"{esc(evidence.get('en', ''))}"
            f"<br><span class=\"zh\">{esc(evidence.get('zh', ''))}</span>"
        )
        note = (
            "This model-flagged company corresponds to a documented real-world ESG controversy "
            "(external validity check)."
        )
        note_zh = "此模型標記公司對應真實世界有據可查的 ESG 爭議（外部效度驗證）。"
        note_class = "danger-note"
    else:
        body = (
            "No major public controversy found in our external validity check; may reflect lower "
            "public ESG scrutiny in this sector."
        )
        body += '<br><span class="zh">外部效度檢查未發現重大公開爭議；可能反映此產業 ESG 公開檢視較少。</span>'
        note = "External validity check: model-flagged company without a strongly validated public controversy match."
        note_zh = "外部效度檢查：模型標記公司尚未對應到強佐證公開爭議。"
        note_class = "warning-note"
    return f"""
<div class="card compact">
  <h3>Real-World Validation / 真實世界佐證</h3>
  <div class="note {note_class}">
    {esc(note)}
    <br><span class="zh">{esc(note_zh)}</span>
  </div>
  <div class="signal-row"><div class="signal-label">Evidence</div><div class="signal-value">{body}</div></div>
</div>
"""


def external_feature_vector_from_row(row) -> dict:
    vector = {feature: float(FALLBACK_MEDIANS.get(feature, 0.0)) for feature in FEATURE_ORDER}
    for feature in FEATURE_ORDER:
        if feature not in row:
            continue
        value = pd.to_numeric(row.get(feature, np.nan), errors="coerce")
        if not pd.isna(value):
            vector[feature] = float(value)
    return vector


def sec_10k_domain_shift_html(details: dict) -> str:
    prob = float(details.get("risk_probability", 0.0) or 0.0)
    return f"""
<div class="card">
  <h3>Very Low / 極低</h3>
  <div class="small-caption">No high-risk signal detected in this SEC 10-K inference profile.</div>
  <div class="zh">此 SEC 10-K 推論檔案未偵測到高風險訊號。</div>
  <div class="metric" style="min-height:auto;text-align:center;margin:12px 0 14px;">
    <div class="label">Raw Model Score</div>
    <div class="value" style="color:#173D2A;font-size:2rem;">{prob:.10f}</div>
    <div>{risk_badge_html("Low")}</div>
  </div>
  <div class="note">
    The model was trained on sustainability reports; full 10-K filings have structurally lower ESG language density. These 10-K scores indicate document-type domain shift, not a fine-grained company ranking.
    <br><span class="zh">模型以永續報告訓練；完整 10-K 申報文件的 ESG 語言密度結構性較低。因此 10-K 分數反映文件類型差異，不代表細緻公司排名。</span>
  </div>
  <div class="signal-row"><div class="signal-label">Company</div><div class="signal-value">{esc(details.get('company', ''))}</div></div>
  <div class="signal-row"><div class="signal-label">Ticker</div><div class="signal-value">{esc(details.get('ticker', '') or 'N/A')}</div></div>
  <div class="signal-row"><div class="signal-label">Document Type</div><div class="signal-value">{esc(details.get('document_type', 'SEC 10-K'))}</div></div>
</div>
"""


def render_memo_generation(details: dict, key_prefix: str, feature_vector: dict | None = None, company_found: bool = True) -> None:
    store_lookup_memo_session(details, feature_vector, company_found)
    if st.button("Generate ESG Credit Review Memo", key=f"{key_prefix}_generate_memo", width="stretch"):
        st.session_state[f"{key_prefix}_memo_ready"] = True
        st.success("✅ ESG Credit Review Memo Ready\n\n報告已產生，可直接檢視或下載 PDF。")
    if st.session_state.get(f"{key_prefix}_memo_ready", True):
        st.markdown(
            f"""
<div class="card compact">
  <h3>ESG Credit Review Memo</h3>
  <div class="zh">ESG 授信查核 Memo</div>
  <div class="small-caption">Prepared for ESG lending, sustainable finance, and investment review.</div>
  <div class="zh">供 ESG 授信、永續金融與投資審查參考。</div>
  {memo_html(details)}
</div>
""",
            unsafe_allow_html=True,
        )
        try:
            st.download_button(
                "Download ESG Credit Review Memo",
                company_memo_pdf_bytes(details),
                memo_pdf_filename(details),
                mime="application/pdf",
                key=f"{key_prefix}_download_memo",
            )
        except Exception:
            st.download_button(
                "Download ESG Credit Review Memo",
                memo_markdown_from_details(details),
                memo_filename(details),
                mime="text/markdown",
                key=f"{key_prefix}_download_memo_md",
            )


def page_prediction_existing_company(df, X, y, meta, model, data_error, model_error):
    st.markdown(
        """
<div class="card compact">
  <h3>Select Existing Company</h3>
  <div class="zh">選擇既有公司</div>
  <p>Internal benchmark lookup from the original formal company universe of 257 companies.</p>
  <div class="zh">內部基準查詢使用 257 家正式公司樣本。</div>
</div>
""",
        unsafe_allow_html=True,
    )
    if model_error or model is None:
        warning_card("Prediction model unavailable", model_error or "Model artifact is missing.")
        return
    lookup_df = internal_lookup_table(df, X, y, meta, model)
    if lookup_df.empty:
        warning_card("Internal company universe unavailable", data_error or "No internal company rows were loaded.")
        return

    st.markdown(
        report_caption(
            "🔍 試試查 Chevron（CVX）或 Tesla（TSLA），看模型怎麼標記它們。",
            "Try looking up Chevron (CVX) or Tesla (TSLA) to see how the model flags them.",
        ),
        unsafe_allow_html=True,
    )
    filter_cols = st.columns([0.32, 0.38, 0.30], gap="large")
    with filter_cols[0]:
        sectors = ["All"] + sorted([s for s in lookup_df["Sector"].dropna().astype(str).unique().tolist() if s and s.lower() != "nan"])
        selected_sector = st.selectbox("Sector filter", sectors, key="annual_lookup_sector_filter")
    with filter_cols[1]:
        search_text = st.text_input("Company / ticker search", value="", key="annual_lookup_search_clean")
    with filter_cols[2]:
        selected_tier = st.selectbox("Risk tier filter", ["All", "Low", "Medium", "High"], key="annual_lookup_risk_filter")

    filtered = lookup_df.copy()
    if selected_sector != "All":
        filtered = filtered[filtered["Sector"].astype(str) == selected_sector]
    if selected_tier != "All":
        filtered = filtered[filtered["Risk Level"].astype(str) == selected_tier]
    query = str(search_text or "").strip().lower()
    if query:
        haystack = filtered[["Company", "Ticker", "Sector", "Industry"]].fillna("").astype(str).agg(" ".join, axis=1).str.lower()
        filtered = filtered[haystack.str.contains(query, regex=False)]

    st.markdown(
        f"""
<div class="note">
  Number of companies shown: {len(filtered)}
  <br><span class="zh">目前顯示公司數：{len(filtered)}</span>
</div>
""",
        unsafe_allow_html=True,
    )
    if filtered.empty:
        warning_card("No companies available for this filter", "No companies match the current sector, search, and risk-tier filters.")
        return

    table_cols = ["Company", "Ticker", "Sector"]
    if filtered["Industry"].astype(str).str.strip().any():
        table_cols.append("Industry")
    table_cols += ["Risk Score", "Risk Level", "Data Source Type"]
    display_table = filtered[table_cols].copy().head(15)
    st.markdown(
        styled_table_html(
            display_table,
            title="Company Risk Screening Table",
            subtitle="Curated view of the filtered company universe. Select a company below to generate the detailed memo and action recommendation.",
            badge_cols=["Risk Level"],
            numeric_cols=["Risk Score"],
        ),
        unsafe_allow_html=True,
    )
    if len(filtered) > 15:
        st.caption(f"Showing top 15 of {len(filtered)} filtered companies. Use filters/search to narrow the list.")

    selected_label = st.selectbox(
        "Select company for scoring and memo",
        filtered["Select Label"].tolist(),
        key="annual_existing_company_selected",
        format_func=display_text,
    )
    selected_row = filtered[filtered["Select Label"] == selected_label].iloc[0]
    medians = project_medians(X)
    profile, found, status_label = company_profile(str(selected_row["Ticker"]), df, X, y, meta, medians)
    feature_vector = {feature: float(profile.get(feature, medians.get(feature, 0.0))) for feature in FEATURE_ORDER}
    prob = predict_probability(model, feature_vector)
    if prob is None:
        warning_card("Selected company not scored", "The trained LightGBM final scorer is unavailable for this company.")
        return
    tier, _, _, _ = risk_tier(prob)
    action, action_zh = risk_review_action(tier)
    details = company_memo_details(
        selected_row["Company"],
        selected_row["Ticker"],
        selected_row["Sector"],
        prob,
        tier,
        "Dataset Company",
        selected_row.get("Industry", ""),
    )
    details.update(
        {
            "lookup_mode": "Select Existing Company",
            "data_source_type": "Internal benchmark profile",
            "source": "Internal benchmark profile",
            "source_zh": "資料來源：內部基準公司資料",
            "model_used": "LightGBM final scorer",
            "recommended_action": action,
            "recommended_action_zh": action_zh,
            "key_input_signals": "Original formal company universe profile; score reflects all 14 Talk, Walk/External, and Context features.",
        }
    )
    memo_final_note(details)
    range_context = feature_range_context(X)
    render_risk_gauge(prob)
    st.markdown(risk_result_html(details, feature_vector, range_context), unsafe_allow_html=True)
    st.markdown(
        external_validity_evidence_html(selected_row["Ticker"], status_label == "High-risk label"),
        unsafe_allow_html=True,
    )
    with st.expander("Input ESG Disclosure Profile / 輸入 ESG 揭露特徵", expanded=False):
        input_rows = pd.DataFrame(
            [
                {
                    "Feature": FEATURE_DISPLAY.get(feature, feature),
                    "Raw Feature": feature,
                    "Value": format_input_feature_value(feature, feature_vector.get(feature, np.nan)),
                    "Source": "Internal company profile",
                }
                for feature in FEATURE_ORDER
            ]
        )
        st.markdown(
            styled_table_html(
                input_rows,
                title="Input ESG Disclosure Profile",
                subtitle="Feature values used for the selected company risk score.\n本表顯示模型評分使用的 ESG 特徵值。",
                badge_cols=["Source"],
            ),
            unsafe_allow_html=True,
        )
    render_memo_generation(details, "annual_existing", feature_vector, found)

    with st.expander("Sector and Industry Coverage / 產業覆蓋範圍", expanded=False):
        st.markdown(
            """
<div class="card compact">
  <h3>Sector and Industry Coverage</h3>
  <div class="small-caption">Original Company Universe by Sector</div>
  <div class="zh">依產業檢視 257 家正式公司樣本範圍。</div>
</div>
""",
            unsafe_allow_html=True,
        )
        coverage = sector_coverage_table(lookup_df if selected_sector == "All" else lookup_df[lookup_df["Sector"].astype(str) == selected_sector])
        if not coverage.empty:
            st.markdown(
                styled_table_html(
                    coverage,
                    title="Sector and Industry Coverage",
                    subtitle="Coverage of the formal 257-company universe by sector.",
                    numeric_cols=[col for col in coverage.columns if "Count" in col or "Companies" in col or "Company" in col],
                ),
                unsafe_allow_html=True,
            )

    st.markdown("<div style='height:22px;'></div>", unsafe_allow_html=True)
    render_company_comparison(lookup_df, df, X, y, meta, model)


def page_prediction_external(X=None):
    case_df, err = combined_external_inference_df()
    if err or case_df.empty:
        warning_card("External inference examples unavailable", err or "No precomputed external inference examples were found.")
        return

    case_df = case_df.copy()
    for col, default in {
        "company": "External Company",
        "ticker": "",
        "source": "SEC EDGAR 10-K",
        "filing_type": "10-K",
        "filing_url": "",
        "model_used": "LightGBM final scorer",
        "document_type": "SEC 10-K",
        "document_label": "SEC 10-K",
        "document_year": "",
        "source_file": "",
        "source_url": "",
        "risk_interpretation": "",
        "recommended_action": "",
        "retrieval_note": "",
        "scoring_note": "",
        "case_note": "",
    }.items():
        if col not in case_df.columns:
            case_df[col] = default
        case_df[col] = case_df[col].fillna(default).astype(str)
    if "cik" not in case_df.columns:
        case_df["cik"] = ""
    case_df["cik"] = case_df["cik"].apply(format_cik)
    case_df["_ticker"] = case_df["ticker"].astype(str).str.upper()
    case_df["_company_sort"] = case_df["company"].astype(str).str.lower()
    if "document_sort" not in case_df.columns:
        case_df["document_sort"] = case_df["document_label"].astype(str).str.lower().map(lambda value: 0 if "sustainability" in value else 1)
    case_df = case_df.sort_values(["document_sort", "_company_sort", "_ticker"]).reset_index(drop=True)

    st.markdown(
        """
<div class="card compact">
  <h3>External Company Inference Examples</h3>
  <div class="zh">外部公司推論案例</div>
  <p>The external lookup combines sustainability-report profiles and SEC 10-K profiles scored by the formal LightGBM model. Sustainability reports match the model training document type; SEC 10-K profiles are shown as document-type domain-shift examples.</p>
  <div class="zh">外部查詢結合永續報告與 SEC 10-K 風險檔案，皆由正式 LightGBM 模型評分。永續報告與模型訓練文件類型一致；SEC 10-K 則作為文件類型差異案例。</div>
  <div class="note" style="margin-top:12px;">External companies are inference only; they are not added to training or the formal holdout test set.<br><span class="zh">外部公司案例只做推論，不參與訓練或正式測試。</span><br>External inference uses document text that has not undergone the identical preprocessing applied to the training corpus, so external scores are conservative and intended for existence-level screening, not precise cross-company ranking.<br><span class="zh">外部推論所用文件未經與訓練語料完全相同的預處理，因此外部分數偏保守，僅供存在性篩選，不作為公司間精確排序。</span></div>
</div>
""",
        unsafe_allow_html=True,
    )

    def _row_label(row) -> str:
        company = str(row.get("company", "External Company")).strip()
        ticker = str(row.get("ticker", "")).strip()
        doc_label = str(row.get("document_label", row.get("document_type", "SEC 10-K"))).strip() or "SEC 10-K"
        return f"{company} / {ticker} — {doc_label}"

    labels = [_row_label(case_df.loc[idx]) for idx in case_df.index]
    label_to_idx = {label: idx for idx, label in zip(case_df.index, labels)}
    default_label = next(
        (
            label
            for label in labels
            if ("NVIDIA" in label.upper() or "NVDA" in label.upper()) and "SUSTAINABILITY" in label.upper()
        ),
        next((label for label in labels if "NVIDIA" in label.upper() or "NVDA" in label.upper()), labels[0]),
    )
    if "annual_external_case_label" not in st.session_state or st.session_state["annual_external_case_label"] not in labels:
        st.session_state["annual_external_case_label"] = default_label
    selected_label = st.selectbox("External Company Profile", labels, key="annual_external_case_label", format_func=display_text)
    st.caption(
        "SEC 10-K: the official annual report that U.S. listed companies are legally required to file with the U.S. Securities and Exchange Commission, covering business, risk, and financial disclosures.\n\n"
        "SEC 10-K：美國上市公司依法每年向美國證券管理委員會（SEC）提交的正式年度報告，涵蓋業務、風險與財務揭露，具法律申報效力。"
    )
    row = case_df.loc[label_to_idx[selected_label]].copy()
    document_label = str(row.get("document_label", row.get("document_type", "SEC 10-K"))).strip() or "SEC 10-K"
    is_sustainability = "sustainability" in document_label.lower() or "sustainability" in str(row.get("document_type", "")).lower()
    details = external_case_memo_details(row)
    if not details:
        warning_card("External profile not scored", "This precomputed profile does not include a model risk score.")
        return
    details["model_used"] = "LightGBM final scorer"
    details["sector"] = "Not assigned in prototype profile"
    details["lookup_mode"] = "External Official Filing Example"
    memo_final_note(details)

    external_feature_vector = external_feature_vector_from_row(row)
    range_context = feature_range_context(X)
    if is_sustainability:
        render_risk_gauge(details.get("risk_probability", 0.0))
        st.markdown(risk_result_html(details, external_feature_vector, range_context), unsafe_allow_html=True)
        st.markdown(
            """
<div class="note">Scored from the company's sustainability report — same document type as model training.<br><span class="zh">以公司永續報告書評分，與模型訓練文件類型一致。</span></div>
""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(sec_10k_domain_shift_html(details), unsafe_allow_html=True)

    filing_url = str(row.get("filing_url", row.get("source_url", ""))).strip()
    if filing_url.lower() in {"nan", "none"}:
        filing_url = ""
    filing_link = f'<a href="{esc(filing_url)}" target="_blank">Source link</a>' if filing_url else "Not available"
    source_file = str(row.get("source_file", "")).strip()
    if source_file.lower() in {"nan", "none"}:
        source_file = ""
    source_file_row = f'<div class="signal-row"><div class="signal-label">Source File</div><div class="signal-value">{esc(source_file)}</div></div>' if source_file else ""
    extraction_note = str(row.get("retrieval_note", row.get("extraction_note", ""))).strip()
    if extraction_note.lower() in {"nan", "none"}:
        extraction_note = ""
    st.markdown(
        f"""
<div class="card compact">
  <h3>Input ESG Disclosure Profile</h3>
  <p>Feature values used for the selected company risk score.</p>
  <div class="zh">本表顯示模型評分使用的 ESG 特徵值。</div>
  <div class="zh">輸入 ESG 揭露特徵資料</div>
  <div class="signal-row"><div class="signal-label">Company</div><div class="signal-value">{esc(details.get('company', ''))}</div></div>
  <div class="signal-row"><div class="signal-label">Ticker</div><div class="signal-value">{esc(details.get('ticker', ''))}</div></div>
  <div class="signal-row"><div class="signal-label">Data Source Type</div><div class="signal-value">{esc(details.get('data_source_type', 'External inference'))}</div></div>
  <div class="signal-row"><div class="signal-label">Document Type</div><div class="signal-value">{esc(details.get('document_type', document_label))}</div></div>
  <div class="signal-row"><div class="signal-label">Model used</div><div class="signal-value">LightGBM final scorer<br><span class="zh">模型：正式 LightGBM 評分器</span></div></div>
  <div class="signal-row"><div class="signal-label">Sector</div><div class="signal-value">Not assigned in prototype profile</div></div>
  <div class="signal-row"><div class="signal-label">Filing Type</div><div class="signal-value">{esc(str(row.get('filing_type', details.get('filing_type', ''))))}</div></div>
  <div class="signal-row"><div class="signal-label">Source</div><div class="signal-value">{esc(str(row.get('source', details.get('external_source', ''))))}</div></div>
  <div class="signal-row"><div class="signal-label">Source URL</div><div class="signal-value">{filing_link}</div></div>
  {source_file_row}
  <div class="note">Individual input signals are shown for interpretation. The final risk score reflects all 14 model features combined.<br><span class="zh">單一特徵只供判讀，最終分數由 14 個特徵共同決定。</span><br>Walk / External / Context features are median-filled for external inference profiles.<br><span class="zh">外部推論檔案的 Walk / External / Context 特徵以訓練資料中位數補齊。</span>{'<br>' + esc(extraction_note) if extraction_note else ''}</div>
</div>
""",
        unsafe_allow_html=True,
    )
    render_memo_generation(details, "annual_external", external_feature_vector, True)


def page_prediction_manual(df, X, y, meta, model, data_error, model_error):
    st.markdown(
        """
<div class="card compact">
  <h3>Manual Company Input</h3>
  <div class="small-caption">Prototype ESG Disclosure Profile Scoring</div>
  <div class="zh">手動輸入公司 ESG 特徵並產生風險分數。</div>
</div>
""",
        unsafe_allow_html=True,
    )
    if model_error or model is None:
        warning_card("Prediction model unavailable", model_error or "Model artifact is missing.")
        return
    medians = project_medians(X)
    st.markdown(
        report_caption(
            "🎛 拉拉看這些滑桿,即時看風險分數怎麼變化——試試把『模糊承諾語言』調高、『具體數字』調低,看會不會被標記為高風險。",
            "Try the sliders and watch the risk score change in real time — raise 'vague language', lower 'numeric disclosure', and see if it gets flagged.",
        ),
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns(3, gap="large")
    with c1:
        company = st.text_input("Company Name", value="Manual Scenario Company", key="annual_manual_company")
        ticker = st.text_input("Ticker", value="MANUAL", key="annual_manual_ticker")
        sector = st.selectbox("Sector", SECTORS, index=SECTORS.index("Technology") if "Technology" in SECTORS else 0, key="annual_manual_sector")
        industry = st.text_input("Industry optional", value="", key="annual_manual_industry")
    with c2:
        e_density = st.slider("Environmental Disclosure Intensity", 0.0, 100.0, float(medians.get("E_density", 0.0)), 0.1, key="annual_manual_e")
        s_density = st.slider("Social Disclosure Intensity", 0.0, 100.0, float(medians.get("S_density", 0.0)), 0.1, key="annual_manual_s")
        g_density = st.slider("Governance Disclosure Intensity", 0.0, 100.0, float(medians.get("G_density", 0.0)), 0.1, key="annual_manual_g")
        hedge_score = st.slider("Hedge / Vague Language Level", 0.0, 1.0, min(max(float(medians.get("hedge_score", 0.0)), 0.0), 1.0), 0.01, key="annual_manual_hedge")
        pct_numeric = st.slider("Quantitative Disclosure Ratio", 0.0, 1.0, min(max(float(medians.get("pct_numeric", 0.0)), 0.0), 1.0), 0.01, key="annual_manual_numeric")
    with c3:
        esg_topic_breadth = st.slider("ESG Topic Breadth", 0.0, 10.0, min(max(float(medians.get("esg_topic_breadth", 3.0)), 0.0), 10.0), 0.1, key="annual_manual_breadth")
        sector_rank = st.slider("External ESG Risk Signal", 0.0, 1.0, min(max(float(medians.get("sector_rank", 0.5)), 0.0), 1.0), 0.01, key="annual_manual_sector_rank")
        planet_signal = st.slider("External Environmental Signal", -100.0, 100.0, float(medians.get("planet_friendly_business", 0.0)), 1.0, key="annual_manual_planet")
        governance_signal = st.slider("External Governance / Ethics Signal", -100.0, 100.0, float(medians.get("honest_fair_business", 0.0)), 1.0, key="annual_manual_governance")

    feature_vector = medians.copy()
    feature_vector.update(
        {
            "E_density": float(e_density),
            "S_density": float(s_density),
            "G_density": float(g_density),
            "hedge_score": float(hedge_score),
            "pct_numeric": float(pct_numeric),
            "talk_total_density": float(e_density + s_density + g_density),
            "esg_topic_breadth": float(esg_topic_breadth),
            "sector_rank": float(sector_rank),
            "planet_friendly_business": float(planet_signal),
            "honest_fair_business": float(governance_signal),
            "sector_encoded": float(SECTOR_ENC.get(sector, medians.get("sector_encoded", 0.0))),
        }
    )
    user_fields = [
        "E_density",
        "S_density",
        "G_density",
        "hedge_score",
        "pct_numeric",
        "talk_total_density",
        "esg_topic_breadth",
        "sector_rank",
        "planet_friendly_business",
        "honest_fair_business",
        "sector_encoded",
    ]
    median_fields = [feature for feature in FEATURE_ORDER if feature not in user_fields]
    st.markdown(
        f"""
<div class="note">
  Manual input mode is a prototype scenario tool. Results depend on user-provided feature assumptions and median-filled values.
  <br><span class="zh">手動輸入模式為原型情境工具，結果取決於使用者輸入的特徵假設與中位數補值。</span>
  <br><strong>User-provided / derived fields:</strong> {esc(', '.join(user_fields))}
  <br><strong>Dataset median defaults:</strong> {esc(', '.join(median_fields))}
</div>
""",
        unsafe_allow_html=True,
    )
    if st.button("Score Manual Company Profile", key="annual_manual_score", width="stretch"):
        st.session_state["annual_manual_scored"] = True
    if not st.session_state.get("annual_manual_scored", False):
        return
    prob = predict_probability(model, feature_vector)
    if prob is None:
        warning_card("Manual profile not scored", "The trained LightGBM final scorer is unavailable.")
        return
    tier, _, _, _ = risk_tier(prob)
    action, action_zh = risk_review_action(tier)
    details = company_memo_details(company, ticker, sector, prob, tier, "Manual Input", industry)
    details.update(
        {
            "lookup_mode": "Manual Company Input",
            "data_source_type": "Manual company input scenario",
            "source": "Manual company input scenario",
            "source_zh": "資料來源：手動輸入公司資料",
            "model_used": "LightGBM final scorer",
            "recommended_action": action,
            "recommended_action_zh": action_zh,
            "key_input_signals": f"User-provided visible inputs plus dataset median defaults for: {', '.join(median_fields)}.",
        }
    )
    memo_final_note(details)
    range_context = feature_range_context(X)
    render_risk_gauge(prob)
    st.markdown(risk_result_html(details, feature_vector, range_context), unsafe_allow_html=True)
    with st.expander("Input ESG Disclosure Profile / 輸入 ESG 揭露特徵", expanded=False):
        rows = []
        for feature in FEATURE_ORDER:
            source = "User-provided or derived" if feature in user_fields else "Dataset median default"
            rows.append({"Feature": FEATURE_DISPLAY.get(feature, feature), "Raw Feature": feature, "Value": format_input_feature_value(feature, feature_vector[feature]), "Source": source})
        comparison_display = pd.DataFrame(rows)
        st.markdown(
            styled_table_html(
                comparison_display,
                title="Input ESG Disclosure Profile",
                subtitle="Feature values used for the selected company risk score.\n本表顯示模型評分使用的 ESG 特徵值。",
                badge_cols=["Source"],
                numeric_cols=[col for col in comparison_display.columns if "Score" in col or "Risk" in col],
            ),
            unsafe_allow_html=True,
        )
    render_memo_generation(details, "annual_manual", feature_vector, False)


def page_prediction(df, X, y, meta, model, data_error, model_error):
    page_header(
        "04",
        "Company Risk Lookup",
        "Company-Level ESG Greenwashing Risk Screening and Review Memo",
        "選擇公司或手動輸入公司資料，產生 ESG 風險查核備忘錄。",
    )
    st.markdown(
        """
<div class="card compact">
  <p>Use this page to select an existing company, review an external filing example, or manually enter an ESG profile to generate an ESG credit review memo.</p>
  <p>Internal lookup is based on 257 formal companies. The 407 setup is only for robustness validation, not the formal company universe or holdout set.</p>
  <div class="zh">可選公司、查看外部案例，或手動輸入 ESG 特徵產生查核 memo；257 為正式公司樣本，407 僅為穩健性檢查用。</div>
</div>
<div class="card compact">
  <h3>Workflow</h3>
  <p>Select company → Review official filing profile → Generate LightGBM risk score → Review input signal notes → Generate ESG Credit Review Memo → Download memo for internal review</p>
  <div class="zh">選擇公司 → 查看風險分數 → 解讀 ESG 訊號 → 產生 ESG 授信查核 Memo</div>
</div>
""",
        unsafe_allow_html=True,
    )
    with st.expander("Methodology Note: Lookup Modes and 257 vs 407", expanded=False):
        st.markdown(
            """
<div class="card compact">
  <p>This page contains three company lookup modes:</p>
  <ol>
    <li><strong>Select Existing Company:</strong> internal benchmark lookup from the original formal company universe of 257 companies.</li>
    <li><strong>External Official Filing Example:</strong> precomputed external official filing inference examples including NVIDIA, Apple, and IBM.</li>
    <li><strong>Manual Company Input:</strong> prototype scenario scoring based on user-provided ESG feature assumptions.</li>
  </ol>
  <p>The 407 setup refers to the largest augmented training setup used for robustness validation. It is not the formal company universe and not the holdout set. The formal final scorer remains LightGBM.</p>
  <div class="zh">內部基準查詢使用 257 家正式公司樣本；407 是樣本補強後的最大訓練設定，用於穩健性檢查，不是正式公司 universe，也不是 holdout。</div>
</div>
""",
            unsafe_allow_html=True,
        )
    mode = st.radio(
        "Lookup Mode",
        ["Select Existing Company", "External Official Filing Example", "Manual Company Input"],
        horizontal=True,
        key="annual_lookup_mode",
    )
    st.markdown('<div class="zh">查詢模式：選擇既有公司 / 外部正式申報案例 / 手動輸入公司資料</div>', unsafe_allow_html=True)

    if mode == "Select Existing Company":
        page_prediction_existing_company(df, X, y, meta, model, data_error, model_error)
    elif mode == "External Official Filing Example":
        page_prediction_external(X)
    else:
        page_prediction_manual(df, X, y, meta, model, data_error, model_error)


def page_explainability():
    page_header(
        "05",
        "Explainability",
        "SHAP-Based Explanation for the Final LightGBM Scorer",
        "使用 SHAP 解釋正式 LightGBM 模型的風險判斷依據。",
    )
    st.markdown(
        """
<div class="card compact">
  <p>This page explains the formal final LightGBM scorer using SHAP global and company-level explanations. SHAP helps ESG analysts and credit officers understand which features increase or decrease model-flagged greenwashing risk.</p>
  <div class="zh">SHAP 用來輔助理解模型判斷，不代表法律上認定漂綠。</div>
  <div class="grid-4" style="margin-top:14px;">
"""
        + metric_card("Final Model", "LightGBM", "正式評分器")
        + metric_card("Formal ROC-AUC", "0.953", "正式模型比較")
        + metric_card("Formal PR-AUC", "0.745", "高風險辨識")
        + metric_card("Formal F1", "0.658", "Precision / Recall 平衡")
        + """
  </div>
  <div class="note" style="margin-top:14px;">
    Features: 14 = 9 Talk + 4 Walk / External + 1 Context. SHAP explanations are generated from the formal final LightGBM scorer, not from the 407 robustness validation setup.
    <br><span class="zh">SHAP 對應正式 LightGBM 模型，不是 407 樣本補強實驗模型。</span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:22px;'></div>", unsafe_allow_html=True)
    st.markdown(
        """
<div class="card compact">
  <h3>How to read SHAP</h3>
  <p>The x-axis shows directional SHAP impact on the model prediction. Color indicates feature value: red = high, blue = low.</p>
  <div class="zh">X 軸代表特徵對模型預測的方向性影響；紅色代表特徵值高，藍色代表特徵值低。</div>
  <div class="grid-4" style="margin-top:14px;">
    <div class="metric"><div class="label">Positive SHAP value</div><div class="value" style="font-size:21px;color:#C94C3A;">Pushes model risk score higher</div><div class="zh">推高模型風險分數</div></div>
    <div class="metric"><div class="label">Negative SHAP value</div><div class="value" style="font-size:21px;color:#2E7D32;">Pushes model risk score lower</div><div class="zh">降低模型風險分數</div></div>
    <div class="metric"><div class="label">Red color</div><div class="value" style="font-size:21px;color:#C94C3A;">Higher feature value</div><div class="zh">特徵值較高</div></div>
    <div class="metric"><div class="label">Blue color</div><div class="value" style="font-size:21px;color:#3F6FB5;">Lower feature value</div><div class="zh">特徵值較低</div></div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:22px;'></div>", unsafe_allow_html=True)
    fi = feature_importance_df().copy()
    feature_col = "feature" if "feature" in fi.columns else fi.columns[0]
    shap_col_name = "mean_abs_shap" if "mean_abs_shap" in fi.columns else None
    fi["Original Feature"] = fi[feature_col].astype(str)
    fi["Display Feature"] = fi["Original Feature"].map(FEATURE_DISPLAY).fillna(fi["Original Feature"])
    fi["Feature Group"] = fi["Original Feature"].map(feature_group)
    if shap_col_name:
        fi["Mean |SHAP|"] = pd.to_numeric(fi[shap_col_name], errors="coerce").fillna(0).round(3)
        fi = fi.sort_values("Mean |SHAP|", ascending=False)
    else:
        fi["Mean |SHAP|"] = "N/A"
    fi["Business Meaning"] = fi["Original Feature"].map(FEATURE_BUSINESS_MEANING).fillna("Supports model interpretation and due diligence review.")
    fi = fi.head(12)

    def highlight_row_attr(idx):
        return ' class="dashboard-highlight"' if idx < 3 else ""

    compact_feature_rows = "".join(
        f"<tr{highlight_row_attr(idx)}>"
        + f"<td><span class='feature-rank-number'>{idx+1}.</span> <strong>{esc(row['Display Feature'])}</strong></td>"
        + f"<td>{table_badge(row['Feature Group'])}</td>"
        + f"<td class='num'>{esc(row['Mean |SHAP|'])}</td>"
        + "</tr>"
        for idx, (_, row) in enumerate(fi.head(8).iterrows())
    )
    full_feature_rows = "".join(
        f"<tr{highlight_row_attr(idx)}>"
        + f"<td><span class='feature-rank-number'>{idx+1}.</span> <strong>{esc(row['Display Feature'])}</strong></td>"
        + f"<td>{esc(row['Original Feature'])}</td>"
        + f"<td>{table_badge(row['Feature Group'])}</td>"
        + f"<td class='num'>{esc(row['Mean |SHAP|'])}</td>"
        + f"<td>{esc(row['Business Meaning'])}</td>"
        + "</tr>"
        for idx, (_, row) in enumerate(fi.iterrows())
    )
    st.markdown(
        f"""
<div class="card">
  <h3>Global SHAP Explanation</h3>
  <div class="zh">全域 SHAP 模型解釋</div>
  <div class="small-caption">The beeswarm chart summarizes the main features that influence risk scores across the dashboard company universe.</div>
  <div class="zh">此圖彙整正式 dashboard 公司資料中，最影響風險分數的主要特徵。</div>
  <div class="note" style="margin-top:12px;">
    Top global drivers: Sector-relative ESG Risk, ESG Disclosure Intensity, and External Governance / Ethics Signal.
    <br><span class="zh">主要驅動因子：同產業相對 ESG 風險、ESG 揭露強度、外部治理 / 誠信訊號。</span>
  </div>
  <div class="shap-beeswarm">{image_html(os.path.join(OUTPUTS, "shap_beeswarm.png"), "SHAP beeswarm")}</div>
  <div class="shap-explain-box">
    <span class="shap-explain-icon">SHAP</span>
    <div>
      <div class="shap-explain-title">Why SHAP Matters</div>
      <div class="shap-explain-text">SHAP explains which features increased or decreased the model-flagged greenwashing risk score.</div>
      <div class="zh">SHAP 可說明哪些特徵提高或降低模型標記之漂綠風險分數。</div>
    </div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:22px;'></div>", unsafe_allow_html=True)
    st.markdown(
        f"""
<div class="card report-panel">
  <div class="insight-strip"><span class="icon">↗</span><div><strong>Top Feature Importance Preview</strong><br><span class="small-caption">Styled analytical table; top drivers are highlighted for presentation.</span></div></div>
  <h3>Top Feature Importance Preview</h3>
  <div class="zh">模型主要判斷依據摘要</div>
  <div class="small-caption">Top 8 features are shown here; the full Top 12 table is available below.</div>
  <div class="zh">此處顯示前 8 個重要特徵，完整前 12 名可展開查看。</div>
  <div class="table-wrap">
    <table class="report-table styled-table" style="font-size:.95rem;min-width:720px;">
      <thead><tr><th>Display Feature</th><th>Feature Group</th><th>Mean |SHAP|</th></tr></thead>
      <tbody>{compact_feature_rows}</tbody>
    </table>
  </div>
  <div class="small-caption">The model combines ESG disclosure-language intensity with external ESG evidence. Higher disclosure intensity combined with weaker external evidence can increase review priority.</div>
  <div class="zh">模型同時結合 ESG 揭露文字強度與外部 ESG 證據。若揭露強度高但外部證據偏弱，可能提高查核優先順序。</div>
  <div class="small-caption">Near-zero SHAP features have limited marginal contribution in the current trained model. This does not mean the features are conceptually useless; it only means their marginal contribution is small in the current trained model.</div>
  <div class="zh">接近零的 SHAP 特徵代表其在目前模型中的邊際貢獻較小，不代表該特徵沒有概念價值。</div>
</div>
""",
        unsafe_allow_html=True,
    )
    with st.expander("View full Top 12 feature table / 查看完整特徵表"):
        st.markdown(
            f"""
<div class="table-wrap">
  <table class="report-table styled-table" style="font-size:.92rem;min-width:760px;">
    <thead><tr><th>Display Feature</th><th>Original Feature</th><th>Feature Group</th><th>Mean |SHAP|</th><th>Business Meaning</th></tr></thead>
    <tbody>{full_feature_rows}</tbody>
  </table>
</div>
""",
            unsafe_allow_html=True,
        )
    with st.expander("View SHAP chart larger"):
        st.markdown(
            f'<div class="shap-expanded">{image_html(os.path.join(OUTPUTS, "shap_beeswarm.png"), "SHAP beeswarm enlarged")}</div>',
            unsafe_allow_html=True,
        )
    st.markdown("<div style='height:22px;'></div>", unsafe_allow_html=True)
    st.markdown(
        """
<div class="card compact">
  <h3>Methodology Note</h3>
  <p>Sector-relative ESG Risk is highly influential because the model is designed as a Talk + Walk / External risk scoring engine. The Talk-only ablation confirms that ESG disclosure language also carries independent signal.</p>
  <div class="zh">模型同時看公司怎麼說 ESG，以及外部 ESG 風險訊號。</div>
  <div class="note" style="margin-top:12px;">Talk-only ablation AUC = 0.741. This confirms that text-based features carry independent signal, but it should be interpreted as a supporting robustness check rather than a formal determination.</div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:22px;'></div>", unsafe_allow_html=True)
    st.markdown(
        """
<div class="section-title-row">
  <div>
    <h2>Company-Level SHAP Review Examples</h2>
    <div class="zh">公司層級 SHAP 查核範例</div>
    <div class="small-caption">These are selected model-flagged high-risk cases for company-level SHAP explanation and due diligence support.</div>
    <div class="zh">以下為選定的模型標記高風險案例，用於公司層級 SHAP 解釋與查核支援。</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
    case_data = [
        {
            "ticker": "CVX",
            "title": "CVX / Chevron / Energy",
            "sector": "Energy",
            "path": "outputs/shap_waterfall_1.png",
            "drivers": "High ESG disclosure intensity combined with weaker sector-relative ESG indicators contributed most to this risk assessment.",
            "drivers_zh": "較高的 ESG 揭露強度，加上相對偏弱的同業 ESG 指標，是本次風險評估的主要驅動因素。",
            "interpretation": "The company's ESG communication appears stronger than the supporting external ESG evidence and should be reviewed further.",
            "interpretation_zh": "企業 ESG 揭露內容相對積極，但外部 ESG 證據支持程度較弱，建議進一步查核。",
            "followup": "Review supporting evidence, peer comparisons, controversy records, and third-party ESG assurance before making ESG-related decisions.",
            "followup_zh": "建議檢視 ESG 佐證資料、同業比較、爭議紀錄及第三方 ESG 確信資訊後，再進行相關決策。",
        },
        {
            "ticker": "MTB",
            "title": "MTB / M&T Bank / Financial Services",
            "sector": "Financial Services",
            "path": "outputs/shap_waterfall_2.png",
            "drivers": "High ESG disclosure intensity combined with weaker sector-relative ESG indicators contributed most to this risk assessment.",
            "drivers_zh": "較高的 ESG 揭露強度，加上相對偏弱的同業 ESG 指標，是本次風險評估的主要驅動因素。",
            "interpretation": "The company's ESG communication appears stronger than the supporting external ESG evidence and should be reviewed further.",
            "interpretation_zh": "企業 ESG 揭露內容相對積極，但外部 ESG 證據支持程度較弱，建議進一步查核。",
            "followup": "Review supporting evidence, peer comparisons, controversy records, and third-party ESG assurance before making ESG-related decisions.",
            "followup_zh": "建議檢視 ESG 佐證資料、同業比較、爭議紀錄及第三方 ESG 確信資訊後，再進行相關決策。",
        },
        {
            "ticker": "WYNN",
            "title": "WYNN / Wynn Resorts / Consumer Cyclical",
            "sector": "Consumer Cyclical",
            "path": "outputs/shap_waterfall_3.png",
            "drivers": "High ESG disclosure intensity combined with weaker sector-relative ESG indicators contributed most to this risk assessment.",
            "drivers_zh": "較高的 ESG 揭露強度，加上相對偏弱的同業 ESG 指標，是本次風險評估的主要驅動因素。",
            "interpretation": "The company's ESG communication appears stronger than the supporting external ESG evidence and should be reviewed further.",
            "interpretation_zh": "企業 ESG 揭露內容相對積極，但外部 ESG 證據支持程度較弱，建議進一步查核。",
            "followup": "Review supporting evidence, peer comparisons, controversy records, and third-party ESG assurance before making ESG-related decisions.",
            "followup_zh": "建議檢視 ESG 佐證資料、同業比較、爭議紀錄及第三方 ESG 確信資訊後，再進行相關決策。",
        },
    ]
    case_rows = "".join(
        f"""
<div class="company-shap-row shap-review-card">
  <div class="company-shap-header">
    <div>
      <div class="company-shap-title">🔍 {esc(case["title"])}</div>
      <div class="company-shap-sector">{esc(case["sector"])}</div>
      <span class="badge-pill risk-pill-high">🚨 model-flagged high-risk case</span>
    </div>
    <div class="company-shap-risk">
      <div class="company-shap-label">Risk Level</div>
      <div class="risk-badge risk-high">Very High (&gt;95%)</div>
      <div class="zh">模型標記高風險｜需進一步 ESG 盡職調查</div>
    </div>
  </div>
  <div class="company-shap-image-wrap">
    {image_html(os.path.join(ROOT, case["path"]), case["ticker"] + " waterfall")}
  </div>
  <div class="company-shap-detail-grid shap-review-grid">
    <div class="company-shap-detail-card shap-mini-card">
      <div class="company-shap-label"><span class="label-icon">01</span>Main Drivers</div>
      <div class="company-shap-text">{esc(case["drivers"])}</div>
      <div class="zh">{esc(case["drivers_zh"])}</div>
    </div>
    <div class="company-shap-detail-card shap-mini-card">
      <div class="company-shap-label"><span class="label-icon">02</span>Business Interpretation</div>
      <div class="company-shap-text">{esc(case["interpretation"])}</div>
      <div class="zh">{esc(case["interpretation_zh"])}</div>
    </div>
    <div class="company-shap-detail-card shap-mini-card">
      <div class="company-shap-label"><span class="label-icon">03</span>Recommended Follow-up</div>
      <div class="company-shap-text">{esc(case["followup"])}</div>
      <div class="zh">{esc(case["followup_zh"])}</div>
    </div>
  </div>
</div>
"""
        for case in case_data
    )
    st.markdown(f'<div class="company-shap-matrix">{case_rows}</div>', unsafe_allow_html=True)


def page_business():
    page_header(
        "06",
        "Business Recommendation",
        "ESG Credit Review Actions and Memo Generation",
        "ESG 授信審查建議與 Memo 產生",
    )
    st.markdown(
        """
<div class="card compact">
  <p>This page converts the model-flagged greenwashing risk score into ESG review actions, evidence requirements, monitoring frequency, and a company-level ESG credit memo.</p>
  <div class="zh">本頁將模型標記的漂綠風險分數轉換為 ESG 審查行動、佐證資料要求、追蹤頻率與公司層級 ESG 授信查核 Memo。</div>
</div>
<div class="business-banner">
  <span class="icon-badge">✅</span>
  <div>
    <div class="business-banner-title">Model flags risk; credit officers make final decisions.</div>
    <div class="business-banner-body">模型提示風險，最終決策由授信人員判斷。</div>
  </div>
</div>
<div class="decision-flow-grid">
  <div class="decision-flow-card"><span class="flow-step-number">1</span><div class="flow-title">🎯 Risk Score</div><div class="flow-body">LightGBM produces a screening score from the 14-feature ESG profile.<br><span class="zh">以 14 個 ESG 特徵產生風險篩選分數。</span></div></div>
  <div class="decision-flow-card"><span class="flow-step-number">2</span><div class="flow-title">📌 Review Priority</div><div class="flow-body">Risk tier determines ESG due diligence priority and evidence depth.<br><span class="zh">風險等級決定查核優先順序與佐證深度。</span></div></div>
  <div class="decision-flow-card"><span class="flow-step-number">3</span><div class="flow-title">📄 ESG Credit Memo</div><div class="flow-body">Analysts receive a structured memo for internal review and documentation.<br><span class="zh">產生授信查核 memo 供內部審查留存。</span></div></div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)


    coverage_df, coverage_err = safe_read_csv(os.path.join(OUTPUTS, "holdout_review_coverage.csv"))
    st.markdown(
        """
<div class="card compact">
  <h3>📈 Review Prioritization Impact</h3>
  <div class="zh">查核排序效益</div>
  <p>In the holdout test, reviewing only the top 10% flagged by the model (8 of 78 companies) captured all 6 high-risk companies, reducing manual review workload by about 90%. The holdout contains only 6 positive cases, so this is screening evidence, not a performance guarantee.</p>
  <div class="zh">在本次 holdout 測試中，僅查核模型標記的前 10%（78 家中的 8 家）即涵蓋全部 6 家高風險公司，人工查核工作量減少約 90%。holdout 僅 6 個正樣本，此為篩選證據，非保證。</div>
  <div class="business-impact-grid" style="margin-top:14px;">
    <div class="insight-card kpi-card-premium"><div class="kpi-card-accent-circle"></div><div class="kpi-card-icon">🎯</div><div class="kpi-card-title">Review Band</div><div class="kpi-card-value">Top 10%</div><div class="kpi-card-subtitle">Highest-priority holdout ranking band</div></div>
    <div class="insight-card kpi-card-premium"><div class="kpi-card-accent-circle"></div><div class="kpi-card-icon">🏢</div><div class="kpi-card-title">Companies Reviewed</div><div class="kpi-card-value">8 / 78</div><div class="kpi-card-subtitle">Companies reviewed in the top band</div></div>
    <div class="insight-card kpi-card-premium"><div class="kpi-card-accent-circle"></div><div class="kpi-card-icon">🚨</div><div class="kpi-card-title">High-Risk Captured</div><div class="kpi-card-value">6 / 6</div><div class="kpi-card-subtitle">High-risk holdout cases captured</div></div>
    <div class="insight-card kpi-card-premium"><div class="kpi-card-accent-circle"></div><div class="kpi-card-icon">✅</div><div class="kpi-card-title">Workload Reduction</div><div class="kpi-card-value">~90%</div><div class="kpi-card-subtitle">Approximate manual review reduction</div></div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
    if coverage_err or coverage_df.empty:
        st.caption("Holdout review coverage table is not available.")
    else:
        display_coverage = coverage_df.copy()
        rename_cols = {
            "review_band": "Review Band",
            "number_companies_reviewed": "Companies Reviewed",
            "true_high_risk_captured": "High-Risk Captured",
            "total_holdout_positives": "Total Holdout Positives",
            "recall": "Recall",
            "workload_reduction_vs_reviewing_all_78": "Workload Reduction",
        }
        display_coverage = display_coverage.rename(columns=rename_cols)
        keep_cols = [col for col in rename_cols.values() if col in display_coverage.columns]
        if "Recall" in display_coverage.columns:
            display_coverage["Recall"] = pd.to_numeric(display_coverage["Recall"], errors="coerce").map(lambda value: f"{value:.0%}" if pd.notna(value) else "")
        if "Workload Reduction" in display_coverage.columns:
            display_coverage["Workload Reduction"] = pd.to_numeric(display_coverage["Workload Reduction"], errors="coerce").map(lambda value: f"{value:.1%}" if pd.notna(value) else "")
        st.markdown(
            styled_table_html(
                display_coverage[keep_cols],
                title="Review Coverage Evidence",
                subtitle="Screening impact by review band; this supports prioritization and is not a guarantee of future outcomes.",
                numeric_cols=["Companies Reviewed", "High-Risk Captured", "Total Holdout Positives", "Recall", "Workload Reduction"],
                highlight_first=True,
            ),
            unsafe_allow_html=True,
        )
    st.markdown("<div style='height:22px;'></div>", unsafe_allow_html=True)

    role = st.selectbox("Review Role", ["ESG Analyst", "Credit Officer", "Investment Committee"])
    st.caption("審查角色")
    role_focus, role_note_zh = {
        "ESG Analyst": (
            "disclosure evidence, controversy context, SHAP drivers.",
            "重點：揭露證據、爭議脈絡與模型解釋。",
        ),
        "Credit Officer": (
            "credit action, required evidence, monitoring frequency.",
            "重點：授信行動、佐證資料與追蹤頻率。",
        ),
        "Investment Committee": (
            "review priority, escalation decision, portfolio implications.",
            "重點：查核優先順序、升級審查與投資組合影響。",
        ),
    }[role]
    st.markdown(
        """
<div class="small-caption">Select a role to adjust the wording of the recommendation.</div>
<div class="zh">選擇審查角色以調整建議語氣。</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
<div class="card compact">
  <div class="risk-policy-label">Current View</div>
  <h3 style="margin-bottom:8px;">{esc(role)}</h3>
  <p><strong>Focus:</strong> {esc(role_focus)}</p>
  <div class="zh">{esc(role_note_zh)}</div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:22px;'></div>", unsafe_allow_html=True)

    st.markdown(
        """
<div class="section-title-row">
  <div>
    <h2>✅ Risk-Based Review Actions</h2>
    <div class="zh">依風險等級設定審查行動</div>
  </div>
</div>
<div style="height:16px;"></div>
<div class="risk-action-grid">
  <div class="risk-action-card low">
    <div class="risk-action-title">✅ Low Risk 0–30%</div>
    <div class="zh">低風險｜標準審查</div>
    <div class="risk-action-section"><div class="risk-action-label">🧾 Credit Action</div><div class="risk-action-text">Standard ESG review.<br><span class="zh">一般 ESG 審查。</span></div></div>
    <div class="risk-action-section"><div class="risk-action-label">📄 Required Evidence</div><div class="risk-action-text">Regular sustainability disclosure and loan-purpose documentation.<br><span class="zh">確認永續揭露與資金用途文件。</span></div></div>
    <div class="risk-action-section"><div class="risk-action-label">📅 Monitoring</div><div class="risk-action-text">Annual ESG review.<br><span class="zh">年度追蹤。</span></div></div>
  </div>
  <div class="risk-action-card medium">
    <div class="risk-action-title">⚠️ Medium Risk 30–70%</div>
    <div class="zh">中風險｜加強查核</div>
    <div class="risk-action-section"><div class="risk-action-label">🧾 Credit Action</div><div class="risk-action-text">Enhanced ESG due diligence.<br><span class="zh">加強 ESG 盡職調查。</span></div></div>
    <div class="risk-action-section"><div class="risk-action-label">📄 Required Evidence</div><div class="risk-action-text">Supporting ESG evidence, peer comparison, and disclosure clarification.<br><span class="zh">要求 ESG 佐證資料、同業比較與揭露補充說明。</span></div></div>
    <div class="risk-action-section"><div class="risk-action-label">📅 Monitoring</div><div class="risk-action-text">Semi-annual review.<br><span class="zh">半年追蹤。</span></div></div>
  </div>
  <div class="risk-action-card high">
    <div class="risk-action-title">🚨 High Risk 70–100%</div>
    <div class="zh">高風險｜升級盡職調查</div>
    <div class="risk-action-section"><div class="risk-action-label">🧾 Credit Action</div><div class="risk-action-text">Escalate to enhanced ESG due diligence before sustainable lending or ESG investment approval.<br><span class="zh">永續授信或 ESG 投資核准前，升級 ESG 盡職調查。</span></div></div>
    <div class="risk-action-section"><div class="risk-action-label">📄 Required Evidence</div><div class="risk-action-text">Third-party assurance; board oversight explanation; controversy history review; emissions or transition evidence where applicable.<br><span class="zh">第三方確信、董事會監督說明、爭議紀錄查核，並視情況要求排放或轉型證據。</span></div></div>
    <div class="risk-action-section"><div class="risk-action-label">📅 Monitoring</div><div class="risk-action-text">Quarterly review or watchlist monitoring.<br><span class="zh">季度追蹤或列入觀察名單。</span></div></div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:22px;'></div>", unsafe_allow_html=True)

    company = st.session_state.get("annual_selected_company")
    prob = st.session_state.get("annual_risk_probability")
    details = None
    if prob is None:
        memo_body = """
<p>Select a company in Company Risk Lookup to generate a customized ESG credit memo.</p>
<div class="zh">請先至公司風險查詢頁選擇公司，即可產生 ESG 授信查核 Memo。</div>
"""
    else:
        details = st.session_state.get("annual_memo_details")
        if not details:
            details = company_memo_details(
                company,
                st.session_state.get("annual_selected_ticker", ""),
                st.session_state.get("annual_selected_sector", ""),
                prob,
                st.session_state.get("annual_risk_tier", "Low"),
                st.session_state.get("profile_source", "Dataset Company"),
            )
        ticker_row = memo_row("Ticker / 股票代碼", details["ticker"]) if details.get("ticker") else ""
        cik_row = memo_row("CIK", details["cik"]) if details.get("cik") else ""
        filing_type_row = memo_row("Filing Type", details["filing_type"]) if details.get("filing_type") else ""
        filing_url_row = memo_row("Filing URL", details["filing_url"]) if details.get("filing_url") else ""
        memo_body = (
            memo_row("Source / 資料來源", details["source"], details["source_zh"])
            + memo_row("Company / 公司", display_company_name(details["company"]))
            + ticker_row
            + cik_row
            + filing_type_row
            + filing_url_row
            + memo_row("Sector / 產業", details["sector"])
            + memo_row("Model Risk Score / 模型風險分數", f"{details['risk_probability']*100:.1f}%")
            + memo_row("Risk Tier / 風險等級", details["risk_tier"])
            + memo_row("Main Concern / 主要疑慮", details["main_concern"], details["main_concern_zh"])
            + memo_row("Recommended Action / 建議行動", details["recommended_action"], details["recommended_action_zh"])
            + memo_row("Required Evidence / 需補充證據", details["required_evidence"], details["required_evidence_zh"])
            + memo_row("Monitoring / 追蹤頻率", details["monitoring"], details["monitoring_zh"])
        )

    st.markdown(
        f"""
<div class="memo-card">
  <h3>📄 ESG Credit Review Memo</h3>
  <div class="zh">ESG 授信查核 Memo</div>
  <div class="small-caption">This memo is generated from the latest company selected in Company Risk Lookup.</div>
  <div class="zh">此 Memo 會依據公司風險查詢頁最新選擇的公司與模型分數產生。</div>
  <div style="margin-top:14px;">{memo_body}</div>
</div>
""",
        unsafe_allow_html=True,
    )
    if details:
        try:
            st.download_button(
                "Download Company ESG Memo PDF",
                company_memo_pdf_bytes(details),
                memo_pdf_filename(details),
                mime="application/pdf",
                key="annual_business_memo_download",
            )
        except Exception as exc:
            st.warning(f"Company ESG Memo PDF is unavailable: {exc}")

    st.markdown("<div style='height:22px;'></div>", unsafe_allow_html=True)
    st.markdown(
        """
<div class="card compact">
  <h3>Future Extension</h3>
  <p>Current prototype supports preprocessed SEC 10-K company risk lookup. Future work can extend the dashboard to on-demand SEC EDGAR retrieval for any listed company.</p>
  <div class="zh">目前 prototype 支援已預先處理的 SEC 10-K 公司風險查詢；未來可擴充為輸入任一上市公司，自動抓取 SEC EDGAR 文件並產生風險分數。</div>
</div>
<div style="height:18px;"></div>
<div class="note">
  Final credit and investment decisions remain subject to internal review and supporting documentation.
  <br><span class="zh">最終授信與投資決策仍須依內部審查與佐證文件辦理。</span>
</div>
""",
        unsafe_allow_html=True,
    )



def main():
    page = sidebar()
    df, X, y, meta, data_error = load_project_data()
    model, model_error = load_project_model()

    if page == "Home":
        page_home()
    elif page == "Data Explorer":
        page_data(df, X, y, meta, data_error)
    elif page == "Model Performance":
        page_model(df, X, y, model, data_error, model_error)
    elif page == "Model Contribution Analysis":
        page_model_contribution()
    elif page == "Company Risk Lookup":
        page_prediction(df, X, y, meta, model, data_error, model_error)
    elif page == "Explainability":
        page_explainability()
    elif page == "Business Recommendation":
        page_business()


if __name__ == "__main__":
    main()

# FINAL_LAPTOP_LAYOUT_RESPONSIVENESS_OVERRIDE
st.markdown(
    """
<style>
/* Final laptop layout responsiveness override. CSS only; no model/data logic changes. */
:root {
  --final-sidebar-width: clamp(340px, 25vw, 390px);
}

/* Keep Streamlit sidebar and its collapse/expand control usable. */
[data-testid="stSidebar"] {
  width: var(--final-sidebar-width) !important;
  min-width: var(--final-sidebar-width) !important;
  max-width: var(--final-sidebar-width) !important;
}
[data-testid="stSidebar"] * {
  visibility: visible !important;
}
[data-testid="stHeader"] {
  background: rgba(247,250,245,.72) !important;
  backdrop-filter: blur(12px);
}

[data-testid="stAppViewContainer"] .main .block-container,
.main .block-container,
.block-container {
  max-width: 1120px !important;
  padding-top: 1.75rem !important;
  padding-left: 1.15rem !important;
  padding-right: 1.15rem !important;
}

.page-head {
  margin-top: 1rem !important;
}

h1,
.hero-title,
.page-title,
.cover-title {
  word-break: normal !important;
  overflow-wrap: normal !important;
  hyphens: none !important;
}

h1 {
  font-size: clamp(36px, 4.2vw, 64px) !important;
  line-height: 1.12 !important;
}

.hero h1,
.home-hero h1,
.landing-hero h1,
.hero-title,
.cover-title {
  font-size: clamp(42px, 5vw, 72px) !important;
  line-height: 1.06 !important;
  max-width: 1120px !important;
}

.hero,
.home-hero,
.landing-hero,
.cover-hero {
  max-width: 100% !important;
  padding-left: 1.25rem !important;
  padding-right: 1.25rem !important;
}

.hero.report-panel {
  min-height: 330px !important;
  gap: 24px !important;
}

.hero .mission {
  font-size: 1.02rem !important;
  line-height: 1.55 !important;
}

.hero h2 {
  font-size: 24px !important;
}

@media (max-width: 1200px) {
  [data-testid="stAppViewContainer"] .main .block-container,
  .main .block-container,
  .block-container {
    max-width: 1040px !important;
    padding-top: 1.5rem !important;
    padding-left: 1rem !important;
    padding-right: 1rem !important;
  }

  h1 {
    font-size: clamp(32px, 4.6vw, 56px) !important;
  }

  .hero h1,
  .home-hero h1,
  .landing-hero h1,
  .hero-title,
  .cover-title {
    font-size: clamp(38px, 5vw, 60px) !important;
  }

  .card,
  .metric-card,
  .kpi-card {
    padding: 1rem !important;
  }

}

@media (max-width: 900px) {
  h1,
  .hero h1,
  .home-hero h1,
  .landing-hero h1,
  .hero-title,
  .cover-title {
    font-size: clamp(30px, 8vw, 44px) !important;
  }

  [data-testid="stAppViewContainer"] .main .block-container,
  .main .block-container,
  .block-container {
    padding-left: 0.85rem !important;
    padding-right: 0.85rem !important;
  }
}
</style>
""",
    unsafe_allow_html=True,
)

# PRESENTATION_POLISH_OVERRIDE
st.markdown(
    """
<style>
/* Presentation polish override. CSS only; preserves all model/data values and app logic. */
:root {
  --polish-bg:#F6FAF5;
  --polish-panel:#FFFFFF;
  --polish-panel-soft:#F9FCF8;
  --polish-ink:#173D2A;
  --polish-muted:#64776A;
  --polish-line:#DCE9DD;
  --polish-green:#1F6A43;
  --polish-green-2:#2E7D32;
  --polish-gold:#D99A2B;
  --polish-red:#C94C3A;
  --polish-radius:20px;
  --polish-shadow:0 14px 34px rgba(10,41,26,.085);
  --polish-shadow-soft:0 8px 20px rgba(10,41,26,.065);
}

html, body, [data-testid="stAppViewContainer"] {
  background:
    linear-gradient(rgba(31,92,58,.020) 1px, transparent 1px),
    linear-gradient(90deg, rgba(31,92,58,.018) 1px, transparent 1px),
    radial-gradient(circle at 92% 2%, rgba(205,230,211,.72) 0, transparent 26%),
    radial-gradient(circle at 4% 92%, rgba(224,239,226,.64) 0, transparent 24%),
    var(--polish-bg) !important;
  background-size:44px 44px,44px 44px,auto,auto,auto !important;
}

[data-testid="stAppViewBlockContainer"],
.main .block-container,
.block-container {
  max-width: min(1180px, calc(100vw - var(--final-sidebar-width) - 42px)) !important;
  padding-bottom: 4rem !important;
}

.page-head {
  grid-template-columns:76px minmax(0,1fr) !important;
  gap:18px !important;
  margin:1.15rem 0 1.55rem !important;
  align-items:center !important;
}
.page-num {
  width:76px;
  height:76px;
  border-radius:24px;
  display:grid;
  place-items:center;
  background:linear-gradient(145deg,#FFFFFF,#EAF4EC);
  border:1px solid rgba(31,92,58,.12);
  box-shadow:var(--polish-shadow-soft);
  color:#B8D8BD !important;
  font-size:48px !important;
  line-height:1 !important;
}
.page-title h1 {
  font-size:clamp(34px,3.6vw,52px) !important;
  letter-spacing:0 !important;
  max-width:1000px !important;
}
.page-title-line {
  display:flex;
  align-items:center;
  gap:13px;
}
.page-title-line h1 {
  margin:0 !important;
}
.icon-badge {
  width:42px;
  height:42px;
  border-radius:16px;
  display:inline-grid;
  place-items:center;
  flex:0 0 auto;
  background:linear-gradient(145deg,#EAF4EC,#FFFFFF);
  border:1px solid rgba(31,106,67,.16);
  box-shadow:var(--polish-shadow-soft);
  color:var(--polish-green) !important;
  font-size:1.18rem;
  line-height:1;
}
.page-title .subtitle {
  color:var(--polish-muted) !important;
  font-size:1.03rem !important;
  font-weight:800 !important;
}

.card,
.metric,
.table-panel,
.company-shap-row,
div[data-testid="stVerticalBlockBorderWrapper"] {
  border-radius:var(--polish-radius) !important;
  border:1px solid var(--polish-line) !important;
  box-shadow:var(--polish-shadow) !important;
}
.card {
  padding:22px 24px !important;
  background:
    linear-gradient(180deg,rgba(255,255,255,.98),rgba(250,253,249,.98)) !important;
}
.card.compact {
  padding:17px 19px !important;
  border-radius:18px !important;
}
.card:hover,
.metric:hover {
  transform:translateY(-1px) !important;
  box-shadow:0 18px 38px rgba(10,41,26,.105) !important;
}
.card h2,
.card h3,
.chart-title,
.table-panel-title {
  letter-spacing:0 !important;
}
.card p,
.small-caption,
.zh,
.note {
  overflow-wrap:break-word;
}
.report-panel {
  border-top:0 !important;
  box-shadow:inset 0 4px 0 rgba(31,106,67,.82), var(--polish-shadow) !important;
}
.card.warn,
.risk-policy-row.medium {
  box-shadow:inset 0 4px 0 rgba(217,154,43,.84), var(--polish-shadow) !important;
}
.card.danger,
.risk-policy-row.high {
  box-shadow:inset 0 4px 0 rgba(201,76,58,.84), var(--polish-shadow) !important;
}

.grid-2,
.grid-3,
.grid-4,
.grid-6,
.home-process-grid,
.company-shap-detail-grid {
  gap:16px !important;
}
.grid-4 .metric,
.home-metrics .metric {
  min-height:128px !important;
}
.model-kpis {
  gap:20px !important;
  align-items:stretch;
}
.model-kpis .metric {
  min-height:150px !important;
}
.model-kpis .kpi-card-value {
  font-size:clamp(2.25rem,2.9vw,2.95rem) !important;
}
.metric {
  display:flex;
  flex-direction:column;
  justify-content:space-between;
  background:linear-gradient(145deg,#FFFFFF 0%,#F8FCF6 100%) !important;
}
.metric .label {
  color:#5F7165 !important;
  font-size:.78rem !important;
  line-height:1.25 !important;
  letter-spacing:.07em !important;
}
.metric .value {
  color:var(--polish-green-2) !important;
  font-size:clamp(25px,2.3vw,34px) !important;
  line-height:1.05 !important;
  overflow-wrap:anywhere !important;
}
.kpi-card-premium {
  position:relative;
  overflow:hidden;
  min-height:132px !important;
  border-radius:26px !important;
  padding:16px 17px 15px !important;
  border:1px solid rgba(31,106,67,.16) !important;
  background:
    linear-gradient(145deg,rgba(255,255,255,.98) 0%,rgba(249,253,248,.98) 52%,rgba(238,247,241,.94) 100%) !important;
  box-shadow:0 10px 24px rgba(10,41,26,.065) !important;
}
.home-metrics .metric.kpi-card-premium {
  background:
    linear-gradient(145deg,rgba(255,255,255,.98) 0%,rgba(249,253,248,.98) 52%,rgba(238,247,241,.94) 100%) !important;
}
.home-metrics .metric.kpi-card-premium:before,
.home-metrics .metric.kpi-card-premium:after {
  display:none !important;
}
.kpi-card-accent-circle {
  position:absolute;
  top:-28px;
  right:-28px;
  width:92px;
  height:92px;
  border-radius:50%;
  background:rgba(184,216,189,.32);
  border:1px solid rgba(31,106,67,.06);
  pointer-events:none;
}
.kpi-card-icon {
  position:relative;
  z-index:1;
  width:34px;
  height:34px;
  border-radius:13px;
  display:grid;
  place-items:center;
  background:linear-gradient(145deg,#EEF4EF,#FFFFFF);
  border:1px solid rgba(31,106,67,.13);
  box-shadow:0 6px 14px rgba(10,41,26,.055);
  font-size:1rem;
  line-height:1;
  margin-bottom:10px;
}
.kpi-card-title {
  position:relative;
  z-index:1;
  max-width:calc(100% - 38px);
  color:#5D7064 !important;
  font-family:var(--sans) !important;
  font-size:.92rem !important;
  line-height:1.24 !important;
  letter-spacing:.015em !important;
  text-transform:none !important;
  font-weight:850 !important;
}
.kpi-card-value {
  position:relative;
  z-index:1;
  margin:8px 0 4px !important;
  color:#1F6A43 !important;
  font-family:var(--serif);
  font-size:clamp(2.05rem,2.6vw,2.65rem) !important;
  line-height:1.02 !important;
  font-weight:950 !important;
  letter-spacing:0 !important;
}
.kpi-card-subtitle {
  position:relative;
  z-index:1;
  color:#6F7F72;
  font-size:.9rem;
  line-height:1.28;
  font-weight:700;
  margin-top:2px;
}
.business-impact-grid .insight-card.kpi-card-premium {
  min-height:138px !important;
}
.business-impact-grid .insight-card.kpi-card-premium:before {
  display:none !important;
}

.hero.report-panel,
.home-hero-panel {
  min-height:300px !important;
  background:
    radial-gradient(circle at 86% 15%,rgba(217,154,43,.16) 0 68px,transparent 69px),
    linear-gradient(136deg,#FFFFFF 0%,#F8FCF6 58%,#EEF7F1 100%) !important;
}
.botanical {
  min-height:230px !important;
  border-radius:22px !important;
  opacity:.96;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
  padding:18px !important;
  background:
    linear-gradient(rgba(31,92,58,.016) 1px, transparent 1px),
    linear-gradient(90deg, rgba(31,92,58,.014) 1px, transparent 1px),
    #FFFFFF !important;
  background-size:24px 24px !important;
}
[data-testid="stPlotlyChart"] {
  border-radius:18px !important;
  border:1px solid var(--polish-line) !important;
  box-shadow:var(--polish-shadow-soft) !important;
  background:#FFFFFF !important;
  padding:10px !important;
}
.holdout-chart-frame {
  width:100%;
  display:flex;
  align-items:center;
  justify-content:center;
  min-height:420px;
  max-height:500px;
  padding:10px;
  border-radius:18px;
  border:1px solid var(--polish-line);
  background:#FFFFFF;
  box-shadow:var(--polish-shadow-soft);
  box-sizing:border-box;
  overflow:hidden;
}
.holdout-img {
  display:block;
  width:100%;
  height:100%;
  max-height:480px;
  object-fit:contain;
}
.chart-title {
  padding:0 2px 8px !important;
  border-bottom:1px solid rgba(31,92,58,.10);
}

.table-panel {
  padding:18px !important;
  background:linear-gradient(180deg,#FFFFFF,#FAFDF9) !important;
  margin:10px 0 18px !important;
}
.styled-table-card {
  border-radius:22px !important;
  border:1px solid var(--polish-line) !important;
  box-shadow:var(--polish-shadow) !important;
}
.dashboard-table-card {
  position:relative;
  overflow:hidden;
  border-radius:22px !important;
  border:1px solid rgba(31,106,67,.14) !important;
  background:linear-gradient(180deg,#FFFFFF,#F9FCF8) !important;
  box-shadow:0 12px 28px rgba(10,41,26,.075) !important;
}
.dashboard-table-card:before {
  content:"";
  position:absolute;
  top:0;
  left:18px;
  width:48px;
  height:4px;
  border-radius:0 0 999px 999px;
  background:linear-gradient(90deg,#1F6A43,#8FBF98);
}
.dashboard-table-header {
  display:block;
  padding:2px 0 8px;
}
.table-section-header {
  display:flex;
  align-items:center;
  gap:10px;
  margin:0 0 12px;
  font-family:var(--serif);
  font-size:1.34rem;
  line-height:1.18;
  font-weight:900;
  color:var(--polish-ink);
}
.table-section-header .icon-badge {
  width:34px;
  height:34px;
  border-radius:13px;
  font-size:1rem;
}
.table-panel-title {
  font-size:1.22rem !important;
  line-height:1.18 !important;
}
.styled-table-title {
  display:flex;
  align-items:center;
  gap:8px;
  color:var(--polish-ink) !important;
}
.dashboard-table-title {
  font-family:var(--serif);
  font-weight:950 !important;
}
.table-panel-subtitle {
  color:var(--polish-muted) !important;
  max-width:920px;
}
.styled-table-subtitle {
  font-size:.92rem !important;
  font-weight:750 !important;
}
.dashboard-table-subtitle {
  color:#64776A !important;
}
.table-wrap {
  overflow-x:auto !important;
  padding:4px 1px 6px !important;
}
.report-table,
.dashboard-table {
  border-spacing:0 9px !important;
  min-width:720px;
}
.styled-table {
  width:100%;
  border-collapse:separate !important;
}
.dashboard-table thead {
  border-radius:16px !important;
}
.report-table th,
.dashboard-table th {
  background:linear-gradient(180deg,#EAF4EC,#DDEFE2) !important;
  color:#173D2A !important;
  font-size:.76rem !important;
  line-height:1.18 !important;
  padding:11px 13px !important;
  white-space:nowrap;
  border-top:1px solid rgba(31,106,67,.12) !important;
  border-bottom:1px solid rgba(31,106,67,.10) !important;
}
.report-table td,
.dashboard-table td {
  padding:12px 13px !important;
  color:#173D2A !important;
  line-height:1.38 !important;
  border-color:#DDEBDD !important;
  vertical-align:top !important;
}
.report-table td:first-child,
.dashboard-table td:first-child {
  min-width:132px;
}
.report-table td.num,
.dashboard-table td.num {
  font-weight:950 !important;
  color:#1F6A43 !important;
}
.dashboard-highlight td {
  box-shadow:inset 4px 0 0 var(--polish-green-2) !important;
  background:linear-gradient(90deg,#EEF9EE,#FFFFFF 46%) !important;
}
.dashboard-warning td {
  box-shadow:inset 4px 0 0 var(--polish-gold) !important;
  background:linear-gradient(90deg,#FFF8E7,#FFFFFF 46%) !important;
}
.badge-pill {
  min-height:25px;
  padding:5px 10px !important;
  border-radius:999px !important;
}
.dashboard-badge {
  display:inline-flex;
  align-items:center;
  justify-content:center;
  gap:5px;
  max-width:100%;
}
.metric-chip {
  display:inline-flex;
  align-items:center;
  justify-content:flex-end;
  min-width:64px;
  padding:4px 9px;
  border-radius:999px;
  border:1px solid #DDEBDD;
  background:#F8FCF6;
  color:var(--polish-ink);
  font-weight:950;
  font-variant-numeric:tabular-nums;
}
.risk-pill-low,
.feature-pill-talk {
  background:#EAF4EC !important;
  color:#17613F !important;
  border-color:#CFE4D2 !important;
}
.risk-pill-medium {
  background:#FFF7E5 !important;
  color:#8A5E11 !important;
  border-color:#F2D6A2 !important;
}
.risk-pill-high {
  background:#FDECEC !important;
  color:#A73627 !important;
  border-color:#E7B5AA !important;
}
.feature-pill-walk,
.model-pill-final {
  background:#E7F1EB !important;
  color:#0B5D42 !important;
  border-color:#BFDCC8 !important;
}
.feature-pill-context {
  background:#EEF3EF !important;
  color:#52655B !important;
  border-color:#D7E2D8 !important;
}
.model-pill-challenger {
  background:#FFF4D6 !important;
  color:#8A5E11 !important;
  border-color:#E9CB7B !important;
}
.model-pill-benchmark {
  background:#EEF3EF !important;
  color:#4D6255 !important;
  border-color:#D7E2D8 !important;
}
.model-pill-baseline {
  background:#EEF1F3 !important;
  color:#4D5961 !important;
  border-color:#D6DEE2 !important;
}
.rank-chip {
  box-shadow:0 4px 10px rgba(31,106,67,.16);
}
.dashboard-rank-chip {
  width:27px !important;
  height:27px !important;
  border-radius:50% !important;
  background:#1F6A43 !important;
  color:#FFFFFF !important;
}
[data-testid="stDataFrame"],
[data-testid="stTable"] {
  border-radius:18px !important;
  border:1px solid var(--polish-line) !important;
  box-shadow:var(--polish-shadow-soft) !important;
  background:#FFFFFF !important;
}

.note,
.report-caption,
.filter-status,
.filter-count,
.insight-strip {
  border-radius:16px !important;
}
.note-strip-green,
.note-strip-blue,
.note-strip-amber {
  display:flex;
  align-items:flex-start;
  gap:12px;
  padding:13px 15px;
  border-radius:16px;
  border:1px solid var(--polish-line);
  font-weight:800;
  line-height:1.45;
  margin:10px 0;
}
.note-strip-green {
  background:linear-gradient(90deg,#EAF4EC,#F9FCF8);
  color:#173D2A;
  border-left:4px solid var(--polish-green-2);
}
.note-strip-blue {
  background:linear-gradient(90deg,#EAF3FA,#F9FCFD);
  color:#173D2A;
  border-left:4px solid #3F6FB5;
}
.note-strip-amber {
  background:linear-gradient(90deg,#FFF7E5,#FFFDF6);
  color:#5E4211;
  border-left:4px solid var(--polish-gold);
}
.note {
  background:linear-gradient(180deg,#F9FCF8,#F1F8F0) !important;
  border:1px solid #DDEBDD !important;
  border-left:4px solid var(--polish-green-2) !important;
}
.warning-note {
  background:linear-gradient(180deg,#FFFDF5,#FFF7E5) !important;
  border-left-color:var(--polish-gold) !important;
}
.danger-note {
  background:linear-gradient(180deg,#FFF7F5,#FDECEC) !important;
  border-left-color:var(--polish-red) !important;
}

.stButton>button,
.stDownloadButton>button {
  border-radius:13px !important;
  background:linear-gradient(135deg,#1F5C3A,#2E7D32) !important;
  box-shadow:0 10px 20px rgba(31,92,58,.18) !important;
}
.stButton>button:hover,
.stDownloadButton>button:hover {
  transform:translateY(-1px);
  box-shadow:0 14px 26px rgba(31,92,58,.23) !important;
}
.stSelectbox div[data-baseweb="select"]>div,
.stTextInput input,
.stNumberInput input,
.stTextArea textarea {
  min-height:43px !important;
  border-radius:13px !important;
  border:1px solid var(--polish-line) !important;
  box-shadow:0 5px 14px rgba(10,41,26,.045) !important;
}
button[role="tab"] {
  padding:10px 15px !important;
  border-radius:14px 14px 0 0 !important;
}
button[role="tab"][aria-selected="true"] {
  background:#EAF4EC !important;
  box-shadow:inset 0 -2px 0 var(--polish-green-2);
}
[data-testid="stExpander"] {
  border-radius:16px !important;
  border:1px solid var(--polish-line) !important;
  box-shadow:var(--polish-shadow-soft) !important;
  overflow:hidden;
}

.risk-result-grid,
.company-shap-detail-grid {
  align-items:stretch;
}
.signal-row {
  grid-template-columns:minmax(160px,.9fr) minmax(0,1.1fr) !important;
  gap:14px !important;
}
.signal-value {
  overflow-wrap:anywhere;
}
.talk-walk-bars {
  background:linear-gradient(180deg,#FFFFFF,#F8FCF6) !important;
}
.tw-track {
  height:12px !important;
}
.decision-flow-card,
.insight-card,
.memo-card {
  position:relative;
  border:1px solid var(--polish-line);
  border-radius:20px;
  background:linear-gradient(180deg,#FFFFFF,#F9FCF8);
  box-shadow:var(--polish-shadow);
  padding:18px;
  color:var(--polish-ink);
}
.decision-flow-card h3,
.insight-card h3,
.memo-card h3 {
  margin:0 0 8px;
  font-family:var(--serif);
  color:var(--polish-ink) !important;
  font-size:1.35rem;
  line-height:1.18;
  font-weight:900;
}
.memo-card .signal-row {
  background:rgba(255,255,255,.68);
  border-radius:14px;
  padding:10px 12px;
  margin-top:6px;
}
.decision-flow-card {
  min-height:132px;
}
.decision-flow-card:before,
.insight-card:before,
.memo-card:before {
  content:"";
  position:absolute;
  left:18px;
  top:0;
  width:42px;
  height:4px;
  border-radius:0 0 999px 999px;
  background:var(--polish-green-2);
}
.decision-flow-grid,
.business-impact-grid,
.memo-grid {
  display:grid;
  gap:14px;
}
.decision-flow-grid {
  grid-template-columns:repeat(3,minmax(0,1fr));
}
.business-impact-grid {
  grid-template-columns:repeat(4,minmax(0,1fr));
}
.memo-grid {
  grid-template-columns:repeat(3,minmax(0,1fr));
}
.credit-summary-grid {
  display:grid;
  grid-template-columns:repeat(3,minmax(0,1fr));
  gap:12px;
  margin:14px 0;
}
.insight-mini-card,
.memo-summary-card {
  border:1px solid rgba(31,106,67,.13);
  border-radius:18px;
  background:linear-gradient(180deg,#FFFFFF,#F8FCF6);
  box-shadow:0 8px 18px rgba(10,41,26,.055);
  padding:13px 14px;
  min-height:88px;
}
.insight-mini-card .memo-label,
.memo-summary-card .memo-label {
  margin-top:0;
  font-family:var(--sans);
  color:#5F7165;
  font-size:.76rem;
  letter-spacing:.06em;
  text-transform:uppercase;
  font-weight:950;
}
.insight-mini-card .memo-value,
.memo-summary-card .memo-value {
  margin-top:7px;
  font-family:var(--serif);
  font-size:1.16rem;
  line-height:1.14;
  color:#173D2A;
  font-weight:950;
  overflow-wrap:anywhere;
}
.flow-step-number {
  display:inline-grid;
  place-items:center;
  width:30px;
  height:30px;
  border-radius:12px;
  background:#EAF4EC;
  border:1px solid #CFE4D2;
  color:#1F6A43;
  font-weight:950;
}
.flow-title,
.insight-title,
.memo-label {
  margin-top:10px;
  font-family:var(--serif);
  color:var(--polish-ink);
  font-size:1.13rem;
  line-height:1.16;
  font-weight:900;
}
.flow-body,
.insight-body,
.memo-value {
  margin-top:7px;
  color:#34453A;
  font-size:.96rem;
  line-height:1.45;
  font-weight:700;
}
.insight-value {
  margin-top:8px;
  font-family:var(--serif);
  font-size:1.8rem;
  line-height:1;
  font-weight:950;
  color:var(--polish-green-2);
}
.risk-action-grid {
  display:grid;
  grid-template-columns:repeat(3,minmax(0,1fr));
  gap:15px;
}
.risk-action-card {
  border-radius:20px;
  border:1px solid var(--polish-line);
  box-shadow:var(--polish-shadow);
  padding:18px;
  background:#FFFFFF;
}
.risk-action-card.low {background:linear-gradient(180deg,#FFFFFF,#F0F8F1);border-color:#CFE4D2;}
.risk-action-card.medium {background:linear-gradient(180deg,#FFFFFF,#FFF7E5);border-color:#F2D6A2;}
.risk-action-card.high {background:linear-gradient(180deg,#FFFFFF,#FDECEC);border-color:#E7B5AA;}
.risk-action-title {
  font-family:var(--serif);
  font-size:1.28rem;
  line-height:1.18;
  font-weight:950;
  color:var(--polish-ink);
}
.risk-action-section {
  margin-top:12px;
  padding-top:10px;
  border-top:1px solid rgba(31,92,58,.12);
}
.risk-action-label {
  font-size:.76rem;
  text-transform:uppercase;
  letter-spacing:.07em;
  color:var(--polish-muted);
  font-weight:950;
}
.risk-action-text {
  margin-top:4px;
  color:#34453A;
  line-height:1.42;
  font-size:.94rem;
  font-weight:700;
}
.business-banner {
  display:grid;
  grid-template-columns:auto minmax(0,1fr);
  gap:14px;
  align-items:center;
  padding:18px 20px;
  border-radius:22px;
  background:linear-gradient(135deg,#173D2A,#1F6A43);
  color:#FFFFFF;
  box-shadow:0 18px 38px rgba(10,41,26,.16);
  margin:0 0 18px;
}
.business-banner * {color:#FFFFFF !important;}
.business-banner-title {
  font-family:var(--serif);
  font-size:1.42rem;
  line-height:1.15;
  font-weight:950;
}
.business-banner-body {
  margin-top:4px;
  font-size:.98rem;
  line-height:1.42;
  font-weight:750;
  opacity:.92;
}
.shap-review-card {
  border:1px solid var(--polish-line);
  border-radius:24px;
  background:#FFFFFF;
  box-shadow:var(--polish-shadow);
  padding:28px 30px;
}
.shap-review-grid {
  display:grid;
  grid-template-columns:repeat(3,minmax(300px,1fr));
  gap:20px;
  align-items:stretch;
}
.shap-mini-card {
  display:flex;
  flex-direction:column;
  min-height:218px;
  border-radius:20px;
  border:1px solid var(--polish-line);
  background:linear-gradient(180deg,#FFFFFF,#F7FCF6);
  box-shadow:0 10px 24px rgba(10,41,26,.055);
  padding:22px 24px;
}
.shap-mini-card .company-shap-label {
  display:flex;
  gap:10px;
  align-items:center;
  color:#173D2A;
  font-size:.88rem;
  letter-spacing:.055em;
  margin-bottom:14px;
}
.shap-mini-card .company-shap-label .label-icon {
  display:inline-grid;
  place-items:center;
  flex:0 0 30px;
  width:30px;
  height:30px;
  border-radius:10px;
  background:#EAF4EC;
  border:1px solid #CFE4D2;
  color:#1F6A43;
  font-size:.8rem;
  font-weight:950;
  line-height:1;
}
.shap-mini-card .company-shap-text {
  color:#25392F;
  font-size:1.08rem;
  line-height:1.58;
  font-weight:780;
}
.shap-mini-card .zh {
  margin-top:10px;
  color:#5F7165;
  font-size:1rem;
  line-height:1.7;
  font-weight:720;
  word-break:keep-all;
  overflow-wrap:normal;
  line-break:strict;
}
.company-shap-image-wrap {
  padding:18px 20px;
  overflow:visible;
}
.company-shap-image-wrap .shap-img {
  height:540px !important;
  max-height:540px !important;
  object-fit:contain !important;
  object-position:center !important;
}

[data-testid="stSidebar"] {
  box-shadow:12px 0 32px rgba(6,28,17,.13);
}
.sidebar-brand,
.sidebar-team,
.sidebar-note-card,
.sidebar-status-card {
  border-radius:18px !important;
}
.sidebar-brand {
  margin-top:10px !important;
}
[data-testid="stSidebar"] div[role="radiogroup"] label {
  transition:background .16s ease,border .16s ease,box-shadow .16s ease;
}

@media (max-width: 1200px) {
  [data-testid="stAppViewBlockContainer"],
  .main .block-container,
  .block-container {
    max-width: min(1060px, calc(100vw - var(--final-sidebar-width) - 28px)) !important;
  }
  .page-title h1 {
    font-size:clamp(32px,4.2vw,46px) !important;
  }
  .hero.report-panel {
    grid-template-columns:1fr !important;
  }
  .botanical {
    min-height:180px !important;
  }
}

@media (max-width: 980px) {
  :root {
    --final-sidebar-width: min(88vw, 360px);
  }
  [data-testid="stAppViewBlockContainer"],
  .main .block-container,
  .block-container {
    max-width:100% !important;
  }
  .page-head {
    grid-template-columns:1fr !important;
  }
  .page-title-line {
    align-items:flex-start;
  }
  .page-num {
    width:58px;
    height:58px;
    border-radius:18px;
    font-size:34px !important;
  }
  .card {
    padding:17px !important;
  }
  .signal-row {
    grid-template-columns:1fr !important;
  }
  .decision-flow-grid,
  .business-impact-grid,
  .memo-grid,
  .credit-summary-grid,
  .risk-action-grid,
  .shap-review-grid {
    grid-template-columns:1fr !important;
  }
  .shap-review-card {
    padding:20px !important;
  }
  .company-shap-image-wrap .shap-img {
    height:480px !important;
    max-height:480px !important;
  }
  .signal-value {
    text-align:left !important;
  }
  .report-table,
  .dashboard-table {
    min-width:680px;
  }
}
</style>
""",
    unsafe_allow_html=True,
)
