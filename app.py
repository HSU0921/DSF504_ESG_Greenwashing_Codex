"""
app.py — Group 2 ESG Dashboard
==============================
AI-Powered ESG Greenwashing Risk Scoring System

Run:  streamlit run app.py
"""

import os, sys, pickle, warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import confusion_matrix, precision_score, recall_score

warnings.filterwarnings("ignore")

ROOT    = os.path.dirname(os.path.abspath(__file__))
OUTPUTS = os.path.join(ROOT, "outputs")
MODELS  = os.path.join(ROOT, "models")
sys.path.insert(0, ROOT)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Group 2 ESG Dashboard",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
:root{
  --deep:#0B3528; --forest:#14532D; --green:#2E7D32;
  --soft:#EAF4EC; --pale:#F7FAF5; --mint:#F1F8F2;
  --yellow:#F4B942; --orange:#D99217; --red:#D64545;
  --text:#374151; --muted:#6B7280; --line:#DCE9DD;
  --white:rgba(255,255,255,.94); --shadow:0 16px 40px rgba(15,61,46,.10);
  --shadow-soft:0 8px 22px rgba(15,61,46,.07);
  --deck-max:90rem;
}
html,body,[data-testid="stAppViewContainer"]{
  background:
    linear-gradient(rgba(15,61,46,.025) 1px, transparent 1px),
    linear-gradient(90deg, rgba(15,61,46,.025) 1px, transparent 1px),
    radial-gradient(circle at 82% 0%, rgba(196,223,202,.50) 0, transparent 28%),
    radial-gradient(circle at 2% 95%, rgba(168,210,181,.36) 0, transparent 24%),
    linear-gradient(135deg,#F7FAF5 0%,#FBFDF9 42%,#EEF7EF 100%);
  background-size:44px 44px,44px 44px,auto,auto,auto;
  color:var(--text); font-family:'Inter','Segoe UI',sans-serif;
}
.app-bg{background:var(--pale);color:var(--text);}
[data-testid="stAppViewBlockContainer"]{
  max-width:min(96vw,var(--deck-max));
  min-height:clamp(42rem,56.25vw,52rem);
  padding:clamp(1.1rem,2vw,2rem) clamp(.75rem,1.6vw,1.5rem) clamp(2rem,3vw,3rem);
}
[data-testid="stHeader"]{background:rgba(247,250,245,.78);backdrop-filter:blur(12px);}
[data-testid="stSidebar"]{
  background:
    radial-gradient(circle at 20% 12%,rgba(255,255,255,.95) 0 38px,transparent 39px),
    radial-gradient(ellipse at 16% 100%,rgba(20,83,45,.18) 0 12%,transparent 13%),
    radial-gradient(ellipse at 62% 100%,rgba(46,125,50,.15) 0 13%,transparent 14%),
    linear-gradient(180deg,#F8FCF7 0%,#EEF7F0 62%,#DDEFE2 100%);
  border-right:1px solid var(--line);
  box-shadow:8px 0 26px rgba(15,61,46,.08);
}
[data-testid="stSidebar"] *{color:var(--deep) !important;}
.sidebar-logo{
  display:flex;align-items:center;gap:12px;padding:16px 10px 26px;
  color:var(--deep);
}
.sidebar-logo .leaf{
  width:48px;height:48px;border-radius:16px;display:grid;place-items:center;
  background:linear-gradient(135deg,#E2F3E5,#fff);
  border:1px solid var(--line);box-shadow:0 8px 22px rgba(15,61,46,.12);
  font-size:1.75rem;
}
.sidebar-logo .title{font-size:1.05rem;font-weight:850;line-height:1.25;}
[data-testid="stSidebar"] div[role="radiogroup"] label{
  border-radius:14px;margin:3px 0;padding:8px 10px;transition:all .18s ease;
}
[data-testid="stSidebar"] div[role="radiogroup"] label:hover{
  background:rgba(46,125,50,.08);
}
[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked){
  background:linear-gradient(135deg,#DDEFE2,#EEF7F0);
  box-shadow:inset 0 0 0 1px rgba(46,125,50,.16);
  font-weight:750;
}
h1{color:var(--deep) !important;font-weight:900;letter-spacing:0;line-height:1.08;}
h2{color:var(--forest) !important;font-weight:800;}
h3{color:var(--forest) !important;font-weight:800;}
p,li,span,div{letter-spacing:0;}
hr{border-color:rgba(20,83,45,.14);margin:1.4rem 0;}
[data-testid="metric-container"]{
  background:linear-gradient(180deg,#FFFFFF,#F7FBF6);
  border:1px solid var(--line);border-radius:18px;padding:15px 18px;
  box-shadow:0 10px 28px rgba(15,61,46,.08);
}
[data-testid="stMetricValue"]{color:var(--deep) !important;font-size:1.85rem !important;font-weight:850;}
[data-testid="stMetricLabel"]{color:var(--muted) !important;font-weight:650;}
[data-testid="stMetricDelta"]{color:var(--green) !important;}
[data-testid="stDataFrame"],[data-testid="stTable"]{
  border:1px solid var(--line);border-radius:18px;overflow:hidden;
  box-shadow:var(--shadow);background:var(--white);
}
[data-testid="stPlotlyChart"]{
  background:
    linear-gradient(rgba(15,61,46,.018) 1px, transparent 1px),
    linear-gradient(90deg, rgba(15,61,46,.018) 1px, transparent 1px),
    var(--white);
  background-size:28px 28px;
  border:1px solid var(--line);border-radius:22px;
  padding:12px;box-shadow:var(--shadow-soft);margin:8px 0 12px;
}
[data-testid="stExpander"]{
  background:var(--white);border:1px solid var(--line);border-radius:18px;
  box-shadow:0 8px 24px rgba(15,61,46,.07);
}
.stButton>button{
  background:linear-gradient(135deg,var(--forest),var(--green));color:#fff !important;
  border:none;border-radius:14px;font-weight:800;font-size:.98rem;
  padding:.62rem 1.35rem;transition:all .18s ease;box-shadow:0 8px 18px rgba(46,125,50,.18);
}
.stButton>button:hover{
  background:linear-gradient(135deg,var(--deep),#3B9A46);transform:translateY(-1px);
  box-shadow:0 12px 26px rgba(46,125,50,.26);
}
.stButton>button:disabled{background:#D6E5D9;color:#7A8B7E !important;box-shadow:none;}
.stTextInput input,.stSelectbox div[data-baseweb="select"]>div{
  background:#fff;border-color:var(--line);border-radius:14px;color:var(--text);
}
.stSlider [data-baseweb="slider"]{color:var(--green);}
button[role="tab"]{border-radius:14px 14px 0 0;color:var(--deep) !important;font-weight:750;}
button[role="tab"][aria-selected="true"]{background:#E5F2E7;color:var(--deep) !important;}
.hero-card,.section-card,.metric-card,.team-card,.sdg-strip,.risk-card,.methodology-card,.memo-card,.case-card,.visual-card,.chart-card{
  background:var(--white);border:1px solid var(--line);border-radius:22px;
  box-shadow:var(--shadow);color:var(--text);
}
.hero-card{overflow:hidden;padding:clamp(1.45rem,2.4vw,2.25rem);min-height:clamp(18rem,31vw,22.5rem);
  background:
    radial-gradient(circle at 92% 16%,rgba(244,185,66,.16) 0 58px,transparent 59px),
    linear-gradient(112deg,#FFFFFF 0%,#F8FCF7 48%,#E9F4EF 100%);
}
.home-hero{display:grid;grid-template-columns:minmax(0,1.32fr) minmax(19rem,.82fr);gap:clamp(1.25rem,2.6vw,2.25rem);align-items:stretch;}
.snapshot-stack{display:grid;grid-template-rows:auto 1fr;gap:1rem;}
.hero-snapshot{margin:0;background:rgba(255,255,255,.86);}
.hero-card h1{font-size:clamp(2rem,3.1vw,2.9rem);margin:0 0 10px;max-width:48rem;}
.hero-card .subtitle{font-size:1.08rem;color:#1F2937;font-weight:650;margin:0 0 6px;}
.hero-card .zh{color:var(--muted);font-size:.95rem;}
.hero-landscape{
  min-height:clamp(8rem,14vw,11rem);border:1px solid #DCE9DD;border-radius:20px;overflow:hidden;
  background:
    radial-gradient(circle at 82% 22%,rgba(244,185,66,.34) 0 24px,transparent 25px),
    linear-gradient(18deg,transparent 0 73%,rgba(15,61,46,.30) 74% 78%,transparent 79%),
    repeating-linear-gradient(90deg,transparent 0 72%,rgba(20,83,45,.35) 73% 74%,transparent 75%),
    linear-gradient(158deg,transparent 36%,rgba(101,152,117,.22) 37% 55%,transparent 56%),
    linear-gradient(205deg,transparent 38%,rgba(56,118,80,.18) 39% 60%,transparent 61%),
    linear-gradient(0deg,rgba(81,142,103,.24) 0 26%,#EEF7F0 27% 100%);
}
.sdg-strip{display:flex;align-items:center;gap:12px;padding:12px 14px;width:fit-content;margin-left:auto;}
.sdg-logo{font-weight:900;color:#0B5790;font-size:.72rem;line-height:1.05;padding:8px 10px;background:#fff;border-radius:14px;border:1px solid #E3EEF2;}
.sdg-card{min-width:58px;height:58px;display:flex;align-items:center;justify-content:center;color:#fff;font-weight:900;font-size:1.28rem;border-radius:12px;box-shadow:0 8px 16px rgba(15,61,46,.12);}
.sdg-more{background:#EEF5EF;color:var(--forest);font-size:.76rem;line-height:1.2;padding:10px 14px;border-radius:12px;font-weight:800;}
.section-card{padding:22px 24px;margin:10px 0;}
.section-card h3,.section-card h2{margin-top:0;}
.section-kicker{font-size:.74rem;text-transform:uppercase;letter-spacing:.08em;color:var(--green);font-weight:900;margin-bottom:6px;}
.section-title-row{display:flex;align-items:center;justify-content:space-between;gap:14px;margin-bottom:10px;}
.section-title-row h3{margin:0;}
.pill-chip,.sdg-chip{
  display:inline-flex;align-items:center;gap:6px;border-radius:999px;padding:7px 11px;
  background:#EEF7F0;border:1px solid #D6EBDD;color:var(--forest);font-size:.78rem;font-weight:850;
}
.sdg-chip{border-radius:12px;background:#F5FAF4;}
.chip-row{display:flex;gap:8px;flex-wrap:wrap;align-items:center;}
.metric-card{padding:16px;text-align:center;border-radius:18px;min-height:96px;}
.metric-card .label{color:var(--muted);font-size:.82rem;font-weight:700;}
.metric-card .value{color:var(--deep);font-size:1.72rem;font-weight:900;margin-top:6px;}
.metric-card .delta{color:var(--green);font-size:.78rem;font-weight:700;margin-top:4px;}
.team-card{padding:14px 16px;display:flex;align-items:center;gap:14px;min-height:94px;}
.team-card .avatar{width:60px;height:60px;border-radius:50%;background:linear-gradient(135deg,#0F3D2E,#3E8B55);
  color:#fff;display:grid;place-items:center;font-weight:900;font-size:1.25rem;box-shadow:0 10px 22px rgba(15,61,46,.20);}
.team-card .sid{color:#4B5563;font-size:.78rem;font-weight:700;}
.team-card .cname{color:var(--deep);font-weight:900;font-size:1.02rem;}
.team-card .ename{color:#1F2937;font-size:.85rem;}
.risk-card{padding:16px;text-align:center;border-radius:18px;}
.risk-card .value{font-size:1.8rem;font-weight:900;color:var(--deep);}
.info-box,.insight-box,.methodology-card{
  background:linear-gradient(180deg,#F5FBF4,#EEF7F0);border:1px solid #D8EBDC;
  border-left:5px solid var(--green);padding:14px 18px;border-radius:18px;margin:10px 0;
  color:var(--text);box-shadow:0 8px 22px rgba(15,61,46,.07);
}
.warn-box{background:#FFF8E6;border:1px solid #F7E3AA;border-left:5px solid var(--yellow);
  padding:14px 18px;border-radius:18px;margin:10px 0;color:var(--text);}
.flag-box{background:#FFF1F1;border:1px solid #F3C4C4;border-left:5px solid var(--red);
  padding:14px 18px;border-radius:18px;margin:10px 0;color:var(--text);}
.memo-card,.case-card{padding:22px 24px;margin:10px 0;}
.model-info-card{
  margin-top:28px;padding:18px 18px;border-radius:18px;
  background:rgba(255,255,255,.72);border:1px solid var(--line);
  box-shadow:0 12px 28px rgba(15,61,46,.08);font-size:.78rem;line-height:1.65;
}
.model-info-card strong{color:var(--deep);font-size:.92rem;}
.console-card{
  background:var(--white);border:1px solid var(--line);border-radius:22px;
  box-shadow:var(--shadow);padding:22px 24px;margin:10px 0;
}
.visual-card,.chart-card{
  padding:20px 22px;margin:10px 0;overflow:hidden;
  background:
    radial-gradient(circle at 96% 4%,rgba(46,125,50,.10),transparent 9rem),
    var(--white);
}
.climate-grid{
  min-height:112px;border-radius:18px;border:1px solid #DCE9DD;
  background:
    linear-gradient(rgba(20,83,45,.05) 1px, transparent 1px),
    linear-gradient(90deg,rgba(20,83,45,.05) 1px, transparent 1px),
    linear-gradient(135deg,#FFFFFF,#EEF7F0);
  background-size:18px 18px;
  display:grid;grid-template-columns:repeat(4,1fr);gap:10px;padding:14px;
}
.grid-stat{background:rgba(255,255,255,.78);border:1px solid #DCE9DD;border-radius:14px;padding:10px;}
.grid-stat strong{display:block;color:var(--deep);font-size:1.25rem;}
.finance-ladder{display:grid;gap:10px;margin-top:10px;}
.ladder-step{display:flex;align-items:center;gap:12px;background:#FFFFFF;border:1px solid #DCE9DD;border-radius:16px;padding:10px 12px;box-shadow:var(--shadow-soft);}
.ladder-num{width:30px;height:30px;border-radius:10px;background:#14532D;color:#fff;display:grid;place-items:center;font-weight:900;}
.ladder-step.medium .ladder-num{background:#D99217;}
.ladder-step.high .ladder-num{background:#D64545;}
.esg-illustration{
  min-height:clamp(9rem,16vw,12rem);border-radius:20px;overflow:hidden;border:1px solid #DCE9DD;
  background:
    radial-gradient(circle at 84% 24%,rgba(244,185,66,.34) 0 1.5rem,transparent 1.6rem),
    linear-gradient(158deg,transparent 35%,rgba(20,83,45,.14) 36% 57%,transparent 58%),
    linear-gradient(205deg,transparent 42%,rgba(46,125,50,.12) 43% 64%,transparent 65%),
    linear-gradient(180deg,#F8FCF7,#E8F4EA);
}
.action-card{
  background:var(--white);border:2px solid var(--line);border-radius:22px;
  box-shadow:var(--shadow);padding:22px 24px;height:100%;color:var(--text);
}
.action-card h3{margin-top:0;}
.action-card.low{background:linear-gradient(180deg,#FFFFFF,#EFF8F0);border-color:#B8DCBF;}
.action-card.medium{background:linear-gradient(180deg,#FFFFFF,#FFF7E5);border-color:#F1D48C;}
.action-card.high{background:linear-gradient(180deg,#FFFFFF,#FFF1F1);border-color:#F0B7B7;}
.action-card .eyebrow{color:var(--muted);font-size:.78rem;font-weight:850;text-transform:uppercase;letter-spacing:.04em;margin:12px 0 5px;}
.action-card ul{margin:0;padding-left:18px;line-height:1.7;color:var(--text);}
.action-card p{color:var(--text);}
.action-footer{font-weight:850;margin-top:12px;border-radius:14px;padding:10px 12px;}
.action-footer.low{background:#E8F5EA;color:#14532D;}
.action-footer.medium{background:#FFF1C9;color:#7A4E00;}
.action-footer.high{background:#FFE4E4;color:#9A1F1F;}
.risk-high{display:inline-block;padding:7px 18px;border-radius:999px;background:#D64545;color:#fff;font-weight:900;font-size:1.02rem;box-shadow:0 8px 20px rgba(214,69,69,.2);}
.risk-medium{display:inline-block;padding:7px 18px;border-radius:999px;background:#F4B942;color:#442E00;font-weight:900;font-size:1.02rem;box-shadow:0 8px 20px rgba(244,185,66,.22);}
.risk-low{display:inline-block;padding:7px 18px;border-radius:999px;background:#2E7D32;color:#fff;font-weight:900;font-size:1.02rem;box-shadow:0 8px 20px rgba(46,125,50,.2);}
.page-hero-small{overflow:hidden;background:
  linear-gradient(rgba(15,61,46,.025) 1px, transparent 1px),
  linear-gradient(90deg,rgba(15,61,46,.025) 1px, transparent 1px),
  radial-gradient(circle at 95% 26%,rgba(244,185,66,.25) 0 1.25rem,transparent 1.35rem),
  linear-gradient(140deg,transparent 70%,rgba(46,125,50,.10) 71% 100%),
  linear-gradient(135deg,#FFFFFF,#EFF7F0);
  background-size:28px 28px,28px 28px,auto,auto,auto;border:1px solid var(--line);border-radius:22px;padding:24px 28px;margin-bottom:18px;box-shadow:var(--shadow);}
.page-hero-small h1{margin:0;font-size:2rem;}
.page-hero-small p{color:var(--muted);margin:8px 0 0;}
.chart-note{font-size:.86rem;color:var(--muted);}
.sdg-pill{display:inline-flex;align-items:center;justify-content:center;min-width:56px;height:56px;border-radius:12px;color:#fff;font-weight:900;font-size:1.2rem;margin-right:8px;box-shadow:0 8px 16px rgba(15,61,46,.12);}
@media (max-width:900px){
  [data-testid="stAppViewBlockContainer"]{max-width:100%;min-height:auto;padding:1rem .8rem 2rem;}
  .home-hero{grid-template-columns:1fr;}
  .hero-card h1{font-size:2rem}
  .sdg-strip{margin-left:0;margin-top:12px;flex-wrap:wrap}
}
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
SECTORS = ["Basic Materials","Communication Services","Consumer Cyclical",
           "Consumer Defensive","Energy","Financial Services","Healthcare",
           "Industrials","Real Estate","Technology","Utilities"]
SECTOR_ENC = {s: i for i, s in enumerate(SECTORS)}

CONTROVERSY_ORDER = ["None Controversy Level","Low Controversy Level",
                     "Moderate Controversy Level","Significant Controversy Level",
                     "High Controversy Level","Severe Controversy Level"]
CONTROVERSY_SHORT = {"None Controversy Level":"None","Low Controversy Level":"Low",
                     "Moderate Controversy Level":"Moderate",
                     "Significant Controversy Level":"Significant",
                     "High Controversy Level":"High","Severe Controversy Level":"Severe"}

CVX_DEFAULTS = {
    "E_density":32.07,"S_density":20.52,"G_density":11.16,
    "hedge_score":0.32,"talk_total_density":63.75,"sector":"Energy",
    "pct_numeric":0.0,"vague_to_specific_ratio":322622.27,
    "text_length":15498.0,"esg_topic_breadth":3.0,
    "e_s_g_dispersion":4.20,"sector_rank":0.80,
    "planet_friendly_business":-40.0,"honest_fair_business":-40.0,
}
MEDIANS = {
    "pct_numeric":0.0,"vague_to_specific_ratio":461947.11,
    "text_length":12471.0,"esg_topic_breadth":3.0,
    "e_s_g_dispersion":3.52,"sector_rank":0.526,
    "planet_friendly_business":-30.0,"honest_fair_business":-10.0,
}
FEATURE_ORDER = [
    "E_density","S_density","G_density","hedge_score","pct_numeric",
    "talk_total_density","vague_to_specific_ratio","text_length",
    "esg_topic_breadth","e_s_g_dispersion","sector_rank",
    "planet_friendly_business","honest_fair_business","sector_encoded",
]
FEATURE_DISPLAY = {
    "sector_rank":             "Sector-relative ESG Risk",
    "talk_total_density":      "ESG Disclosure Intensity",
    "e_s_g_dispersion":        "ESG Pillar Imbalance",
    "pct_numeric":             "Quantitative Disclosure Ratio",
    "vague_to_specific_ratio": "Vague vs Specific Language Ratio",
    "E_density":               "Climate Keyword Density",
    "S_density":               "Social Keyword Density",
    "G_density":               "Governance Keyword Density",
    "hedge_score":             "Vague Aspiration Phrases",
    "planet_friendly_business":"External Environmental Signal",
    "honest_fair_business":    "External Governance / Ethics Signal",
    "text_length":             "Report Length (words)",
    "esg_topic_breadth":       "ESG Topic Breadth",
    "sector_encoded":          "Sector Context",
}
DATASET_MEDIANS_TALK = {"talk_total_density": 55.37, "E_density": 14.51,
                        "S_density": 27.54, "G_density": 11.71,
                        "hedge_score": 0.46}

# ── Data loaders ───────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading dataset …")
def load_dataset():
    from src.data_loader import load_merged_dataset
    from src.feature_engineering import build_feature_matrix
    df = load_merged_dataset(verbose=False)
    X, y, meta = build_feature_matrix(df, verbose=False)
    return df, X, y, meta

@st.cache_resource(show_spinner="Loading model …")
def load_model():
    with open(os.path.join(MODELS,"lgbm_best.pkl"),"rb") as f:
        return pickle.load(f)

@st.cache_data
def load_comparison():
    return pd.read_csv(os.path.join(OUTPUTS,"model_comparison.csv"))

@st.cache_data
def load_leakage():
    return pd.read_csv(os.path.join(OUTPUTS,"leakage_check.csv"))

@st.cache_data
def load_feature_importance():
    return pd.read_csv(os.path.join(OUTPUTS,"feature_importance.csv"))

# ── Helpers ────────────────────────────────────────────────────────────────────
def predict_risk(pipeline, fdict):
    row = np.array([[fdict[f] for f in FEATURE_ORDER]], dtype=float)
    return float(pipeline.predict_proba(row)[0,1])

def risk_badge(prob):
    p = prob*100
    if prob >= 0.70:
        return f'<span class="risk-high">🔴 HIGH RISK {p:.1f}%</span>'
    elif prob >= 0.30:
        return f'<span class="risk-medium">🟠 MEDIUM RISK {p:.1f}%</span>'
    else:
        return f'<span class="risk-low">🟢 LOW RISK {p:.1f}%</span>'

def risk_action(prob):
    if prob >= 0.70:
        return ("🔴 Escalate to Enhanced ESG Due Diligence",
                "Escalate to enhanced ESG due diligence. Request third-party assurance, "
                "senior ESG / credit committee review, and controversy history before "
                "sustainable lending approval. 高風險：升級盡職調查，並由主管或 ESG / "
                "授信委員會審查。", "flag-box")
    elif prob >= 0.30:
        return ("🟠 Request Supporting Evidence",
                "Request supplementary ESG documentation (quantitative targets vs. "
                "actuals) and conduct peer comparison. Monitor semi-annually. "
                "中度風險：要求補件，同業比較，半年監控一次。", "warn-box")
    else:
        return ("🟢 Standard ESG Review",
                "Proceed with standard ESG review. Annual monitoring sufficient. "
                "May be considered for sustainable finance if loan purpose also meets "
                "green taxonomy requirements. 低風險：正常授信審查流程。", "info-box")

def pdk():
    return dict(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#374151"))

def insight(en, zh):
    st.markdown(f'<div class="insight-box">💡 {en}<br>'
                f'<span style="color:#8b949e;">{zh}</span></div>',
                unsafe_allow_html=True)

def climate_data_grid(stats, title="Climate Data Grid", subtitle="ESG signals for risk screening"):
    cells = "".join(
        f'<div class="grid-stat"><span style="color:#6B7280;font-size:.75rem;font-weight:800;">{label}</span>'
        f'<strong>{value}</strong><span style="color:#6B7280;font-size:.76rem;">{note}</span></div>'
        for label, value, note in stats
    )
    st.markdown(
        f"""
<div class="visual-card">
  <div class="section-kicker">AI ESG Analytics</div>
  <div class="section-title-row">
    <h3>{title}</h3>
    <span class="pill-chip">sustainable finance</span>
  </div>
  <p style="color:#6B7280;margin-top:-4px;">{subtitle}</p>
  <div class="climate-grid">{cells}</div>
</div>
""",
        unsafe_allow_html=True,
    )

def finance_action_ladder(compact=False):
    min_height = "auto" if compact else "clamp(14rem,22vw,18rem)"
    st.markdown(
        f"""
<div class="visual-card" style="min-height:{min_height};">
  <div class="section-kicker">Decision Support</div>
  <h3>Green Finance Action Ladder</h3>
  <div class="finance-ladder">
    <div class="ladder-step"><div class="ladder-num">1</div><div><strong>Screen</strong><br><span style="color:#6B7280;">Prioritize ESG due diligence queue</span></div></div>
    <div class="ladder-step medium"><div class="ladder-num">2</div><div><strong>Review</strong><br><span style="color:#6B7280;">Request evidence and peer comparison</span></div></div>
    <div class="ladder-step high"><div class="ladder-num">3</div><div><strong>Escalate</strong><br><span style="color:#6B7280;">Enhanced ESG / credit committee review</span></div></div>
  </div>
  <p style="color:#6B7280;font-size:.86rem;margin-top:12px;">模型輔助排序與查核，不做自動授信決策。</p>
</div>
""",
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ══════════════════════════════════════════════════════════════════════════════
def page_home():
    # ── Section 1: Hero ──────────────────────────────────────────────────────
    st.markdown("""
<div class="hero-card home-hero">
  <div>
    <h1>AI-Powered ESG Greenwashing Risk Scoring System</h1>
    <p class="subtitle">Group 2 | DSF504 Final Project</p>
    <p class="zh">永續授信與 ESG 投資的綠漂風險輔助判斷工具</p>
    <p style="max-width:760px;margin-top:18px;color:#374151;font-size:1.02rem;line-height:1.75;">
      Help banks and investors prioritize companies requiring deeper ESG due diligence before sustainable lending or ESG investment decisions.
    </p>
    <div style="display:flex;gap:10px;flex-wrap:wrap;margin-top:18px;">
      <span class="sdg-more" style="padding:8px 12px;">ESG Risk</span>
      <span class="sdg-more" style="padding:8px 12px;">Sustainable Lending</span>
      <span class="sdg-more" style="padding:8px 12px;">Explainable AI</span>
      <span class="sdg-more" style="padding:8px 12px;">SHAP</span>
      <span class="sdg-more" style="padding:8px 12px;">LightGBM</span>
    </div>
  </div>
  <div class="snapshot-stack">
    <div class="section-card hero-snapshot">
      <h3 style="margin-bottom:14px;">Project Snapshot</h3>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;">
        <div class="metric-card"><div class="label">Best Model</div><div class="value" style="font-size:1.35rem;">LightGBM</div></div>
        <div class="metric-card"><div class="label">ROC-AUC</div><div class="value">0.953</div></div>
        <div class="metric-card"><div class="label">PR-AUC</div><div class="value">0.745</div></div>
        <div class="metric-card"><div class="label">Dataset</div><div class="value" style="font-size:1.35rem;">257</div><div class="delta">companies</div></div>
        <div class="metric-card" style="grid-column:1 / -1;"><div class="label">Positive Rate</div><div class="value">7.4%</div><div class="delta">high-risk training label</div></div>
      </div>
    </div>
    <div class="hero-landscape"></div>
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("<div style='height:22px;'></div>", unsafe_allow_html=True)

    home_visual, home_ladder = st.columns([1.25, .85])
    with home_visual:
        climate_data_grid(
            [
                ("Talk", "ESG text", "keyword density"),
                ("Walk", "external risk", "sector-relative"),
                ("Model", "LightGBM", "final scorer"),
                ("Output", "risk score", "review priority"),
            ],
            "AI Risk Signal Map",
            "A finance-oriented view of ESG disclosure, external risk signals, and model output.",
        )
    with home_ladder:
        finance_action_ladder(compact=True)

    # ── Section 2: Key Results ───────────────────────────────────────────────
    st.markdown("""
<div class="section-card">
  <h3>Key Results</h3>
  <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:18px;">
    <div class="metric-card"><div class="label">ROC-AUC</div><div class="value">0.953</div><div class="delta">overall ranking</div></div>
    <div class="metric-card"><div class="label">PR-AUC</div><div class="value">0.745</div><div class="delta">rare class detection</div></div>
    <div class="metric-card"><div class="label">F1-Score</div><div class="value">0.658</div><div class="delta">precision / recall balance</div></div>
    <div class="metric-card"><div class="label">Positive Rate</div><div class="value">7.4%</div><div class="delta">19 / 257 companies</div></div>
  </div>
  <div class="methodology-card" style="margin-top:18px;">
    Because high-risk companies are rare, PR-AUC and F1 are reported alongside ROC-AUC.<br>
    <span style="color:#6B7280;">高風險公司比例低，因此不能只看 accuracy。</span>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── Section 3: Business Problem + Definition ─────────────────────────────
    col_problem, col_definition = st.columns(2)
    with col_problem:
        st.markdown("""
<div class="section-card" style="min-height:260px;">
  <h3>Business Problem</h3>
  <p>ESG reports may sound strong, but external ESG risk indicators may tell a different story.</p>
  <p>This dashboard helps banks and investors prioritize companies requiring deeper ESG due diligence before sustainable lending or ESG investment decisions.</p>
  <div class="info-box" style="margin-top:18px;">
    輔助永續授信與 ESG 投資前的風險篩選。
  </div>
</div>
""", unsafe_allow_html=True)
    with col_definition:
        st.markdown("""
<div class="section-card" style="min-height:260px;">
  <h3>Greenwashing Definition</h3>
  <div style="font-size:1.35rem;font-weight:900;color:#0F3D2E;margin:12px 0 16px;">High Talk + Weak Walk = Greenwashing Risk</div>
  <p><strong>Talk</strong> = ESG disclosure intensity</p>
  <p><strong>Walk</strong> = external ESG risk / controversy indicators</p>
  <p><strong>Label</strong> = sector-relative top 1/3 Talk + bottom 1/3 Walk</p>
  <div class="info-box" style="margin-top:18px;">
    同產業內：說很多，但外部表現偏弱。
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div class="section-card" style="padding:16px 20px;">
  <strong>Dataset Summary:</strong> Dataset: 257 companies across 11 sectors, with 7.4% labeled as high-risk under the project training definition.<br>
  <span style="color:#6B7280;">資料集包含 257 家公司，高風險標籤比例為 7.4%。</span>
</div>
""", unsafe_allow_html=True)

    # ── Section 4: Workflow ──────────────────────────────────────────────────
    workflow = [
        ("1", "Data Explorer", "Check dataset and Talk-Walk gap"),
        ("2", "Model Performance", "Compare models and validation results"),
        ("3", "Prediction Demo", "Generate company risk score"),
        ("4", "Explainability", "Review SHAP explanations"),
        ("5", "Business Recommendation", "Translate risk into ESG credit actions"),
        ("6", "AI Audit Trail", "Show responsible AI usage"),
    ]
    st.markdown("<h3 style='margin-top:20px;'>Dashboard Workflow</h3>", unsafe_allow_html=True)
    cols = st.columns(6)
    for col, (num, title, body) in zip(cols, workflow):
        col.markdown(
            f"""
<div class="section-card" style="padding:16px 14px;text-align:left;min-height:145px;">
  <div style="width:34px;height:34px;border-radius:50%;background:#14532D;color:#fff;display:grid;place-items:center;font-weight:900;margin-bottom:10px;">{num}</div>
  <strong style="color:#0F3D2E;">{title}</strong>
  <p style="color:#6B7280;font-size:.82rem;line-height:1.45;margin-top:8px;">{body}</p>
</div>
""",
            unsafe_allow_html=True,
        )
    st.markdown(
        "<p style='color:#6B7280;margin-top:4px;'>依序看資料、模型、預測、解釋、建議與 AI 使用紀錄。</p>",
        unsafe_allow_html=True,
    )

    # ── Supporting sustainability context: small chips only ──────────────────
    st.markdown("""
<div class="section-card" style="padding:14px 18px;display:flex;align-items:center;gap:10px;flex-wrap:wrap;">
  <strong>Supporting sustainability context:</strong>
  <span class="sdg-more">SDG 3</span><span class="sdg-more">SDG 7</span><span class="sdg-more">SDG 8</span><span class="sdg-more">SDG 12</span><span class="sdg-more">SDG 13</span><span class="sdg-more">SDG 16</span>
  <span style="color:#6B7280;">SDGs 只作為永續議題輔助背景，模型主軸是綠漂風險評分。</span>
</div>
""", unsafe_allow_html=True)

    # ── Section 5: Team Members ──────────────────────────────────────────────
    st.markdown("<h3 style='margin-top:20px;'>Team Members — 第二組 Group 2</h3>", unsafe_allow_html=True)
    members = [
        ("SC", "M14B020807", "鄭仰甫", "Stanley Cheng", "Team Leader"),
        ("CW", "M14B020803", "吳立詮", "Chester Wu", "Team Member"),
        ("SH", "M14B020809", "許椀筑", "Sophia Hsu", "Team Member"),
        ("AH", "M14B020812", "許乃云", "Alice Hsu", "Team Member"),
    ]
    for col, (initials, sid, cname, ename, role) in zip(st.columns(4), members):
        col.markdown(
            f"""
<div class="team-card" style="min-height:84px;padding:12px 14px;">
  <div class="avatar" style="width:50px;height:50px;font-size:1.05rem;">{initials}</div>
  <div>
    <div class="sid">{sid}</div>
    <div class="cname">{cname}</div>
    <div class="ename">{ename}</div>
    <div style="color:#2E7D32;font-weight:800;font-size:.78rem;margin-top:2px;">{role}</div>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — DATA EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
def page_data(df, X, y, meta):
    def compact_insight(en, zh):
        st.markdown(
            f"""
<div class="insight-box" style="margin-top:10px;">
  <strong>Key Insight</strong><br>
  {en}<br><span style="color:#6B7280;">{zh}</span>
</div>
""",
            unsafe_allow_html=True,
        )

    st.markdown(
        "<div class='page-hero-small'>"
        "<h1>Data Explorer — Dataset Summary & Talk-Walk Gap</h1>"
        "<p>Explore the dataset, sector distribution, class imbalance, and "
        "Talk-Walk gap behind the greenwashing label.<br>"
        "資料探索：檢查產業分布、類別不平衡與 Talk-Walk 落差。</p>"
        "</div>",
        unsafe_allow_html=True,
    )
    climate_data_grid(
        [
            ("Companies", "257", "S&P 500 sample"),
            ("Sectors", "11", "peer-relative label"),
            ("High Risk", "19", "training label"),
            ("Positive Rate", "7.4%", "imbalanced class"),
        ],
        "Dataset Signal Overview",
        "Use this page to validate data composition before interpreting model results.",
    )

    # ── Sector filter ─────────────────────────────────────────────────────────
    sector_opts = ["All sectors"] + sorted(df["Sector"].dropna().unique().tolist())
    with st.container(border=True):
        st.markdown("""
### Interactive Sector Filter
Select a sector to compare Talk-Walk patterns, ESG risk distribution, and controversy levels.  
<span style="color:#6B7280;">選擇產業後，下方圖表會即時更新。</span>
""", unsafe_allow_html=True)
        sel_sector  = st.selectbox("Filter by Sector", sector_opts, key="data_sector")

    tickers_all = pd.Series(meta["ticker"]).astype(str).reset_index(drop=True)
    sectors_all = pd.Series(meta["sector"]).astype(str).reset_index(drop=True)
    if sel_sector != "All sectors":
        mask = df["Sector"] == sel_sector
        mask_np = mask.to_numpy()
        df_f = df.loc[mask].reset_index(drop=True)
        X_f  = X.loc[mask].reset_index(drop=True)
        y_f  = y.loc[mask].reset_index(drop=True)
        tickers_f = tickers_all[mask_np].reset_index(drop=True)
        sectors_f = sectors_all[mask_np].reset_index(drop=True)
    else:
        df_f = df.reset_index(drop=True)
        X_f  = X.reset_index(drop=True)
        y_f  = y.reset_index(drop=True)
        tickers_f = tickers_all
        sectors_f = sectors_all

    companies = len(df_f)
    greenwashers = int(y_f.sum())
    sector_count = 1 if sel_sector != "All sectors" and companies else int(df_f["Sector"].nunique())
    positive_rate = (greenwashers / companies * 100) if companies else 0
    avg_report_length = int(X_f["text_length"].mean()) if companies else 0

    # ── KPI cards: filtered dataset summary ──────────────────────────────────
    with st.container(border=True):
        st.markdown("""
### Filtered Dataset KPI
<span style="color:#6B7280;">KPI 會依目前選擇的產業更新。</span>
""", unsafe_allow_html=True)
        k1,k2,k3,k4,k5 = st.columns(5)
        k1.metric("Companies",        f"{companies}")
        k2.metric("Greenwashers",     f"{greenwashers}")
        k3.metric("Sectors",          f"{sector_count}")
        k4.metric("Positive Rate",    f"{positive_rate:.1f}%")
        k5.metric("Avg Report Length",f"{avg_report_length:,} words")

        if sel_sector == "All sectors":
            imbalance_text = (
                "Only 19 out of 257 companies are labeled as high greenwashing risk. "
                "Because the positive class is rare, PR-AUC and F1 are reported alongside ROC-AUC."
            )
        else:
            imbalance_text = (
                f"In the selected sector, {greenwashers} out of {companies} companies are "
                "labeled high risk. Because the positive class is rare, PR-AUC and F1 "
                "are reported alongside ROC-AUC."
            )
        st.markdown(f"""
<div class="info-box" style="margin-top:12px;">
<strong>Class Imbalance Note</strong>
<p style="margin:8px 0 4px;">{imbalance_text}</p>
<span style="color:#8b949e;font-size:.9rem;">
高風險樣本少，因此不能只看 accuracy。
</span>
</div>
""", unsafe_allow_html=True)

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

    # ── Talk vs Walk scatter — CORE STORY (moved to top) ──────────────────────
    plot_df = pd.DataFrame({
        "ESG Disclosure Intensity":  X_f["talk_total_density"].values,
        "Sector-relative ESG Risk":  X_f["sector_rank"].values,
        "label": y_f.map({0:"No Greenwashing",1:"⚠️ Greenwashing"}).values,
        "ticker": tickers_f, "sector": sectors_f,
    })
    fig_scatter = px.scatter(
        plot_df, x="ESG Disclosure Intensity", y="Sector-relative ESG Risk",
        color="label",
        color_discrete_map={"No Greenwashing":"#3fb950","⚠️ Greenwashing":"#da3633"},
        hover_data=["ticker","sector"], opacity=0.8,
        labels={"ESG Disclosure Intensity":"Talk Score (keyword density / 1,000 words)",
                "Sector-relative ESG Risk":"Walk Score (sector sub-score percentile, high = risky)"},
    )
    talk_cut = float(plot_df["ESG Disclosure Intensity"].quantile(0.67)) if companies else 0
    talk_max = float(plot_df["ESG Disclosure Intensity"].max()) if companies else 1
    fig_scatter.add_shape(type="rect",x0=talk_cut,x1=max(talk_max * 1.05, talk_cut + 1),y0=0.67,y1=1.0,
        fillcolor="rgba(218,54,51,.08)",
        line=dict(color="#da3633",dash="dash",width=1.5))
    fig_scatter.add_annotation(xref="paper",yref="paper",x=0.98,y=0.96,
        text="Greenwashing Zone<br>high Talk + weak Walk",
        font=dict(color="#D64545",size=11),showarrow=False,align="right",
        bgcolor="rgba(255,255,255,.86)",bordercolor="#D64545",borderwidth=1)
    fig_scatter.update_layout(**pdk(), legend_title_text="Label")

    with st.container(border=True):
        st.markdown("""
### Talk vs Walk Gap — Core Greenwashing Signal
<div style="font-size:1.15rem;font-weight:900;color:#0F3D2E;margin:8px 0;">
High Talk + Weak Walk = Greenwashing Risk
</div>
This chart visualizes the project’s label logic: companies with high ESG disclosure intensity and weak sector-relative external ESG performance are flagged as high risk.  
<span style="color:#6B7280;">這張圖就是本專案標籤邏輯的視覺化。</span>
""", unsafe_allow_html=True)
        st.plotly_chart(fig_scatter, use_container_width=True)
        compact_insight(
            "Greenwashing cases concentrate where ESG disclosure is high but sector-relative ESG risk is also high.",
            "高風險公司集中在「說多但外部風險也高」的區域。",
        )

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

    # ── Row 1: ESG Risk Score histogram + Controversy Level ───────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        fig_hist = px.histogram(df_f, x="Total ESG Risk score", nbins=30,
            color_discrete_sequence=["#3fb950"],
            labels={"Total ESG Risk score":"Total ESG Risk Score"})
        fig_hist.update_layout(**pdk(), bargap=0.05,
            xaxis_title="ESG Risk Score (higher = riskier)", yaxis_title="Companies")
        with st.container(border=True):
            st.markdown("""
### ESG Risk Score Distribution
Higher ESG risk scores indicate weaker external ESG performance, not stronger ESG quality.  
<span style="color:#6B7280;">分數越高代表外部 ESG 風險越高，不是 ESG 表現越好。</span>
""", unsafe_allow_html=True)
            st.plotly_chart(fig_hist, use_container_width=True)
            compact_insight("Higher-risk sectors such as Energy and Utilities tend to cluster above 25.",
                            "能源、公用事業等高風險產業分數常集中在 25 以上。")

    with col_b:
        ctrl = (
            df_f["Controversy Level"]
            .value_counts()
            .reindex(CONTROVERSY_ORDER, fill_value=0)
            .reset_index()
        )
        ctrl.columns = ["level","count"]
        ctrl["short"] = ctrl["level"].map(CONTROVERSY_SHORT)
        fig_ctrl = px.bar(ctrl, x="count", y="short", orientation="h",
            color="count", color_continuous_scale="RdYlGn_r",
            labels={"count":"Companies","short":"Controversy Level"},
            category_orders={"short":[CONTROVERSY_SHORT[v] for v in CONTROVERSY_ORDER]})
        fig_ctrl.update_layout(**pdk(), coloraxis_showscale=False,
            yaxis=dict(categoryorder="array",
                       categoryarray=[CONTROVERSY_SHORT[v] for v in CONTROVERSY_ORDER]))
        with st.container(border=True):
            st.markdown("""
### Controversy Level Distribution
Controversy levels provide an external signal for reputational and ESG incident risk.  
<span style="color:#6B7280;">爭議程度可作為外部聲譽風險訊號。</span>
""", unsafe_allow_html=True)
            st.plotly_chart(fig_ctrl, use_container_width=True)
            compact_insight("Most companies report low or no controversy, with a tail of significant cases.",
                            "大多數公司爭議程度低，少數有顯著爭議事件。")

    # ── Row 2: Sector distribution + Sub-score comparison ─────────────────────
    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
    col_c, col_d = st.columns(2)

    with col_c:
        sec_counts = df_f["Sector"].value_counts().reset_index()
        sec_counts.columns = ["Sector","Count"]
        sec_counts = sec_counts.sort_values("Count", ascending=False)
        fig_sec = px.bar(sec_counts, x="Count", y="Sector", orientation="h",
            color="Count", color_continuous_scale="Greens", text="Count")
        fig_sec.update_layout(**pdk(), coloraxis_showscale=False,
            yaxis=dict(categoryorder="array",
                       categoryarray=sec_counts["Sector"].tolist(),
                       autorange="reversed"),
            xaxis_title="Companies")
        fig_sec.update_traces(textposition="outside", cliponaxis=False)
        with st.container(border=True):
            st.markdown("""
### Company Count by Sector
Sector composition matters because the greenwashing label is defined within sector peers.  
<span style="color:#6B7280;">產業結構會影響同產業排名與標籤判斷。</span>
""", unsafe_allow_html=True)
            st.plotly_chart(fig_sec, use_container_width=True)
            if len(sec_counts) > 1:
                top3 = sec_counts.head(3)["Sector"].tolist()
                compact_insight(
                    f"{', '.join(top3)} are the three largest sectors in the current view.",
                    f"{'、'.join(top3)} 是目前視圖中公司數最多的三大產業。",
                )
            elif len(sec_counts) == 1:
                sector_name = sec_counts.iloc[0]["Sector"]
                sector_n = int(sec_counts.iloc[0]["Count"])
                compact_insight(
                    f"Selected sector: {sector_name}, with {sector_n} companies in the current view.",
                    f"目前篩選：{sector_name}，共 {sector_n} 家公司。",
                )

    with col_d:
        radar_df = pd.DataFrame({
            "E": df_f["Environment Risk Score"].values,
            "S": df_f["Social Risk Score"].values,
            "G": df_f["Governance Risk Score"].values,
            "label": y_f.values,
        })
        gm  = radar_df[radar_df["label"]==1][["E","S","G"]].mean().fillna(0)
        nm  = radar_df[radar_df["label"]==0][["E","S","G"]].mean().fillna(0)
        cats = ["E Risk","S Risk","G Risk"]
        fig_sub = go.Figure()
        fig_sub.add_trace(go.Bar(name="Greenwashing (label=1)", x=cats, y=gm.values,
            marker_color="#da3633", opacity=0.85))
        fig_sub.add_trace(go.Bar(name="Normal (label=0)", x=cats, y=nm.values,
            marker_color="#3fb950", opacity=0.85))
        fig_sub.update_layout(**pdk(), barmode="group", yaxis_title="Mean Sub-Score",
            legend=dict(orientation="h",y=1.12))
        with st.container(border=True):
            st.markdown("""
### Mean ESG Risk Sub-scores by Greenwashing Label
Higher sub-scores mean higher ESG risk.  
<span style="color:#6B7280;">子分數越高，代表該面向風險越高。</span>
""", unsafe_allow_html=True)
            st.plotly_chart(fig_sub, use_container_width=True)
            compact_insight("Greenwashing companies tend to have higher E and S risk scores despite extensive disclosures.",
                            "高綠漂風險企業在環境與社會風險分數上普遍偏高。")

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)
    with st.expander("Raw Data Preview — Key Columns", expanded=False):
        st.caption("僅顯示與標籤和模型解釋最相關的欄位。")
        base_cols = []
        for candidates in [
            ["ticker"],
            ["Name", "company_name", "Company", "company"],
            ["Sector", "sector"],
            ["year"],
            ["Total ESG Risk score"],
            ["Controversy Level"],
        ]:
            for col in candidates:
                if col in df_f.columns:
                    base_cols.append(col)
                    break
        show = df_f[base_cols].copy()
        show["greenwashing_label"] = y_f.values
        for col in ["talk_total_density", "sector_rank"]:
            if col in X_f.columns:
                show[col] = X_f[col].values
        st.dataframe(show.head(20), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
def page_model(pipeline, X, y):
    st.markdown(
        "<div class='page-hero-small'><h1>Model Performance</h1>"
        "<p>Model validation, threshold interpretation, and feature importance.</p></div>",
        unsafe_allow_html=True)
    s1, s2, s3 = st.columns(3)
    summary_cards = [
        (s1, "Final Model",
         "LightGBM selected for highest ROC-AUC and stable CV."),
        (s2, "Minority-Class Detection",
         "XGBoost achieves best F1 and PR-AUC."),
        (s3, "Business Use",
         "The model helps ESG analysts and credit officers prioritize deeper due diligence."),
    ]
    for col, title, body in summary_cards:
        with col:
            st.markdown(f"""
<div class="section-card" style="min-height:150px;">
  <h3 style="margin-bottom:10px;">{title}</h3>
  <p style="margin:0;color:#374151;line-height:1.65;">{body}</p>
</div>
""", unsafe_allow_html=True)
    st.markdown(
        "<div class='info-box'>協助授信與 ESG 投資人員排序需要優先查核的公司。</div>",
        unsafe_allow_html=True)
    climate_data_grid(
        [
            ("Final", "0.9528", "LightGBM ROC-AUC"),
            ("Baseline", "0.4746", "DummyClassifier"),
            ("Δ AUC", "+0.4782", "vs baseline"),
            ("Rare class", "7.4%", "positive rate"),
        ],
        "Validation Dashboard",
        "Nested CV metrics are used for model selection; full-dataset charts are for demonstration.",
    )
    st.markdown("---")

    comp = load_comparison()
    leak = load_leakage()
    fi_df = load_feature_importance()

    # Add Role column
    role_map = {
        "DummyClassifier":"Naive Baseline",
        "Logistic Regression":"Interpretable Benchmark",
        "Random Forest":"Tree Ensemble Benchmark",
        "XGBoost":"Minority-Class Detection Challenger",
        "LightGBM":"Final Risk Scoring Model",
    }
    comp["Role"] = comp["model"].map(role_map)

    # KPIs
    best = comp.iloc[comp["mean_roc_auc"].argmax()]
    base = comp[comp["model"]=="DummyClassifier"].iloc[0]
    k1,k2,k3,k4 = st.columns(4)
    k1.metric("Best Model",   best["model"])
    k2.metric("Best AUC",     f"{best['mean_roc_auc']:.4f}", f"±{best['std_roc_auc']:.4f}")
    k3.metric("Baseline AUC", f"{base['mean_roc_auc']:.4f}", "DummyClassifier")
    k4.metric("Δ vs Baseline",   f"+{best['delta_auc_vs_baseline']:.4f}", "improvement")
    st.markdown("---")

    # ── Comparison table ───────────────────────────────────────────────────────
    st.markdown("### 📋 Model Comparison (5-fold Nested CV)")
    disp = comp[["model","Role","mean_roc_auc","std_roc_auc",
                 "mean_f1","std_f1","mean_pr_auc","std_pr_auc",
                 "delta_auc_vs_baseline"]].copy()
    disp.columns = ["Model","Role","ROC-AUC","ROC-AUC Std","F1","F1 Std",
                    "PR-AUC","PR-AUC Std","Δ vs Baseline"]
    disp = disp.round(4)
    st.table(disp)
    st.markdown("---")

    # ── Metric charts in tabs ──────────────────────────────────────────────────
    st.markdown("### 📈 Metric Visualizations")
    st.markdown("""
<div class="info-box">
ROC-AUC evaluates overall ranking ability, while PR-AUC and F1 are more informative
for rare high-risk cases.<br>
<span style="color:#8b949e;font-size:.9rem;">
高風險公司比例低，因此 PR-AUC 與 F1 特別重要。
</span>
</div>
""", unsafe_allow_html=True)
    tab1,tab2,tab3,tab4 = st.tabs(
        ["ROC-AUC","PR-AUC","F1-Score","Confusion Matrix"])

    def hbar(col_y, col_err, title, x_label):
        clrs = ["#8b949e" if m=="DummyClassifier"
                else "#3fb950" if m==best["model"] else "#58a6ff"
                for m in comp["model"]]
        fig = go.Figure(go.Bar(
            x=comp[col_y], y=comp["model"], orientation="h",
            error_x=dict(type="data",array=comp[col_err],visible=True,color="#555"),
            marker_color=clrs,
            text=comp[col_y].round(4), textposition="outside",
            textfont=dict(color="#374151"),
        ))
        fig.add_vline(x=0.5,line_dash="dash",line_color="#da3633")
        fig.update_layout(**pdk(), xaxis_range=[0,1.1], xaxis_title=x_label,
            yaxis=dict(autorange="reversed"), showlegend=False)
        return fig

    with tab1:
        st.plotly_chart(hbar("mean_roc_auc","std_roc_auc",
            "ROC-AUC","Mean ROC-AUC"), use_container_width=True)
    with tab2:
        st.plotly_chart(hbar("mean_pr_auc","std_pr_auc",
            "PR-AUC","Mean PR-AUC"), use_container_width=True)
    with tab3:
        st.plotly_chart(hbar("mean_f1","std_f1",
            "F1-Score","Mean F1"), use_container_width=True)

    with tab4:
        st.markdown("#### Confusion Matrix — Full Dataset Predictions")
        st.caption(
            "Note: This confusion matrix is for threshold demonstration only, "
            "based on full-dataset predictions from the final retrained model. "
            "Model selection and performance claims are based on the 5-fold "
            "Nested CV metrics above. 混淆矩陣只用來展示門檻調整，模型好壞以 "
            "Nested CV 結果為準。")
        threshold = st.slider("Decision threshold", 0.05, 0.95, 0.50, 0.05,
                              key="cm_thresh")
        X_arr = X.values.astype(float)
        proba = pipeline.predict_proba(X_arr)[:,1]
        pred  = (proba >= threshold).astype(int)
        tn,fp,fn,tp = confusion_matrix(y, pred).ravel()
        prec = precision_score(y, pred, zero_division=0)
        rec  = recall_score(y, pred, zero_division=0)

        c1,c2,c3,c4,c5,c6 = st.columns(6)
        c1.metric("TP",  tp)
        c2.metric("FP",  fp)
        c3.metric("TN",  tn)
        c4.metric("FN",  fn)
        c5.metric("Precision", f"{prec:.3f}")
        c6.metric("Recall",    f"{rec:.3f}")

        fig_cm = px.imshow([[tn,fp],[fn,tp]],
            text_auto=True, color_continuous_scale="RdYlGn",
            x=["Predicted 0","Predicted 1"],
            y=["Actual 0","Actual 1"],
            labels=dict(color="Count"))
        fig_cm.update_layout(**pdk(), height=320,
            coloraxis_showscale=False)
        st.plotly_chart(fig_cm, use_container_width=True)
        st.markdown("""
<div class="warn-box">
Perfect full-dataset classification is expected because the final model is retrained
on all data. Use nested CV metrics for fair performance evaluation.<br>
<span style="color:#8b949e;">全資料重訓結果只供展示，公平評估請看 Nested CV。</span>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Leakage check ──────────────────────────────────────────────────────────
    st.markdown("### 🔬 Reduced Walk-Proxy Ablation（removed sector_rank and e_s_g_dispersion）")
    st.markdown("""
<div class="warn-box">
<strong>Experiment:</strong> Removed selected Walk / External proxy features and
re-ran LightGBM with the same nested CV design.<br><br>
<strong>Result:</strong> AUC dropped from 0.953 to 0.741, but remained well above
the baseline AUC of 0.475. This suggests ESG report text still carries independent
predictive signal.<br>
<span style="color:#8b949e;font-size:.9rem;">
移除部分外部指標後，文字特徵仍有預測能力。
</span><br><br>
<em>Note: A complete Talk-only test would remove all Walk / External features.</em>
</div>
""", unsafe_allow_html=True)
    full_auc = leak.loc[leak["feature_set"]=="Full (14 features)", "mean_roc_auc"].iloc[0]
    reduced_auc = leak.loc[leak["feature_set"]!="Full (14 features)", "mean_roc_auc"].iloc[0]
    leak_d = pd.DataFrame({
        "Feature Set": ["Full (14 features)", "Reduced Walk-Proxy Ablation"],
        "# Features": [14, 12],
        "ROC-AUC": [full_auc, reduced_auc],
        "Interpretation": [
            "Best overall risk ranking model",
            "Text signal remains above baseline after removing selected Walk / External proxies",
        ],
    })
    leak_d["ROC-AUC"] = leak_d["ROC-AUC"].map(lambda v: f"{v:.4f}")
    st.markdown(f"""
<table style="width:100%;border-collapse:separate;border-spacing:0;overflow:hidden;
border:1px solid #DCE9DD;border-radius:16px;background:#fff;box-shadow:0 10px 28px rgba(15,61,46,.08);">
  <thead>
    <tr style="background:#EAF4EC;color:#0F3D2E;text-align:left;">
      <th style="padding:12px 14px;">Feature Set</th>
      <th style="padding:12px 14px;"># Features</th>
      <th style="padding:12px 14px;">ROC-AUC</th>
      <th style="padding:12px 14px;">Interpretation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="padding:12px 14px;border-top:1px solid #DCE9DD;">{leak_d.iloc[0]["Feature Set"]}</td>
      <td style="padding:12px 14px;border-top:1px solid #DCE9DD;">{leak_d.iloc[0]["# Features"]}</td>
      <td style="padding:12px 14px;border-top:1px solid #DCE9DD;">{leak_d.iloc[0]["ROC-AUC"]}</td>
      <td style="padding:12px 14px;border-top:1px solid #DCE9DD;">{leak_d.iloc[0]["Interpretation"]}</td>
    </tr>
    <tr>
      <td style="padding:12px 14px;border-top:1px solid #DCE9DD;">{leak_d.iloc[1]["Feature Set"]}</td>
      <td style="padding:12px 14px;border-top:1px solid #DCE9DD;">{leak_d.iloc[1]["# Features"]}</td>
      <td style="padding:12px 14px;border-top:1px solid #DCE9DD;">{leak_d.iloc[1]["ROC-AUC"]}</td>
      <td style="padding:12px 14px;border-top:1px solid #DCE9DD;">{leak_d.iloc[1]["Interpretation"]}</td>
    </tr>
  </tbody>
</table>
""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Feature importance ─────────────────────────────────────────────────────
    st.markdown("### 🏆 Feature Importance (mean |SHAP|)")
    fi_show = fi_df[fi_df["mean_abs_shap"] > 0].head(11).copy()
    fi_show["display"] = fi_show["feature"].map(
        lambda f: FEATURE_DISPLAY.get(f, f))
    walk_external_features = {
        "sector_rank",
        "e_s_g_dispersion",
        "planet_friendly_business",
        "honest_fair_business",
    }
    fi_show["group_display"] = np.where(
        fi_show["feature"].eq("sector_encoded"), "Context",
        np.where(fi_show["feature"].isin(walk_external_features), "Walk / External", "Talk")
    )
    clr_map = {"Talk":"#3fb950","Walk / External":"#58a6ff","Context":"#f0883e"}
    colours = [clr_map.get(g,"#8b949e") for g in fi_show["group_display"]]
    fig_fi = go.Figure(go.Bar(
        x=fi_show["mean_abs_shap"], y=fi_show["display"], orientation="h",
        marker_color=colours,
        text=fi_show["mean_abs_shap"].round(3),
        textposition="outside", textfont=dict(color="#374151"),
        name="Feature Importance",
        showlegend=False,
    ))
    for grp,clr in clr_map.items():
        fig_fi.add_trace(go.Bar(x=[None],y=[None],name=grp,marker_color=clr))
    fig_fi.update_layout(**pdk(), yaxis=dict(autorange="reversed"),
        xaxis_title="Mean |SHAP|", showlegend=True,
        legend=dict(orientation="h",y=-0.15), margin=dict(l=10))
    st.plotly_chart(fig_fi, use_container_width=True)
    st.markdown("""
<div class="insight-box">
💡 SHAP results show that both ESG disclosure intensity and sector-relative
external ESG risk drive the model. This is consistent with the project design:
greenwashing risk is defined as a mismatch between strong ESG communication and
weak external ESG signals. The reduced ablation test confirms that text-based
features still carry independent signal.<br>
<span style="color:#8b949e;">
模型同時看「說了什麼」與「外部表現如何」，不是只看單一分數。
</span><br>
Near-zero features are retained for financial logic but not shown in the chart.<br>
<span style="color:#8b949e;">
貢獻接近 0 的特徵因具備金融邏輯仍保留，但不列入圖中。
</span>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — PREDICTION DEMO
# ══════════════════════════════════════════════════════════════════════════════
def page_predict(pipeline):
    st.markdown(
        "<div class='page-hero-small'><h1>ESG Risk Scoring Console</h1>"
        "<p>Select a company, load its ESG disclosure profile, adjust Talk features, "
        "and generate a model-estimated greenwashing risk score.<br>"
        "Final risk score reflects all 14 features combined.</p></div>",
        unsafe_allow_html=True)
    pred_visual, pred_ladder = st.columns([1.15, .85])
    with pred_visual:
        climate_data_grid(
            [
                ("Input", "Talk", "sliders"),
                ("Fixed", "Walk / External", "company profile"),
                ("Model", "14 features", "same order"),
                ("Output", "risk tier", "review action"),
            ],
            "Scoring Console Flow",
            "Load a company profile, simulate disclosure inputs, and translate score into review action.",
        )
    with pred_ladder:
        finance_action_ladder(compact=True)
    st.markdown("---")

    try:
        df_demo, X_demo, y_demo, meta_demo = load_dataset()
    except Exception as exc:
        st.error(f"Unable to load prediction dataset: {exc}")
        return

    if pipeline is None:
        st.error("Unable to load prediction model. Please check models/lgbm_best.pkl.")
        return

    ticker_series = pd.Series(meta_demo["ticker"]).astype(str).str.strip().str.upper()
    ticker_list = sorted(ticker_series[ticker_series != ""].unique())
    ticker_set = set(ticker_list)
    feature_medians = X_demo[FEATURE_ORDER].apply(pd.to_numeric, errors="coerce").median()

    def safe_float(value, fallback=0.0):
        try:
            val = float(value)
            if np.isfinite(val):
                return val
        except (TypeError, ValueError):
            pass
        return float(fallback)

    def clip_value(value, lo, hi):
        return float(np.clip(safe_float(value, lo), lo, hi))

    def company_name_from_row(row, ticker):
        for col in ["Name", "company_name", "Company", "company"]:
            if col in row.index and pd.notna(row[col]) and str(row[col]).strip():
                return str(row[col]).strip()
        return ticker

    ticker_display = {}
    for pos, ticker in ticker_series.items():
        if ticker and ticker not in ticker_display:
            row = df_demo.iloc[pos]
            company = company_name_from_row(row, ticker)
            sector_name = str(meta_demo["sector"][pos])
            ticker_display[ticker] = f"{ticker} — {company} — {sector_name}"

    select_options = ["Manual / Median defaults"] + ticker_list

    def display_ticker_option(option):
        if option == "Manual / Median defaults":
            return option
        return ticker_display.get(option, option)

    def clear_ticker_search():
        st.session_state["ticker_input"] = ""

    def risk_level(prob):
        if prob >= 0.70:
            return "High"
        if prob >= 0.30:
            return "Medium"
        return "Low"

    def prediction_action(prob):
        if prob >= 0.70:
            return (
                "🔴 Escalate to Enhanced ESG Due Diligence",
                "This company is flagged as high greenwashing risk by the model — "
                "further due diligence is recommended. Request third-party assurance, "
                "senior ESG / credit committee review, and controversy history before "
                "sustainable lending approval. 模型標記為高風險，建議進一步查核，不代表直接定罪。",
                "flag-box",
            )
        if prob >= 0.30:
            return (
                "🟠 Request Supporting Evidence",
                "Ask for additional ESG evidence, third-party data, and peer comparison "
                "before approval. 中度風險：要求補充佐證。",
                "warn-box",
            )
        return (
            "🟢 Standard ESG Review",
            "May be considered for sustainable finance if loan purpose also meets green "
            "taxonomy requirements. 低風險：標準 ESG 審查。",
            "info-box",
        )

    if "pending_demo_ticker" in st.session_state:
        pending = str(st.session_state.pop("pending_demo_ticker")).strip().upper()
        st.session_state["ticker_input"] = pending
        st.session_state["ticker_select"] = pending if pending in ticker_set else "Manual / Median defaults"

    st.session_state.setdefault("ticker_input", "")
    st.session_state.setdefault("ticker_select", "Manual / Median defaults")

    st.markdown("""
<div class="console-card">
  <h3>Company Ticker Search</h3>
  <p>Enter or select a company ticker to load its ESG disclosure profile and generate a greenwashing risk score.<br>
  <span style="color:#6B7280;">輸入公司代碼，即時載入資料並產生風險分數。</span></p>
</div>
""", unsafe_allow_html=True)
    search_col, select_col = st.columns([1, 1])
    with search_col:
        ticker_input = st.text_input(
            "Enter ticker",
            placeholder="AAPL, MSFT, TSLA, CVX",
            key="ticker_input",
        )
    with select_col:
        selected_from_list = st.selectbox(
            "Select company from dataset",
            select_options,
            format_func=display_ticker_option,
            on_change=clear_ticker_search,
            key="ticker_select",
        )

    st.markdown("<div class='console-card' style='padding:16px 18px;'><strong>Quick Demo Buttons</strong></div>", unsafe_allow_html=True)
    quick_cols = st.columns(4)
    for qcol, ticker in zip(quick_cols, ["CVX", "AAPL", "MSFT", "TSLA"]):
        available = ticker in ticker_set
        if qcol.button(ticker, disabled=not available, key=f"quick_{ticker}"):
            st.session_state["pending_demo_ticker"] = ticker
            st.rerun()
        if not available:
            qcol.caption("Ticker is not available in the current dataset.")

    typed_ticker = str(ticker_input).strip().upper()
    selected_ticker = typed_ticker if typed_ticker else selected_from_list
    if selected_ticker == "Manual / Median defaults":
        selected_ticker = None

    ticker_valid = selected_ticker in ticker_set if selected_ticker else False
    if selected_ticker and not ticker_valid:
        st.warning(
            "Ticker not found in the current dataset. Please select another company.  \n"
            "目前資料集中找不到此公司代碼。")

    selected_idx = None
    selected_row = None
    selected_company = "Not selected"
    selected_sector = st.session_state.get("sector_select", SECTORS[0])
    selected_actual_label = "Not available"

    profile_features = feature_medians.copy()
    if ticker_valid:
        selected_pos = int(ticker_series[ticker_series == selected_ticker].index[0])
        selected_row = df_demo.iloc[selected_pos]
        selected_company = company_name_from_row(selected_row, selected_ticker)
        selected_sector = str(meta_demo["sector"][selected_pos])
        selected_actual_label = "High-risk label" if int(y_demo.iloc[selected_pos]) == 1 else "Non-high-risk label"
        profile_features = X_demo.iloc[selected_pos][FEATURE_ORDER].copy()

    def feature_value(feature):
        return safe_float(profile_features.get(feature, np.nan), feature_medians.get(feature, 0.0))

    profile_key = selected_ticker if ticker_valid else "MANUAL"
    if st.session_state.get("loaded_profile_key") != profile_key:
        st.session_state["e_dens"] = clip_value(feature_value("E_density"), 0.0, 75.0)
        st.session_state["s_dens"] = clip_value(feature_value("S_density"), 0.0, 55.0)
        st.session_state["g_dens"] = clip_value(feature_value("G_density"), 0.0, 50.0)
        st.session_state["hedge"] = clip_value(feature_value("hedge_score"), 0.0, 3.0)
        st.session_state["ttl"] = clip_value(feature_value("talk_total_density"), 0.0, 90.0)
        sector_default = selected_sector if selected_sector in SECTORS else SECTORS[0]
        st.session_state["sector_select"] = sector_default
        st.session_state["loaded_profile_key"] = profile_key
        st.session_state.demo_loaded = ticker_valid and selected_ticker == "CVX"

    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
    col_inp, col_res = st.columns([2,3])

    with col_inp:
        st.markdown("""
<div class="console-card">
  <h3>Input ESG Disclosure Profile — Talk Features</h3>
  <p>These sliders represent ESG report language intensity and wording style.<br>
  <span style="color:#6B7280;">這些欄位代表 ESG 報告「說了什麼、說多少、是否模糊」。</span></p>
  <p style="color:#6B7280;">Selected company values are loaded as defaults. Users can adjust disclosure inputs to simulate how the risk score changes.<br>
  先載入公司真實資料，再手動調整輸入觀察風險變化。Walk / External features are fixed from the selected company profile.</p>
</div>
""", unsafe_allow_html=True)

        e_dens = st.slider("🌍 Climate Keyword Density (E_density)",
            0.0, 75.0, step=0.5, key="e_dens",
            help="climate, emission, carbon, transition, net zero per 1 000 words")
        s_dens = st.slider("👥 Social Keyword Density (S_density)",
            0.0, 55.0, step=0.5, key="s_dens",
            help="diversity, safety, human rights, community per 1 000 words")
        g_dens = st.slider("🏛 Governance Keyword Density (G_density)",
            0.0, 50.0, step=0.5, key="g_dens",
            help="governance, board, audit, compliance, transparency per 1 000 words")
        hedge = st.slider("💬 Vague Aspiration Phrases (hedge_score)",
            0.0, 3.0, step=0.05, key="hedge",
            help="may, might, could, aspire, intend to, aim to per 1 000 words")
        ttl = st.slider("📄 ESG Disclosure Intensity (talk_total_density)",
            0.0, 90.0, step=1.0, key="ttl",
            help="E + S + G keyword density combined")
        sector = st.selectbox("🏢 Sector", SECTORS, key="sector_select",
                              disabled=ticker_valid)

    hidden = {
        "pct_numeric": feature_value("pct_numeric"),
        "vague_to_specific_ratio": feature_value("vague_to_specific_ratio"),
        "text_length": feature_value("text_length"),
        "esg_topic_breadth": feature_value("esg_topic_breadth"),
        "e_s_g_dispersion": feature_value("e_s_g_dispersion"),
        "sector_rank": feature_value("sector_rank"),
        "planet_friendly_business": feature_value("planet_friendly_business"),
        "honest_fair_business": feature_value("honest_fair_business"),
        "sector_encoded": feature_value("sector_encoded") if ticker_valid else float(SECTOR_ENC.get(sector, 0)),
    }
    feat_vec = {
        "E_density": e_dens, "S_density": s_dens, "G_density": g_dens,
        "hedge_score": hedge,
        "pct_numeric":             hidden["pct_numeric"],
        "talk_total_density":      ttl,
        "vague_to_specific_ratio": hidden["vague_to_specific_ratio"],
        "text_length":             hidden["text_length"],
        "esg_topic_breadth":       hidden["esg_topic_breadth"],
        "e_s_g_dispersion":        hidden["e_s_g_dispersion"],
        "sector_rank":             hidden["sector_rank"],
        "planet_friendly_business":hidden["planet_friendly_business"],
        "honest_fair_business":    hidden["honest_fair_business"],
        "sector_encoded":          hidden["sector_encoded"],
    }
    try:
        prob = predict_risk(pipeline, feat_vec)
    except Exception as exc:
        st.error(f"Unable to generate prediction: {exc}")
        return

    current_risk_level = risk_level(prob)

    # Persist for Business Recommendation page
    st.session_state["selected_ticker"] = selected_ticker if ticker_valid else None
    st.session_state["selected_company_name"] = selected_company
    st.session_state["selected_sector"] = selected_sector if ticker_valid else sector
    st.session_state["selected_actual_label"] = selected_actual_label
    st.session_state["last_prob"] = prob
    st.session_state["last_risk_level"] = current_risk_level
    st.session_state["last_feature_vector"] = feat_vec
    # Backward-compatible keys used by the existing Business Recommendation page.
    st.session_state["last_sector"] = st.session_state["selected_sector"]
    st.session_state["last_feat_vec"] = feat_vec

    with col_res:
        st.markdown("### Selected Company")
        if ticker_valid:
            st.markdown(
                f"<div class='info-box'>"
                f"<strong>Ticker:</strong> {selected_ticker}<br>"
                f"<strong>Company:</strong> {selected_company}<br>"
                f"<strong>Sector:</strong> {selected_sector}<br>"
                f"<strong>Actual Label:</strong> {selected_actual_label}<br>"
                f"<span style='color:#8b949e;font-size:.9rem;'>"
                f"Actual label is based on the project’s training definition and does not "
                f"represent a legal finding. 訓練標籤不代表法律認定。"
                f"</span></div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div class='info-box'>"
                f"<strong>Current Mode:</strong> Manual input / median defaults<br>"
                f"<strong>Selected Company:</strong> Not selected<br>"
                f"<strong>Sector Assumption:</strong> {sector}<br>"
                f"<strong>Actual Label:</strong> Not available<br>"
                f"<span style='color:#8b949e;font-size:.9rem;'>"
                f"尚未選擇公司，目前使用手動輸入與中位數預設值。"
                f"</span></div>",
                unsafe_allow_html=True,
            )
        st.markdown("### Risk Score Result")

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=prob*100,
            number={"suffix":"%","font":{"color":"#0F3D2E","size":42}},
            delta={"reference":7.4,"suffix":"%","font":{"color":"#6B7280"}},
            title={"text":"Greenwashing Risk Score",
                   "font":{"color":"#374151","size":16}},
            gauge={
                "axis":{"range":[0,100],"tickcolor":"#6B7280",
                        "tickfont":{"color":"#6B7280"}},
                "bar":{"color":"#D64545" if prob>=.7 else "#F4B942" if prob>=.3 else "#2E7D32"},
                "bgcolor":"#FFFFFF","bordercolor":"#DCE9DD",
                "steps":[{"range":[0,30],"color":"#E8F5EA"},
                         {"range":[30,70],"color":"#FFF3CD"},
                         {"range":[70,100],"color":"#FFE1E1"}],
                "threshold":{"line":{"color":"#F59E0B","width":3},
                             "thickness":0.8,"value":7.4},
            },
        ))
        fig_gauge.update_layout(**pdk(),
            height=280, margin=dict(l=30,r=30,t=50,b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)

        st.markdown(f"<div style='text-align:center;margin:10px 0;'>"
                    f"{risk_badge(prob)}</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='warn-box' style='margin-top:8px;'>"
            "This is a model-estimated risk score, not a legal conclusion.<br>"
            "<span style='color:#8b949e;font-size:.9rem;'>"
            "模型分數僅作為風險篩選參考，不代表法律判定。</span></div>",
            unsafe_allow_html=True,
        )

        if round(prob*100, 1) == 0.0:
            st.markdown(
                '<div class="info-box">ℹ️ The model does not detect a strong '
                'Talk-Walk mismatch under current assumptions.</div>',
                unsafe_allow_html=True)

        # ── Recommended Action ─────────────────────────────────────────────────
        action_title, action_text, action_cls = prediction_action(prob)
        st.markdown(f"#### ⚡ Recommended Action: {action_title}")
        st.markdown(f'<div class="{action_cls}">{action_text}</div>',
                    unsafe_allow_html=True)

        # ── Input Signal Notes ─────────────────────────────────────────────────
        if current_risk_level == "High":
            signal_title = "Why is this company flagged?"
            signal_zh = "為什麼模型標記高風險？"
        elif current_risk_level == "Medium":
            signal_title = "What signals require review?"
            signal_zh = "哪些訊號需要進一步審查？"
        else:
            signal_title = "Input Signal Notes"
            signal_zh = "目前輸入訊號摘要。"
        st.markdown(f"#### 🔎 {signal_title}")
        st.markdown(
            f"<span style='color:#8b949e;font-size:.9rem;'>{signal_zh}</span>",
            unsafe_allow_html=True,
        )
        st.caption("Input Signal Notes")
        st.markdown(
            "Individual input signals. Final risk score reflects all 14 features combined.<br>"
            "<span style='color:#8b949e;'>模型會綜合 Talk、Walk / External 與產業脈絡，不只看單一指標。</span>",
            unsafe_allow_html=True,
        )

        median_ttl = safe_float(feature_medians["talk_total_density"], 0.0)
        median_e   = safe_float(feature_medians["E_density"], 0.0)
        median_sr  = safe_float(feature_medians["sector_rank"], 0.0)
        median_hfb = safe_float(feature_medians["honest_fair_business"], 0.0)
        median_pfb = safe_float(feature_medians["planet_friendly_business"], 0.0)

        flags = []
        if ttl > median_ttl:
            flags.append(f"📢 ESG Disclosure Intensity ({ttl:.1f}) is above dataset median ({median_ttl:.1f})")
        if e_dens > median_e:
            flags.append(f"🌡 Climate Keyword Density ({e_dens:.1f}) above median ({median_e:.1f})")
        if hidden["sector_rank"] > median_sr:
            flags.append(f"📉 Sector-relative ESG Risk ({hidden['sector_rank']:.2f}) above median ({median_sr:.2f})")
        if hidden["honest_fair_business"] < median_hfb:
            flags.append(f"⚖️ External Governance / Ethics Signal ({hidden['honest_fair_business']:.0f}) below median ({median_hfb:.0f})")
        if hidden["planet_friendly_business"] < median_pfb:
            flags.append(f"🌍 External Environmental Signal ({hidden['planet_friendly_business']:.0f}) below median ({median_pfb:.0f})")

        if prob >= 0.70:
            st.markdown(
                '<div class="flag-box">ESG disclosure is high, but sector-relative '
                'external ESG risk is also high and external governance/environmental '
                'signals are weak.<br>'
                '<span style="color:#8b949e;">說很多 ESG，但外部風險與治理 / 環境訊號偏弱。</span>'
                '</div>',
                unsafe_allow_html=True,
            )
        elif prob < 0.30:
            st.markdown(
                '<div class="info-box">The model does not detect a strong '
                'Talk-Walk mismatch under current assumptions.<br>'
                '<span style="color:#8b949e;">目前未偵測到明顯 Talk-Walk 落差。</span>'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="warn-box">Some input signals require review before '
                'sustainable lending or ESG investment decisions.<br>'
                '<span style="color:#8b949e;">部分訊號需要進一步審查。</span>'
                '</div>',
                unsafe_allow_html=True,
            )

        if flags:
            for flag in flags:
                st.markdown(f'<div class="flag-box" style="margin:4px 0;">{flag}</div>',
                            unsafe_allow_html=True)
        elif current_risk_level != "Low":
            st.markdown(
                '<div class="info-box">✅ No inputs exceed dataset medians '
                'under current settings.</div>', unsafe_allow_html=True)

        expander_title = (
            "Fixed Walk / External Features from Selected Company — 使用所選公司的外部特徵值。"
            if ticker_valid
            else "Median Walk / External Assumptions — 目前使用資料集中位數作為預設假設。"
        )
        with st.expander(expander_title):
            hidden_df = pd.DataFrame([
                (FEATURE_DISPLAY.get(k,k), v)
                for k,v in feat_vec.items()
                if k not in ["E_density","S_density","G_density",
                             "hedge_score","talk_total_density","sector_encoded"]
            ], columns=["Feature","Value"])
            st.dataframe(hidden_df, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════════════
def page_shap():
    st.markdown(
        "<div class='page-hero-small'><h1>Explainability — SHAP Analysis</h1>"
        "<p>Model interpretation for ESG due diligence and credit review support.</p></div>",
        unsafe_allow_html=True)
    st.markdown("""
<div class="warn-box">
⚠️ <strong>Disclaimer:</strong> This model flags companies for further ESG due diligence.
It provides a risk screening signal, not a legal finding of greenwashing or wrongdoing. /
本模型僅作為風險篩選工具，不代表法律認定。
</div>
""", unsafe_allow_html=True)
    climate_data_grid(
        [
            ("Global", "beeswarm", "all companies"),
            ("Local", "waterfall", "case evidence"),
            ("Drivers", "SHAP", "directional impact"),
            ("Action", "follow-up", "credit review"),
        ],
        "Explainable AI Evidence Layer",
        "SHAP connects model scores to business-readable due diligence questions.",
    )
    st.markdown("---")

    # ── Global beeswarm ───────────────────────────────────────────────────────
    st.markdown("## 🐝 Global Feature Impact (all 257 companies)")
    st.markdown("""
<div class="info-box">
The x-axis shows directional SHAP impact on greenwashing risk prediction.
Color indicates feature value: <strong style="color:#ef5350;">red = high value</strong>,
<strong style="color:#42a5f5;">blue = low value</strong>.
Each dot represents one company.
</div>
""", unsafe_allow_html=True)

    bee_path = os.path.join(OUTPUTS,"shap_beeswarm.png")
    if os.path.exists(bee_path):
        st.image(bee_path, use_container_width=True)
    else:
        st.warning("shap_beeswarm.png not found — run src/explainability.py first.")

    st.markdown("""
<div class="info-box" style="border-left-color:#d2a8ff;margin-top:12px;">
<strong>📌 Methodology Note:</strong>
<code>sector_rank</code> (Sector-relative ESG Risk) is the most impactful feature
because the model is designed to use both Talk and Walk / External indicators as a
risk scoring engine. Reduced Walk-Proxy Ablation (AUC = 0.741) confirms that ESG
disclosure language still carries independent signal after removing selected
Walk / External proxy features.<br>
<span style="color:#8b949e;">移除部分外部代理特徵後，文字特徵仍有訊號。</span>
</div>
""", unsafe_allow_html=True)
    st.markdown("---")

    # ── Company waterfall plots ───────────────────────────────────────────────
    st.markdown("## 🔍 Company-Level Evidence — Top 3 Model-Flagged High-Risk Cases")

    companies = [
        {
            "rank":1, "ticker":"CVX", "name":"Chevron Corporation",
            "sector":"Energy", "risk_label":"Risk Level: Very High (>95%)",
            "img": os.path.join(OUTPUTS,"shap_waterfall_1.png"),
            "main_drivers": [
                "Sector-relative ESG Risk (sector_rank=0.80) — bottom 20% in Energy",
                "ESG Disclosure Intensity (talk_total_density=63.75) — top tier",
                "External Environmental / Governance signals both at −40",
            ],
            "interpretation": (
                "CVX combines the highest ESG keyword density in the Energy sector "
                "with the worst sector-relative ESG risk rating and the lowest "
                "external governance / environmental signals. This creates the maximum "
                "Talk-Walk mismatch."
            ),
            "followup": (
                "Request independent ESG assurance. Review scope 1/2/3 "
                "emission target credibility. Consider senior ESG / credit committee "
                "review before sustainable lending approval. 建議提高至主管或 ESG / "
                "授信委員會層級審查。"
            ),
        },
        {
            "rank":2, "ticker":"MTB", "name":"M&T Bank Corporation",
            "sector":"Financial Services", "risk_label":"Risk Level: Very High (>95%)",
            "img": os.path.join(OUTPUTS,"shap_waterfall_2.png"),
            "main_drivers": [
                "Poor sector-relative performance (sector_rank) within Financial Services",
                "External Governance / Ethics Signal (honest_fair_business) below peers",
                "Elevated ESG Disclosure Intensity relative to actual performance",
            ],
            "interpretation": (
                "MTB publishes extensive ESG disclosures but Walk / External signals "
                "do not support the narrative. The external governance / ethics signal "
                "is a decisive negative indicator, suggesting a credibility gap."
            ),
            "followup": (
                "Conduct peer comparison vs. regional bank ESG leaders. "
                "Request governance controversy history. "
                "Review alignment between ESG commitments and actual lending practices."
            ),
        },
        {
            "rank":3, "ticker":"WYNN", "name":"Wynn Resorts Ltd.",
            "sector":"Consumer Cyclical", "risk_label":"Risk Level: Very High (>95%)",
            "img": os.path.join(OUTPUTS,"shap_waterfall_3.png"),
            "main_drivers": [
                "Sector-relative ESG Risk — poor within Consumer Cyclical",
                "ESG Pillar Imbalance — elevated, suggesting selective disclosure",
                "ESG Disclosure Intensity above sector average",
            ],
            "interpretation": (
                "WYNN's high sub-score dispersion indicates selective reporting: "
                "stronger disclosure in some ESG pillars while weaknesses in other "
                "pillars may be underrepresented."
            ),
            "followup": (
                "Verify ESG pillar coverage in the disclosure report. "
                "Identify which ESG dimension shows the largest gap. "
                "Request consistent multi-pillar ESG evidence before sustainable "
                "finance review."
            ),
        },
    ]

    for c in companies:
        with st.expander(
            f"#{c['rank']} · {c['ticker']} — {c['name']} [{c['sector']}]  "
            f"| {c['risk_label']}",
            expanded=(c["rank"]==1)):

            col_img, col_txt = st.columns([3,2])
            with col_img:
                if os.path.exists(c["img"]):
                    st.image(c["img"], use_container_width=True)
                else:
                    st.warning(f"Waterfall image not found: {c['img']}")
            with col_txt:
                st.markdown(f"#### {c['ticker']} — {c['name']}")

                st.markdown("**🔑 Main Drivers**")
                for d in c["main_drivers"]:
                    st.markdown(f"- {d}")

                st.markdown("**💡 Business Interpretation**")
                st.markdown(f'<div class="warn-box" style="margin:6px 0;">'
                            f'{c["interpretation"]}</div>', unsafe_allow_html=True)

                st.markdown("**📋 Recommended Follow-up**")
                st.markdown(f'<div class="info-box" style="margin:6px 0;">'
                            f'{c["followup"]}</div>', unsafe_allow_html=True)

    st.markdown("---")

    # ── Feature importance table ───────────────────────────────────────────────
    fi_df = load_feature_importance()
    fi_df["Business Name"] = fi_df["feature"].map(
        lambda f: FEATURE_DISPLAY.get(f, f))
    walk_external_features = {
        "sector_rank",
        "e_s_g_dispersion",
        "planet_friendly_business",
        "honest_fair_business",
    }
    description_map = {
        "sector_rank": "Company's external ESG risk position within its sector.",
        "talk_total_density": "Total ESG keyword density in the company's disclosure text.",
        "honest_fair_business": "External governance and ethics credibility signal.",
        "planet_friendly_business": "External environmental credibility signal.",
        "text_length": "Length of the ESG disclosure text.",
        "sector_encoded": "Industry context used by the model.",
        "hedge_score": "Frequency of vague or aspirational ESG language.",
        "E_density": "Climate and environmental keyword density.",
        "e_s_g_dispersion": "Imbalance across E, S, and G pillar scores.",
        "S_density": "Social responsibility keyword density.",
        "G_density": "Governance keyword density.",
        "pct_numeric": "Share of quantitative disclosure signals.",
        "vague_to_specific_ratio": "Vague language relative to specific disclosure signals.",
        "esg_topic_breadth": "Breadth of ESG topics covered in the disclosure text.",
    }
    fi_df["Feature Group"] = np.where(
        fi_df["feature"].eq("sector_encoded"), "Context",
        np.where(fi_df["feature"].isin(walk_external_features), "Walk / External", "Talk")
    )
    fi_df["mean_abs_shap"] = pd.to_numeric(
        fi_df["mean_abs_shap"], errors="coerce").round(3)
    fi_df["description"] = fi_df["feature"].map(
        lambda f: description_map.get(f, FEATURE_DISPLAY.get(f, f)))
    fi_show = fi_df[fi_df["mean_abs_shap"] > 0].copy()

    st.markdown("## 📊 Top Feature Importance")
    st.caption(
        "Non-zero SHAP features are shown below. Near-zero features are retained "
        "for financial logic but not shown in the table.  \n"
        "貢獻接近 0 的特徵因具備金融邏輯仍保留，但不列入表格。")
    st.dataframe(
        fi_show[["rank","Business Name","Feature Group","mean_abs_shap","description"]],
        use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — ENGAGEMENT PRIORITY AGENT
# ══════════════════════════════════════════════════════════════════════════════
def _build_model_linked_engagement_portfolio(df):
    import engagement_agent as ea

    configs = [
        ("CVX", "石油", True, 180, 12_000, 35_000_000, False, False, False, 3),
        ("XOM", "石油", True, 160, 15_000, 32_000_000, True, False, False, 3),
        ("NUE", "鋼鐵", True, 95, 4_800, 12_000_000, False, False, False, 3),
        ("VMC", "水泥", True, 80, 3_200, 7_000_000, False, True, False, 4),
        ("FCX", "金屬加工", False, 70, 4_000, 5_500_000, True, True, False, 3),
        ("TSLA", "汽車", False, 65, 8_500, 2_500_000, True, True, False, 2),
        ("AAPL", "科技", False, 55, 30_000, 1_800_000, True, True, False, 2),
        ("SME01", "金屬加工", False, 8, None, None, None, None, False, 5),
    ]
    name_lookup = {}
    if "ticker" in df.columns:
        for _, row in df.iterrows():
            ticker = str(row.get("ticker", "")).strip().upper()
            if not ticker or ticker in name_lookup:
                continue
            for col in ["Name", "company_name", "Company", "company"]:
                if col in row.index and pd.notna(row[col]) and str(row[col]).strip():
                    name_lookup[ticker] = str(row[col]).strip()
                    break

    portfolio = []
    for ticker, industry, high_risk, exposure, company_value, emissions, midterm, decarb, policy, dq in configs:
        company_name = name_lookup.get(ticker, "示意中小企業")
        display_name = f"{company_name}（模型示意）" if ticker != "SME01" else "某精密中小企業（待補件示意）"
        portfolio.append(ea.Client(
            display_name,
            ticker,
            industry,
            high_risk,
            exposure,
            company_value,
            emissions,
            has_midterm_temp_target=midterm,
            has_decarbon_target=decarb,
            is_policy_investment=policy,
            talk_walk_gap=None,
            pcaf_data_quality=dq,
        ))
    return portfolio


def page_engagement(df):
    import engagement_agent as ea
    import engagement_addons as ad

    st.markdown(
        "<div class='page-hero-small'><h1>優先議合 Agent</h1>"
        "<p>Use the LightGBM Talk-Walk Gap score, PCAF financed emissions, "
        "and lending exposure to rank engagement priorities.</p></div>",
        unsafe_allow_html=True)
    st.markdown("""
<div class="warn-box">
AI Decision Support Only: ranking and memos support ESG due diligence. They do not
replace credit approval review or make legal findings.<br>
<span style="color:#8b949e;">本頁僅供責任授信議合排序與覆核準備。</span>
</div>
""", unsafe_allow_html=True)
    climate_data_grid(
        [
            ("Model", "LightGBM", "Talk-Walk Gap"),
            ("PCAF", "scope 1+2", "financed emissions"),
            ("HITL", "exposure", "human review"),
            ("Output", "ranking", "engagement memo"),
        ],
        "Engagement Agent Flow",
        "Model score and bank-side exposure data are combined into a review-first engagement list.",
    )
    st.markdown("---")

    control_col, status_col = st.columns([1.2, .8])
    with control_col:
        mode = st.radio(
            "Portfolio",
            ["Model-linked S&P demo portfolio", "Taiwan illustrative portfolio"],
            horizontal=True,
            label_visibility="collapsed",
        )
        top_n = st.slider("議合包產出件數", 1, 3, 1)
    with status_col:
        st.markdown("""
<div class="info-box" style="margin-top:0;">
<strong>Scoring Source</strong><br>
LightGBM 漂綠模型 via <code>greenwashing_scorer.score_company()</code><br>
<span style="color:#8b949e;">查無 ticker 時會標 NEED_HUMAN_REVIEW，不補假分數。</span>
</div>
""", unsafe_allow_html=True)

    portfolio = (
        _build_model_linked_engagement_portfolio(df)
        if mode == "Model-linked S&P demo portfolio"
        else ea.sample_portfolio()
    )
    state = ea.run_engagement_workflow(portfolio, top_n=top_n, use_llm=False)
    state["ranked"] = ad.annotate_tiers(state["ranked"])
    ranked = pd.DataFrame(state["ranked"])

    metric_cols = st.columns(4)
    metric_cols[0].metric("Portfolio", f"{len(ranked)}")
    metric_cols[1].metric("Priority", f"{int(ranked['eligible'].fillna(False).sum())}")
    metric_cols[2].metric("NEED_HUMAN_REVIEW", f"{len(state['needs_review'])}")
    metric_cols[3].metric("HITL", f"{len(state['hitl'])}")

    show_cols = [
        "rank", "name", "ticker", "industry", "exposure_e_twd",
        "financed_emissions_tco2e", "twgap_tier", "has_target",
        "is_policy", "priority_score", "eligible", "engagement_method",
        "status", "reasons",
    ]
    st.markdown("## 優先議合排序清單")
    st.dataframe(
        ranked[show_cols],
        use_container_width=True,
        hide_index=True,
        column_config={
            "rank": "排序",
            "name": "客戶",
            "ticker": "Ticker",
            "industry": "產業",
            "exposure_e_twd": st.column_config.NumberColumn("暴險(億)", format="%.0f"),
            "financed_emissions_tco2e": st.column_config.NumberColumn("融資排放(tCO2e)", format="%.0f"),
            "twgap_tier": "TWGap分級",
            "has_target": "已有目標",
            "is_policy": "政策性",
            "priority_score": st.column_config.NumberColumn("優先分數", format="%.3f"),
            "eligible": "優先議合",
            "engagement_method": "建議方式",
            "status": "狀態",
            "reasons": "排序理由",
        },
    )

    ok = ranked[ranked["status"] == "OK"].copy()
    chart_col, review_col = st.columns([1.35, .65])
    with chart_col:
        if ok.empty:
            st.warning("目前沒有可計分案件；請先補齊模型特徵、PCAF 或目標資料。")
        else:
            fig = px.scatter(
                ok,
                x="financed_emissions_tco2e",
                y="talk_walk_gap",
                size="exposure_e_twd",
                color="eligible",
                hover_name="name",
                hover_data=["ticker", "industry", "priority_score", "engagement_method"],
                labels={
                    "financed_emissions_tco2e": "Financed Emissions (tCO2e)",
                    "talk_walk_gap": "Talk-Walk Gap",
                    "eligible": "Priority",
                },
                color_discrete_map={True: "#D64545", False: "#2E7D32"},
            )
            fig.update_layout(**pdk(), height=390, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig, use_container_width=True)
    with review_col:
        st.markdown("### 人工覆核")
        if state["needs_review"]:
            st.markdown(
                "<div class='warn-box'><strong>NEED_HUMAN_REVIEW</strong><br>"
                + "<br>".join(state["needs_review"])
                + "</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown("<div class='info-box'>無待補件案件。</div>", unsafe_allow_html=True)
        if state["hitl"]:
            st.markdown(
                "<div class='flag-box'><strong>HITL Checkpoint</strong><br>"
                + "<br>".join(state["hitl"])
                + "</div>",
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.markdown("## 議合成效追蹤")
    comparison = ad.compare_snapshots()
    st.markdown(ad.render_comparison(comparison))
    if state.get("snapshot_path"):
        st.caption(f"本次快照：{state['snapshot_path']}")
    elif state.get("snapshot_error"):
        st.warning(f"快照儲存失敗：{state['snapshot_error']}")

    st.markdown("---")
    st.markdown("## 議合包")
    if not state["packages"]:
        st.markdown(
            "<div class='warn-box'>目前沒有可產出議合包的案件；排序前段案件仍需人工補件。</div>",
            unsafe_allow_html=True,
        )
    else:
        package_options = [f"{p['ticker']} — {p['name']}" for p in state["packages"]]
        selected = st.selectbox("選擇議合包", package_options)
        selected_pkg = state["packages"][package_options.index(selected)]
        st.markdown(f"<div class='memo-card'>{selected_pkg['package']}</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 7 — BUSINESS RECOMMENDATION
# ══════════════════════════════════════════════════════════════════════════════
def page_business(pipeline):
    st.markdown(
        "<div class='page-hero-small'><h1>Business Recommendation</h1>"
        "<p>Translate model risk into ESG credit review actions and governance notes.</p></div>",
        unsafe_allow_html=True)
    st.markdown("""
<div class="warn-box">
Model flags risk; credit officers make final decisions. /
模型只做風險提示，最終由授信人員判斷。
</div>
""", unsafe_allow_html=True)
    business_visual, business_ladder = st.columns([1, 1])
    with business_visual:
        climate_data_grid(
            [
                ("Low", "0–30%", "standard review"),
                ("Medium", "30–70%", "evidence request"),
                ("High", "70–100%", "enhanced due diligence"),
                ("Human", "final", "credit decision"),
            ],
            "Credit Review Decision Support",
            "Model output is translated into ESG lending review actions, not automatic decisions.",
        )
    with business_ladder:
        finance_action_ladder(compact=True)
    st.markdown("---")

    # ── Governance and limitation notes ───────────────────────────────────────
    st.markdown("## Governance & Limitation Notes")
    st.caption("治理與限制說明")
    gov_notes = [
        (
            "1. Screening Tool, Not Legal Judgment",
            "This dashboard is designed as an early-warning and risk screening tool. "
            "It supports ESG due diligence review and does not make legal findings.",
            "模型只做風險提示，供進一步查核使用。",
        ),
        (
            "2. Human Review Required",
            "Final lending or investment decisions should be made by ESG analysts, credit officers, "
            "and responsible decision-makers after reviewing supporting evidence.",
            "最終授信或投資判斷仍需由專業人員審查。",
        ),
        (
            "3. Sector-Relative Limitation",
            "The greenwashing label is based on sector-relative Talk and Walk rankings. Results may "
            "be affected by the number of companies and data quality within each sector.",
            "模型使用同產業排名，因此會受產業樣本數與資料品質影響。",
        ),
        (
            "4. External Data Limitation",
            "External ESG ratings, controversy indicators, and integrity scores may also contain "
            "bias, delays, or inconsistent methodologies.",
            "外部 ESG 評分本身也可能有偏誤或時間落差。",
        ),
        (
            "5. AI Tool Dependency Risk",
            "AI tools were used to assist coding, feature design, and dashboard development. All "
            "AI-generated outputs should be reviewed, tested, and explained by the team.",
            "AI 協助產出程式與內容，但仍需人工檢查、測試與理解。",
        ),
    ]
    for start in range(0, len(gov_notes), 2):
        gov_cols = st.columns([1,1])
        for col, (title, en, zh) in zip(gov_cols, gov_notes[start:start+2]):
            col.markdown(
                f"<div class='info-box' style='min-height:150px;margin:8px 0;'>"
                f"<strong>{title}</strong>"
                f"<p style='margin:8px 0 4px;'>{en}</p>"
                f"<span style='color:#8b949e;font-size:.9rem;'>{zh}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # ── Three-tier action cards ────────────────────────────────────────────────
    st.markdown("## 🏦 ESG Credit Action Framework")
    r1,r2,r3 = st.columns(3)

    with r3:  # High on the right
        st.markdown("""
<div class="action-card high">
<h3 style="color:#D64545;">HIGH RISK 70–100%</h3>
<div class="eyebrow">Credit Action</div>
<p>Escalate to enhanced ESG due diligence. Request third-party assurance,
senior ESG / credit committee review, and controversy history before sustainable lending approval.<br>
<span style="color:#6B7280;">升級盡職調查，並由主管或 ESG / 授信委員會審查。</span></p>
<div class="eyebrow">Required Evidence</div>
<ul>
  <li>Third-party ESG assurance report</li>
  <li>Senior ESG / credit committee review record</li>
  <li>Quantified emission / social targets with baselines</li>
  <li>Controversy history and remediation plan</li>
</ul>
<div class="eyebrow">Monitoring</div>
<p>Quarterly review recommended. ESG analysts should verify supporting
evidence before sustainable lending decisions.</p>
<div class="action-footer high">
Senior ESG / Credit Committee Review Recommended<br>
<span style="font-weight:650;">建議提高至主管或 ESG / 授信委員會層級審查。</span>
</div>
</div>
""", unsafe_allow_html=True)

    with r2:
        st.markdown("""
<div class="action-card medium">
<h3 style="color:#C98300;">MEDIUM RISK 30–70%</h3>
<div class="eyebrow">Credit Action</div>
<p>Request supplementary ESG documentation and conduct peer comparison
before finalising credit terms.<br><span style="color:#6B7280;">要求補件與同業比較，供授信審查參考。</span></p>
<div class="eyebrow">Required Evidence</div>
<ul>
  <li>Quantitative ESG targets vs. actuals table</li>
  <li>Peer (same sector) ESG performance comparison</li>
  <li>Explanation of any controversy events</li>
</ul>
<div class="eyebrow">Monitoring</div>
<p>Semi-annual ESG review. Track controversy news flow and verify
supporting evidence before final credit judgment.</p>
<div class="action-footer medium">
Supplementary Filing + Peer Comparison
</div>
</div>
""", unsafe_allow_html=True)

    with r1:  # Low on the left
        st.markdown("""
<div class="action-card low">
<h3 style="color:#2E7D32;">LOW RISK 0–30%</h3>
<div class="eyebrow">Credit Action</div>
<p>Proceed with standard ESG review process.<br>
<span style="color:#6B7280;">依標準授信審查流程辦理。</span></p>
<div class="eyebrow">Required Evidence</div>
<ul>
  <li>Standard ESG disclosure review</li>
  <li>Annual sustainability report assessment</li>
</ul>
<div class="eyebrow">Monitoring</div>
<p>Annual monitoring. May be considered for sustainable finance
if loan purpose also meets green taxonomy requirements.</p>
<div class="action-footer low">
Standard Review — Sustainable Finance Eligibility Requires Taxonomy Alignment
</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")

    # ── ESG Credit Memo — dynamic from session state ──────────────────────────
    st.markdown("## 🎯 ESG Credit Memo")
    selected_ticker = st.session_state.get("selected_ticker")
    prob_memo = st.session_state.get("last_prob")

    if selected_ticker is None or prob_memo is None:
        st.markdown(
            '<div class="info-box">🔁 Select a company in '
            '<strong>Prediction Demo</strong> to generate a customized ESG credit memo.<br>'
            '<span style="color:#8b949e;">請先到 Prediction Demo 選擇公司。</span>'
            '</div>', unsafe_allow_html=True)
    else:
        company_name = st.session_state.get("selected_company_name", selected_ticker)
        sector_memo = st.session_state.get("selected_sector", "—")
        actual_label = st.session_state.get("selected_actual_label", "—")
        risk_level = st.session_state.get("last_risk_level", "High" if prob_memo >= .7 else "Medium" if prob_memo >= .3 else "Low")
        feat_memo = st.session_state.get("last_feature_vector", {})
        memo_color = "#da3633" if prob_memo >= 0.7 else "#d29922" if prob_memo >= 0.3 else "#3fb950"

        memo_actions = {
            "Low": "Standard ESG Review",
            "Medium": "Request Supporting Evidence and Peer Comparison",
            "High": "Escalate to Enhanced ESG Due Diligence",
        }
        memo_evidence = {
            "Low": "Standard ESG disclosure and sustainability report review",
            "Medium": "Quantitative ESG targets, peer comparison, and controversy explanation",
            "High": "Third-party assurance, senior ESG / credit committee review, and controversy remediation plan",
        }
        st.markdown(
            f"""
<div class="memo-card">
<h3 style="color:#14532D;margin-top:0;">ESG Credit Memo</h3>
<p><strong>Company:</strong><br>{company_name} ({selected_ticker})</p>
<p><strong>Sector:</strong><br>{sector_memo}</p>
<p><strong>Model-estimated Greenwashing Risk:</strong><br>
<span style="color:{memo_color};font-size:2rem;font-weight:800;">{prob_memo*100:.1f}%</span></p>
<p><strong>Risk Level:</strong><br>{risk_level}</p>
<p><strong>Training Label:</strong><br>{actual_label}</p>
<p><strong>Recommended Credit Action:</strong><br>{memo_actions.get(risk_level, memo_actions["High"])}</p>
<p><strong>Required Evidence:</strong><br>{memo_evidence.get(risk_level, memo_evidence["High"])}</p>
<p><strong>Human Review Disclaimer:</strong><br>
This memo is generated for ESG due diligence support only. Final lending or investment decisions require human review.<br>
<span style="color:#6B7280;">此 memo 僅供 ESG 查核輔助，最終決策仍需人工審查。</span></p>
</div>
""",
            unsafe_allow_html=True,
        )
        if feat_memo:
            with st.expander("📄 Feature Profile Used in This Memo"):
                memo_rows = [(FEATURE_DISPLAY.get(k,k), round(v,3))
                             for k,v in feat_memo.items()]
                st.dataframe(pd.DataFrame(memo_rows, columns=["Feature","Value"]),
                             use_container_width=True)

    st.markdown("---")

    # ── AI Audit Trail ─────────────────────────────────────────────────────────
    st.markdown("## 📝 AI Audit Trail")
    st.markdown("""
This audit trail documents how AI-generated suggestions were reviewed, modified,
tested, and translated into project learning.<br>
<span style="color:#6B7280;">紀錄 AI 輔助內容如何經過人工檢查、修改與學習反思。</span>
""", unsafe_allow_html=True)

    audit = pd.DataFrame([
        {
            "Date/Week": "2025-05-15 / Week 8",
            "Task": "Data Loading & Merging",
            "AI Tool": "Claude Code",
            "Prompt Summary": "Load three CSV files, merge by ticker, keep the latest ESG report year, and remove rows with missing ESG scores.",
            "AI Output Used?": "Partially",
            "Human Modification": "Verified ticker matching, checked company counts, and confirmed missing ESG score handling.",
            "Learning Reflection": "Learned how data integration affects model sample size and downstream validity.",
        },
        {
            "Date/Week": "2025-05-15 / Week 8",
            "Task": "Feature Engineering",
            "AI Tool": "Claude Code",
            "Prompt Summary": "Create Talk features from ESG report text and Walk / External features from ESG scores and integrity indicators.",
            "AI Output Used?": "Partially",
            "Human Modification": "Reviewed feature definitions, renamed feature groups, and ensured Total ESG Risk was not directly used as a model feature.",
            "Learning Reflection": "Learned that domain-driven features must avoid circularity and should reflect financial logic.",
        },
        {
            "Date/Week": "2025-05-15 / Week 8",
            "Task": "Label Construction",
            "AI Tool": "Claude Code",
            "Prompt Summary": "Define greenwashing risk label using sector-relative Talk-Walk mismatch.",
            "AI Output Used?": "Partially",
            "Human Modification": "Checked that the label is based on within-sector rankings and added caveats about sector-relative limitations.",
            "Learning Reflection": "Learned that greenwashing labels are not legal facts but model training definitions.",
        },
        {
            "Date/Week": "2025-05-15 / Week 8",
            "Task": "Model Training + Optuna",
            "AI Tool": "Claude Code",
            "Prompt Summary": "Train baseline, Logistic Regression, Random Forest, XGBoost, and LightGBM using nested cross-validation and Optuna tuning.",
            "AI Output Used?": "Yes",
            "Human Modification": "Verified SMOTE was applied only inside training folds, reviewed model comparison results, and selected LightGBM for final scoring.",
            "Learning Reflection": "Learned why nested CV is important for separating tuning from unbiased evaluation.",
        },
        {
            "Date/Week": "2025-05-15 / Week 9",
            "Task": "SHAP Explainability",
            "AI Tool": "Claude Code",
            "Prompt Summary": "Generate global SHAP beeswarm plot, company-level waterfall plots, and feature importance table.",
            "AI Output Used?": "Yes",
            "Human Modification": "Revised feature names into business-friendly terms and added disclaimers that the model is for risk screening only.",
            "Learning Reflection": "Learned how explainability helps convert model scores into ESG due diligence reasoning.",
        },
        {
            "Date/Week": "2025-05-15 / Week 9",
            "Task": "Dashboard",
            "AI Tool": "Claude Code",
            "Prompt Summary": "Build a six-page Streamlit dashboard with data exploration, model performance, prediction demo, explainability, business recommendation, and AI audit trail.",
            "AI Output Used?": "Yes",
            "Human Modification": "Modified wording, reduced overly technical explanations, added bilingual short notes, and tested the dashboard flow.",
            "Learning Reflection": "Learned how to translate machine learning outputs into business decision support for ESG analysts and credit officers.",
        },
    ])

    audit_columns = [
        "Date/Week",
        "Task",
        "AI Tool",
        "Prompt Summary",
        "AI Output Used?",
        "Human Modification",
        "Learning Reflection",
    ]
    audit = audit[audit_columns]
    st.dataframe(audit, use_container_width=True, hide_index=True)

    for _, row in audit.iterrows():
        with st.expander(f"{row['Date/Week']} — {row['Task']}", expanded=True):
            st.markdown(f"""
<div class="case-card" style="margin:0;background:#FFFFFF;">
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:14px;">
    <div class="info-box" style="margin:0;">
      <strong>Date/Week</strong><br>{row['Date/Week']}
    </div>
    <div class="info-box" style="margin:0;">
      <strong>AI Tool</strong><br>{row['AI Tool']}
    </div>
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:14px;">
    <div class="methodology-card" style="margin:0;">
      <strong>Task</strong><br>{row['Task']}
    </div>
    <div class="methodology-card" style="margin:0;">
      <strong>AI Output Used?</strong><br>{row['AI Output Used?']}
    </div>
  </div>
  <div class="info-box">
    <strong>Prompt Summary</strong><br>{row['Prompt Summary']}
  </div>
  <div class="warn-box">
    <strong>Human Modification</strong><br>{row['Human Modification']}
  </div>
  <div class="insight-box">
    <strong>Learning Reflection</strong><br>{row['Learning Reflection']}
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div class="info-box" style="margin-top:16px;">
<strong>IFRS S1/S2 Alignment Note:</strong>
This system is conceptually aligned with IFRS S1/S2 decision-useful sustainability
disclosure. The greenwashing risk score supports ESG disclosure credibility review
before sustainable lending or ESG investment decisions.<br>
<span style="color:#8b949e;">用於輔助永續授信與 ESG 投資前的揭露可信度審查。</span>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# NAVIGATION + MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    with st.sidebar:
        st.markdown(
            """
<div class="sidebar-logo">
  <div class="leaf">🌿</div>
  <div class="title">Group 2<br>ESG Dashboard</div>
</div>
""", unsafe_allow_html=True)

        page = st.radio("Navigation", [
            "🏠  Home",
            "📊  Data Explorer",
            "🤖  Model Performance",
            "🔴  Prediction Demo",
            "🔬  Explainability",
            "💼  Business Recommendation",
        ], label_visibility="collapsed")

        st.markdown("---")
        st.markdown(
            """
<div class="model-info-card">
  <strong>Model Information</strong><br>
  Model: LightGBM<br>
  AUC: 0.953&nbsp;&nbsp;|&nbsp;&nbsp;PR-AUC: 0.745<br>
  Dataset: 257 companies<br>
  Features: 14 = 9 Talk + 4 Walk / External + 1 Context<br>
  CV: Outer 5-fold · Inner 3-fold<br>
  Optuna: n_trials=30<br>
  SMOTE: k_neighbors=3<br>
  Risk tiers:<br>Low 0–30% / Med 30–70% / High 70–100%
</div>
""", unsafe_allow_html=True)

    df, X, y, meta = load_dataset()
    pipeline       = load_model()

    if   "Home"              in page: page_home()
    elif "Data Explorer"     in page: page_data(df, X, y, meta)
    elif "Model Performance" in page: page_model(pipeline, X, y)
    elif "Prediction Demo"   in page: page_predict(pipeline)
    elif "Explainability"    in page: page_shap()
    elif "Business"          in page: page_business(pipeline)


if __name__ == "__main__":
    main()
