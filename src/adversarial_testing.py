"""
Deterministic adversarial rewrite robustness test for ESG disclosure signals.

This script uses regex and lexicon matching only. It does not call an LLM API.
It compares original ESG disclosure snippets with AI-style rewritten versions to
show how surface keyword signals, semantic signals, and specificity change.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

import matplotlib.pyplot as plt
import pandas as pd


OUTPUTS_DIR = "outputs"
CSV_PATH = os.path.join(OUTPUTS_DIR, "adversarial_rewrite_test.csv")
PNG_PATH = os.path.join(OUTPUTS_DIR, "adversarial_rewrite_signal_comparison.png")


KEYWORD_PATTERNS = [
    re.compile(r"\bnet[-\s]?zero\b", re.IGNORECASE),
    re.compile(r"\bcarbon\b", re.IGNORECASE),
    re.compile(r"\bemissions?\b", re.IGNORECASE),
    re.compile(r"\bclimate\b", re.IGNORECASE),
    re.compile(r"\bESG\b", re.IGNORECASE),
    re.compile(r"\bsustainability\b", re.IGNORECASE),
    re.compile(r"\brenewable\b", re.IGNORECASE),
    re.compile(r"\btransitions?\b", re.IGNORECASE),
    re.compile(r"\bgreen\b", re.IGNORECASE),
]

SEMANTIC_PATTERNS = [
    re.compile(r"\blong[-\s]?term\s+sustainable\s+transitions?\b", re.IGNORECASE),
    re.compile(r"\benvironmental\s+responsibilit(y|ies)\b", re.IGNORECASE),
    re.compile(r"\bresponsible\s+business(es)?\b", re.IGNORECASE),
    re.compile(r"\bgovernance\s+process(es)?\b", re.IGNORECASE),
    re.compile(r"\bclimate\s+resilien(ce|t)\b", re.IGNORECASE),
    re.compile(r"\baccountabilit(y|ies)\b", re.IGNORECASE),
    re.compile(r"\bmaterialit(y|ies)\b", re.IGNORECASE),
    re.compile(r"\bstakeholder\s+engagements?\b", re.IGNORECASE),
]

MEASURABLE_PATTERNS = [
    re.compile(r"\d+\s*%", re.IGNORECASE),
    re.compile(r"\b20\d{2}\b", re.IGNORECASE),
    re.compile(r"\bbaseline\s+years?\b", re.IGNORECASE),
    re.compile(r"\bscience[-\s]?based\s+targets?\b", re.IGNORECASE),
    re.compile(r"\bverified\b", re.IGNORECASE),
    re.compile(r"\baudited\b", re.IGNORECASE),
    re.compile(r"\bthird[-\s]?party\s+assurance\b", re.IGNORECASE),
]

VAGUE_PATTERNS = [
    re.compile(r"\bcommitted\s+to\b", re.IGNORECASE),
    re.compile(r"\bmeaningful\s+progress\b", re.IGNORECASE),
    re.compile(r"\bcontinues?\s+to\s+improve\b", re.IGNORECASE),
    re.compile(r"\baspir(e|es|ed|ing)\s+to\b", re.IGNORECASE),
    re.compile(r"\baim(s|ed|ing)?\s+to\b", re.IGNORECASE),
    re.compile(r"\bsupport(s|ed|ing)?\s+a\s+better\s+future\b", re.IGNORECASE),
]


def _count_patterns(text: str, patterns: list[re.Pattern]) -> int:
    """Count deterministic regex matches in text."""
    if not isinstance(text, str):
        text = "" if pd.isna(text) else str(text)
    return sum(len(pattern.findall(text)) for pattern in patterns)


def keyword_signal(text: str) -> int:
    """Count surface ESG keywords."""
    return _count_patterns(text, KEYWORD_PATTERNS)


def semantic_signal(text: str) -> int:
    """Count meaning-level ESG phrases."""
    return _count_patterns(text, SEMANTIC_PATTERNS)


def measurable_commitment(text: str) -> int:
    """Count concrete evidence and measurable commitment patterns."""
    return _count_patterns(text, MEASURABLE_PATTERNS)


def vague_commitment(text: str) -> int:
    """Count vague commitment language."""
    return _count_patterns(text, VAGUE_PATTERNS)


@dataclass(frozen=True)
class RewriteCase:
    case_id: int
    category: str
    original_text: str
    ai_rewritten_text: str
    expected_issue: str


CASES = [
    RewriteCase(
        1,
        "Specific number -> vague progress language",
        "We achieved a 20% reduction in carbon emissions by 2025 compared with our baseline year.",
        "We made meaningful progress on sustainability and continue to improve our environmental responsibility.",
        "Specific quantified reduction is softened into vague progress language.",
    ),
    RewriteCase(
        2,
        "Specific number -> vague progress language",
        "The company reduced Scope 2 emissions by 15% and expanded renewable electricity procurement.",
        "The company is committed to sustainability and aims to support a better future through responsible business practices.",
        "Concrete emissions and renewable progress are rewritten as broad commitments.",
    ),
    RewriteCase(
        3,
        "Hard target year -> soft long-term language",
        "We target net zero emissions by 2050 and will cut carbon intensity by 2030.",
        "We support a long-term sustainable transition and climate resilience across our operations.",
        "Hard dated targets become softer transition language.",
    ),
    RewriteCase(
        4,
        "Hard target year -> soft long-term language",
        "Our climate strategy commits to net-zero operations by 2040 with interim targets by 2030.",
        "Our organization aspires to long-term sustainable transition through stronger stakeholder engagement.",
        "Dated net-zero commitments are replaced by long-term aspiration.",
    ),
    RewriteCase(
        5,
        "Third-party assurance -> governance-process language",
        "Our ESG data was verified by Deloitte and covered by third-party assurance.",
        "Our ESG information is supported by appropriate governance processes and accountability practices.",
        "External assurance is replaced with internal process language.",
    ),
    RewriteCase(
        6,
        "Third-party assurance -> governance-process language",
        "The carbon inventory was audited by an independent assurance provider in 2024.",
        "The climate information is managed through governance process improvements and materiality review.",
        "Audited evidence is rewritten as governance process disclosure.",
    ),
    RewriteCase(
        7,
        "Quantified Scope emissions -> abstract climate strategy",
        "Scope 1 emissions were 1.2M tCO2e and Scope 2 emissions were 0.4M tCO2e in 2023.",
        "We continue to improve climate strategy implementation and environmental responsibility.",
        "Quantified emissions data is abstracted into strategy language.",
    ),
    RewriteCase(
        8,
        "Quantified Scope emissions -> abstract climate strategy",
        "We reported 950,000 tons of carbon emissions and a 12% emissions reduction in 2023.",
        "We are committed to climate resilience and responsible business through ongoing strategy implementation.",
        "Specific emissions amounts and reductions are removed.",
    ),
    RewriteCase(
        9,
        "Specific baseline year -> undated commitment",
        "We delivered a 30% reduction versus the 2019 baseline year for operational emissions.",
        "We are committed to emission reduction and meaningful progress in sustainability.",
        "Baseline year and reduction percentage are replaced with undated commitment.",
    ),
    RewriteCase(
        10,
        "Specific baseline year -> undated commitment",
        "Water intensity declined 18% from the 2020 baseline year after renewable energy investments.",
        "We aim to improve resource efficiency and support a better future through environmental responsibility.",
        "Baseline comparison and quantified reduction disappear.",
    ),
    RewriteCase(
        11,
        "Concrete board oversight -> generic accountability",
        "The ESG committee is chaired by an independent director and reviews climate targets quarterly.",
        "Accountability is embedded across the organization through governance processes.",
        "Concrete board oversight is generalized into accountability language.",
    ),
    RewriteCase(
        12,
        "Concrete board oversight -> generic accountability",
        "The audit committee reviewed third-party assurance findings on carbon emissions in 2024.",
        "Our governance process supports accountability and stakeholder engagement on sustainability matters.",
        "Board-level review and assurance findings are softened.",
    ),
    RewriteCase(
        13,
        "Industry-specific metric -> industry-agnostic vision",
        "Our plants reached 70% renewable electricity by 2030 under our energy transition plan.",
        "We pursue responsible business practices and environmental responsibility across our value chain.",
        "Industry-specific renewable electricity target becomes generic vision language.",
    ),
    RewriteCase(
        14,
        "Industry-specific metric -> industry-agnostic vision",
        "The bank financed $5 billion in green bonds and verified 100% of eligible sustainable loans in 2023.",
        "The bank supports stakeholder engagement and responsible business practices in sustainable finance.",
        "Quantified finance metric and verification are rewritten as broad business practice.",
    ),
    RewriteCase(
        15,
        "Industry-specific metric -> industry-agnostic vision",
        "Fleet fuel carbon intensity improved 9% by 2024 through renewable diesel and route optimization.",
        "We continue to improve environmental responsibility and support a better future.",
        "Operational performance metrics become high-level responsibility language.",
    ),
]


def interpret_case(keyword_drop: int, semantic_drop: int, specificity_change: int) -> str:
    """Create a concise deterministic interpretation for each rewrite case."""
    if keyword_drop > semantic_drop and specificity_change > 0:
        return "Surface keyword signal fell more than semantic signal, and the rewrite became vaguer."
    if keyword_drop > semantic_drop:
        return "Surface keyword signal fell more than semantic signal after rewriting."
    if specificity_change > 0:
        return "The rewrite became vaguer even though keyword and semantic drops were similar."
    return "This case shows limited signal loss under the deterministic test."


def build_results(cases: list[RewriteCase]) -> pd.DataFrame:
    """Compute adversarial rewrite comparison rows."""
    rows = []
    for case in cases:
        keyword_original = keyword_signal(case.original_text)
        keyword_rewritten = keyword_signal(case.ai_rewritten_text)
        semantic_original = semantic_signal(case.original_text)
        semantic_rewritten = semantic_signal(case.ai_rewritten_text)
        measurable_original = measurable_commitment(case.original_text)
        measurable_rewritten = measurable_commitment(case.ai_rewritten_text)
        vague_original = vague_commitment(case.original_text)
        vague_rewritten = vague_commitment(case.ai_rewritten_text)

        keyword_drop = keyword_original - keyword_rewritten
        semantic_drop = semantic_original - semantic_rewritten
        gap_original = vague_original - measurable_original
        gap_rewritten = vague_rewritten - measurable_rewritten
        gap_change = gap_rewritten - gap_original

        rows.append({
            "case_id": case.case_id,
            "category": case.category,
            "original_text": case.original_text,
            "ai_rewritten_text": case.ai_rewritten_text,
            "expected_issue": case.expected_issue,
            "keyword_signal_original": keyword_original,
            "keyword_signal_rewritten": keyword_rewritten,
            "semantic_signal_original": semantic_original,
            "semantic_signal_rewritten": semantic_rewritten,
            "measurable_commitment_original": measurable_original,
            "measurable_commitment_rewritten": measurable_rewritten,
            "vague_commitment_original": vague_original,
            "vague_commitment_rewritten": vague_rewritten,
            "keyword_signal_drop": keyword_drop,
            "semantic_signal_drop": semantic_drop,
            "specificity_gap_original": gap_original,
            "specificity_gap_rewritten": gap_rewritten,
            "specificity_gap_change": gap_change,
            "robustness_interpretation": interpret_case(keyword_drop, semantic_drop, gap_change),
        })
    return pd.DataFrame(rows)


def save_comparison_chart(df: pd.DataFrame, output_path: str) -> None:
    """Save grouped bar chart with average robustness summary metrics."""
    labels = [
        "Avg keyword\nsignal drop",
        "Avg semantic\nsignal drop",
        "Avg specificity\ngap change",
    ]
    values = [
        df["keyword_signal_drop"].mean(),
        df["semantic_signal_drop"].mean(),
        df["specificity_gap_change"].mean(),
    ]
    colors = ["#C94C3A", "#2E7D32", "#D99A2B"]

    fig, ax = plt.subplots(figsize=(9, 5.4), facecolor="white")
    bars = ax.bar(labels, values, color=colors, width=0.56)
    ax.set_title("AI-Rewritten ESG Disclosure Robustness Test", fontsize=15, weight="bold", pad=16)
    ax.set_ylabel("Average signal change")
    ax.axhline(0, color="#5F7165", linewidth=0.9)
    ax.grid(axis="y", color="#DDEBDD", linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)

    ymax = max(values + [0])
    ymin = min(values + [0])
    span = max(ymax - ymin, 1)
    ax.set_ylim(ymin - 0.18 * span, ymax + 0.36 * span)

    for bar, value in zip(bars, values):
        offset = 0.04 * span if value >= 0 else -0.08 * span
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + offset,
            f"{value:.2f}",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=11,
            weight="bold",
            color="#173D2A",
        )

    ax.annotate(
        "Larger keyword drop than semantic drop -> keyword-only models are fragile",
        xy=(0.5, 0.96),
        xycoords="axes fraction",
        ha="center",
        va="center",
        fontsize=10.5,
        color="#34453A",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#FFF8E8", edgecolor="#D99A2B"),
    )

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#BFD3C4")
    ax.spines["bottom"].set_color("#BFD3C4")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    df = build_results(CASES)
    df.to_csv(CSV_PATH, index=False)
    save_comparison_chart(df, PNG_PATH)

    avg_keyword_drop = df["keyword_signal_drop"].mean()
    avg_semantic_drop = df["semantic_signal_drop"].mean()
    avg_gap_change = df["specificity_gap_change"].mean()
    interpretation = (
        "Keyword signal dropped more than semantic signal on average, suggesting keyword-only models are more fragile to AI rewriting."
        if avg_keyword_drop > avg_semantic_drop
        else "Keyword signal did not drop more than semantic signal on average; robustness is mixed in this deterministic test."
    )

    print("=== Adversarial Rewrite Test Summary ===")
    print(f"Total cases: {len(df)}")
    print(f"Average keyword signal drop:    {avg_keyword_drop:.2f}")
    print(f"Average semantic signal drop:   {avg_semantic_drop:.2f}")
    print(f"Average specificity gap change: {avg_gap_change:.2f} (positive = vaguer after rewrite)")
    print(f"Interpretation: {interpretation}")


if __name__ == "__main__":
    main()
