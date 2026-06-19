# Cross-Source Inconsistency Index

## Purpose

This index is a transparent review-priority ranking. It is not optimized for AUC and does not legally determine greenwashing.

## Formula

`final_inconsistency_index = 0.35*talk + 0.30*walk + 0.20*integrity + 0.15*contradiction`

Weights are configurable in `INDEX_WEIGHTS`.

## Component Meaning

- Talk Intensity Score: higher ESG disclosure intensity.
- Walk Weakness Score: weaker external ESG risk evidence.
- Integrity Weakness Score: weaker integrity or governance signal.
- Cross-Source Contradiction Score: disagreement across the Talk, Walk, and integrity sources.

## V1 Baseline Context

| model_setting              |   mean_roc_auc |   mean_pr_auc |   mean_f1 |
|:---------------------------|---------------:|--------------:|----------:|
| Full Model                 |         0.9413 |        0.7223 |    0.6474 |
| External-only Model        |         0.754  |        0.3096 |    0.2751 |
| Text-only Model            |         0.6635 |        0.1582 |    0.1844 |
| No-ranking Model           |         0.7316 |        0.2322 |    0.2161 |
| Strict No-Walk-Proxy Model |         0.6635 |        0.1582 |    0.1844 |

## V1 Ranking Signal Used for Overlap

V1 ranking signal: saved LightGBM model probability from models/lgbm_best.pkl.

## Top Review-Priority Cases

| ticker   | company_name                     | sector             |   final_inconsistency_index | review_priority_tier   | explanation                                                                                                                            |
|:---------|:---------------------------------|:-------------------|----------------------------:|:-----------------------|:---------------------------------------------------------------------------------------------------------------------------------------|
| EFX      | Equifax, Incorporated            | Industrials        |                      0.7994 | High priority          | Flagged for high ESG disclosure intensity, weak external ESG evidence, weak integrity signal; requires further ESG due diligence.      |
| WFC      | Wells Fargo & Co.                | Financial Services |                      0.7634 | High priority          | Flagged for high ESG disclosure intensity, weak external ESG evidence, weak integrity signal; requires further ESG due diligence.      |
| NRG      | Nrg Energy, Inc.                 | Utilities          |                      0.7574 | High priority          | Flagged for high ESG disclosure intensity, weak external ESG evidence, weak integrity signal; requires further ESG due diligence.      |
| MMM      | 3m Company                       | Industrials        |                      0.7534 | High priority          | Flagged for high ESG disclosure intensity, weak external ESG evidence, weak integrity signal; requires further ESG due diligence.      |
| WYNN     | Wynn Resorts Ltd                 | Consumer Cyclical  |                      0.7446 | High priority          | Flagged for high ESG disclosure intensity, weak external ESG evidence, weak integrity signal; requires further ESG due diligence.      |
| CVX      | Chevron Corporation              | Energy             |                      0.7332 | High priority          | Flagged for high ESG disclosure intensity, weak external ESG evidence, weak integrity signal; requires further ESG due diligence.      |
| ATO      | Atmos Energy Corporation         | Utilities          |                      0.7266 | High priority          | Flagged for high ESG disclosure intensity, weak external ESG evidence; requires further ESG due diligence.                             |
| OXY      | Occidental Petroleum Corporation | Energy             |                      0.7219 | High priority          | Flagged for weak external ESG evidence, weak integrity signal; requires further ESG due diligence.                                     |
| CB       | Chubb Limited                    | Financial Services |                      0.7166 | High priority          | Flagged for high ESG disclosure intensity, weak external ESG evidence, cross-source inconsistency; requires further ESG due diligence. |
| CNP      | Centerpoint Energy, Inc.         | Utilities          |                      0.7144 | High priority          | Flagged for high ESG disclosure intensity, weak external ESG evidence; requires further ESG due diligence.                             |
| DLTR     | Dollar Tree Inc.                 | Consumer Defensive |                      0.7107 | High priority          | Flagged for high ESG disclosure intensity, weak integrity signal; requires further ESG due diligence.                                  |
| DVN      | Devon Energy Corporation         | Energy             |                      0.7098 | High priority          | Flagged for high ESG disclosure intensity, weak external ESG evidence, weak integrity signal; requires further ESG due diligence.      |

## Overlap Analysis

| comparison                         |   count | tickers                                                                       | interpretation                                                                                                         |
|:-----------------------------------|--------:|:------------------------------------------------------------------------------|:-----------------------------------------------------------------------------------------------------------------------|
| overlap_v1_top_and_index_high      |      15 | ATO, AVGO, CB, CNP, CTRA, CVX, ITW, MMM, NDSN, NRG, PNC, PNW, TSLA, WFC, WYNN | Non-overlap is useful because it shows the new index is not merely copying the v1 model or a single ESG rating source. |
| overlap_v1_top_and_v2_positive     |       9 | AVGO, CVX, ITW, MMM, NRG, PCAR, TSLA, WFC, WYNN                               | Non-overlap is useful because it shows the new index is not merely copying the v1 model or a single ESG rating source. |
| overlap_index_high_and_v2_positive |       8 | AVGO, CVX, ITW, MMM, NRG, TSLA, WFC, WYNN                                     | Non-overlap is useful because it shows the new index is not merely copying the v1 model or a single ESG rating source. |
| v1_top_not_index                   |      11 | AAPL, ABT, AIZ, AMZN, CINF, EQIX, EXR, MCHP, MTB, PCAR, SPG                   | Non-overlap is useful because it shows the new index is not merely copying the v1 model or a single ESG rating source. |
| index_high_not_v1                  |      11 | CMS, DLTR, DUK, DVN, EFX, FTV, HBAN, OXY, STT, VLO, XOM                       | Non-overlap is useful because it shows the new index is not merely copying the v1 model or a single ESG rating source. |
| v2_positive_not_v1                 |       0 |                                                                               | Non-overlap is useful because it shows the new index is not merely copying the v1 model or a single ESG rating source. |

## Interpretation

Non-overlap between v1 top-risk companies, v2 clean-label positives, and index high-priority companies is useful. It shows the index is not simply copying the v1 supervised model or one external ESG rating source.

AI Decision Support Only: This system is designed for ESG risk triage and review prioritization. It does not legally determine greenwashing, credit approval, or investment exclusion.
