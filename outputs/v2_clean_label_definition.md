# V2 Clean Label Definition

## Definition

`high_risk_v2 = High Talk AND Multi-source Weak Evidence`

High Talk uses sustainability-report text-derived disclosure intensity:

- `talk_total_density` in the top 33% of all companies.

Multi-source Weak Evidence requires both:

- Sustainalytics weakness: weak sector-relative ESG risk evidence from `Total ESG Risk score` or `sector_rank`.
- Integrity weakness: weak integrity signal from `planet_friendly_business` and `honest_fair_business`.

This label is cleaner than v1 because it does not rely on a single external rating signal alone. It requires weak evidence from more than one source.

## Label Counts

- Positive count: 9
- Negative count: 248
- Positive rate: 3.5%

## Sector Distribution of V2 Positives

| sector             |   v2_positive_count |
|:-------------------|--------------------:|
| Industrials        |                   3 |
| Consumer Cyclical  |                   2 |
| Financial Services |                   1 |
| Energy             |                   1 |
| Utilities          |                   1 |
| Technology         |                   1 |

## Top V2 Positive Cases

| ticker   | company_name             | sector             |   talk_percentile |   total_esg_risk_sector_percentile |   integrity_weakness_percentile |
|:---------|:-------------------------|:-------------------|------------------:|-----------------------------------:|--------------------------------:|
| ITW      | Illinois Tool Works Inc. | Industrials        |          0.933852 |                           0.720588 |                        0.715953 |
| AVGO     | Broadcom Inc.            | Technology         |          0.929961 |                           0.875    |                        0.846304 |
| WYNN     | Wynn Resorts Ltd         | Consumer Cyclical  |          0.92607  |                           0.892857 |                        0.846304 |
| NRG      | Nrg Energy, Inc.         | Utilities          |          0.914397 |                           0.923077 |                        0.715953 |
| CVX      | Chevron Corporation      | Energy             |          0.793774 |                           0.8      |                        0.955253 |
| PCAR     | Paccar Inc               | Industrials        |          0.754864 |                           0.794118 |                        0.922179 |
| TSLA     | Tesla, Inc.              | Consumer Cyclical  |          0.727626 |                           0.857143 |                        0.846304 |
| WFC      | Wells Fargo & Co.        | Financial Services |          0.688716 |                           1        |                        0.922179 |
| MMM      | 3m Company               | Industrials        |          0.680934 |                           0.970588 |                        0.986381 |

## Governance Disclaimer

AI Decision Support Only: This system is designed for ESG risk triage and review prioritization. It does not legally determine greenwashing, credit approval, or investment exclusion.
