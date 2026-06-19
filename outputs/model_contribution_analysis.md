# Model Contribution Analysis and Response to Professor Feedback

## Purpose

This analysis responds to professor feedback that the current Full Model may rely strongly on already-processed external ESG ranking features, such as `sector_rank`, `planet_friendly_business`, and `honest_fair_business`.

The goal is not to replace the production model. The goal is to separate the contribution of external ESG ranking signals from sustainability report text-derived signals.

## Governance Disclaimer

AI Decision Support Only: This model is designed for ESG risk triage and review prioritization. It does not legally determine greenwashing, credit approval, or investment exclusion.

## Model Versions

- **Full Model:** all original 14 features.
- **External-only Model:** Walk / external ranking features only.
- **Text-only Model:** sustainability report Talk features only.
- **No-ranking Model:** removes `sector_rank`, `planet_friendly_business`, and `honest_fair_business`.
- **Strict No-Walk-Proxy Model:** removes all Walk-proxy and context features. In this project, it is identical to the Text-only Model if the feature list matches.

## Results Table

| model_version              | feature_group                              |   n_features |   roc_auc_mean |   pr_auc_mean |   f1_mean |   precision_mean |   recall_mean |   accuracy_mean | interpretation                                                                                                                                                                                                  |
|:---------------------------|:-------------------------------------------|-------------:|---------------:|--------------:|----------:|-----------------:|--------------:|----------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Full Model                 | Talk + Walk / External + Context           |           14 |         0.9413 |        0.7223 |    0.6474 |           0.6722 |        0.6833 |          0.9419 | Reference risk triage model. Strong performance should be interpreted cautiously because it combines text features with external ESG ranking signals.                                                           |
| External-only Model        | Walk / External ranking features           |            5 |         0.754  |        0.3096 |    0.2751 |           0.4019 |        0.25   |          0.8754 | External ESG ranking features explain a meaningful share of performance. This supports repositioning the system as ESG risk triage and rating inconsistency consolidation.                                      |
| Text-only Model            | Sustainability report Talk features        |            9 |         0.6635 |        0.1582 |    0.1844 |           0.1321 |        0.3167 |          0.8053 | Text-derived sustainability report features retain modest independent signal. The text model captures about 37% of the Full Model ROC-AUC lift over random ranking.                                             |
| No-ranking Model           | No sector_rank / integrity ranking outputs |           11 |         0.7316 |        0.2322 |    0.2161 |           0.2    |        0.2667 |          0.8756 | Removing sector_rank, planet_friendly_business, and honest_fair_business changes ROC-AUC by -0.210 versus the Full Model. This shows how much performance remains after removing the most rating-like features. |
| Strict No-Walk-Proxy Model | Talk-only strict ablation                  |            9 |         0.6635 |        0.1582 |    0.1844 |           0.1321 |        0.3167 |          0.8053 | This feature set is identical to the Text-only Model in the current 14-feature design, because all Walk-proxy and context features are removed.                                                                 |

## Interpretation

- External ESG ranking features account for a meaningful, but not complete, share of performance.
- The Text-only Model shows modest independent signal from sustainability report language.
- The No-ranking Model ROC-AUC is 0.732, a drop of 0.210 from the Full Model.
- The most defensible positioning is ESG risk triage and rating inconsistency consolidation, not legal greenwashing detection.

## Final Wording for Report / Slides

The Full Model may partially reflect the structure of external ESG rating systems. Therefore, we reposition the dashboard as an ESG risk triage and rating inconsistency consolidation tool. The system helps analysts prioritize companies for further ESG due diligence, but it does not legally determine greenwashing, credit approval, or investment exclusion.
