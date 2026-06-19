# Enhanced Text Feature Validation

## Purpose

This additive validation tests one primary greenwashing text feature and minimal supporting features. It does not use sector ranking, ESG rating scores, integrity scores, or Sustainalytics-style circular features.

## Baseline Text Model

- Existing text features only ROC-AUC: 0.7775
- Existing text features only PR-AUC: 0.2136

## Validation Table

| feature                       | role       |   spearman_corr | corr_sign   | fold_signs   | sign_stable_all_5_folds   |   baseline_roc_auc |   candidate_roc_auc |   delta_roc_auc |   baseline_pr_auc |   candidate_pr_auc |   delta_pr_auc | decision   | rationale                                                  | description                                                         |
|:------------------------------|:-----------|----------------:|:------------|:-------------|:--------------------------|-------------------:|--------------------:|----------------:|------------------:|-------------------:|---------------:|:-----------|:-----------------------------------------------------------|:--------------------------------------------------------------------|
| promotional_to_specific_ratio | PRIMARY    |          0.2371 | +           | +,-,-,+,-    | False                     |             0.7775 |              0.7793 |          0.0018 |            0.2136 |             0.2002 |        -0.0134 | DROP       | Dropped because sign was unstable or OOF PR-AUC decreased. | Promotional language density divided by specific evidence density.  |
| sector_adj_promotional        | SUPPORTING |          0.2311 | +           | +,+,-,+,+    | False                     |             0.7775 |              0.7926 |          0.015  |            0.2136 |             0.2269 |         0.0134 | DROP       | Dropped because sign was unstable or OOF PR-AUC decreased. | Promotional density adjusted by train-fold sector median.           |
| promo_density                 | SUPPORTING |          0.2353 | +           | +,-,+,+,-    | False                     |             0.7775 |              0.7771 |         -0.0004 |            0.2136 |             0.2122 |        -0.0014 | DROP       | Dropped because sign was unstable or OOF PR-AUC decreased. | Promotional ESG claim density.                                      |
| specific_density              | SUPPORTING |         -0.0182 | -           | +,-,+,+,-    | False                     |             0.7775 |              0.7543 |         -0.0232 |            0.2136 |             0.2177 |         0.0041 | DROP       | Dropped because sign was unstable or OOF PR-AUC decreased. | Specific ESG evidence density.                                      |
| concreteness_ratio            | SUPPORTING |         -0.0277 | -           | +,+,-,+,-    | False                     |             0.7775 |              0.7747 |         -0.0029 |            0.2136 |             0.2168 |         0.0033 | DROP       | Dropped because sign was unstable or OOF PR-AUC decreased. | Target, quantified, and framework density divided by vague density. |

## Kept Features

No candidate features passed the keep rule.

## Dropped / Flagged Features

promotional_to_specific_ratio, sector_adj_promotional, promo_density, specific_density, concreteness_ratio

## Governance Disclaimer

These features support ESG risk screening and due diligence prioritization. They do not legally determine greenwashing, credit approval, or investment exclusion.
