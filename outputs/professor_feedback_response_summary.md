# Professor Feedback Response Summary

## English Interpretation

The v2 label has only 9 positive cases. With such a small positive sample, cross-validation estimates for the indirect text-only model are highly unstable, and the resulting AUC is near or below random. The correct reading is that indirect text features alone carry no reliable signal for the stricter v2 label — which is consistent with our ablation finding that text-only signal is limited and must be combined with external evidence. The v2 experiment is a governance check, not a candidate production model.

## 中文說明

v2 label 僅 9 個正樣本，間接文字特徵在極小樣本下的交叉驗證極不穩定，AUC 接近甚至低於隨機。正確解讀是：單靠間接文字特徵無法可靠預測更嚴格的 v2 label——這與 ablation 結論一致：純文字訊號有限，必須結合外部證據。v2 是治理檢查，不是候選正式模型。

## V2 Clean Model Metrics

| model_name             | feature_group                                                                             |   n_features | features_used                                                                                                      |   roc_auc_mean |   roc_auc_std |   pr_auc_mean |   pr_auc_std |   f1_mean |   f1_std |   precision_mean |   recall_mean |   accuracy_mean | confusion_matrix     |   positive_class_support |   negative_class_support | interpretation                                                                                                                                   | governance_disclaimer                                                                                                                                                                  |
|:-----------------------|:------------------------------------------------------------------------------------------|-------------:|:-------------------------------------------------------------------------------------------------------------------|---------------:|--------------:|--------------:|-------------:|----------:|---------:|-----------------:|--------------:|----------------:|:---------------------|-------------------------:|-------------------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| v2_governance_baseline | DummyClassifier baseline                                                                  |            0 | No features (DummyClassifier)                                                                                      |         0.4778 |        0.0039 |        0.035  |       0.0077 |    0      |   0      |           0      |           0   |          0.9222 | [[237, 11], [9, 0]]  |                        9 |                      248 | Governance baseline. Any useful v2 model should perform better than this naive reference.                                                        | AI Decision Support Only: This system is designed for ESG risk triage and review prioritization. It does not legally determine greenwashing, credit approval, or investment exclusion. |
| v2_text_model          | Primary clean model: indirect text features only                                          |            5 | hedge_score, pct_numeric, vague_to_specific_ratio, esg_topic_breadth, text_length                                  |         0.2764 |        0.0763 |        0.0392 |       0.009  |    0.0154 |   0.0308 |           0.0083 |           0.1 |          0.4897 | [[125, 123], [8, 1]] |                        9 |                      248 | The v2 label has only 9 positive cases. With such a small positive sample, cross-validation estimates for the indirect text-only model are highly unstable, and the resulting AUC is near or below random. The correct reading is that indirect text features alone carry no reliable signal for the stricter v2 label — which is consistent with our ablation finding that text-only signal is limited and must be combined with external evidence. The v2 experiment is a governance check, not a candidate production model.               | AI Decision Support Only: This system is designed for ESG risk triage and review prioritization. It does not legally determine greenwashing, credit approval, or investment exclusion. |
| v2_explainable_model   | Sensitivity analysis: explainable text model with potentially label-related Talk features |            8 | E_density, S_density, G_density, hedge_score, pct_numeric, vague_to_specific_ratio, esg_topic_breadth, text_length |         0.7336 |        0.082  |        0.2194 |       0.1954 |    0.1271 |   0.0777 |           0.0732 |           0.5 |          0.7357 | [[184, 64], [4, 5]]  |                        9 |                      248 | Sensitivity analysis only. Includes E/S/G text features that may be related to label construction; do not present it as the primary clean model. | AI Decision Support Only: This system is designed for ESG risk triage and review prioritization. It does not legally determine greenwashing, credit approval, or investment exclusion. |

## Cross-Source Top Cases

| ticker   | company_name                     | sector             |   final_inconsistency_index | review_priority_tier   |
|:---------|:---------------------------------|:-------------------|----------------------------:|:-----------------------|
| EFX      | Equifax, Incorporated            | Industrials        |                      0.7994 | High priority          |
| WFC      | Wells Fargo & Co.                | Financial Services |                      0.7634 | High priority          |
| NRG      | Nrg Energy, Inc.                 | Utilities          |                      0.7574 | High priority          |
| MMM      | 3m Company                       | Industrials        |                      0.7534 | High priority          |
| WYNN     | Wynn Resorts Ltd                 | Consumer Cyclical  |                      0.7446 | High priority          |
| CVX      | Chevron Corporation              | Energy             |                      0.7332 | High priority          |
| ATO      | Atmos Energy Corporation         | Utilities          |                      0.7266 | High priority          |
| OXY      | Occidental Petroleum Corporation | Energy             |                      0.7219 | High priority          |

## V1 Ranking Signal Used for Overlap

V1 ranking signal: saved LightGBM model probability from models/lgbm_best.pkl.

## Governance Disclaimer

AI Decision Support Only: This system is designed for ESG risk triage and review prioritization. It does not legally determine greenwashing, credit approval, or investment exclusion.
