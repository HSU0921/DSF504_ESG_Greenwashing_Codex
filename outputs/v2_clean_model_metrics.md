# V2 Clean Model Metrics

These models test whether a cleaner v2 label can be predicted without simply copying external ESG rating outputs.

| model_name             | feature_group                                                                             |   n_features |   roc_auc_mean |   pr_auc_mean |   f1_mean |   precision_mean |   recall_mean |   accuracy_mean | interpretation                                                                                                                                   |
|:-----------------------|:------------------------------------------------------------------------------------------|-------------:|---------------:|--------------:|----------:|-----------------:|--------------:|----------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------|
| v2_governance_baseline | DummyClassifier baseline                                                                  |            0 |         0.4778 |        0.035  |    0      |           0      |           0   |          0.9222 | Governance baseline. Any useful v2 model should perform better than this naive reference.                                                        |
| v2_text_model          | Primary clean model: indirect text features only                                          |            5 |         0.2764 |        0.0392 |    0.0154 |           0.0083 |           0.1 |          0.4897 | The v2 label has only 9 positive cases. With such a small positive sample, cross-validation estimates for the indirect text-only model are highly unstable, and the resulting AUC is near or below random. The correct reading is that indirect text features alone carry no reliable signal for the stricter v2 label — which is consistent with our ablation finding that text-only signal is limited and must be combined with external evidence. The v2 experiment is a governance check, not a candidate production model.               |
| v2_explainable_model   | Sensitivity analysis: explainable text model with potentially label-related Talk features |            8 |         0.7336 |        0.2194 |    0.1271 |           0.0732 |           0.5 |          0.7357 | Sensitivity analysis only. Includes E/S/G text features that may be related to label construction; do not present it as the primary clean model. |

## Interpretation

The v2 label has only 9 positive cases. With such a small positive sample, cross-validation estimates for the indirect text-only model are highly unstable, and the resulting AUC is near or below random. The correct reading is that indirect text features alone carry no reliable signal for the stricter v2 label — which is consistent with our ablation finding that text-only signal is limited and must be combined with external evidence. The v2 experiment is a governance check, not a candidate production model.

v2 label 僅 9 個正樣本，間接文字特徵在極小樣本下的交叉驗證極不穩定，AUC 接近甚至低於隨機。正確解讀是：單靠間接文字特徵無法可靠預測更嚴格的 v2 label——這與 ablation 結論一致：純文字訊號有限，必須結合外部證據。v2 是治理檢查，不是候選正式模型。

AI Decision Support Only: This system is designed for ESG risk triage and review prioritization. It does not legally determine greenwashing, credit approval, or investment exclusion.
