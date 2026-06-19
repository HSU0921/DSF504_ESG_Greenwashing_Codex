# Professor Feedback Revision Pack

## 1. Why These Files Were Added

These files were added to document the Day 1 professor-feedback revision for the DSF504 Team 5 ESG Greenwashing Risk Scoring System. The goal is to keep the original v1 supervised model as a benchmark while adding a more transparent and auditable response to concerns about external ESG rating dependence.

## 2. Professor's Concern

The professor's concern is that the original v1 model may rely too much on external ESG rating-related features. Because the v1 label and some strong predictors are partly tied to external ESG risk structures, a high AUC alone may not prove that the model independently detects greenwashing from sustainability report text.

## 3. What We Did in Response

- V1 self-audit / ablation: We compared Full Model, External-only, Text-only, No-ranking, and Strict No-Walk-Proxy settings.
- V2 clean-label experiment: We created `high_risk_v2 = High Talk AND Multi-source Weak Evidence`, requiring both disclosure intensity and weak evidence from more than one source.
- Cross-Source Inconsistency Index: We created a transparent rule-based review-priority index combining Talk, Walk, integrity, and cross-source contradiction scores.

## 4. Key Results

- V2 positives: 9 / 257, 3.5%.
- `v2_text_model` ROC-AUC: 0.2764.
- `v2_explainable_model` ROC-AUC: 0.7336.
- Cross-source index top cases include EFX, WFC, NRG, MMM, WYNN, CVX, ATO, OXY, CB, CNP.
- Overlap analysis: 15 overlap, 11 v1-only, 11 index-only.

## 5. Governance Conclusion

The final contribution is not high AUC alone, but an auditable ESG review-priority system combining supervised modeling, ablation governance, clean-label testing, and cross-source inconsistency analysis.

This system supports ESG risk triage and review prioritization. It does not legally determine greenwashing, credit approval, or investment exclusion.
