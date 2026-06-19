# Holdout Robustness Validation

## Design

- Split real companies before model fitting: 70% real training companies and 30% real holdout companies.
- The holdout set is not used in Optuna tuning, model selection, synthetic augmentation, or threshold selection.
- Synthetic ESG disclosure samples are added only to the training set.
- Professor suggested increasing company-level training samples.
- DAX company documents are converted into weak-labeled training augmentation samples when the DAX dataset is available.
- Greenwashing_Score_Data.xlsx is used to add company-level score-label weak samples when available.
- SEC EDGAR files are inspected; they are used only if full filing or risk-factor text is available.
- Additional Kaggle-style ESG / greenwashing files are converted into weak-labeled company samples when available.
- DAX, SEC EDGAR, and additional Kaggle samples are weak-labeled training augmentation samples.
- Weak labels are not legal ground truth.
- DAX, SEC EDGAR, and Kaggle samples are used only for training augmentation.
- The model is evaluated only on real holdout companies.
- No synthetic, DAX, SEC EDGAR, or Kaggle samples enter the test set.
- Walk / External / Context fields for synthetic, DAX, SEC EDGAR, or Kaggle samples are filled with real training-set medians; the real holdout set remains unchanged.
- Greenwashing score records without report text use conservative content placeholders and are reported as score-label augmentation, not disclosure-text augmentation.
- The output is a risk screening signal, not a legal determination of greenwashing.

## DAX Weak-Label Augmentation Status

DAX weak-label generation created 26 company-level training samples from esg_documents_for_dax_companies.csv. DAX distribution-aware weak labeling produced 13 high-risk and 13 low-risk company samples; 13 ambiguous companies were skipped. SDG reference file found (sdg_descriptions_with_targetsText.csv) and treated only as keyword context; it was not used as company-level training data. Columns: id, name, description, targets, targets_json_array, progress.

## DAX Augmentation Interpretation

DAX augmentation did not improve holdout PR-AUC versus synthetic-only augmentation. This is reported honestly as a limitation of weak-label external augmentation.

## Additional External Augmentation Status

- External files found: Greenwashing_Score_Data.xlsx, SEC_EDGAR_10-K_METADATA.json, SEC_EDGAR_10-K_MASTER_DATA_SAMPLE.csv, SEC_EDGAR_10-K_MASTER_DATA.parquet
- Combined external weak-labeled company samples: 204 total (26 DAX, 178 Kaggle / SEC / score-label samples). Greenwashing_Score_Data.xlsx: created 178 company-level score-label samples from latest company years (103 high-risk, 75 low-risk); skipped 37 ambiguous companies with 0.50 < GW_SCORE < 0.75. This is company-level score-label augmentation, not disclosure-text augmentation. SEC_EDGAR_10-K_METADATA.json: inspected metadata keys ['keyword', 'total_files_processed', 'data_type', 'unique_entities', 'filing_date_range']; not used as training samples. SEC_EDGAR_10-K_MASTER_DATA_SAMPLE.csv: inspected 10000 rows and columns ['cik', 'company_name', 'form_type', 'filing_date', 'filename', 'source_filename', 'data_collection_date', 'is_major_filing']; metadata/index only, no recognized full filing text column, so no labels were created. SEC_EDGAR_10-K_MASTER_DATA.parquet: inspected 476226 rows and columns ['cik', 'company_name', 'form_type', 'filing_date', 'filename', 'source_filename', 'data_collection_date', 'is_major_filing']; metadata/index only, no recognized full filing text column, so no labels were created.

## All-External Augmentation Interpretation

All-external augmentation did not improve PR-AUC or F1 versus synthetic-only augmentation. This is reported honestly as a limitation of weak-label external company augmentation.

## Holdout Results

| model_version                                                      |   train_real_n |   train_synthetic_n |   train_dax_labeled_n |   train_external_labeled_n |   holdout_n_real_companies |   holdout_positive_count |   holdout_negative_count |   roc_auc |   pr_auc |     f1 |   precision |   recall | confusion_matrix   | evaluation_note                                                                  |
|:-------------------------------------------------------------------|---------------:|--------------------:|----------------------:|---------------------------:|---------------------------:|-------------------------:|-------------------------:|----------:|---------:|-------:|------------:|---------:|:-------------------|:---------------------------------------------------------------------------------|
| original_real_training_only                                        |            179 |                   0 |                     0 |                          0 |                         78 |                        6 |                       72 |    0.8924 |   0.5708 | 0.5    |      0.5    |   0.5    | [[69, 3], [3, 3]]  | Evaluated only on the real-company holdout set; no synthetic samples in holdout. |
| augmented_training_with_synthetic_disclosures                      |            179 |                  24 |                     0 |                          0 |                         78 |                        6 |                       72 |    0.9421 |   0.7663 | 0.8333 |      0.8333 |   0.8333 | [[71, 1], [1, 5]]  | Evaluated only on the real-company holdout set; no synthetic samples in holdout. |
| augmented_training_with_synthetic_and_dax_labeled_company_samples  |            179 |                  24 |                    26 |                         26 |                         78 |                        6 |                       72 |    0.963  |   0.6001 | 0.6667 |      0.6667 |   0.6667 | [[70, 2], [2, 4]]  | Evaluated only on the real-company holdout set; no synthetic samples in holdout. |
| augmented_training_with_synthetic_and_all_external_company_samples |            179 |                  24 |                    26 |                        204 |                         78 |                        6 |                       72 |    0.9514 |   0.6319 | 0.6    |      0.75   |   0.5    | [[71, 1], [3, 3]]  | Evaluated only on the real-company holdout set; no synthetic samples in holdout. |

## External Case Study

| company   |        cik | source_url                                                                           | case_study_model                                                   |   model_flagged_greenwashing_risk_score | retrieval_note                                             | case_study_note                                                            | governance_disclaimer                                                                                                                                   |
|:----------|-----------:|:-------------------------------------------------------------------------------------|:-------------------------------------------------------------------|----------------------------------------:|:-----------------------------------------------------------|:---------------------------------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------|
| NVIDIA    | 0001045810 | https://www.sec.gov/Archives/edgar/data/1045810/000104581026000021/nvda-20260125.htm | augmented_training_with_synthetic_and_all_external_company_samples |                                  0.0227 | Fetched latest NVIDIA 10-K from SEC EDGAR submissions API. | External company case study only; not formal proof or legal determination. | This experiment reports model-flagged greenwashing risk as a risk screening signal. It does not legally determine greenwashing or corporate wrongdoing. |

## Governance Disclaimer

This experiment reports model-flagged greenwashing risk as a risk screening signal. It does not legally determine greenwashing or corporate wrongdoing.
