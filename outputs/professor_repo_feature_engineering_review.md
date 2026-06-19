# Professor Repo Feature Engineering Review

## 1. Executive Summary

This review compares the professor's ESG feature engineering design with the current local DSF504 ESG Greenwashing Risk Scoring project.

The current project should keep the existing 14-feature baseline for the final presentation. The design is already aligned with the core business framing:

**High Talk + Weak Walk = greenwashing risk screening signal**

The professor repo provides useful ideas, especially TF-IDF phrase features, richer text statistics, interaction features, and stricter train-only fitting patterns. However, most of these should be treated as future improvements rather than added immediately, because the current dashboard, model results, and explanation flow already depend on a stable and interpretable 14-feature design.

Chinese support line:
目前模型應維持 14 個核心特徵，作為風險篩選工具，而不是法律指控工具。

## 2. File Existence Check

| File | Expected Path | Status |
|---|---|---|
| Professor reference file | `/Users/xuwanzhu/Desktop/dsf504-ml-platform/use_case_F_esg/03_feature_engineering.py` | Not found locally at the requested path |
| My local project file | `/Users/xuwanzhu/Desktop/DSF504_ESG_Greenwashing_Codex/src/feature_engineering.py` | Found |

Because the local professor reference file was not available at the requested Desktop path, this comparison uses the official GitHub reference file:

`https://github.com/kyw1000/dsf504-ml-platform/blob/main/use_case_F_esg/03_feature_engineering.py`

No local professor repo file was created or copied.

## 3. Professor Repo Feature Inventory

The professor's ESG feature engineering design is broader and more general-purpose. It creates or proposes the following feature families:

### TF-IDF Features

- Uses `TfidfVectorizer`.
- Uses unigram and bigram text features.
- Selects top TF-IDF features with chi-square feature selection.
- This is useful for discovering which disclosure phrases are associated with the target.

### Text Statistics

- Text length.
- Word count.
- Average word length.
- These features capture report size and language complexity.

### Claim Density / ESG Keyword Features

- Counts climate-positive or ESG-related keywords.
- Converts keyword counts into a density-style feature.
- This is similar to the idea of measuring how strongly a company "talks" about ESG.

### ESG Gap Features

- Creates gap-based features comparing reported or claimed ESG signals against assessed ESG signals.
- Uses clipped positive gaps so only suspicious positive gaps are emphasized.
- Includes gap variation and maximum gap features.

### Absolute ESG Score Features

- Uses absolute ESG score levels.
- Creates low/high ESG score flags.
- These features are useful in the professor's synthetic or structured use case, but must be handled carefully in this project because raw ESG risk scores are used for label construction.

### Financial Features

- Includes variables such as market cap, revenue, and emissions intensity where available.
- These are not currently part of this project's merged dataset.

### Sector Encoding

- Uses sector one-hot style features.
- Also includes train-only sector risk encoding logic.
- This is useful but requires careful cross-validation handling to avoid leakage.

### Interaction Features

- Creates interaction features such as claim intensity multiplied by ESG gap.
- This idea is conceptually strong for a Talk-Walk project because high ESG claims become more meaningful when paired with weak external evidence.

### Train-Only Fitting Logic

- Fits vectorizers, selectors, and encoders on training data only.
- Applies fitted transformations to validation/test data.
- This is an important leakage-prevention pattern if TF-IDF, selectors, target encoders, or scalers are added later.

## 4. My Local Project Feature Inventory

The local project creates a stable and interpretable baseline feature matrix with 14 model features.

### 9 Talk Features

| Feature | Meaning |
|---|---|
| `E_density` | Environmental keyword density per 1,000 words |
| `S_density` | Social keyword density per 1,000 words |
| `G_density` | Governance keyword density per 1,000 words |
| `hedge_score` | Vague or aspirational language density |
| `pct_numeric` | Fraction of sentences containing numbers or percentages |
| `talk_total_density` | Combined E + S + G disclosure intensity |
| `vague_to_specific_ratio` | Vague language relative to quantitative specificity |
| `text_length` | Sustainability report word count |
| `esg_topic_breadth` | Number of ESG pillars covered |

### 4 Walk / External Features

| Feature | Meaning |
|---|---|
| `e_s_g_dispersion` | Dispersion across E/S/G sub-risk scores |
| `sector_rank` | Sector-relative ESG risk proxy based on E/S/G sub-scores |
| `planet_friendly_business` | External environmental integrity signal |
| `honest_fair_business` | External governance / ethics integrity signal |

### 1 Context Feature

| Feature | Meaning |
|---|---|
| `sector_encoded` | Encoded sector context |

### Greenwashing Label Logic

The label is created from a sector-relative Talk-Walk gap:

- Talk side: `talk_total_density`
- Walk side: `Total ESG Risk score + Controversy Score`
- Label = 1 when the company is in the top third of Talk and the bottom third of Walk within its sector.

This supports the project framing:

**High Talk + Weak Walk = greenwashing risk screening signal**

### Sector-Relative Comparison

The project compares companies within the same sector. This is important because ESG risk levels naturally differ by industry.

### Leakage-Aware Label Construction

The raw columns used to construct the label are not included in the final model feature matrix.

## 5. Feature-by-Feature Comparison Table

| Professor Repo Feature | My Project Equivalent | Add Now? | Reason |
|---|---|---|---|
| TF-IDF unigram/bigram features | No direct equivalent; current project uses curated ESG keyword densities | No | Useful future appendix, but adding sparse TF-IDF now may reduce interpretability and require retraining/CV comparison |
| Top TF-IDF selection with chi-square | No direct equivalent | No | Requires train-only selection inside CV to avoid leakage; better as future robustness experiment |
| Text length | `text_length` | Already covered | Current project already captures report size |
| Word count | `text_length` | Already covered | `text_length` is word count in the local file |
| Average word length | No direct equivalent | Future only | May capture jargon, but less business-clear than current features |
| ESG claim density | `E_density`, `S_density`, `G_density`, `talk_total_density` | Already covered | Current project has a richer E/S/G split instead of one claim-density score |
| Climate keyword list | Partly covered by `E_KEYWORDS` | Future only | Could add terms such as `ghg`, `footprint`, `circular`, or `offset`, but changing dictionaries changes model inputs |
| ESG gap features | Label logic and dashboard Talk-Walk framing | Partly covered | Current project uses sector-relative Talk-Walk label logic rather than reported-vs-assessed synthetic gaps |
| Clipped positive gap | No direct final-X equivalent | Future only | Could be useful for EDA, but should not replace current label logic |
| Absolute ESG score features | Not included in X | Do not add now | Raw ESG risk fields are tied to label construction and may create leakage risk |
| Financial features | No equivalent | Do not add now | Market cap, revenue, and emissions intensity are not available in current merged dataset |
| Sector one-hot encoding | `sector_encoded` | Future only | One-hot is statistically cleaner than label encoding, but requires retraining and validation |
| Sector target/risk encoding | No equivalent | Do not add now | Higher leakage risk unless fitted strictly inside each training fold |
| Claim x gap interaction | No direct equivalent | Future only | Useful idea: `talk_total_density * sector_rank`, but must be tested before adding |
| Gap x emissions interaction | No equivalent | Do not add now | Emissions intensity is unavailable in current dataset |
| Train-only vectorizer/selector fitting | Current deterministic features do not need fitted text vectorizers | Future process rule | Essential if TF-IDF or feature selectors are added later |

## 6. Leakage Risk Check

A runtime-safe check of the current baseline feature matrix produced:

- Feature matrix shape: `257 x 14`
- Final X columns:
  - `E_density`
  - `S_density`
  - `G_density`
  - `hedge_score`
  - `pct_numeric`
  - `talk_total_density`
  - `vague_to_specific_ratio`
  - `text_length`
  - `esg_topic_breadth`
  - `e_s_g_dispersion`
  - `sector_rank`
  - `planet_friendly_business`
  - `honest_fair_business`
  - `sector_encoded`

### Forbidden / Reference Columns

| Column | Appears in final X feature matrix? | How it is used |
|---|---:|---|
| `Total ESG Risk score` | No | Used for label construction and EDA/meta diagnostics, not as model input |
| `Controversy Score` | No | Used for label construction and EDA/meta diagnostics, not as model input |
| `ESG Risk Percentile` | No | Excluded from final X |
| `ESG Risk Level` | No | Excluded from final X |

Conclusion: the current 14-feature baseline does not include the major raw label/reference columns in the model feature matrix.

Important nuance: the model does use safer Walk / External features such as E/S/G sub-score dispersion, sector-relative sub-score rank, and third-party integrity scores. These are not the same as directly feeding the raw label-construction fields into the model.

## 7. Final Recommendation

Keep the current 14-feature baseline for the final demo and report.

Do not replace the model with the professor repo's full feature set before submission. The current design is more interpretable, easier to defend, and already matches the dashboard story.

Recommended future improvements:

1. Add a TF-IDF phrase audit as an appendix or robustness experiment.
2. Compare sector one-hot encoding against the current `sector_encoded`.
3. Test interaction features such as `talk_total_density * sector_rank`.
4. Expand climate keyword coverage only in a separate branch and re-run validation.
5. Keep semantic robustness features as complementary evidence, not as a replacement for the baseline model.

Do not add now:

1. Absolute ESG risk score features.
2. ESG Risk Percentile or ESG Risk Level as predictors.
3. Sector target encoding without fold-safe implementation.
4. Financial or emissions features unless reliable columns are added to the dataset.

## 8. Suggested Final Report Paragraph

We reviewed the professor's ESG feature engineering design and compared it with our local project. The professor repo includes TF-IDF text features, text statistics, ESG claim density, gap-based features, sector encoding, and interaction features. Our final dashboard keeps a more focused and interpretable 14-feature design: 9 Talk features from sustainability report text, 4 Walk / External features from ESG risk structure and third-party integrity data, and 1 sector context feature. The label follows the project logic: **High Talk + Weak Walk = greenwashing risk screening signal**. We intentionally exclude `Total ESG Risk score`, `Controversy Score`, `ESG Risk Percentile`, and `ESG Risk Level` from the final model inputs to reduce leakage risk. Future work can test TF-IDF phrases, interaction terms, and one-hot sector encoding, but the submitted model remains a risk-ranking and due diligence support tool, not a legal judgment of greenwashing.

## 9. Suggested PPT Bullet Points

- Compared our feature design with the professor's ESG use case.
- Professor repo uses TF-IDF, text statistics, ESG claim density, gap features, sector encoding, and interactions.
- Our project keeps 14 interpretable features: 9 Talk + 4 Walk / External + 1 Context.
- We use sector-relative comparison because ESG risk differs by industry.
- We exclude raw label-construction fields from final model inputs.
- **High Talk + Weak Walk = greenwashing risk screening signal**.
- Future improvements: TF-IDF phrase audit, sector one-hot encoding, and Talk x Walk interaction features.
- The dashboard supports ESG due diligence prioritization; it does not accuse a company of greenwashing.

中文簡短說明：
模型是風險篩選工具，不是法律定論。

## 10. Suggested AI Audit Trail Row

| Date / Week | Task | AI Tool | Prompt Summary | AI Output Used? | Human Modification | Learning Reflection |
|---|---|---|---|---|---|---|
| Final Review | Professor repo feature engineering comparison | Codex | Compare professor ESG feature engineering design with local 14-feature Talk-Walk project and check leakage risk. | Partially | Verified local feature matrix columns, checked missing local professor file path, and framed recommendations as future improvements rather than changing the final model. | Learned how to compare external reference feature ideas without overwriting the current interpretable project design or creating leakage risk. |
