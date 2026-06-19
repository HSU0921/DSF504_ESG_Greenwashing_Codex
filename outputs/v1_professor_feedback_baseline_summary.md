# V1 Professor Feedback Baseline Summary

## Baseline Finding

The original v1 supervised model remains a useful benchmark, but it should be interpreted as an ESG risk triage model rather than a legal greenwashing determination.

| model_setting              |   mean_roc_auc |   mean_pr_auc |   mean_f1 |
|:---------------------------|---------------:|--------------:|----------:|
| Full Model                 |         0.9413 |        0.7223 |    0.6474 |
| External-only Model        |         0.754  |        0.3096 |    0.2751 |
| Text-only Model            |         0.6635 |        0.1582 |    0.1844 |
| No-ranking Model           |         0.7316 |        0.2322 |    0.2161 |
| Strict No-Walk-Proxy Model |         0.6635 |        0.1582 |    0.1844 |

## Professor Feedback Interpretation

- Full Model performance is high.
- External-only performance is also meaningful, which supports the professor's concern that processed external ESG evidence carries strong signal.
- Text-only performance is weaker but not zero, so sustainability report language still contains useful signal.
- No-ranking and Strict No-Walk-Proxy settings are governance checks, not replacements for the full model.

Therefore, v1 should be presented as a risk-ranking and due diligence support model, not as proof that any company is greenwashing.

AI Decision Support Only: This system is designed for ESG risk triage and review prioritization. It does not legally determine greenwashing, credit approval, or investment exclusion.
