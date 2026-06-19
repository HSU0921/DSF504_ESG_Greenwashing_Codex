# Robustness Against AI-Rewritten ESG Disclosure

## 1. Professor's Concern

A key concern is that ESG disclosure text can be rewritten by AI. If a model relies only on surface keywords, a company could keep the same business meaning while changing the wording enough to reduce keyword signals.

## 2. Current Model Limitation

The original 14-feature baseline is transparent and easy to explain, but several Talk-side variables are still based on observable ESG terms, densities, and wording patterns. These features are useful for interpretation, but surface keyword signals are attackable when disclosure language is rewritten.

## 3. Improvement Added

We added 5 semantic robustness features that look for climate transition language, measurable commitments, vague commitments, assurance language, and a semantic specificity gap. These features are complementary to the original model, not a replacement for the 14-feature baseline.

## 4. Experiment Design

We created a deterministic 15-case adversarial rewrite test. Each case rewrites a more specific ESG claim into softer or more abstract language. The test compares keyword signal, semantic signal, and specificity gap before and after rewriting.

The results were:

- average keyword signal drop = 1.00
- average semantic signal drop = -1.60
- average specificity gap change = 2.93

The negative semantic signal drop means semantic signal increased after rewriting. In other words, rewritten disclosures used fewer surface keywords, but still contained meaning-level ESG language. At the same time, the specificity gap increased, showing that the rewritten disclosures became vaguer.

## 5. Business Interpretation

The result supports the professor's concern: keyword-only ESG models can be fragile when language is rewritten. The semantic features help identify whether companies preserve ESG-sounding meaning while reducing specific, measurable commitments. This is useful for ESG due diligence because vague AI-polished disclosure may still require follow-up evidence.

## 6. Governance Recommendation

The dashboard should continue to combine Talk features with Walk / External evidence. Financial institutions should not rely on company wording alone. When the model flags risk, analysts should request supporting documents, third-party assurance, board oversight evidence, and controversy review before making sustainable lending or investment decisions.

## 7. Disclaimer

This model flags companies for further ESG due diligence. It does NOT legally determine greenwashing or corporate wrongdoing.
