# AI-Powered ESG Greenwashing Risk Scoring Dashboard

DSF504 final project Streamlit dashboard for screening potential ESG greenwashing risk among S&P 500 companies.

The system compares ESG disclosure intensity ("Talk") with external ESG risk signals ("Walk") and produces model-based risk triage outputs for financial institution ESG due diligence.

## Dashboard Pages

- Home
- Data Explorer
- Model Performance
- Model Contribution Analysis
- Company Risk Lookup
- Explainability
- Business Recommendation

## Data And Model Artifacts

Required local folders:

- `data/` - source ESG datasets
- `models/` - trained model pickle files
- `outputs/` - model metrics, SHAP images, validation outputs, and dashboard tables

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app is designed for Streamlit Cloud deployment from the repository root with `app.py` as the entry point.

## Governance Note

This dashboard is an AI-assisted ESG risk screening tool. It supports prioritization and due diligence review; it does not make legal determinations of greenwashing, credit approval, or investment exclusion.
