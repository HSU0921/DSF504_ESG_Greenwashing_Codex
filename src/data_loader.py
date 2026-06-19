"""
data_loader.py
==============
Loads and merges the three source CSVs into a single clean working dataset.

Pipeline:
  1. Load SP_500_ESG_Risk_Ratings.csv  → esg_df    (Walk performance)
  2. Load preprocessed_content.csv.gz  → report_df  (Talk text, one row per report-year)
  3. Load sp500_integrity_scores.csv   → integrity_df (third-party cross-validation)
  4. For each ticker in report_df, keep only the most recent year
  5. Inner-join all three on ticker
  6. Drop rows where Total ESG Risk score or sub-scores are missing
  7. Return a single DataFrame of ~258 companies

Column naming convention after merge:
  - Ticker column: 'ticker'
  - All ESG ratings columns kept as-is from source files
"""

import os
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def _resolve(filename: str) -> str:
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Expected data file not found: {path}")
    return path


def load_esg_ratings() -> pd.DataFrame:
    """Load Sustainalytics ESG risk ratings (Walk data)."""
    df = pd.read_csv(_resolve("SP_500_ESG_Risk_Ratings.csv"))
    df = df.rename(columns={"Symbol": "ticker"})
    df["ticker"] = df["ticker"].str.strip().str.upper()
    return df


def load_reports() -> pd.DataFrame:
    """Load ESG report text data and keep only the most recent year per ticker."""
    df = pd.read_csv(_resolve("preprocessed_content.csv.gz"), index_col=0)
    df["ticker"] = df["ticker"].str.strip().str.upper()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year", "preprocessed_content"])
    df = df[df["preprocessed_content"].str.strip().str.len() > 0]
    df = df.sort_values("year", ascending=False)
    df = df.drop_duplicates(subset="ticker", keep="first")
    return df.reset_index(drop=True)


def load_integrity_scores() -> pd.DataFrame:
    """Load third-party ethical integrity scores."""
    df = pd.read_csv(_resolve("sp500_integrity_scores.csv"))
    df["ticker"] = df["ticker"].str.strip().str.upper()
    return df


def load_merged_dataset(verbose: bool = True) -> pd.DataFrame:
    """
    Master loader: merge all three sources and return a clean dataset.

    Returns
    -------
    pd.DataFrame
        One row per company (~258 companies after inner join + null drop).
        Contains all source columns; downstream feature engineering reads this.
    """
    esg = load_esg_ratings()
    reports = load_reports()
    integrity = load_integrity_scores()

    # Inner join: only companies present in all three sources
    df = (
        reports
        .merge(esg, on="ticker", how="inner")
        .merge(integrity, on="ticker", how="inner", suffixes=("", "_integrity"))
    )

    # Drop rows where the Walk scoring columns are missing
    # (these are used to build y; without them the label cannot be constructed)
    required_walk_cols = [
        "Total ESG Risk score",
        "Environment Risk Score",
        "Social Risk Score",
        "Governance Risk Score",
    ]
    before = len(df)
    df = df.dropna(subset=required_walk_cols)
    after = len(df)

    if verbose:
        print(f"[data_loader] Loaded {before} companies after 3-way inner join.")
        print(f"[data_loader] Dropped {before - after} rows with missing ESG scores.")
        print(f"[data_loader] Final working dataset: {after} companies.")
        print(f"[data_loader] Columns: {list(df.columns)}")

    return df.reset_index(drop=True)


if __name__ == "__main__":
    df = load_merged_dataset(verbose=True)
    print(df[["ticker", "year", "Sector", "Total ESG Risk score"]].head(10))
