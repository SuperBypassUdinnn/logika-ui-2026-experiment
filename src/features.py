"""Feature builders: all aggregations must respect decision cutoff.

Each builder takes pre-cutoff event frames and returns one row per
application_id. IDs are join keys only, never numeric features.
"""

from __future__ import annotations

import pandas as pd


def bureau_features(bureau_prefiltered: pd.DataFrame) -> pd.DataFrame:
    if bureau_prefiltered.empty:
        return pd.DataFrame(columns=["application_id"])
    g = bureau_prefiltered.groupby("application_id")
    out = pd.DataFrame(
        {
            "bureau_n_reports": g.size(),
            "bureau_score_min": g["score"].min(),
            "bureau_score_mean": g["score"].mean(),
            "bureau_outstanding_sum": g["outstanding_balance"].sum(),
            "bureau_delinq_max": g["delinquent_accounts_last_12m"].max(),
            "bureau_inq_sum": g["inquiries_last_6m"].sum(),
        }
    ).reset_index()
    return out


def account_features(account_prefiltered: pd.DataFrame) -> pd.DataFrame:
    if account_prefiltered.empty:
        return pd.DataFrame(columns=["application_id"])
    g = account_prefiltered.groupby("application_id")
    out = pd.DataFrame(
        {
            "acct_n_months": g.size(),
            "acct_closing_mean": g["closing_balance"].mean(),
            "acct_inflows_sum": g["inflows"].sum(),
            "acct_low_days_sum": g["low_balance_days"].sum(),
        }
    ).reset_index()
    return out


def base_app_features(apps: pd.DataFrame) -> pd.DataFrame:
    df = apps.copy()
    df["loan_to_income"] = df["loan_amount"] / df.get(
        "declared_monthly_income", pd.Series(1, index=df.index)
    )
    return df
