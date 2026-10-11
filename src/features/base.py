from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.features.common import CUTOFF


def scored_apps(data: Path) -> pd.DataFrame:
    train = pd.read_csv(
        data / "train_applications.csv",
        usecols=[
            "application_id",
            "customer_id",
            "application_datetime",
            "decision_datetime",
            "loan_amount",
            "term_months",
            "annual_interest_rate",
            "proposed_monthly_installment",
            "loan_purpose",
            "target",
        ],
        parse_dates=["application_datetime", "decision_datetime"],
    )
    train["is_train"] = True
    test = pd.read_csv(
        data / "test_applications.csv",
        usecols=[
            "application_id",
            "customer_id",
            "application_datetime",
            "decision_datetime",
            "loan_amount",
            "term_months",
            "annual_interest_rate",
            "proposed_monthly_installment",
            "loan_purpose",
        ],
        parse_dates=["application_datetime", "decision_datetime"],
    )
    test["is_train"] = False
    return pd.concat([train, test], ignore_index=True)


def base_features(apps: pd.DataFrame, data: Path) -> pd.DataFrame:
    profile = pd.read_csv(data / "customer_profiles.csv")
    feat = apps.merge(profile, on="customer_id", how="left")
    feat["loan_to_income"] = feat["loan_amount"] / feat["declared_monthly_income"].clip(
        lower=1
    )
    feat["installment_to_income"] = feat["proposed_monthly_installment"] / feat[
        "declared_monthly_income"
    ].clip(lower=1)
    feat["rate_x_loan"] = feat["annual_interest_rate"] * feat["loan_amount"]
    feat["rate_x_term"] = feat["annual_interest_rate"] * feat["term_months"]
    feat["decision_lag_h"] = (
        feat[CUTOFF] - feat["application_datetime"]
    ).dt.total_seconds() / 3600
    feat["decision_month"] = feat[CUTOFF].dt.month.astype("int8")
    feat["decision_weekday"] = feat[CUTOFF].dt.weekday.astype("int8")
    return feat
