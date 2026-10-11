"""Cross-table ratios and finalization (runs after per-table merges)."""

from __future__ import annotations

import numpy as np
import pandas as pd

COUNT_PREFIXES = (
    "prev_n",
    "repay_n",
    "repay_overdue_n",
    "acct_n_months",
    "bureau_n",
    "eco_n_orders",
    "eco_n_cancelled",
    "prev_n_approved",
    "prev_n_rejected",
)


def apply_cross_ratios(feat: pd.DataFrame) -> pd.DataFrame:
    inc = feat["declared_monthly_income"].clip(lower=1)
    feat["x_total_inst_to_income"] = (
        feat["prev_active_inst_sum"].fillna(0) + feat["proposed_monthly_installment"]
    ) / inc
    feat["x_loan_to_prev_max"] = feat["loan_amount"] / (
        feat["prev_approved_max"].replace(0, np.nan)
    )
    feat["x_rate_diff_prev"] = feat["annual_interest_rate"] - feat["prev_appr_rate_max"]
    feat["x_income_vs_observed"] = feat["declared_monthly_income"] / (
        feat["acct_income_cr_mean"].replace(0, np.nan)
    )
    feat["x_income_vs_avgbal"] = feat["declared_monthly_income"] / (
        feat["acct_avg_bal_mean"].replace(0, np.nan)
    )
    feat["eco_spend_to_income"] = feat["eco_spend_sum"] / inc
    feat["bureau_outstanding_to_income"] = feat["bureau_outstanding_sum"] / inc
    feat["repay_due_to_income"] = feat["repay_due_sum"] / inc
    feat["repay_paid_to_income"] = feat["repay_paid_sum"] / inc
    feat["prev_requested_to_income"] = feat["prev_requested_sum"] / inc
    feat["acct_outflows_to_income"] = feat["acct_outflows_sum"] / inc
    return feat


def finalize(feat: pd.DataFrame) -> pd.DataFrame:
    for c in feat.columns:
        if c.startswith(COUNT_PREFIXES):
            feat[c] = feat[c].fillna(0)
    feat["has_repay_history"] = (feat["repay_n"] > 0).astype("int8")
    feat["has_account_history"] = (feat["acct_n_months"] > 0).astype("int8")
    return feat
