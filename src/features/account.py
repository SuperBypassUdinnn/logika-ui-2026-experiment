from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.features.common import CUTOFF, with_cutoff


def agg_account(data: Path, apps: pd.DataFrame) -> pd.DataFrame:
    acc = pd.read_csv(
        data / "account_behavior.csv",
        usecols=[
            "customer_id",
            "month",
            "opening_balance",
            "closing_balance",
            "inflows",
            "outflows",
            "income_like_credits",
            "average_balance",
            "minimum_balance",
            "low_balance_days",
        ],
        parse_dates=["month"],
    )
    m = with_cutoff(acc, apps)
    m = m[
        m["month"] < m[CUTOFF].values.astype("datetime64[M]").astype("datetime64[ns]")
    ]
    g = m.groupby("application_id")
    out = pd.DataFrame(
        {
            "acct_n_months": g.size(),
            "acct_closing_mean": g["closing_balance"].mean(),
            "acct_inflows_sum": g["inflows"].sum(),
            "acct_outflows_sum": g["outflows"].sum(),
            "acct_low_days_sum": g["low_balance_days"].sum(),
            "acct_closing_std": g["closing_balance"].std(),
            "acct_closing_min": g["closing_balance"].min(),
            "acct_income_cr_mean": g["income_like_credits"].mean(),
            "acct_income_cr_sum": g["income_like_credits"].sum(),
            "acct_avg_bal_mean": g["average_balance"].mean(),
            "acct_min_bal_min": g["minimum_balance"].min(),
            "acct_flow_dir": (
                g["closing_balance"].mean() - g["opening_balance"].mean()
            ),
        }
    ).reset_index()

    def acct_traj(d):
        d = d.sort_values("month")
        bal = d["closing_balance"].to_numpy(dtype=float)
        last3 = bal[-3:]
        older = bal[:-3] if len(bal) > 3 else bal[:1]
        return pd.Series(
            {
                "acct_closing_last": float(bal[-1]),
                "acct_drawdown": float(older.mean() - last3.mean()),
                "acct_low_days_last3": float(
                    d["low_balance_days"].to_numpy()[-3:].sum()
                ),
            }
        )

    out = out.merge(
        m.groupby("application_id")[["month", "closing_balance", "low_balance_days"]]
        .apply(acct_traj)
        .reset_index(),
        on="application_id",
        how="left",
    )
    last_month = (
        m.groupby("application_id")["month"].max().reset_index(name="acct_last_month")
    )
    out = out.merge(last_month, on="application_id", how="left")
    out = out.merge(apps[["application_id", CUTOFF]], on="application_id", how="left")
    out["acct_months_since"] = (
        out[CUTOFF].dt.year * 12
        + out[CUTOFF].dt.month
        - out["acct_last_month"].dt.year * 12
        - out["acct_last_month"].dt.month
    )
    out = out.drop(columns=["acct_last_month", CUTOFF])
    return out
