from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.features.common import CUTOFF, with_cutoff


def agg_repay(data: Path, apps: pd.DataFrame) -> pd.DataFrame:
    loan_customer = pd.read_csv(
        data / "previous_applications.csv",
        usecols=["loan_id", "customer_id", "decision_datetime"],
        parse_dates=["decision_datetime"],
    ).dropna(subset=["loan_id"])
    loan_customer = loan_customer.drop_duplicates("loan_id")
    loan_customer = loan_customer.rename(columns={"decision_datetime": "orig_date"})
    rep = pd.read_csv(
        data / "repayment_history.csv",
        usecols=[
            "loan_id",
            "due_date",
            "amount_due",
            "amount_paid",
            "days_past_due",
            "max_days_past_due",
            "last_payment_datetime",
        ],
        parse_dates=["due_date", "last_payment_datetime"],
    )
    rep = rep.merge(loan_customer, on="loan_id", how="inner")
    m = with_cutoff(rep, apps)
    m = m[m["due_date"] < m[CUTOFF]]
    m["overdue"] = (m["days_past_due"] > 0).astype("int8")
    m["overdue_amt"] = ((m["amount_due"] - m["amount_paid"]).clip(lower=0)) * m[
        "overdue"
    ]
    m["rec_w"] = 1.0 / (1.0 + (m[CUTOFF] - m["due_date"]).dt.days.clip(lower=0) / 30.0)
    m["cured"] = (
        (m["days_past_due"] > 0) & (m["amount_paid"] >= m["amount_due"])
    ).astype("int8")
    m["recent"] = (m["due_date"] >= m[CUTOFF] - pd.Timedelta(days=90)).astype("int8")
    g = m.groupby("application_id")
    out = pd.DataFrame(
        {
            "repay_n": g.size(),
            "repay_due_sum": g["amount_due"].sum(),
            "repay_paid_sum": g["amount_paid"].sum(),
            "repay_dpd_max": g["days_past_due"].max(),
            "repay_overdue_n": g["overdue"].sum(),
            "repay_overdue_amt_sum": g["overdue_amt"].sum(),
            "repay_rec_wlate": g.apply(lambda d: (d["overdue"] * d["rec_w"]).sum()),
            "repay_rec_wdpd": g.apply(
                lambda d: (d["days_past_due"] * d["rec_w"]).sum()
            ),
            "repay_rec_wsum": g["rec_w"].sum(),
            "repay_maxdpd_max": g["max_days_past_due"].max(),
            "repay_maxdpd_mean": g["max_days_past_due"].mean(),
            "repay_cured_n": g["cured"].sum(),
            "repay_recent_n": g["recent"].sum(),
            "repay_recent_dpd_max": g.apply(
                lambda d: (
                    d.loc[d["recent"] == 1, "days_past_due"].max()
                    if (d["recent"] == 1).any()
                    else np.nan
                )
            ),
            "repay_recent_paid_sum": g.apply(
                lambda d: (d["amount_paid"] * d["recent"]).sum()
            ),
            "repay_recent_due_sum": g.apply(
                lambda d: (d["amount_due"] * d["recent"]).sum()
            ),
        }
    ).reset_index()
    out["repay_pay_ratio"] = out["repay_paid_sum"] / out["repay_due_sum"].clip(lower=1)
    out["repay_late_ratio"] = out["repay_overdue_n"] / out["repay_n"].clip(lower=1)
    out["repay_overdue_amt_ratio"] = out["repay_overdue_amt_sum"] / out[
        "repay_due_sum"
    ].clip(lower=1)
    out["repay_recent_pay_ratio"] = out["repay_recent_paid_sum"] / out[
        "repay_recent_due_sum"
    ].clip(lower=1)

    per_loan = (
        m.groupby(["application_id", "loan_id"])
        .agg(
            loan_dpd_max=("days_past_due", "max"),
            loan_due=("amount_due", "sum"),
            loan_paid=("amount_paid", "sum"),
            loan_n_inst=("days_past_due", "size"),
        )
        .reset_index()
    )
    per_loan["loan_pay_ratio"] = per_loan["loan_paid"] / per_loan["loan_due"].clip(
        lower=1
    )
    per_loan["loan_bad90"] = (per_loan["loan_dpd_max"] >= 90).astype("int8")
    per_loan = per_loan.merge(
        loan_customer[["loan_id", "orig_date"]],
        on="loan_id",
        how="left",
    )
    gl = per_loan.groupby("application_id")
    loan_agg = pd.DataFrame(
        {
            "repay_n_loans": gl.size(),
            "repay_worst_loan_dpd": gl["loan_dpd_max"].max(),
            "repay_n_loans90": gl["loan_bad90"].sum(),
            "repay_worst_loan_pay_ratio": gl["loan_pay_ratio"].min(),
            "repay_mean_loan_pay_ratio": gl["loan_pay_ratio"].mean(),
            "repay_mean_loan_n_inst": gl["loan_n_inst"].mean(),
        }
    ).reset_index()
    out = out.merge(loan_agg, on="application_id", how="left")

    def loan_trend(d):
        d = d.sort_values("orig_date")
        if len(d) < 2 or d["orig_date"].isna().all():
            return pd.Series({"repay_loan_dpd_trend": 0.0, "repay_loan_pay_trend": 0.0})
        first = d.iloc[0]
        last = d.iloc[-1]
        return pd.Series(
            {
                "repay_loan_dpd_trend": float(
                    last["loan_dpd_max"] - first["loan_dpd_max"]
                ),
                "repay_loan_pay_trend": float(
                    last["loan_pay_ratio"] - first["loan_pay_ratio"]
                ),
            }
        )

    out = out.merge(
        per_loan.groupby("application_id")[
            ["orig_date", "loan_dpd_max", "loan_pay_ratio"]
        ]
        .apply(loan_trend)
        .reset_index(),
        on="application_id",
        how="left",
    )

    last_dates = (
        m.groupby("application_id")
        .agg(last_due=("due_date", "max"), last_paid=("last_payment_datetime", "max"))
        .reset_index()
    )
    out = out.merge(last_dates, on="application_id", how="left")
    out = out.merge(apps[["application_id", CUTOFF]], on="application_id", how="left")
    out["repay_days_since_due"] = (out[CUTOFF] - out["last_due"]).dt.days
    out["repay_days_since_paid"] = (out[CUTOFF] - out["last_paid"]).dt.days
    out = out.drop(columns=["last_due", "last_paid", CUTOFF])

    last = (
        m.sort_values("due_date")
        .groupby("application_id")
        .tail(1)[["application_id", "days_past_due"]]
    ).rename(columns={"days_past_due": "repay_last_dpd"})
    out = out.merge(last, on="application_id", how="left")

    def traj(d):
        d = d.sort_values("due_date")
        dpd = d["days_past_due"].to_numpy()
        last3 = dpd[-3:]
        older = dpd[:-3] if len(dpd) > 3 else dpd[:1]
        first_del = d.loc[d["days_past_due"] > 0, "due_date"].min()
        months_since = (
            (d[CUTOFF].iloc[0] - first_del).days / 30.44
            if pd.notna(first_del)
            else np.nan
        )
        x = np.arange(len(last3), dtype=float)
        slope = (
            float(np.polyfit(x, last3.astype(float), 1)[0]) if len(last3) >= 2 else 0.0
        )
        trail = 0
        for v in dpd[::-1]:
            if v > 0:
                trail += 1
            else:
                break
        bands = pd.cut(dpd, [-1, 0, 30, 90, 10**9], labels=False).astype(int)
        if len(bands) >= 2:
            diff = bands[1:] - bands[:-1]
            up = int((diff > 0).sum())
            down = int((diff < 0).sum())
        else:
            up, down = 0, 0
        return pd.Series(
            {
                "repay_last3_dpd_mean": float(last3.mean()),
                "repay_dpd_slope": slope,
                "repay_worsening": int(last3.max() > older.max()),
                "repay_months_since_delinq": months_since,
                "repay_trail_late_n": trail,
                "repay_last3_late_ratio": float((last3 > 0).mean()),
                "repay_n_worsen_trans": up,
                "repay_n_improve_trans": down,
            }
        )

    out = out.merge(
        m.groupby("application_id")[[CUTOFF, "due_date", "days_past_due"]]
        .apply(traj)
        .reset_index(),
        on="application_id",
        how="left",
    )
    return out
