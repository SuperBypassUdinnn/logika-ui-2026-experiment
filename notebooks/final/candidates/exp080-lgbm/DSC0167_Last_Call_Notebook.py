# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # DSC0167_Last_Call_Notebook
#
# Notebook final tim Last Call (DSC0167) untuk eliminasi LOGIKA UI 2026.
# Model: LightGBM 5-fold (OOF KS 0,5150). Seluruh fitur leakage-safe terhadap `decision_datetime`.
#
# **Kandidat final 1 dari `notebooks/final/candidates/`.**

# %%
# !pip install -q lightgbm scikit-learn pandas pyarrow scipy

# %%
import os
CANDIDATES = [
    "/kaggle/input/competitions/data-science-competition-logika-ui-2026",
    "/kaggle/input/data-science-competition-logika-ui-2026",
]
def find_data():
    for c in CANDIDATES:
        if os.path.isfile(os.path.join(c, "train_applications.csv")):
            return c
    for root in ["/kaggle/input", "data"]:
        if not os.path.isdir(root):
            continue
        for dirpath, _, files in os.walk(root):
            if "train_applications.csv" in files:
                return dirpath
    d = os.getcwd()
    for _ in range(6):
        cand = os.path.join(d, "data", "train_applications.csv")
        if os.path.isfile(cand):
            return os.path.join(d, "data")
        nd = os.path.dirname(d)
        if nd == d:
            break
        d = nd
    return None
DATA = find_data()
assert DATA is not None, ("data tidak ditemukan; "
"pastikan dataset kompetisi ter-attach di notebook Kaggle")
print("DATA:", DATA)

# %%
import numpy as np

import pandas as pd

from pathlib import Path

CUTOFF = "decision_datetime"

def with_cutoff(aux: pd.DataFrame, apps: pd.DataFrame) -> pd.DataFrame:
    """Attach per-application cutoff to event rows (no filtering)."""
    cut = apps[["customer_id", "application_id", CUTOFF]]
    return aux.merge(cut, on="customer_id", how="inner")

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

def agg_prev(data: Path, apps: pd.DataFrame) -> pd.DataFrame:
    prev = pd.read_csv(
        data / "previous_applications.csv",
        usecols=[
            "application_id",
            "customer_id",
            "decision_datetime",
            "requested_amount",
            "term_months",
            "decision",
            "approved_amount",
            "annual_interest_rate",
            "scheduled_monthly_installment",
        ],
        parse_dates=["decision_datetime"],
    ).rename(columns={"application_id": "hist_app_id"})
    m = prev.merge(
        apps[["customer_id", "application_id", CUTOFF]].rename(
            columns={CUTOFF: "cutoff"}
        ),
        on="customer_id",
        how="inner",
    )
    m = m[
        (m["hist_app_id"] != m["application_id"])
        & (m["decision_datetime"] < m["cutoff"])
    ]
    g = m.groupby("application_id")
    out = pd.DataFrame(
        {
            "prev_n": g.size(),
            "prev_n_approved": g["decision"].apply(lambda s: (s == "approved").sum()),
            "prev_n_rejected": g["decision"].apply(lambda s: (s == "rejected").sum()),
            "prev_requested_sum": g["requested_amount"].sum(),
            "prev_approved_sum": g["approved_amount"].sum(),
            "prev_term_mean": g["term_months"].mean(),
            "prev_term_max": g["term_months"].max(),
            "prev_sched_sum": g["scheduled_monthly_installment"].sum(),
            "prev_req_appr_gap": g.apply(
                lambda d: (d["requested_amount"] - d["approved_amount"].fillna(0)).sum()
            ),
            "prev_days_since_last": g.apply(
                lambda d: (d["cutoff"].iloc[0] - d["decision_datetime"].max()).days
            ),
            "prev_approval_rate": g["decision"].apply(
                lambda s: (s == "approved").mean()
            ),
        }
    ).reset_index()
    last_dec = (
        m.sort_values("decision_datetime")
        .groupby("application_id")
        .tail(1)[["application_id", "decision"]]
    ).rename(columns={"decision": "prev_last_decision"})
    out = out.merge(last_dec, on="application_id", how="left")
    last3 = (
        m.sort_values("decision_datetime")
        .groupby("application_id")
        .tail(3)
        .groupby("application_id")["decision"]
        .apply(list)
        .reset_index(name="dec_list")
    )
    last3["prev_reject_last3"] = last3["dec_list"].apply(
        lambda s: sum(1 for d in s if d == "rejected")
    )
    last3["prev_approved_last"] = last3["dec_list"].apply(
        lambda s: int(len(s) > 0 and s[-1] == "approved")
    )
    out = out.merge(
        last3[["application_id", "prev_reject_last3", "prev_approved_last"]],
        on="application_id",
        how="left",
    )
    ap = m[m["decision"] == "approved"].copy()
    ap["end_date"] = ap["decision_datetime"] + pd.to_timedelta(
        ap["term_months"] * 30.4, unit="D"
    )
    act = ap[ap["end_date"] > ap["cutoff"]]
    gact = act.groupby("application_id")
    active = pd.DataFrame(
        {
            "prev_cnt_active": gact.size(),
            "prev_active_inst_sum": gact["scheduled_monthly_installment"].sum(),
            "prev_active_amt_sum": gact["approved_amount"].sum(),
        }
    ).reset_index()
    out = out.merge(active, on="application_id", how="left")
    out["prev_approved_max"] = (
        m[m["decision"] == "approved"]
        .groupby("application_id")["approved_amount"]
        .max()
        .reindex(out["application_id"])
        .to_numpy()
    )
    out["prev_appr_rate_max"] = (
        m[m["decision"] == "approved"]
        .groupby("application_id")["annual_interest_rate"]
        .max()
        .reindex(out["application_id"])
        .to_numpy()
    )
    return out

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

def agg_bureau(data: Path, apps: pd.DataFrame) -> pd.DataFrame:
    bur = pd.read_csv(
        data / "bureau_observations.csv",
        usecols=[
            "application_id",
            "customer_id",
            "bureau_provider",
            "report_datetime",
            "available_datetime",
            "score",
            "active_accounts",
            "outstanding_balance",
            "revolving_credit_limit",
            "revolving_outstanding_balance",
            "utilization",
            "inquiries_last_6m",
            "delinquent_accounts_last_12m",
        ],
        parse_dates=["report_datetime", "available_datetime"],
    )
    cut = apps[["application_id", CUTOFF, "is_train"]]
    m = bur.merge(cut, on="application_id", how="inner")
    m = m[m["available_datetime"] <= m[CUTOFF]]
    prov_stats = (
        m[m["is_train"]]
        .groupby("bureau_provider")["score"]
        .agg(["mean", "std"])
        .reset_index()
    )
    prov_map = prov_stats.set_index("bureau_provider")
    m["score_z"] = (m["score"] - m["bureau_provider"].map(prov_map["mean"])) / m[
        "bureau_provider"
    ].map(prov_map["std"]).replace(0, np.nan)
    g = m.groupby("application_id")
    out = pd.DataFrame(
        {
            "bureau_n": g.size(),
            "bureau_score_min": g["score"].min(),
            "bureau_score_mean": g["score"].mean(),
            "bureau_outstanding_sum": g["outstanding_balance"].sum(),
            "bureau_util_mean": g["utilization"].mean(),
            "bureau_inq_sum": g["inquiries_last_6m"].sum(),
            "bureau_delinq_max": g["delinquent_accounts_last_12m"].max(),
            "bureau_active_sum": g["active_accounts"].sum(),
            "bureau_rev_limit_sum": g["revolving_credit_limit"].sum(),
            "bureau_rev_out_sum": g["revolving_outstanding_balance"].sum(),
        }
    ).reset_index()
    per_provider = (
        m.groupby(["application_id", "bureau_provider"])
        .agg(
            n=("score", "size"),
            score_min=("score", "min"),
            outstanding=("outstanding_balance", "sum"),
            delinq=("delinquent_accounts_last_12m", "max"),
        )
        .reset_index()
    )
    wide = per_provider.pivot(index="application_id", columns="bureau_provider").fillna(
        0
    )
    wide.columns = [
        f"bureau_{prov}_{metric}" for metric, prov in wide.columns.to_flat_index()
    ]
    out = out.merge(wide.reset_index(), on="application_id", how="left")

    def score_trend(d):
        d = d.sort_values("available_datetime")
        s = d["score"].dropna().to_numpy(dtype=float)
        return pd.Series(
            {
                "bureau_score_trend": float(s[-1] - s[0]) if len(s) >= 2 else 0.0,
                "bureau_score_last": float(s[-1]) if len(s) >= 1 else np.nan,
            }
        )

    out = out.merge(
        m.groupby("application_id")[["available_datetime", "score"]]
        .apply(score_trend)
        .reset_index(),
        on="application_id",
        how="left",
    )
    spread = (
        m.groupby("application_id")["score"].agg(["max", "min", "count"]).reset_index()
    )
    out["bureau_score_spread"] = spread["max"] - spread["min"]
    out["bureau_n_providers"] = spread["count"]
    out["bureau_score_nmiss"] = 3 - spread["count"]
    gz = m.groupby("application_id")["score_z"]
    out["bureau_z_mean"] = gz.mean().reindex(out["application_id"]).to_numpy()
    out["bureau_z_min"] = gz.min().reindex(out["application_id"]).to_numpy()
    out["bureau_rev_util"] = out["bureau_rev_out_sum"] / out[
        "bureau_rev_limit_sum"
    ].replace(0, np.nan)
    mh = bur.merge(
        apps[["customer_id", "application_id", CUTOFF]], on="customer_id", how="inner"
    )
    mh = mh[
        (mh["available_datetime"] <= mh[CUTOFF])
        & (mh["application_id_x"] != mh["application_id_y"])
    ]
    mh["score_z"] = (mh["score"] - mh["bureau_provider"].map(prov_map["mean"])) / mh[
        "bureau_provider"
    ].map(prov_map["std"]).replace(0, np.nan)
    gh = mh.groupby("application_id_y")
    hist = pd.DataFrame(
        {
            "bureau_hist_z_mean": gh["score_z"].mean(),
            "bureau_hist_n": gh.size(),
            "bureau_hist_delinq_max": gh["delinquent_accounts_last_12m"].max(),
        }
    ).reset_index(names="application_id")
    out = out.merge(hist, on="application_id", how="left")
    out["bureau_z_delta"] = out["bureau_z_mean"] - out["bureau_hist_z_mean"]
    last_rep = (
        m.groupby("application_id")["available_datetime"]
        .max()
        .reset_index(name="bureau_last_available")
    )
    out = out.merge(last_rep, on="application_id", how="left")
    out = out.merge(apps[["application_id", CUTOFF]], on="application_id", how="left")
    out["bureau_days_since_report"] = (
        out[CUTOFF] - out["bureau_last_available"]
    ).dt.days
    out = out.drop(columns=["bureau_last_available", CUTOFF])
    return out

def agg_eco(data: Path, apps: pd.DataFrame) -> pd.DataFrame:
    eco = pd.read_csv(
        data / "ecommerce_transactions.csv",
        usecols=[
            "order_id",
            "customer_id",
            "transaction_datetime",
            "product_category",
            "quantity",
            "item_total",
            "order_total",
            "payment_method",
            "order_status",
        ],
        parse_dates=["transaction_datetime"],
    )
    m = with_cutoff(eco, apps)
    m = m[m["transaction_datetime"] < m[CUTOFF]]
    m["completed"] = (m["order_status"] == "completed").astype("int8")
    m["spend"] = m["item_total"] * m["completed"]
    m["days_since"] = (m[CUTOFF] - m["transaction_datetime"]).dt.days
    m["rec_w"] = 1.0 / (1.0 + m["days_since"].clip(lower=0) / 30.0)
    m["recent90"] = (
        m["transaction_datetime"] >= m[CUTOFF] - pd.Timedelta(days=90)
    ).astype("int8")
    m["recent30"] = (
        m["transaction_datetime"] >= m[CUTOFF] - pd.Timedelta(days=30)
    ).astype("int8")
    orders = (
        m.drop_duplicates(["application_id", "order_id"])
        .groupby("application_id")
        .agg(
            eco_n_orders=("order_id", "size"),
            eco_order_total_sum=("order_total", "sum"),
        )
        .reset_index()
    )
    g = m.groupby("application_id")
    items = pd.DataFrame(
        {
            "eco_spend_sum": g["spend"].sum(),
            "eco_qty_sum": g["quantity"].sum(),
            "eco_n_cancelled": g["completed"].apply(lambda s: (s == 0).sum()),
            "eco_n_categories": g["product_category"].nunique(),
            "eco_cod_n": g["payment_method"].apply(lambda s: (s == "COD").sum()),
            "eco_ewallet_n": g["payment_method"].apply(
                lambda s: (s == "e_wallet").sum()
            ),
            "eco_card_n": g["payment_method"].apply(
                lambda s: s.isin(["credit_card", "debit_card"]).sum()
            ),
            "eco_spend_90d": g.apply(lambda d: (d["spend"] * d["recent90"]).sum()),
            "eco_spend_30d": g.apply(lambda d: (d["spend"] * d["recent30"]).sum()),
            "eco_rec_wspend": g.apply(lambda d: (d["spend"] * d["rec_w"]).sum()),
            "eco_n_orders_90d": g.apply(
                lambda d: d.loc[d["recent90"] == 1, "order_id"].nunique()
            ),
            "eco_max_order": g["order_total"].max(),
        }
    ).reset_index()
    cat_spend = (
        m[m["completed"] == 1]
        .groupby(["application_id", "product_category"])["spend"]
        .sum()
        .reset_index()
    )
    cat_wide = cat_spend.pivot(
        index="application_id", columns="product_category", values="spend"
    ).fillna(0)
    cat_wide.columns = [f"eco_spend_{c}" for c in cat_wide.columns]
    items = items.merge(cat_wide.reset_index(), on="application_id", how="left")
    for w in (30, 90, 180, 365):
        wm = m[m["days_since"] <= w]
        wm_c = wm[wm["completed"] == 1]
        wm_c = wm[wm["completed"] == 1]
        items[f"eco_sum_done_{w}d"] = (
            wm_c.groupby("application_id")["order_total"]
            .sum()
            .reindex(items["application_id"])
            .fillna(0)
            .to_numpy()
        )
        items[f"eco_cnt_done_{w}d"] = (
            wm_c.groupby("application_id")
            .size()
            .reindex(items["application_id"])
            .fillna(0)
            .to_numpy()
        )
        for cat in ("groceries", "fashion", "electronics", "personal_care"):
            items[f"eco_{cat}_{w}d"] = (
                wm_c[wm_c["product_category"] == cat]
                .groupby("application_id")["spend"]
                .sum()
                .reindex(items["application_id"])
                .fillna(0)
                .to_numpy()
            )
    items["eco_vel_30_180"] = items["eco_sum_done_30d"] / (
        items["eco_sum_done_180d"] / 6
    ).replace(0, np.nan)
    items["eco_vel_90_365"] = items["eco_sum_done_90d"] / (
        items["eco_sum_done_365d"] / 4
    ).replace(0, np.nan)
    done = m[m["completed"] == 1].groupby("application_id")["order_total"]
    items["eco_done_median"] = done.median().reindex(items["application_id"]).to_numpy()
    items["eco_done_std"] = done.std().reindex(items["application_id"]).to_numpy()
    n_items = m.groupby("application_id").size()
    items["eco_cancel_rate"] = (
        items["eco_n_cancelled"] / n_items.reindex(items["application_id"]).to_numpy()
    ).fillna(0)
    last_txn = (
        m.groupby("application_id")
        .agg(last_txn=("transaction_datetime", "max"), cutoff=(CUTOFF, "first"))
        .reset_index()
    )
    items = items.merge(last_txn, on="application_id", how="left")
    items["eco_days_since_txn"] = (items["cutoff"] - items["last_txn"]).dt.days
    items = items.drop(columns=["last_txn", "cutoff"])

    def gap_stats(d):
        t = d.sort_values("transaction_datetime")["transaction_datetime"]
        if len(t) < 2:
            return pd.Series({"eco_gap_mean": np.nan, "eco_n_gaps": 0})
        gaps = t.diff().dt.days.dropna()
        spend = d.sort_values("transaction_datetime")["spend"].to_numpy(dtype=float)
        half = len(spend) // 2
        accel = float(spend[half:].mean() - spend[:half].mean()) if half >= 1 else 0.0
        return pd.Series(
            {
                "eco_gap_mean": float(gaps.mean()),
                "eco_n_gaps": len(gaps),
                "eco_spend_accel": accel,
            }
        )

    items = items.merge(
        m.groupby("application_id")[["transaction_datetime", "spend"]]
        .apply(gap_stats)
        .reset_index(),
        on="application_id",
        how="left",
    )
    first_txn = (
        m.groupby("application_id")
        .agg(first_txn=("transaction_datetime", "min"), cutoff=(CUTOFF, "first"))
        .reset_index()
    )
    items = items.merge(first_txn, on="application_id", how="left")
    items["eco_tenure_days"] = (items["cutoff"] - items["first_txn"]).dt.days
    items = items.drop(columns=["first_txn", "cutoff"])
    n_orders = m.groupby("application_id")["order_id"].nunique()
    items["eco_items_per_order"] = items["eco_qty_sum"] / n_orders.reindex(
        items["application_id"]
    ).to_numpy().clip(min=1)

    done = m["completed"] == 1
    win = {}
    for w in (7, 14, 30, 90, 180):
        inw = m["days_since"] <= w
        win[f"cd_{w}"] = (inw & done).astype("float32")
        win[f"cx_{w}"] = (inw & ~done).astype("float32")
        win[f"sd_{w}"] = (m["order_total"] * (inw & done)).astype("float32")
    g = pd.DataFrame(win, index=m.index).groupby(m["application_id"]).sum()
    nz = lambda s: s.replace(0, np.nan)
    ec3 = pd.DataFrame(index=g.index)
    for w in (7, 14):
        ec3[f"ec3_win_cnt_done_{w}d"] = g[f"cd_{w}"]
        ec3[f"ec3_win_cnt_cancel_{w}d"] = g[f"cx_{w}"]
        ec3[f"ec3_win_sum_done_{w}d"] = g[f"sd_{w}"]
        ec3[f"ec3_win_cancel_ratio_{w}d"] = g[f"cx_{w}"] / (
            g[f"cd_{w}"] + g[f"cx_{w}"]
        ).replace(0, np.nan)
    ec3["ec3_win_spk_sum_7_30"] = g["sd_7"] / nz(g["sd_30"] * 7 / 30)
    ec3["ec3_win_spk_sum_14_90"] = g["sd_14"] / nz(g["sd_90"] * 14 / 90)
    ec3["ec3_win_spk_sum_30_180"] = g["sd_30"] / nz(g["sd_180"] * 30 / 180)
    ec3["ec3_win_spk_ord_7_30"] = g["cd_7"] / nz(g["cd_30"] * 7 / 30)
    ec3["ec3_win_spk_ord_14_90"] = g["cd_14"] / nz(g["cd_90"] * 14 / 90)
    cr = lambda w: g[f"cx_{w}"] / nz(g[f"cd_{w}"] + g[f"cx_{w}"])
    ec3["ec3_win_spk_cancel_30_180"] = cr(30) / nz(cr(180))
    items = items.merge(
        ec3.reset_index(names="application_id"), on="application_id", how="left"
    )

    t = m["transaction_datetime"]
    h, dow, dom = t.dt.hour.to_numpy(), t.dt.dayofweek.to_numpy(), t.dt.day.to_numpy()
    ang = 2 * np.pi * h / 24
    tb = pd.DataFrame(
        {
            "night": (h < 6).astype("float32"),
            "late": (h >= 20).astype("float32"),
            "wkend": (dow >= 5).astype("float32"),
            "payday": ((dom >= 25) | (dom <= 5)).astype("float32"),
            "hsin": np.sin(ang).astype("float32"),
            "hcos": np.cos(ang).astype("float32"),
        },
        index=m.index,
    )
    key = m["application_id"]
    tmean = tb.groupby(key).mean().add_prefix("ec3_time_")
    r90 = (m["days_since"] <= 90).to_numpy()
    t90 = tb[r90].groupby(key[r90]).mean().add_prefix("ec3_time_").add_suffix("_90d")
    tfeat = tmean.join(t90)
    items = items.merge(
        tfeat.reset_index(names="application_id"), on="application_id", how="left"
    )

    od = m[m["completed"] == 1].copy()
    od["bucket"] = np.floor(od["days_since"].to_numpy() / 30).astype("int8")
    od = od[od["bucket"] < 6]
    gb = od.groupby(["application_id", "bucket"], observed=True)
    S = (
        gb["order_total"]
        .sum()
        .unstack(fill_value=0)
        .reindex(columns=range(6), fill_value=0)
    )
    N = (
        gb.size()
        .unstack(fill_value=0)
        .fillna(0)
        .reindex(columns=range(6), fill_value=0)
    )
    Sv, Nv = S.to_numpy("float64"), N.to_numpy("float64")
    x = -(np.arange(6) - 2.5)
    slope = lambda M: (M @ x) / (x**2).sum()
    mean = Sv.mean(axis=1)
    nz = lambda a: np.where(a == 0, np.nan, a)
    trd = pd.DataFrame(
        {
            "ec3_trd_slope_amt": slope(Sv),
            "ec3_trd_slope_amt_rel": slope(Sv) / nz(mean),
            "ec3_trd_slope_ord": slope(Nv),
            "ec3_trd_cv_amt": Sv.std(axis=1) / nz(mean),
            "ec3_trd_zero_buckets": (Sv == 0).sum(axis=1),
            "ec3_trd_recent_vs_prior": Sv[:, 0] / nz(Sv[:, 1:].mean(axis=1)),
            "ec3_trd_last3_vs_prev3": Sv[:, :3].sum(axis=1) / nz(Sv[:, 3:].sum(axis=1)),
        },
        index=S.index,
    )
    items = items.merge(
        trd.reset_index(names="application_id"), on="application_id", how="left"
    )

    is_tr_ids = set(apps.loc[apps["is_train"], "application_id"])
    d = m[m["completed"] & (m["quantity"] > 0)].copy()
    up = (d["item_total"] / d["quantity"]).to_numpy("float64")
    is_tr = d["application_id"].isin(is_tr_ids).to_numpy()
    cat_med = (
        (d[is_tr]["item_total"] / d[is_tr]["quantity"])
        .groupby(d[is_tr]["product_category"], observed=True)
        .median()
        .to_dict()
    )
    d = d.assign(
        up=up, idx=up / d["product_category"].map(cat_med).astype("float64").to_numpy()
    )
    gd = d.groupby("application_id")
    prc = pd.DataFrame(
        {
            "ec3_prc_idx_mean": gd["idx"].mean(),
            "ec3_prc_idx_std": gd["idx"].std(),
            "ec3_prc_idx_max": gd["idx"].max(),
            "ec3_prc_up_mean": gd["up"].mean(),
            "ec3_prc_qty_mean": gd["quantity"].mean(),
        }
    ).reset_index(names="application_id")
    items = items.merge(prc, on="application_id", how="left")
    return orders.merge(items, on="application_id", how="outer")

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
    bs = feat["bureau_score_mean"].fillna(650)
    feat["x_pay_x_score"] = feat["repay_pay_ratio"].fillna(1) * bs
    feat["x_dpd_x_score"] = feat["repay_dpd_max"].fillna(0) * (850 - bs)
    feat["x_spend_x_score"] = feat["eco_spend_sum"].fillna(0) * (850 - bs) / 1e9
    feat["x_pay_x_rate"] = (
        feat["repay_pay_ratio"].fillna(1) * feat["annual_interest_rate"]
    )
    return feat

def finalize(feat: pd.DataFrame) -> pd.DataFrame:
    for c in feat.columns:
        if c.startswith(COUNT_PREFIXES):
            feat[c] = feat[c].fillna(0)
    feat["has_repay_history"] = (feat["repay_n"] > 0).astype("int8")
    feat["has_account_history"] = (feat["acct_n_months"] > 0).astype("int8")
    return feat

COUNT_PREFIXES = (
    "prev_n", "repay_n", "repay_overdue_n", "acct_n_months", "bureau_n",
    "eco_n_orders", "eco_n_cancelled", "prev_n_approved", "prev_n_rejected",
)


# %%
from scipy.stats import ks_2samp
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb

SEED = 42
PARAMS = {'n_estimators': 5000, 'learning_rate': 0.01845, 'num_leaves': 21, 'min_child_samples': 203, 'feature_fraction': 0.742, 'bagging_fraction': 0.679, 'bagging_freq': 1, 'lambda_l1': 3.68, 'lambda_l2': 16.9, 'scale_pos_weight': 3}
data = Path(DATA)
apps = scored_apps(data)
feat = base_features(apps, data)
for name, fn in [("prev", agg_prev), ("repay", agg_repay), ("account", agg_account), ("bureau", agg_bureau), ("eco", agg_eco)]:
    part = fn(data, apps)
    print(name, len(part))
    feat = feat.merge(part, on="application_id", how="left")
feat = apply_cross_ratios(feat)
feat = finalize(feat)
train_apps = pd.read_csv(data / "train_applications.csv")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
folds = train_apps[["application_id"]].copy()
folds["fold"] = -1
for f, (_, va) in enumerate(skf.split(folds, train_apps["target"])):
    folds.loc[va, "fold"] = f
train = feat.merge(folds, on="application_id")
test = feat[~feat["application_id"].isin(set(train["application_id"]))].copy()
drop = {"application_id", "customer_id", "target", "fold", "decision_datetime", "application_datetime", "is_train"}
features = [c for c in train.columns if c not in drop]
for c in train[features].select_dtypes(include=["object", "string"]).columns:
    train[c] = train[c].astype("category")
    test[c] = test[c].astype("category")
print("features:", len(features), "train:", train.shape, "test:", test.shape)

# %% [markdown]
# ## 1. Eksplorasi data
#
# Target 13,58%, stabil per bulan. Sinyal terkuat: skor bureau dan DPD historis (lihat `notebooks/eda/` untuk EDA lengkap).

# %%
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
tr = pd.read_csv(Path(DATA) / 'train_applications.csv')
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
tr.groupby(tr.decision_datetime.str[:7]).target.mean().plot(ax=ax[0], marker='o')
ax[0].set_title('Target rate per bulan')
tr.groupby('loan_purpose').target.mean().plot.bar(ax=ax[1])
ax[1].set_title('Target rate per tujuan')
plt.tight_layout()
plt.savefig('eda_target.png', dpi=100)
print('plot tersimpan: eda_target.png')
from IPython.display import Image
Image('eda_target.png')

# %%
y = train_apps["target"].to_numpy()
oof = np.zeros(len(train_apps))
test_pred = np.zeros(len(test))
ks_metric = lambda yt, yp: ('ks', float(ks_2samp(np.asarray(yp)[np.asarray(yt) == 1], np.asarray(yp)[np.asarray(yt) == 0]).statistic), True)
SEEDS = [42]
for seed in SEEDS:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    folds = train_apps[["application_id"]].copy()
    folds["fold"] = -1
    for f, (_, va) in enumerate(skf.split(folds, train_apps["target"])):
        folds.loc[va, "fold"] = f
    tr_ids = train.drop(columns=["fold"]).merge(folds, on="application_id")
    oof_s = np.zeros(len(train_apps))
    for fold in sorted(tr_ids["fold"].unique()):
        tr = (tr_ids["fold"] != fold).to_numpy()
        va = ~tr
        model = lgb.LGBMClassifier(**PARAMS, random_state=seed)
        model.fit(train.loc[tr, features], y[tr], eval_set=[(train.loc[~tr, features], y[~tr])], eval_metric=ks_metric, callbacks=[lgb.early_stopping(200, verbose=False)], )
        oof_s[va] = model.predict_proba(train.loc[va, features])[:, 1]
        test_pred += model.predict_proba(test[features])[:, 1] / (5 * len(SEEDS))
    oof += oof_s / len(SEEDS)
    print(f"seed {seed} done", flush=True)
ks = ks_2samp(oof[y == 1], oof[y == 0]).statistic
print(f"OOF KS: {ks:.5f} (acuan repo: 0.51502)")
pd.DataFrame({"application_id": test["application_id"], "default_probability": test_pred}).to_csv("submission.csv", index=False)
print("submission.csv:", test_pred.shape)
