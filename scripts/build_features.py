"""Build leakage-safe application-level features for all tables.

Every auxiliary row is used only if its event timestamp is strictly
before the scored application's decision_datetime (bureau uses
available_datetime <= decision_datetime).

Usage:
    python scripts/build_features.py --data data \
        --out-train data/train_features.parquet \
        --out-test data/test_features.parquet

Outputs are ignored by git (data/*.parquet); they are regenerable.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CUTOFF = "decision_datetime"


def scored_apps(data: Path) -> pd.DataFrame:
    train = pd.read_csv(
        data / "train_applications.csv",
        usecols=[
            "application_id",
            "customer_id",
            "decision_datetime",
            "loan_amount",
            "term_months",
            "annual_interest_rate",
            "proposed_monthly_installment",
            "loan_purpose",
            "target",
        ],
        parse_dates=["decision_datetime"],
    )
    train["is_train"] = True
    test = pd.read_csv(
        data / "test_applications.csv",
        usecols=[
            "application_id",
            "customer_id",
            "decision_datetime",
            "loan_amount",
            "term_months",
            "annual_interest_rate",
            "proposed_monthly_installment",
            "loan_purpose",
        ],
        parse_dates=["decision_datetime"],
    )
    test["is_train"] = False
    return pd.concat([train, test], ignore_index=True)


def with_cutoff(aux: pd.DataFrame, apps: pd.DataFrame) -> pd.DataFrame:
    cut = apps[["customer_id", "application_id", CUTOFF]]
    return aux.merge(cut, on="customer_id", how="inner")


def agg_prev(data: Path, apps: pd.DataFrame) -> pd.DataFrame:
    prev = pd.read_csv(
        data / "previous_applications.csv",
        usecols=[
            "application_id",
            "customer_id",
            "decision_datetime",
            "requested_amount",
            "decision",
            "approved_amount",
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
    return out


def agg_repay(data: Path, apps: pd.DataFrame) -> pd.DataFrame:
    loan_customer = pd.read_csv(
        data / "previous_applications.csv",
        usecols=["loan_id", "customer_id"],
    ).dropna(subset=["loan_id"])
    loan_customer = loan_customer.drop_duplicates("loan_id")
    rep = pd.read_csv(
        data / "repayment_history.csv",
        usecols=[
            "loan_id",
            "due_date",
            "amount_due",
            "amount_paid",
            "days_past_due",
        ],
        parse_dates=["due_date"],
    )
    rep = rep.merge(loan_customer, on="loan_id", how="inner")
    m = with_cutoff(rep, apps)
    m = m[m["due_date"] < m[CUTOFF]]
    m["overdue"] = (m["days_past_due"] > 0).astype("int8")
    m["dpd_band"] = pd.cut(
        m["days_past_due"], [-1, 0, 30, 90, 10**9], labels=["ok", "mild", "bad", "severe"]
    )
    m["cured"] = ((m["days_past_due"] > 0) & (m["amount_paid"] >= m["amount_due"])).astype(
        "int8"
    )
    m["recent"] = (m["due_date"] >= m[CUTOFF] - pd.Timedelta(days=90)).astype("int8")
    g = m.groupby("application_id")
    out = pd.DataFrame(
        {
            "repay_n": g.size(),
            "repay_due_sum": g["amount_due"].sum(),
            "repay_paid_sum": g["amount_paid"].sum(),
            "repay_dpd_max": g["days_past_due"].max(),
            "repay_overdue_n": g["overdue"].sum(),
            "repay_mild_n": g["dpd_band"].apply(lambda s: (s == "mild").sum()),
            "repay_bad_n": g["dpd_band"].apply(lambda s: (s == "bad").sum()),
            "repay_severe_n": g["dpd_band"].apply(lambda s: (s == "severe").sum()),
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
    out["repay_recent_pay_ratio"] = out["repay_recent_paid_sum"] / out[
        "repay_recent_due_sum"
    ].clip(lower=1)
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
        return pd.Series(
            {
                "repay_last3_dpd_mean": float(last3.mean()),
                "repay_dpd_slope": slope,
                "repay_worsening": int(last3.max() > older.max()),
                "repay_months_since_delinq": months_since,
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
            "closing_balance",
            "inflows",
            "outflows",
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
    return out


def agg_bureau(data: Path, apps: pd.DataFrame) -> pd.DataFrame:
    bur = pd.read_csv(
        data / "bureau_observations.csv",
        usecols=[
            "application_id",
            "report_datetime",
            "available_datetime",
            "score",
            "outstanding_balance",
            "utilization",
            "inquiries_last_6m",
            "delinquent_accounts_last_12m",
        ],
        parse_dates=["report_datetime", "available_datetime"],
    )
    cut = apps[["application_id", CUTOFF]]
    m = bur.merge(cut, on="application_id", how="inner")
    m = m[m["available_datetime"] <= m[CUTOFF]]
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
        }
    ).reset_index()

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
        }
    ).reset_index()
    return orders.merge(items, on="application_id", how="outer")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out-train", default="data/train_features.parquet")
    ap.add_argument("--out-test", default="data/test_features.parquet")
    args = ap.parse_args()
    data = Path(args.data)

    apps = scored_apps(data)
    print(f"scored apps: {len(apps)}")

    profile = pd.read_csv(data / "customer_profiles.csv")
    feat = apps.merge(profile, on="customer_id", how="left")
    feat["loan_to_income"] = feat["loan_amount"] / feat["declared_monthly_income"].clip(
        lower=1
    )
    feat["decision_month"] = feat[CUTOFF].dt.month.astype("int8")
    feat["decision_weekday"] = feat[CUTOFF].dt.weekday.astype("int8")

    for name, fn in [
        ("prev", agg_prev),
        ("repay", agg_repay),
        ("account", agg_account),
        ("bureau", agg_bureau),
        ("eco", agg_eco),
    ]:
        part = fn(data, apps)
        print(f"{name}: {len(part)} applications with history")
        feat = feat.merge(part, on="application_id", how="left")

    for c in feat.columns:
        if c.startswith(
            (
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
        ):
            feat[c] = feat[c].fillna(0)
    feat["has_repay_history"] = (feat["repay_n"] > 0).astype("int8")
    feat["has_account_history"] = (feat["acct_n_months"] > 0).astype("int8")

    train = feat[feat["is_train"]].drop(columns=["is_train"])
    test = feat[~feat["is_train"]].drop(columns=["is_train", "target"])
    train.to_parquet(args.out_train, index=False)
    test.to_parquet(args.out_test, index=False)
    print(f"train {train.shape} -> {args.out_train}")
    print(f"test {test.shape} -> {args.out_test}")
    print(f"train positive rate: {train['target'].mean():.4f}")


if __name__ == "__main__":
    main()
