from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.features.common import CUTOFF


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
