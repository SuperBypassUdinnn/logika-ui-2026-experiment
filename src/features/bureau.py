from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.features.common import CUTOFF


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
