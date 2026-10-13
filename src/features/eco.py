from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.features.common import CUTOFF, with_cutoff


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
        accel = (
            float(spend[half:].mean() - spend[:half].mean()) if half >= 1 else 0.0
        )
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
    items = items.merge(tfeat.reset_index(names="application_id"), on="application_id", how="left")

    od = m[m["completed"] == 1].copy()
    od["bucket"] = np.floor(od["days_since"].to_numpy() / 30).astype("int8")
    od = od[od["bucket"] < 6]
    gb = od.groupby(["application_id", "bucket"], observed=True)
    S = gb["order_total"].sum().unstack(fill_value=0).reindex(columns=range(6), fill_value=0)
    N = gb.size().unstack(fill_value=0).fillna(0).reindex(columns=range(6), fill_value=0)
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
    items = items.merge(trd.reset_index(names="application_id"), on="application_id", how="left")

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
    d = d.assign(up=up, idx=up / d["product_category"].map(cat_med).astype("float64").to_numpy())
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
