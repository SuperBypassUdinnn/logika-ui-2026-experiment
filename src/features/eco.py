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
    return orders.merge(items, on="application_id", how="outer")
