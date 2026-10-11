# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
# ---

# %% [markdown]
# EDA Awal: Target, Sinyal, dan Cakupan (fadhlur)
#
# Ringkasan temuan awal yang mengarahkan strategi fitur.
# Data: `data/*.csv`. Semua angka dihitung di notebook ini.

# %% [markdown]
# ## 0. Setup

# %%
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def find_repo_root(marker="data/train_applications.csv"):
    d = Path.cwd()
    for _ in range(6):
        if (d / marker).exists():
            return d / "data"
        d = d.parent
    raise FileNotFoundError("repo root dengan data/ tidak ditemukan")

DATA = str(find_repo_root())
SEED = 42

# %% [markdown]
# ## 1. Target overall dan stabilitas waktu
#
# Target 13,58%. Stabil per bulan (12-15%) — stratified 5-fold sah,
# time-split hanya robustness check.

# %%
tr = pd.read_csv(f"{DATA}/train_applications.csv", parse_dates=["decision_datetime"])
print("train rows:", len(tr), "| positive rate:", round(tr.target.mean(), 4))
tr["ym"] = tr.decision_datetime.dt.to_period("M").astype(str)
print(tr.groupby("ym").target.agg(["mean", "count"]).round(4).to_string())
print(tr.groupby("loan_purpose").target.mean().round(4).to_string())

# %% [markdown]
# ## 2. Sinyal tunggal terkuat: skor bureau dan DPD historis
#
# Skor bureau monoton (50,7% → 6,2%). DPD maksimum historis monoton
# (8,5% → 51,8%). Keduanya dihitung strictly sebelum cutoff.

# %%
cut = tr.set_index("customer_id").decision_datetime
bur = pd.read_csv(
    f"{DATA}/bureau_observations.csv",
    usecols=["application_id", "available_datetime", "score"],
    parse_dates=["available_datetime"],
)
m = bur.merge(tr[["application_id", "decision_datetime", "target"]], on="application_id")
m = m[m.available_datetime <= m.decision_datetime]
print(m.groupby(pd.cut(m.score, [0, 500, 600, 700, 850])).target.agg(["mean", "count"]).round(4).to_string())

lc = pd.read_csv(f"{DATA}/previous_applications.csv", usecols=["loan_id", "customer_id"]).dropna().drop_duplicates("loan_id")
rp = pd.read_csv(f"{DATA}/repayment_history.csv", usecols=["loan_id", "due_date", "days_past_due"], parse_dates=["due_date"]).merge(lc, on="loan_id")
rp["cut"] = rp.customer_id.map(cut)
rp = rp[rp.cut.notna() & (rp.due_date < rp.cut)]
g = rp.groupby("customer_id").days_past_due.max()
both = pd.DataFrame({"dpdmax": g, "target": tr.set_index("customer_id").target}).dropna()
print(both.groupby(pd.cut(both.dpdmax, [-1, 0, 30, 90, 9999])).target.agg(["mean", "count"]).round(4).to_string())

# %% [markdown]
# ## 3. Ranking KS univariat fitur engineered
#
# Dibangun oleh `scripts/build_features.py` (87 fitur exp018).
# Teratas: `repay_pay_ratio` 0,322, `repay_dpd_max` 0,315,
# `bureau_score_mean` 0,253, `annual_interest_rate` 0,230.

# %%
f = pd.read_parquet(f"{DATA}/train_features.parquet")
y = f.target.to_numpy()
rows = []
for c in f.columns:
    if c in ("application_id", "customer_id", "target"):
        continue
    v = f[c]
    try:
        v = v.astype(float).to_numpy()
    except (ValueError, TypeError):
        v = pd.factorize(v.astype(str))[0].astype(float)
    msk = ~np.isnan(v)
    if msk.sum() < 100 or len(np.unique(v[msk])) < 2:
        continue
    ks = ks_2samp(v[(y == 1) & msk], v[(y == 0) & msk]).statistic
    rows.append((c, round(float(ks), 4)))
for r in sorted(rows, key=lambda r: -r[1])[:15]:
    print(r)

# %% [markdown]
# ## 4. Cakupan histori dan audit leakage
#
# Riwayat repayment hanya ada di 46k/70k train — segmen tanpa histori
# butuh penanganan khusus. Audit: 0 baris bureau melewati cutoff
# (filter `available_datetime <= decision_datetime` bekerja).

# %%
print("apps with repay history:", int((f["repay_n"] > 0).sum()), "/", len(f))
bur2 = pd.read_csv(f"{DATA}/bureau_observations.csv", usecols=["application_id", "available_datetime"], parse_dates=["available_datetime"])
cutapp = dict(zip(tr.application_id, tr.decision_datetime))
linked = bur2[bur2.application_id.isin(cutapp)]
future = (linked.available_datetime > linked.application_id.map(cutapp)).sum()
print("bureau rows linked:", len(linked), "| future rows:", int(future))
print("DONE - temuan diringkas di docs/STRATEGI_AWAL.md")
