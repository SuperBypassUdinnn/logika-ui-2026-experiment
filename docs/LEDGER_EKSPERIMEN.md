# Ledger Eksperimen (Bahasa Indonesia)

Satu baris per eksperimen. ID = path notebook
(misal `fadhlur/exp001`). Kolom: ID | Tanggal | Fitur/Model | OOF KS | Fold KS | LB | Status | Log.

| ID | Tanggal | Fitur/Model | OOF KS | Fold KS | LB | Status | Log |
|:---|:---|:---|:---:|:---|:---:|:---:|:---|
| exp000 | - | baseline mean 0.125 | - | - | - | planned | - |
| fadhlur/exp001 | 2026-10-11 | LightGBM default, 44 fitur join leakage-safe | 0.3823 | 0.379-0.397 | - | done | outputs/experiments/fadhlur/exp001/log.md |
| fadhlur/exp002 | 2026-10-11 | Tuning 12 cfg, best lr0.03 leaves31 min100 n200 | 0.3853 | 0.377-0.402 | - | done | outputs/experiments/fadhlur/exp002/log.md |
| fadhlur/exp003 | 2026-10-11 | +6 fitur recency repayment 90d, params exp002 | 0.3930 | 0.385-0.411 | - | done | outputs/experiments/fadhlur/exp003/log.md |
| fadhlur/exp004 | 2026-10-11 | Interaksi rate + bureau per-provider (69 fitur) | 0.3917 | 0.385-0.402 | - | done | outputs/experiments/fadhlur/exp004/log.md |
| fadhlur/exp005 | 2026-10-11 | +9 fitur trajektori DPD/saldo/skor, 63 fitur | 0.3923 | 0.380-0.409 | - | done | outputs/experiments/fadhlur/exp005/log.md |
| fadhlur/exp007 | 2026-10-11 | Model per segmen histori/tanpa (negatif) | 0.3871 | 0.379-0.401 | - | done | outputs/experiments/fadhlur/exp007/log.md |
| fadhlur/exp008 | 2026-10-11 | Single model + flag segmen histori | 0.3910 | 0.381-0.404 | - | done | outputs/experiments/fadhlur/exp008/log.md |
| fadhlur/exp009 | 2026-10-11 | Episode DPD band + prev recency + volatilitas saldo (74 fitur) | 0.3926 | 0.384-0.404 | - | done | outputs/experiments/fadhlur/exp009/log.md |
| fadhlur/exp010 | 2026-10-11 | Long-train n1000 lr0.02 (negatif, overfit) | 0.3905 | 0.380-0.404 | - | done | outputs/experiments/fadhlur/exp010/log.md |
| fadhlur/exp011 | 2026-10-11 | scale_pos_weight 3 | 0.3945 | 0.388-0.412 | - | done | outputs/experiments/fadhlur/exp011/log.md |
| fadhlur/exp012 | 2026-10-11 | Drop 15 fitur gain-rendah (negatif) | 0.3908 | 0.383-0.411 | - | done | outputs/experiments/fadhlur/exp012/log.md |
| fadhlur/exp013 | 2026-10-11 | XGBoost mirror setup LGBM | 0.3949 | 0.388-0.406 | - | done | outputs/experiments/fadhlur/exp013/log.md |
| fadhlur/exp014 | 2026-10-11 | Blend 60/40 exp011+exp013 | 0.3964 | - | - | done | outputs/experiments/fadhlur/exp014/log.md |
| fadhlur/exp015 | 2026-10-11 | Agregat per-loan worst/pay-ratio (redundan) | 0.3936 | 0.381-0.412 | - | done | outputs/experiments/fadhlur/exp015/log.md |
| fadhlur/exp016 | 2026-10-11 | Bagging 3 seed LGBM (negatif) | 0.3941 | - | - | done | outputs/experiments/fadhlur/exp016/log.md |
| fadhlur/exp017 | 2026-10-11 | CatBoost kategorikal natif | 0.3934 | - | - | done | outputs/experiments/fadhlur/exp017/log.md |
| fadhlur/exp018 | 2026-10-11 | Eco kategori+payment, streak prev, spread bureau (87 fitur) | 0.4543 | 0.444-0.475 | - | done | outputs/experiments/fadhlur/exp018/log.md |
| fadhlur/exp019 | 2026-10-11 | days_since_* + ES200 n5000 reg5 (108 fitur) | 0.4730 | 0.456-0.491 | - | done | outputs/experiments/fadhlur/exp019/log.md |
| fadhlur/exp006 | - | Validasi time-split | - | - | - | planned | docs/STRATEGI_AWAL.md |

Aturan: tambah baris di PR yang sama dengan `outputs/experiments/expXXX/log.md`.
