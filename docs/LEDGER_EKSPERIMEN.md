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
| fadhlur/exp020 | 2026-10-11 | XGBoost di 108 fitur | 0.4717 | - | - | done | outputs/experiments/fadhlur/exp020/log.md |
| fadhlur/exp021 | 2026-10-11 | CatBoost kategorikal natif 108 fitur | 0.4694 | - | - | done | outputs/experiments/fadhlur/exp021/log.md |
| fadhlur/exp022 | 2026-10-11 | Blend 50/30/20 exp019+020+021 | 0.4757 | - | - | done | outputs/experiments/fadhlur/exp022/log.md |
| fadhlur/exp023 | 2026-10-11 | Stacker logistik (di bawah blend) | 0.4725 | - | - | done | outputs/experiments/fadhlur/exp023/log.md |
| fadhlur/exp024 | 2026-10-11 | Rasio lintas-tabel ke income (119 fitur) | 0.4709 | - | - | done | outputs/experiments/fadhlur/exp024/log.md |
| fadhlur/exp025 | 2026-10-11 | Drop rate_x (ablasi) | 0.4734 | 0.453-0.485 | - | done | outputs/experiments/fadhlur/exp025/log.md |
| fadhlur/exp026 | 2026-10-11 | Eco recency 30/90d + max order | 0.4744 | - | - | done | outputs/experiments/fadhlur/exp026/log.md |
| fadhlur/exp027 | 2026-10-11 | Eco velocity + maxdpd + overdue amt (133 fitur) | 0.4972 | 0.489-0.510 | - | done | outputs/experiments/fadhlur/exp027/log.md |
| fadhlur/exp028 | 2026-10-11 | XGBoost di 133 fitur | 0.4957 | - | - | done | outputs/experiments/fadhlur/exp028/log.md |
| fadhlur/exp029 | 2026-10-11 | CatBoost 133 fitur | 0.4963 | - | - | done | outputs/experiments/fadhlur/exp029/log.md |
| fadhlur/exp030 | 2026-10-11 | Blend 40/30/30 (027+028+029) | 0.4987 | - | - | done | outputs/experiments/fadhlur/exp030/log.md |
| fadhlur/exp031 | 2026-10-11 | Retune min_child 20 di 133 fitur | 0.4984 | - | - | done | outputs/experiments/fadhlur/exp031/log.md |
| fadhlur/exp032 | 2026-10-11 | Blend 40/30/30 (031+028+029) | 0.4990 | - | - | done | outputs/experiments/fadhlur/exp032/log.md |
| fadhlur/exp033 | 2026-10-11 | Streak + nmiss bureau (datar) | 0.4984 | - | - | done | outputs/experiments/fadhlur/exp033/log.md |
| fadhlur/exp034 | 2026-10-11 | Agregat recency-weighted | 0.4978 | - | - | done | outputs/experiments/fadhlur/exp034/log.md |
| fadhlur/exp036 | 2026-10-11 | late_ratio + overdue_amt_ratio | 0.4971 | - | - | done | outputs/experiments/fadhlur/exp036/log.md |
| fadhlur/exp037 | 2026-10-11 | Gap order + z-score bureau (146 fitur) | 0.5003 | 0.487-0.512 | - | done | outputs/experiments/fadhlur/exp037/log.md |
| fadhlur/exp038 | 2026-10-11 | XGBoost di 146 fitur | 0.4983 | - | - | done | outputs/experiments/fadhlur/exp038/log.md |
| fadhlur/exp039 | 2026-10-11 | CatBoost 146 fitur | 0.4994 | - | - | done | outputs/experiments/fadhlur/exp039/log.md |
| farhan/exp-final | 2026-10-11 | Repro notebook farhan (OOF referensi) | 0.5050 | 0.482-0.519 | 0.53221 | done | outputs/experiments/farhan/exp-final/log.md |
| fadhlur/exp040 | 2026-10-11 | Blend 55/20/15/10 farhan+LGBM+XGB+CAT | 0.5056 | 0.495-0.518 | - | done | outputs/experiments/fadhlur/exp040/log.md |
| fadhlur/exp041 | 2026-10-11 | Eco kategori×recency + tenure (167 fitur) | 0.5039 | 0.488-0.518 | - | done | outputs/experiments/fadhlur/exp041/log.md |
| fadhlur/exp042 | 2026-10-11 | Blend 70/30 farhan+exp041 | 0.5061 | - | - | done | outputs/experiments/fadhlur/exp042/log.md |
| fadhlur/exp043 | 2026-10-11 | Kolom account + x_income_vs_observed, ES-KS (eval KS ≈ AUC) | 0.5028 | 0.496-0.517 | - | done | outputs/experiments/fadhlur/exp043/log.md |
| fadhlur/exp044 | 2026-10-11 | Active-loan burden + rasio beban (186 fitur) | 0.5051 | 0.498-0.518 | - | done | outputs/experiments/fadhlur/exp044/log.md |
| fadhlur/exp045 | 2026-10-11 | Bureau lintas aplikasi + z-delta | 0.5056 | - | - | done | outputs/experiments/fadhlur/exp045/log.md |
| fadhlur/exp046 | 2026-10-11 | XGBoost di 186 fitur | 0.5018 | - | - | done | outputs/experiments/fadhlur/exp046/log.md |
| fadhlur/exp047 | 2026-10-11 | CatBoost 186 fitur | 0.5058 | - | - | done | outputs/experiments/fadhlur/exp047/log.md |
| fadhlur/exp048 | 2026-10-11 | Blend 40/25/20/15 farhan+LGBM+XGB+CAT | 0.5088 | - | - | done | outputs/experiments/fadhlur/exp048/log.md |
| fadhlur/exp049 | 2026-10-12 | Optuna 40 trial + setup penuh | 0.5080 | 0.497-0.522 | - | done | outputs/experiments/fadhlur/exp049/log.md |
| fadhlur/exp050 | 2026-10-12 | Blend 45/30/15/10 farhan+049+XGB+CAT | 0.5086 | - | - | done | outputs/experiments/fadhlur/exp050/log.md |
| fadhlur/exp051 | 2026-10-12 | Bagging 5 seed config optuna (di bawah single) | 0.5074 | - | - | done | outputs/experiments/fadhlur/exp051/log.md |
| fadhlur/exp052 | 2026-10-12 | Detail prev term/sched/gap (190 fitur) | 0.5103 | 0.499-0.524 | - | done | outputs/experiments/fadhlur/exp052/log.md |
| fadhlur/exp053 | 2026-10-12 | XGBoost di 190 fitur | 0.5030 | - | - | done | outputs/experiments/fadhlur/exp053/log.md |
| fadhlur/exp054 | 2026-10-12 | CatBoost 190 fitur | 0.5052 | - | - | done | outputs/experiments/fadhlur/exp054/log.md |
| fadhlur/exp055 | 2026-10-12 | Blend 30/35/20/15 (jenuh, di bawah single) | 0.5100 | - | - | done | outputs/experiments/fadhlur/exp055/log.md |
| fadhlur/exp056 | 2026-10-12 | Tren lintas-loan (mayoritas 1 loan, datar) | 0.5074 | 0.495-0.521 | - | done | outputs/experiments/fadhlur/exp056/log.md |
| fadhlur/exp071 | 2026-10-12 | Cross top-signal eksplisit (negatif) | 0.5084 | - | - | done | outputs/experiments/fadhlur/exp071/log.md |
| fadhlur/exp057 | 2026-10-12 | LambdaRank chunk 10k (negatif, query artifisial) | 0.3448 | 0.390-0.433 | - | done | outputs/experiments/fadhlur/exp057/log.md |
| fadhlur/exp058 | 2026-10-12 | Prune acct+provider+rate | 0.5091 | - | - | done | outputs/experiments/fadhlur/exp058/log.md |
| fadhlur/exp059 | 2026-10-12 | Blend 35/35/15/15 farhan+052+XGB+CAT (jenuh) | 0.5107 | - | - | done | outputs/experiments/fadhlur/exp059/log.md |
| fadhlur/exp060 | 2026-10-12 | XGBoost matriks berjalan | 0.5019 | - | - | done | outputs/experiments/fadhlur/exp060/log.md |
| fadhlur/exp061 | 2026-10-12 | CatBoost matriks berjalan | 0.5038 | - | - | done | outputs/experiments/fadhlur/exp061/log.md |
| fadhlur/exp062 | 2026-10-12 | Optuna 40 trial n100 (lemah, tanpa ES) | 0.4962 | - | - | done | outputs/experiments/fadhlur/exp062/log.md |
| fadhlur/exp064 | 2026-10-12 | Hyper optuna + setup penuh ES | 0.5103 | - | - | done | outputs/experiments/fadhlur/exp064/log.md |
| fadhlur/exp065 | 2026-10-12 | Port ec3 + optuna hypers (negatif: -0,004) | 0.5063 | - | - | done | outputs/experiments/fadhlur/exp065/log.md |
| fadhlur/exp066 | 2026-10-12 | XGBoost matriks 206 | 0.5019 | - | - | done | outputs/experiments/fadhlur/exp066/log.md |
| fadhlur/exp067 | 2026-10-12 | CatBoost matriks 206 | 0.5082 | - | - | done | outputs/experiments/fadhlur/exp067/log.md |
| fadhlur/exp068 | 2026-10-12 | Blend 40/30/15/15 ec3+064+XGB+CAT | 0.5128 | - | - | done | outputs/experiments/fadhlur/exp068/log.md |
| fadhlur/exp069 | 2026-10-12 | Optuna 40 trial matriks 206 n100 | 0.5008 | - | - | done | outputs/experiments/fadhlur/exp069/log.md |
| fadhlur/exp070 | 2026-10-12 | Config optuna-206 + setup penuh | 0.5070 | 0.502-0.521 | - | done | outputs/experiments/fadhlur/exp070/log.md |
| fadhlur/exp006 | 2026-10-11 | Validasi time-split (robust, gap 0.003) | 0.4954 | - | - | done | outputs/experiments/fadhlur/exp006/log.md |
| fadhlur/exp006 | - | Validasi time-split | - | - | - | planned | docs/STRATEGI_AWAL.md |
| farhan/exp-ec3 | 2026-10-12 | Modul ec3 kernel v8 (window 7/14d + spike) | 0.5121 | - | - | done | outputs/experiments/farhan/exp-ec3/log.md |
| farhan/exp-blend | 2026-10-12 | Kernel v14: 5x5 bag + LR + rank blend w0.3 | 0.5173 | 0.55173 | - | done | outputs/experiments/farhan/ver14/ |
| fadhlur/exp063 | 2026-10-12 | Blend 60/40 farhan-ec3+exp052 | 0.5132 | - | - | done | outputs/experiments/fadhlur/exp063/log.md |

Aturan: tambah baris di PR yang sama dengan `outputs/experiments/expXXX/log.md`.
