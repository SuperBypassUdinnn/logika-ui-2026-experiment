# Ledger Eksperimen (Bahasa Indonesia)

Satu baris per eksperimen. ID = path notebook
(misal `fadhlur/exp001`). Kolom: ID | Tanggal | Fitur/Model | OOF KS | Fold KS | LB | Status | Log.

| ID | Tanggal | Fitur/Model | OOF KS | Fold KS | LB | Status | Log |
|:---|:---|:---|:---:|:---|:---:|:---:|:---|
| exp000 | - | baseline mean 0.125 | - | - | - | planned | - |
| fadhlur/exp001 | 2026-10-11 | LightGBM default, 44 fitur join leakage-safe | 0.3823 | 0.379-0.397 | - | done | outputs/experiments/fadhlur/exp001/log.md |
| fadhlur/exp002 | 2026-10-11 | Tuning 12 cfg, best lr0.03 leaves31 min100 n200 | 0.3853 | 0.377-0.402 | - | done | outputs/experiments/fadhlur/exp002/log.md |
| fadhlur/exp003 | 2026-10-11 | +6 fitur recency repayment 90d, params exp002 | 0.3930 | 0.385-0.411 | - | done | outputs/experiments/fadhlur/exp003/log.md |
| fadhlur/exp005 | 2026-10-11 | +9 fitur trajektori DPD/saldo/skor, 63 fitur | 0.3923 | 0.380-0.409 | - | done | outputs/experiments/fadhlur/exp005/log.md |
| fadhlur/exp007 | 2026-10-11 | Model per segmen histori/tanpa (negatif) | 0.3871 | 0.379-0.401 | - | done | outputs/experiments/fadhlur/exp007/log.md |
| fadhlur/exp008 | 2026-10-11 | Single model + flag segmen histori | 0.3910 | 0.381-0.404 | - | done | outputs/experiments/fadhlur/exp008/log.md |
| fadhlur/exp009 | 2026-10-11 | Episode DPD band + prev recency + volatilitas saldo (74 fitur) | 0.3926 | 0.384-0.404 | - | done | outputs/experiments/fadhlur/exp009/log.md |
| fadhlur/exp010 | 2026-10-11 | Long-train n1000 lr0.02 (negatif, overfit) | 0.3905 | 0.380-0.404 | - | done | outputs/experiments/fadhlur/exp010/log.md |
| fadhlur/exp004 | - | Interaksi rate + bureau per-provider | - | - | - | planned | docs/STRATEGI_AWAL.md |
| fadhlur/exp005 | - | Perilaku ecommerce | - | - | - | planned | docs/STRATEGI_AWAL.md |
| fadhlur/exp006 | - | Validasi time-split | - | - | - | planned | docs/STRATEGI_AWAL.md |

Aturan: tambah baris di PR yang sama dengan `outputs/experiments/expXXX/log.md`.
