# Ledger Eksperimen (Bahasa Indonesia)

Satu baris per eksperimen. ID = path notebook
(misal `fadhlur/exp001`). Kolom: ID | Tanggal | Fitur/Model | OOF KS | Fold KS | LB | Status | Log.

| ID | Tanggal | Fitur/Model | OOF KS | Fold KS | LB | Status | Log |
|:---|:---|:---|:---:|:---|:---:|:---:|:---|
| exp000 | - | baseline mean 0.125 | - | - | - | planned | - |
| fadhlur/exp001 | 2026-10-11 | LightGBM default, 44 fitur join leakage-safe | 0.3823 | 0.379-0.397 | - | done | outputs/experiments/fadhlur/exp001/log.md |
| fadhlur/exp003 | 2026-10-11 | +6 fitur recency repayment 90d, params exp002 | 0.3930 | 0.385-0.411 | - | done | outputs/experiments/fadhlur/exp003/log.md |

Aturan: tambah baris di PR yang sama dengan `outputs/experiments/expXXX/log.md`.
