# Ledger Eksperimen (Bahasa Indonesia)

Satu baris per eksperimen. ID = path notebook
(misal `fadhlur/exp001`). Kolom: ID | Tanggal | Fitur/Model | OOF KS | Fold KS | LB | Status | Log.

| ID | Tanggal | Fitur/Model | OOF KS | Fold KS | LB | Status | Log |
|:---|:---|:---|:---:|:---|:---:|:---:|:---|
| exp000 | - | baseline mean 0.125 | - | - | - | planned | - |
| fadhlur/exp001 | 2026-10-11 | LightGBM default, 44 fitur join leakage-safe | 0.3823 | 0.379-0.397 | - | done | outputs/experiments/fadhlur/exp001/log.md |
| fadhlur/exp002 | - | Tuning LightGBM | - | - | - | planned | docs/STRATEGI_AWAL.md |
| fadhlur/exp003 | - | Fitur tren repayment recency | - | - | - | planned | docs/STRATEGI_AWAL.md |
| fadhlur/exp004 | - | Interaksi rate + bureau per-provider | - | - | - | planned | docs/STRATEGI_AWAL.md |
| fadhlur/exp005 | - | Perilaku ecommerce | - | - | - | planned | docs/STRATEGI_AWAL.md |
| fadhlur/exp006 | - | Validasi time-split | - | - | - | planned | docs/STRATEGI_AWAL.md |

Aturan: tambah baris di PR yang sama dengan `outputs/experiments/expXXX/log.md`.
