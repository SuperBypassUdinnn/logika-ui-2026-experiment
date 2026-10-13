# Submission Log (Bahasa Indonesia)

Berat: KS. Kuota: 3/hari. OOF KS adalah gate, bukan LB probing.

| Sub | Tanggal | File | OOF KS | Public KS | Kuota | Notebook | Notes |
|:---|:---|:---|:---:|:---:|:---:|:---|:---|
| sub01 | 2026-10-11 | outputs/submissions/sub01/submission.csv | 0.4543 | 0.50641 | 11 Okt (1/3) | notebooks/current/fadhlur (81e92a3) | exp018, ref 57070953 |
| sub02 | 2026-10-12 | outputs/submissions/sub02/submission.csv | 0.5103 | 0.53885 | 12 Okt | outputs/experiments/fadhlur/exp052 | exp052, ref 57087574 |
| sub03 | 2026-10-12 | outputs/submissions/sub03/submission.csv | 0.5128 | 0.54504 | 12 Okt | outputs/experiments/fadhlur/exp068 | exp068 blend, ref 57112235 |
| sub05 | 2026-10-13 | outputs/submissions/sub05/submission.csv | 0.5208 | 0.55424 | 13 Okt | outputs/experiments/fadhlur/exp085 | exp085 3-way rank, ref 57158818 |
| farhan-sub | 2026-10-10 | - | 0.5049 | 0.53221 | 10 Okt | notebooks/current/farhan/dsc0167-last-call-notebook.ipynb | ref 57046866, LGBM n5000+ES200 |
| farhan-bag5 | 2026-10-12 | outputs/current/farhan/submissions/sub_lgbm_ec3_bag5_ks0.5161.csv | 0.5161 | 0.54716 | 12 Okt | kernel v8 (ec3 + seed bagging) | ref 57093334 |
| farhan-blend | 2026-10-12 | outputs/experiments/farhan/ver14/submissions/sub_blend_lgbm_lr_w0.3_ks0.5173.csv | 0.5173 | 0.55173 | 12 Okt | kernel v14 (5x5 bag + LR + rank blend) | ref 57121199 |

Setiap submission wajib: `audit_submission.py` lolos, `log.md` blend (jika ensemble),
dan rationale 4 poin (delta, hipotesis, dampak OOF, risiko).
