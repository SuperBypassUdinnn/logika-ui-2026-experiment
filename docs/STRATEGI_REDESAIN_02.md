# Strategi Redesain 02: Tutup Kesenjangan Data (Bahasa Indonesia)

Penulis: Fadhlur, 2026-10-11. Berdasarkan `docs/ANALISIS_STAGNASI_OPUS.md`
(Opus 4.6 via Antigravity) + verifikasi lokal. `STRATEGI_AWAL.md` dan
`STRATEGI_REDESAIN_01.md` tetap frozen.

## Koreksi terhadap analisis

- Klaim perlunya GroupKFold (D2) **tidak berlaku**: train berisi 70.000
  customer unik (1 aplikasi per customer), overlap customer train-test 0%.
  StratifiedKFold sah, tidak ada kebocoran antar fold. D2 dibuang.
- Estimasi delta (+0,005 dll.) adalah hipotesis, bukan fakta. Dieksekusi
  berurutan dan diukur, bukan borongan.

## Fase A: tutup gap kolom (eksekusi sekarang)

| ID | Isi | Dasar |
|:---|:---|:---|
| fadhlur/exp043 | `income_like_credits`, `average_balance`, `minimum_balance`, `opening_balance` + rasio `x_income_vs_observed` | A1+A2, sinyal inflasi income |
| fadhlur/exp044 | Active-loan burden: pinjaman approved yang masih jalan saat cutoff (count, cicilan, pokok) + `x_total_inst_to_income` | A2+A3, blind spot terbesar |
| fadhlur/exp045 | Bureau lintas aplikasi (histori z-score per customer, delta) + kolom revolving | B1, trajektori kredit |
| fadhlur/exp046 | Blend 3-model di matriks baru + evaluasi KS kustom untuk ES | C1+C4 |

## Fase B: optimasi model (setelah A terbukti)

- Optuna 50 trial (C2), multi-seed 5x (C3), adversarial validation (C6, diagnostik).
- Ablasi grup dengan metrik KS (ikuti Farhan, bukan gain).

## Aturan main tetap

Satu exp satu artefak + baris ledger + log.md. Gate submit: OOF > terbaik,
spread fold < 0,03, audit lolos. Target OOF 0,53.
