# Temuan Eksperimen exp001–exp042 (Bahasa Indonesia)

Disusun 2026-10-11 oleh Fadhlur. Perjalanan OOF KS 0,3823 → 0,5061.
Detail per eksperimen: `docs/LEDGER_EKSPERIMEN.md` + `outputs/experiments/*/*/log.md`.

## Papan skor (OOF KS)

| Rentang | Best | Catatan |
|:---|:---:|:---|
| exp001–exp017 (agregat dasar, tuning, 1 model) | 0,3964 | Blend LGBM+XGB pertama |
| exp018–exp026 (eco mix, ES, recency, ablasi) | 0,4744 | Lompatan dari spend kategori |
| exp027–exp032 (velocity, 3 model, stacker) | 0,4990 | Blend 3 arah |
| exp033–exp042 (streak, z-score, tenure, blend Farhan) | 0,5061 | Terbaik saat ini (exp042) |
| Referensi Farhan (repro bit-identik) | 0,5050 | Public LB 0,53221 |

## Temuan 1: Ecommerce adalah penentu (ablasi −0,089)

Group-drop pada 146 fitur (baseline ~0,500):

| Grup dibuang | OOF KS | Delta |
|:---|:---:|:---:|
| eco_* | 0,4116 | −0,089 |
| repay_* | 0,4568 | −0,043 |
| bureau_* | 0,4861 | −0,014 |
| prev_* | 0,4977 | −0,002 |
| acct_* | 0,4994 | netral |

Spend per kategori (groceries, personal_care, fashion), velocity
30/180 hari, dan recency transaksi adalah fitur non-repayment terkuat.
Implikasi: gali perilaku belanja lebih dalam sebelum menyentuh grup lain.

## Temuan 2: Early stopping wajib, n fixed menipu

exp010 (n=1000 fixed) overfit 0,3905. Farhan (n=5000 + ES200,
`best_iteration` ~550–680) + reg_lambda 5,0 membawa 0,4730.
Jangan pernah train fixed-n besar tanpa ES.

## Temuan 3: Segmentasi model gagal, segmentasi sinyal berhasil

- Model per segmen (exp007): 0,3871 < model tunggal. Split menghilangkan
  cross-segment learning; segmen lemah makin lemah.
- Yang berhasil: flag histori + fitur khusus segmen dalam satu model.
- Bukti(segmen, OOF exp003): berhistori 0,4162 vs tanpa histori 0,3441.
  Segmen tanpa histori miskin sinyal (bureau/eco/profil saja) — butuh
  sinyal baru, bukan arsitektur baru.

## Temuan 4: Tuning dan seleksi fitur jenuh cepat

- Grid 12 config: +0,003. Retune di 133 fitur: +0,001.
- Drop 15 fitur gain-rendah: −0,004 (kombinasi yang dibuang ternyata berguna).
- Bagging 3 seed: +0,000. Long-train: negatif.
- Rank-average blend ≈ plain average (skala antar model tidak masalah).

## Temuan 5: Diversitas model membantu kecil tapi konsisten

LGBM ≈ XGB ≈ CatBoost (selisih < 0,005), blend 3–4 arah +0,002–0,004.
Stacker logistik/LGBM di bawah blend rata-rata. Korelasi OOF antar
model ~0,92 — blend ada plafonnya.

## Temuan 6: Validasi time-split robust

exp006 (fold per kuartal): 0,4954 vs stratified 0,4984 (gap 0,003).
Tidak ada drift temporal yang mengkhawatirkan.

## Temuan 7: Reproduksi notebook Farhan bit-identik
Pipeline Farhan dijalankan lokal menghasilkan OOF 0,5049 persis
(fold per fold sama). Artefak OOF/test-nya diarsipkan di
`outputs/experiments/farhan/exp-final/` untuk blend reproducible.
Pelajaran metode darinya: ES, `days_since_*`, velocity, maxdpd,
overdue-amount ratio, ablasi grup.

## Jalan terbuka ke 0,53 (gap 0,024)

1. Sinyal untuk segmen tanpa histori (prioritas 1, pemilik: Ilham?).
2. Sequence-level repayment (transisi bucket DPD, bukan agregat).
3. Blend dengan OOF Ilham saat tersedia (diversitas manusia > diversitas model).
4. Interaksi silang baru yang belum dicoba (burden-to-income temporal).

## Temuan 8: Metrik bawaan menentukan early stopping, bukan feval

- Notebook Kaggle OOF 0,474 vs repo 0,510 dengan fitur/fold/params "sama".
  Diagnosis (best_iteration): notebook berhenti iter ~43, repo ~477.
- Penyebab: PARAMS notebook memakai default `binary_logloss`, repo memakai
  `metric='auc'` dari `src/model.py`. Pilihan metrik bawaan mengubah lintasan
  ES meski KS feval identik terpasang di keduanya. Determinisme/threads
  terbukti tidak berpengaruh (uji pemisah).
- Setelah PARAMS disamakan plek: notebook 0,5086 vs repo 0,5074 (paritas).
- Aturan: PARAMS notebook generator tidak boleh disederhanakan; harus mirror
  `make_lgbm` (metric, deterministic, num_threads).

## Temuan 9: Fitur ec3 spesifik-pipeline (2026-10-12)

- Port window 7/14 + spike Farhan ke pipeline sendiri: 0,5103 -> 0,5063
  (-0,004) di dua konfigurasi (optuna-reg dan Farhan-like). Ablasi
  konfirmasi: tanpa ec3_* kembali 0,5103.
- ec3 membantu model Farhan (+0,007) tapi noise di pipeline Dusina:
  interaksi dengan regularisasi/kapasitas berbeda. Fitur tidak selalu
  transferable antar pipeline — uji ablasi sebelum mengadopsi.
- Kode port tetap di `src/features/eco.py` (atribusi), di-drop saat train.

## Temuan 10: Resep stabil Farhan v14 (2026-10-12)

- LGBM 5 seed x 5 fold (25 model) + bagging: OOF 0,5161.
- Pendamping regresi logistik (QuantileTransformer + OneHot, C=0,03,
  OOF per seed yang sama) + rank-blend 0,7 LGBM / 0,3 LR.
- Gerbang bootstrap 200x sebelum menerima blend (selisih harus > 0
  dengan keyakinan 95%).
- Hasil blend OOF 0,5173 -> public 0,55173 (ref 57121199).
- Mengapa stabil: penghancur varians (25 model) + diversitas linear
  + ruang peringkat (kebal kalibrasi) + disiplin statistik.
- Modul ec3 v14 melebar: time (porsi order malam), trd (tren bulanan),
  prc (indeks harga vs median), reg (ritme), cncl rinci.
- Ditiru berikutnya: pendamping LR + rank-blend + gerbang bootstrap
  (`scripts/train_lr.py` sudah ada, tinggal blend peringkat).

## Temuan 11: Ablasi union dua pipeline (2026-10-13)

Drop grup Farhan dari matriks union 445 kolom (baseline ~0,507):

| Grup dibuang | OOF KS |
|:---|:---:|
| (penuh) | ~0,507 |
| ec2_* | 0,5096 |
| bur2_* | 0,5101 |
| rep2_* | 0,5093 |
| prev2_* | 0,5096 |
| x_* | 0,5088 |

Semua drop netral/positif: grup Farhan redundan penuh dengan fitur
kita (dan sebaliknya, blend jenuh korelasi 0,92). Dua pipeline
konvergen ke ranking yang sama. Sisa gap ke 0,53 butuh sinyal baru,
bukan reshuffle fitur lama. Top milik kita di union: x_pay_x_score,
repay_maxdpd_mean, eco window/tenure/gap, hist_z.
