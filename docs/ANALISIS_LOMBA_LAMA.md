# Analisis Ringkas Lomba Vision Lama (Bahasa Indonesia)

Sumber: `~/.clone/Kaggle/data-science-infest-2026` (7 kelas aksara,
3.771 train, 1.220 test, 16 submission sub01-sub16, 26 eksperimen exp000-exp025).

## Hasil

- OOF Macro-F1 0.9569 -> 0.9794, Public LB 0.64271 -> 0.84178 (peak sub14).
- Pola: tiap lompatan besar dari scaling model (ResNet -> DenseNet/Swin ->
  ConvNeXt-V2-Base + Swin-V2-Base), bukan dari trik.
- Blend 50:50 (`sub06,07,09,11,16`) memberi +0.003-0.011 tapi tanpa kode beku
  untuk sub03,06,07,11,12,13.

## Penyebab gap OOF-LB dan pelajaran tabular

1. OOF 0.97 vs LB 0.64 awal: distribusi train/test beda + model bias mayoritas
   (Jawa over, Sunda/Bali under). Pelajaran: jangan percaya satu OOF;
   wajib fold-wise + time-split + distribusi prediksi.
2. Optimal Transport balancing (+0.08 hipotesis) ternyata kalah vs raw argmax
   (+0.033): asumsi distribusi test seragam salah. Pelajaran: uji asumsi via
   A/B kecil, jangan kalibrasi buta (relevan untuk KS: jangan tuning prior
   tanpa bukti OOF).
3. Kelas tersulit Pegon (F1 0.92 vs 0.99 lain): diakritik 2px. Solusi efektif
   resolusi + arsitektur modern, bukan ensemble lebih banyak. Pelajaran tabular:
   fokus ke fitur yang memisahkan kelas sulit (bureau delinq, repayment DPD,
   account low-balance), bukan stacking buta.
4. Scaling law valid sampai Base (~177M), Colossal (~395M) stagnan
   (sub11 0.812 < sub10 0.821). Pelajaran: tetapkan budget compute + early stop
   via OOF, hindari overnight raksasa tanpa gate.
5. Workflow gagal: tanpa git, artefak ganda, builder 500+ baris, ledger stale.
   Sudah diperbaiki di workspace ini (lihat WORKFLOW.md).

## Teknik yang diwariskan

- Seed 42 deterministik, stratified 5-fold, OOF sebagai truth.
- Audit submission + consensus anchor (>=98% stabil) -> diadaptasi jadi
  audit KS + stabilitas probabilitas.
- Zero weight export, hanya csv/npy ringan.

## Yang tidak diwariskan

- Augmentasi vision, TTA shift, letterbox, dual-GPU fold queue.
- Prior calibration gamma tanpa bukti (khusus tabular: butuh justifikasi KS).
