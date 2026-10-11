# Outputs (Bahasa Indonesia)

Artefak eksperimen. Yang ringan dan deterministik di-commit,
yang besar/regenerable tidak.

- `experiments/<member>/<exp>/` — satu folder per eksperimen:
  `oof.csv`, `test_pred.npy`, `submission.csv`, `log.md`. Di-commit.
- `submissions/subXX/` — kandidat submission beku + `log.md`. Di-commit.
- `analysis/` — lintas eksperimen (bukan milik satu exp):
  perbandingan OOF, konsensus prediksi test, grafik, ringkasan EDA.
  File kecil di-commit, dump besar jangan.
- `current/<nama>/` — log run Kaggle per anggota (misal `.log` dari
  kernel) agar kegagalan cloud bisa didiagnosis dari repo.
  Jangan taruh prediksi/fitur di sini.

Dilarang: duplikat artefak di root repo, bobot model (`.pth`),
dan data mentah (sudah di-ignore).
