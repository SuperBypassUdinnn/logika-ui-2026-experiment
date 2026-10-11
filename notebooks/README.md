# Notebooks

Notebook tipis yang memanggil `src/`. Kode berat dilarang di notebook.

## Aturan nama (anti-bentrok antar anggota)

- Eksperimen: `experiments/<member>/expNNN_<slug>.ipynb`
  (nomor jalan per anggota, contoh `experiments/fadhlur/exp001_lgbm_baseline.ipynb`).
  ID unik = path lengkap, bukan nomor saja.
- EDA: `eda/<member>_<topik>.ipynb`
  (contoh `eda/ilham_target_leakage_check.ipynb`).
- Submission: `submissions/subXX_<slug>.ipynb`
  (namespace tunggal milik tim, nomor diterbitkan leader via ledger,
  contoh `submissions/sub01_lgbm_baseline.ipynb`).

Eksplorasi bebas boleh di `notebooks/current/<member>/` masing-masing;
yang masuk `notebooks/` hanya hasil kurasi yang lolos audit + ledger.
Lihat `docs/WORKFLOW.md` dan aturan pairing jupytext di sana.
