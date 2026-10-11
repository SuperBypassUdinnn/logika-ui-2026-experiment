# Workflow Lomba (Bahasa Indonesia)

Dokumen ini hasil penyempurnaan dari workflow lomba vision lama.

## Kekurangan lomba lama yang diperbaiki

1. Tanpa git -> sekarang wajib git + branch + PR.
2. Artefak ganda di root (`submission.csv`, `oof_*.npy` duplikat `outputs/`) ->
   sekarang hanya `outputs/` yang kanonis, root bersih.
3. Notebook builder 500-730 baris per eksperimen, overlap `scratch/` vs `scripts/` ->
   sekarang logika di `src/`, notebook tipis, tidak ada builder raksasa.
4. Blend komposit (`sub03,06,07,11,12,13`) tanpa kode beku ->
   sekarang setiap submission wajib ada `log.md` + script blend yang di-commit.
5. Ledger manual dan stale (`STRATEGY_ROADMAP` tertinggal di Day 5) ->
   sekarang `LEDGER_EKSPERIMEN.md` satu baris per exp, update via PR yang sama dengan artefak.
6. OOF Macro-F1 0.97 vs LB 0.64-0.84 (gap besar, overfit + probing) ->
   sekarang OOF KS + fold-wise KS + time-split check sebagai gate.
7. `__pycache__`, `old_work/` duplikat, `outputs/experiments` bolong ->
   sekarang `.gitignore` ketat, tidak ada arsip duplikat.

## Alur baku

1. Buat fold: `python scripts/make_folds.py`
2. Bangun fitur: `python scripts/build_features.py` (join leakage-safe
   semua 7 tabel bantu ke `data/*_features.parquet`, otomatis ter-ignore).
3. Kembangkan fitur di `src/features.py` dengan cutoff `decision_datetime`.
4. Eksperimen lokal full (laptop sanggup 5-7GB peak): train LightGBM 5-fold.
5. Simpan ke `outputs/experiments/expXXX/` (`oof.csv`, `test_pred.npy`, `submission.csv`, `log.md`).
6. Tambah satu baris ke `docs/LEDGER_EKSPERIMEN.md` di PR yang sama.
7. Audit: `python scripts/audit_submission.py outputs/.../submission.csv --oof outputs/.../oof.csv`
8. Minta persetujuan user, submit maksimal 3/hari.
9. Sinkron `current/<nama>/` via `scripts/kaggle_sync.py` + PR harian.

## Aturan notebook (adopsi skill notebook-workflow)

Notebook boleh dipakai untuk EDA dan narasi, tapi logika tetap di `src/`:

1. Pairing sekali: `jupytext --set-formats ipynb,py:percent <notebook.ipynb>`
2. Edit file `.py` pasangan (batas sel `# %%`), jangan edit JSON `.ipynb`.
3. Sinkron: `jupytext --sync`, eksekusi penuh `jupyter nbconvert --execute`.
4. Commit keduanya (`.ipynb` + `.py`), output dipertahankan.
5. Cek: `ruff check` pada `.py` pasangan sebelum PR.

## Aturan nama notebook (anti-bentrok)

- Eksperimen per anggota: `notebooks/experiments/<member>/expNNN_<slug>.ipynb`.
- EDA: `notebooks/eda/<member>_<topik>.ipynb`.
- Submission bersama: `notebooks/submissions/subXX_<slug>.ipynb`,
  nomor diterbitkan leader. Detail: `notebooks/README.md`.

## Peran

- Farhan (leader): final merge + submission.
- Fadhlur (`fadhlurrahmanalaudin`): pipeline + fitur.
- Ilham: EDA + validasi.
