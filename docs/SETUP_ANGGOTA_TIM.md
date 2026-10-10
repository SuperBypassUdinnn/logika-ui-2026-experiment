# Setup Anggota Tim (Bahasa Indonesia)

Panduan mulai untuk Farhan, Fadhlur, dan Ilham di workspace
`logika-ui-2026-experiment` (tabular loan-default, metrik KS).
Pengguna Windows baca juga `SETUP_WINDOWS.md`.

## 1. Prasyarat

- Git + GitHub CLI (`gh auth login`)
- Python 3.11
- Akun Kaggle + `kaggle.json` di `~/.kaggle/` (chmod 600)
- Ikuti kompetisi `data-science-competition-logika-ui-2026` di Kaggle (Accept Rules)

## 2. Clone

```bash
git clone --recurse-submodules https://github.com/SuperBypassUdinnn/logika-ui-2026-experiment.git
cd logika-ui-2026-experiment
```

Jika sudah clone tanpa submodule:

```bash
git submodule update --init --recursive
```

## 3. Install

```bash
pip install -r requirements.txt
```

## 4. Data (tidak di-commit, ~1GB)

```bash
kaggle competitions download -c data-science-competition-logika-ui-2026 -p data/ --unzip
ls -lh data/
```

Kerja di folder `data/` di repo ini (bukan `/tmp`, karena `/tmp` adalah tmpfs makan RAM).
Jangan commit file CSV/parquet.

## 5. Folds + baseline lokal

```bash
python scripts/make_folds.py --input data/train_applications.csv --out train_folds.csv --seed 42
python -c "from src.evaluate_ks import ks_score; print('KS module OK')"
```

Baseline LightGBM ada di `src/model.py` (`make_lgbm`). Laptop tim sanggup
full-lokal (peak 5-7GB): olah per-file streaming, dtype hemat, agregasi dulu
baru train.

## 6. Alur eksperimen

1. Buat branch: `git checkout -b exp/<nama>-<fitur>`
2. Tambah fitur leakage-safe di `src/features.py` (wajib filter
   `decision_datetime`, lihat `docs/ANTI_LEAKAGE.md`)
3. Train 5-fold, simpan ke `outputs/experiments/expXXX/`
   (`oof.csv`, `test_pred.npy`, `submission.csv`, `log.md`)
4. Tambah satu baris ke `docs/LEDGER_EKSPERIMEN.md`
5. Audit:

```bash
python scripts/audit_submission.py outputs/experiments/expXXX/submission.csv --oof outputs/experiments/expXXX/oof.csv
```

6. Push branch + buka PR (template otomatis). Tunggu 1 review + CI hijau.
   Leader (Farhan) yang merge ke `main`.

## 7. Folder `current/` per anggota

- `current/fadhlur/` sudah submodule aktif
  (Kaggle `fadhlurrahmanalaudin/logika-ui-2026-notebook-experiment`).
- `current/farhan/` dan `current/ilham/` masih placeholder.

Untuk Farhan/Ilham setelah kernel Kaggle dibuat:

```bash
# 1. Buat notebook di Kaggle, catat slug user/kernel
# 2. Minta owner menambahkan submodule:
git submodule add https://github.com/<org>/<notebook-repo>.git current/farhan
# 3. Isi kaggle_slug di current/members.json
python scripts/kaggle_sync.py --member farhan --pull
```

Sinkron harian:

```bash
python scripts/kaggle_sync.py --member <nama> --pull
git add current/<nama> && git commit -m "sync(current): ..."
```

## 8. Submission (maks 3/hari)

1. Audit lolos 100%
2. Tampilkan scorecard OOF KS + distribusi + checklist leakage di PR/issue
3. Minta persetujuan eksplisit di chat/PR
4. Submit manual via Kaggle, catat di `docs/SUBMISSIONS.md`

Agen dilarang submit tanpa persetujuan.

## 9. OpenCode dan Antigravity

- Skill project: `ds-experiment-workflow` dan `submission-gatekeeper`
  (di `.opencode/skills/` dan `.agents/skills/`).
- Buka OpenCode dari folder repo ini agar skill terdeteksi.
- Di Antigravity, aturan di `.agents/rules/` + `AGENTS.md` terbaca otomatis.

## 10. Masalah umum

- `kaggle 403`: belum join kompetisi atau `kaggle.json` salah akun.
- OOM lokal: tutup browser, olah per-file dengan `chunksize`, pakai
  `float32/int32` + `pyarrow string`.
- Submodule kosong: jalankan `git submodule update --init --recursive`.
- Kuota habis: berhenti, lanjut besok 07:00 WIB.
