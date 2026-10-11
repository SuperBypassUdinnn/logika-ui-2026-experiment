# Setup Anggota Tim (Bahasa Indonesia)

Panduan mulai untuk Farhan, Fadhlur, dan Ilham di workspace
`logika-ui-2026-experiment` (tabular loan-default, metrik KS).
Pengguna Windows baca juga `SETUP_WINDOWS.md`.

## 1. Prasyarat umum (wajib dua opsi)

- Git + GitHub CLI (`gh auth login`)
- Akun Kaggle, join kompetisi `data-science-competition-logika-ui-2026`
  (Accept Rules)
- SSH key terdaftar di GitHub (URL submodule memakai SSH).
  Tanpa SSH key, pakai override HTTPS lokal (bagian 3).

## 2. Pilih cara kerja

| | Opsi A: Lokal | Opsi B: Kaggle |
|:---|:---|:---|
| Cocok untuk | Laptop RAM 8GB+, ingin iterasi cepat | RAM kecil / ingin CPU Kaggle gratis |
| Data | Download ~1GB ke `data/` | Tetap di Kaggle input, tanpa download |
| Venv | Bebas di mana saja (contoh: `~/.venvs/tabular`) | Tidak perlu, install per notebook |
| Ketergantungan | Python 3.11 + pip | Browser + koneksi |

Boleh campur: EDA di Kaggle, training di lokal, atau sebaliknya.
Aturan main (ledger, audit, PR, submission) sama untuk dua-duanya
(bagian 6-8).

## 3. Opsi A: kerja lokal

```bash
git clone --recurse-submodules https://github.com/SuperBypassUdinnn/logika-ui-2026-experiment.git
cd logika-ui-2026-experiment
```

Tanpa SSH key, paksa submodule lewat HTTPS (lokal saja):

```bash
git config submodule.notebooks/current/fadhlur.url https://github.com/SuperBypassUdinnn/logika-ui-2026-notebook-experiment.git
git submodule update --init --recursive
```

Buat venv di lokasi bebas pilihanmu, lalu install:

```bash
python3.11 -m venv <lokasi-bebas>/tabular
<lokasi-bebas>/tabular/bin/pip install -r requirements.txt
```

Download data (~1GB, jangan di-commit):

```bash
kaggle competitions download -c data-science-competition-logika-ui-2026 -p data/ --unzip
```

Kerja di folder `data/` repo ini (bukan `/tmp` yang berupa tmpfs makan RAM).

Buat folds + cek modul:

```bash
python scripts/make_folds.py --input data/train_applications.csv --out train_folds.csv --seed 42
python -c "from src.evaluate_ks import ks_score; print('KS module OK')"
```

Baseline LightGBM ada di `src/model.py` (`make_lgbm`).
Tips hemat RAM: olah per-file streaming, dtype `float32/int32` +
`pyarrow string`, agregasi dulu baru train.

## 4. Opsi B: kerja di Kaggle

1. Di halaman kompetisi, buat Notebook baru (Code > New Notebook).
2. Add Input > Competition > `data-science-competition-logika-ui-2026`.
   Data tampil di `/kaggle/input/data-science-competition-logika-ui-2026/`.
3. Nyalakan Internet (untuk `pip install`), lalu sel install di awal:

```python
!pip install -q lightgbm pandas scikit-learn scipy pyarrow
```

4. Bawa kode `src/` agar notebook tetap tipis. Cara termudah: upload folder
   `src/` sekali sebagai Dataset Kaggle pribadimu, lalu di notebook:

```python
import sys
sys.path.insert(0, '/kaggle/input/<nama-dataset-src-kamu>')
from evaluate_ks import ks_score
```

5. Tulis eksperimen memanggil fungsi `src/` (fitur leakage-safe, model,
   evaluasi KS). Jangan duplikat logika besar di dalam notebook.
6. Save Version (Run All). Download hasilnya (`oof.csv`,
   `submission.csv`) lalu taruh di `outputs/experiments/expXXX/` di repo
   dan catat ledger seperti Opsi A.
7. Notebook yang bagus di-push ke repo notebook masing-masing
   (`scripts/kaggle_sync.py`, bagian 5) agar masuk `notebooks/current/<nama>/`.

## 5. Folder `notebooks/current/` per anggota

- `notebooks/current/fadhlur/` sudah submodule aktif
  (Kaggle `fadhlurrahmanalaudin/logika-ui-2026-notebook-experiment`).
- `notebooks/current/farhan/` dan `notebooks/current/ilham/` masih placeholder.

Untuk Farhan (`C4R4MeL`)/Ilham (`MuhIlham420`) setelah kernel Kaggle dibuat:

```bash
# 1. Buat notebook di Kaggle, catat slug user/kernel
# 2. Buat repo notebook GitHub masing-masing, minta owner pasang deploy key
#    read-only untuk CI (1 key per repo)
# 3. Minta owner menambahkan submodule (pakai URL SSH agar CI bisa clone):
git submodule add git@github.com:<org>/<notebook-repo>.git notebooks/current/farhan
# 4. Isi kaggle_slug + github_repo di notebooks/current/members.json
python scripts/kaggle_sync.py --member farhan --pull
```

Sinkron harian:

```bash
python scripts/kaggle_sync.py --member <nama> --pull
git add notebooks/current/<nama> && git commit -m "sync(current): ..."
```

## 6. Alur eksperimen (berlaku dua opsi)

1. Buat branch: `git checkout -b exp/<nama>-<fitur>`
2. Tambah fitur leakage-safe di `src/features.py` (wajib filter
   `decision_datetime`, lihat `docs/ANTI_LEAKAGE.md`)
3. Train 5-fold (seed 42), simpan ke `outputs/experiments/expXXX/`
   (`oof.csv`, `test_pred.npy`, `submission.csv`, `log.md`)
4. Tambah satu baris ke `docs/LEDGER_EKSPERIMEN.md`
5. Audit:

```bash
python scripts/audit_submission.py outputs/experiments/expXXX/submission.csv --oof outputs/experiments/expXXX/oof.csv
```

6. Push branch + buka PR (template otomatis). Tunggu 1 review + CI hijau.
   Leader (Farhan) yang merge ke `main`.

## 7. Submission (maks 3/hari)

1. Audit lolos 100%
2. Tampilkan scorecard OOF KS + distribusi + checklist leakage di PR/issue
3. Minta persetujuan eksplisit di chat/PR
4. Submit manual via Kaggle, catat di `docs/SUBMISSIONS.md`

Agen dilarang submit tanpa persetujuan.

## 8. OpenCode dan Antigravity

- Skill project: `ds-experiment-workflow` dan `submission-gatekeeper`
  (di `.agents/skills/`, terbaca OpenCode dan Antigravity).
- Buka OpenCode dari folder repo ini agar skill terdeteksi.
- Di Antigravity, aturan di `.agents/rules/` + `AGENTS.md` terbaca otomatis.

## 9. Masalah umum

- `kaggle 403`: belum join kompetisi atau `kaggle.json` salah akun.
- OOM lokal: tutup browser, olah per-file dengan `chunksize`, pakai
  `float32/int32` + `pyarrow string`. Atau pindah ke Opsi B.
- Submodule kosong: jalankan `git submodule update --init --recursive`.
  Jika error `Permission denied (publickey)`, daftarkan SSH key ke GitHub
  atau pakai override HTTPS di bagian 3.
- CI merah di checkout: pastikan URL submodule SSH dan secret
  `SUBMODULE_SSH_KEY` masih valid (deploy key read-only di repo notebook).
- Kuota habis: berhenti, lanjut besok 07:00 WIB.
