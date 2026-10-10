# Setup Windows (Bahasa Indonesia)

Panduan untuk anggota tim yang memakai Windows 10/11.
Rekomendasi: pakai Git Bash + Python 3.11. Untuk training lokal besar,
tetap bisa, tapi jaga RAM dan path.

## 1. Install

1. Git for Windows (aktifkan Git Bash + `gh` opsional via `winget install GitHub.cli`)
2. Python 3.11 dari python.org (centang Add to PATH)
3. Verifikasi di PowerShell atau Git Bash:

```bash
git --version
python --version
pip --version
```

## 2. Kredensial Kaggle

1. Login kaggle.com, buat API token (Avatar > Settings > API > Create New Token).
2. Simpan `kaggle.json` ke `%USERPROFILE%\.kaggle\kaggle.json`:

```powershell
mkdir $env:USERPROFILE\.kaggle -Force
# pindahkan file unduhan ke sana
```

3. Join kompetisi `data-science-competition-logika-ui-2026` (Accept Rules).

## 3. Clone (Git Bash disarankan)

```bash
git clone --recurse-submodules https://github.com/SuperBypassUdinnn/logika-ui-2026-experiment.git
cd logika-ui-2026-experiment
pip install -r requirements.txt
```

Catatan path Windows:

- Hindari folder dengan spasi (misal `Documents/Lomba UI/` bisa bermasalah di script).
- Pakai path pendek, misal `C:\lomba\logika-ui-2026-experiment`.
- Jika `kaggle` tidak dikenal, pakai `python -m kaggle ...`.

## 4. Data

```bash
kaggle competitions download -c data-science-competition-logika-ui-2026 -p data/ --unzip
```

Jika unzip gagal di PowerShell, ekstrak manual via Explorer ke `data/`.
Jangan commit CSV. Jika RAM <16GB, mulai dari sampel:

```bash
python -c "import pandas as pd; df=pd.read_csv('data/train_applications.csv', nrows=5000); print(df.shape)"
```

## 5. Folds + audit (sama seperti Linux)

```bash
python scripts/make_folds.py --input data/train_applications.csv --out train_folds.csv --seed 42
python scripts/audit_submission.py outputs/experiments/expXXX/submission.csv --oof outputs/experiments/expXXX/oof.csv
```

Jika error `ModuleNotFoundError: lightgbm` di Windows, install wheel:

```bash
pip install lightgbm
```

## 6. Submodule `current/`

Di Git Bash:

```bash
git submodule update --init --recursive
python scripts/kaggle_sync.py --member farhan --pull
```

Jika `kaggle kernels pull` gagal di Windows (path), pull manual dari web Kaggle
lalu copy file ke `current/<nama>/`.

## 7. Troubleshooting Windows

- Baris `CRLF` vs `LF`: biarkan default Git, jangan edit `.ipynb` manual.
- `Permission denied` saat submodule: tutup VSCode/Explorer yang mengunci folder.
- Python menunjuk Microsoft Store: install dari python.org dan perbaiki PATH.
- Untuk training penuh 1GB, colok charger + tutup browser agar RAM lega.
