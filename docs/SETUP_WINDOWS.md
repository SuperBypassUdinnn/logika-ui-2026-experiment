# Setup Windows (Bahasa Indonesia)

Panduan untuk anggota tim yang memakai Windows 10/11.
Rekomendasi: pakai Git Bash + Python 3.11.
Pilih Opsi A (lokal) atau Opsi B (Kaggle) seperti di `SETUP_ANGGOTA_TIM.md`.

## 1. Install umum (wajib dua opsi)

1. Git for Windows (aktifkan Git Bash + `gh` opsional via `winget install GitHub.cli`)
2. Verifikasi di PowerShell atau Git Bash:

```bash
git --version
```

## 2. Opsi A: kerja lokal di Windows

1. Python 3.11 dari python.org (centang Add to PATH). Verifikasi:

```bash
python --version
pip --version
```

2. Kredensial Kaggle: login kaggle.com, buat API token
   (Avatar > Settings > API > Create New Token).
   Simpan `kaggle.json` ke `%USERPROFILE%\.kaggle\kaggle.json`:

```powershell
mkdir $env:USERPROFILE\.kaggle -Force
# pindahkan file unduhan ke sana
```

3. Join kompetisi `data-science-competition-logika-ui-2026` (Accept Rules).
4. Clone di Git Bash (pakai path pendek tanpa spasi,
   misal `C:\lomba\logika-ui-2026-experiment`):

```bash
git clone --recurse-submodules https://github.com/SuperBypassUdinnn/logika-ui-2026-experiment.git
cd logika-ui-2026-experiment
```

Submodule memakai URL SSH, jadi butuh SSH key (Git Bash):

```bash
ssh-keygen -t ed25519 -C "email@kamu"
cat ~/.ssh/id_ed25519.pub
# tambah ke GitHub > Settings > SSH and GPG keys, lalu tes:
ssh -T git@github.com
```

Tanpa SSH key, paksa submodule lewat HTTPS (lokal saja):

```bash
git config submodule.current/fadhlur.url https://github.com/SuperBypassUdinnn/logika-ui-2026-notebook-experiment.git
git submodule update --init --recursive
```

5. Venv di lokasi bebas pilihanmu:

```powershell
py -3.11 -m venv <lokasi-bebas>\tabular
<lokasi-bebas>\tabular\Scripts\pip install -r requirements.txt
```

6. Download data lalu folds + audit:

```bash
kaggle competitions download -c data-science-competition-logika-ui-2026 -p data/ --unzip
# jika 'kaggle' tidak dikenal, pakai: python -m kaggle ...
python scripts/make_folds.py --input data/train_applications.csv --out train_folds.csv --seed 42
python scripts/audit_submission.py outputs/experiments/expXXX/submission.csv --oof outputs/experiments/expXXX/oof.csv
```

Jika unzip gagal di PowerShell, ekstrak manual via Explorer ke `data/`.
Jika RAM <16GB, mulai dari sampel atau pindah ke Opsi B:

```bash
python -c "import pandas as pd; df=pd.read_csv('data/train_applications.csv', nrows=5000); print(df.shape)"
```

## 3. Opsi B: kerja di Kaggle (sama di semua OS)

Ikuti bagian 4 `SETUP_ANGGOTA_TIM.md` dari browser: attach data kompetisi,
install `lightgbm`, bawa `src/` sebagai dataset, Save Version, download
hasil ke `outputs/experiments/expXXX/`. Tidak perlu Python lokal sama sekali.

Jika `kaggle kernels pull` gagal di Windows (path), pull manual dari web
Kaggle lalu copy file ke `current/<nama>/`.

## 4. Troubleshooting Windows

- Baris `CRLF` vs `LF`: biarkan default Git, jangan edit `.ipynb` manual.
- `Permission denied` saat submodule: tutup VSCode/Explorer yang mengunci folder.
  Jika `Permission denied (publickey)`, daftarkan SSH key atau pakai override HTTPS.
- Python menunjuk Microsoft Store: install dari python.org dan perbaiki PATH.
- Untuk training penuh 1GB, colok charger + tutup browser agar RAM lega.
