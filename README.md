# Logika UI 2026 — Experiment Workspace

Workspace utama lomba tabular prediksi gagal bayar (KS statistic).
Tim 3 orang: Farhan (leader), Fadhlurrahman Alaudin (`fadhlurrahmanalaudin`), Ilham.

## Struktur

- `src/` — kode reusable (data, fitur anti-leakage, model, evaluasi KS)
- `scripts/` — otomasi (audit submission, folds, sync Kaggle)
- `docs/` — dokumentasi Bahasa Indonesia (workflow, ledger, anti-leakage, analisis lomba lama)
- `notebooks/` — pointer EDA/eksperimen (kode berat di `src/`, bukan builder 500+ baris)
- `outputs/` — artefak per eksperimen/submission (satu-satunya lokasi kanonis)
- `data/` — tidak di-commit, diunduh via Kaggle API
- `notebooks/current/` — submodule per anggota: `notebooks/current/fadhlur`, `notebooks/current/farhan`, `notebooks/current/ilham`
- `.agents/rules/` — aturan agen (English, technical)
- `.agents/knowledge/` — spek lomba kanonis (overview, rules)

## Quickstart

```bash
pip install -r requirements.txt
kaggle competitions files -c data-science-competition-logika-ui-2026
python scripts/make_folds.py --input data/train_applications.csv --out train_folds.csv --seed 42
python scripts/audit_submission.py outputs/submissions/sub01/submission.csv
```

## Aturan

- Dilarang submit tanpa audit + persetujuan eksplisit user.
- OOF KS adalah kebenaran utama, bukan LB probing.
- Semua fitur harus memakai cutoff `decision_datetime` (anti-leakage).
- Satu eksperimen = satu folder `outputs/experiments/expXXX/` + satu baris ledger.
- Sinkron `notebooks/current/<nama>/` via PR harian.

Lihat `docs/WORKFLOW.md` dan `AGENTS.md`.
Anggota baru mulai dari `docs/SETUP_ANGGOTA_TIM.md`.
