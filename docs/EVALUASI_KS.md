# Evaluasi KS (Bahasa Indonesia)

Metrik resmi: Kolmogorov-Smirnov (KS) statistic.

Fungsi `score` resmi:

```python
from scipy.stats import ks_2samp
ks_2samp(preds_bad, preds_good).statistic
```

`preds_bad` = prediksi untuk `target==1`, `preds_good` = `target==0`.
KS = selisih maksimum CDF kedua grup. Lebih tinggi lebih baik.
Leaderboard awal 0.543-0.568 konsisten dengan KS (bukan AUC).

## Aturan evaluasi lokal

- OOF KS 5-fold stratified adalah kebenaran utama.
- Laporkan KS per fold + OOF untuk deteksi fold anomali.
- Tambahkan time-split check (sort by `decision_datetime`) sebagai robustness,
  bukan pengganti.
- Jangan optimasi threshold; KS memakai ranking probabilitas.
- Kalibrasi prior (seperti gamma di lomba lama) dilarang kecuali ada bukti OOF.

Implementasi: `src/evaluate_ks.py`.
