# Strategi Awal Eksperimen (Bahasa Indonesia)

Disusun 2026-10-11 dari analisis data + KS univariat fitur exp001.
Acuan: OOF KS, baseline `fadhlur/exp001` = 0.3823.

## Temuan analisis

1. Target stabil per bulan (12-15%), purpose/region/employment/housing datar.
   Stratified 5-fold sah; time-split hanya robustness check.
2. Ranking KS univariat (70k train):
   `repay_pay_ratio` 0.322, `repay_dpd_max` 0.315, `bureau_score_mean` 0.253,
   `annual_interest_rate` 0.230, `bureau_util_mean` 0.150,
   `eco_spend_sum` 0.137. Sisanya < 0.10.
3. DPD maksimum historis monoton: 8.5% (DPD 0) -> 51.8% (DPD 90+).
   Riwayat repayment hanya ada di 46k/70k train — segmen tanpa histori
   butuh penanganan khusus (indikator missing, bukan sekadar 0).
4. Model baseline (0.382) hanya sedikit di atas fitur univariat terbaik
   (0.322): interaksi dan tren temporal belum tergali.

## Urutan eksperimen

| ID | Delta | Hipotesis | Target OOF KS | Risiko |
|:---|:---|:---|:---:|:---|
| fadhlur/exp002 | Tuning LightGBM (lr, leaves, min_data, feature_fraction) | Baseline default underfit; tuning murni tanpa fitur baru | 0.40-0.42 | Overfit fold, mitigasi via fold-wise spread |
| fadhlur/exp003 | Fitur tren repayment (DPD 3 bln terakhir, tren pay-ratio, DPD saat cutoff) | Sinyal terkuat ada di perilaku terbaru, bukan agregat lifetime | 0.43-0.46 | Agregat recency sensitif cutoff; uji leakage ganda |
| fadhlur/exp004 | Interaksi rate (rate x loan, rate x tenor) + bureau per-provider | Rate KS 0.23 univariat; provider A/B/C beda kualitas skor | 0.40-0.43 | Multikolinearitas, gain kecil |
| fadhlur/exp005 | Perilaku ecommerce (mix payment, kategori, frekuensi 3 bln) | Spend KS 0.137 masih kasar; pola COD vs kartu memisahkan risiko | 0.39-0.41 | Noise tinggi, butuh regularisasi |
| fadhlur/exp006 | Validasi time-split (train bulan awal, valid bulan akhir) | Pastikan ranking tidak bergantung periode; bukan pengganti stratified | selisih < 0.03 vs stratified | Jika gap besar, strategi feature ulang |

Aturan eksekusi: satu exp satu PR (artefak + baris ledger + log.md),
audit wajib, tanpa submit sebelum OOF > 0.45 dan fold stabil.
