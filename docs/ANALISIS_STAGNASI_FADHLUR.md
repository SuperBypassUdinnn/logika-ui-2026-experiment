# Analisis Stagnasi Pipeline Fadhlur vs Farhan

**Tanggal**: 2026-10-13
**Konteks**: 71 eksperimen fadhlur (exp001-exp071) vs 3 entry Farhan
**Best OOF Fadhlur**: 0.5128 (exp068, blend)  |  **Best standalone**: 0.5103 (exp052)
**Best OOF Farhan**: 0.5173 (v14 blend)  |  **Best standalone**: 0.5161 (ec3 + bag5)
**Gap**: 0.0045 OOF, tetapi **0.007 di Public LB** (0.545 vs 0.552)

---

## Ringkasan Masalah Inti

Masalahnya **bukan** kamu kurang fitur atau kurang rajin. Kamu sudah menjalankan 71 eksperimen dan sudah mengadopsi hampir semua kolom yang Farhan pakai (account, bureau lintas-aplikasi, active-loan burden, cross-ratio). Gap OOF sekarang hanya 0.005. Masalah sebenarnya ada di **tiga area yang lebih mendasar**:

1. **Variance reduction** -- Farhan menghancurkan varians, kamu tidak
2. **Model diversity** -- Farhan blend GBDT + linear model, kamu blend sesama tree
3. **Rank-space blending** -- Farhan blend di ruang peringkat, kamu blend probabilitas mentah

---

## Masalah 1: Farhan Menghancurkan Varians (25 Model vs 5 Model)

### Apa yang Farhan lakukan

Farhan menjalankan **5 seed x 5 fold = 25 model LightGBM**, lalu rata-ratakan prediksi. Ini disebut "seed bagging". Hasilnya: OOF naik dari ~0.5050 ke **0.5161** murni dari variance reduction.

### Apa yang kamu lakukan

- exp016 (bagging 3 seed): **negatif**, lalu diabaikan
- exp051 (bagging 5 seed config optuna): **0.5074**, di bawah single model 0.5080

### Mengapa milikmu gagal tapi Farhan berhasil

Masalah ada di script [`train_bagging.py`](file:///home/superbypassudin/.clone/Github/logika-ui-2026-experiment/scripts/train_bagging.py):

```
Baris 59: callbacks=[lgb.early_stopping(args.early_stopping, verbose=False)],
```

Di `train_bagging.py`, early stopping **tidak menggunakan eval_metric KS kustom**. Bandingkan dengan `train_baseline.py` baris 77-82 yang menggunakan `eval_metric=lambda yt, yp: ("ks", ks_score(...), True)`. Script bagging kamu jatuh ke default `metric='auc'` dari `make_lgbm`, jadi ES berhenti di iterasi optimal-AUC bukan optimal-KS, lalu rata-ratanya jadi noise.

Farhan menjalankan bagging dengan early stopping yang konsisten karena dia hanya punya satu notebook monolitik -- tidak ada diskrepansi konfigurasi antar script.

> [!IMPORTANT]
> **Kesimpulan**: Bagging 5 seed harusnya selalu membantu (mengurangi varians), tapi bagging kamu terdegradasi oleh bug early stopping yang hilang di `train_bagging.py`.

---

## Masalah 2: Kamu Hanya Blend Sesama Tree Model

### Apa yang Farhan lakukan

Farhan **menambah regresi logistik** (QuantileTransformer + OneHot, C=0.03) sebagai pendamping LGBM. Lalu rank-blend 0.7 LGBM / 0.3 LR.

Mengapa ini kuat: LR dan GBDT membuat **error yang berbeda secara struktural**. LR bersifat linear -- ia lebih stabil di area data sparse dan menangkap hubungan monoton yang GBDT potensial over-split. Blending keduanya mengurangi error yang saling ortogonal.

### Apa yang kamu lakukan

Semua blend kamu (exp014, exp022, exp030, exp040, exp048, exp050, exp055, exp059, exp063, exp068) berisi kombinasi:
- LGBM + XGBoost + CatBoost
- Atau LGBM + Farhan-OOF

Masalahnya: **LGBM, XGBoost, dan CatBoost membuat error yang sangat mirip** (korelasi OOF ~0.92 per Temuan 5). Blend sesama tree punya plafon karena ketiga model melihat data dengan cara yang hampir identik.

Kamu sudah punya `scripts/train_lr.py` tapi **belum pernah menggunakannya dalam eksperimen yang tercatat di ledger**.

> [!IMPORTANT]
> **Kesimpulan**: Diversitas model yang sebenarnya membutuhkan model family yang berbeda (tree + linear), bukan hanya implementasi tree yang berbeda.

---

## Masalah 3: Blend di Ruang Probabilitas vs Ruang Peringkat

### Apa yang Farhan lakukan

Farhan blend menggunakan **rank percentile** (`pd.Series(pred).rank(pct=True)`), lalu rata-rata tertimbang di ruang peringkat.

### Apa yang kamu lakukan

`scripts/blend.py` mendukung `--rank` (baris 49-57), tapi **tidak ada satu eksperimen pun yang menggunakan flag ini**. Semua blend dicatat sebagai weighted average biasa.

Mengapa ini penting untuk KS: Metrik KS mengukur pemisahan CDF. Yang penting adalah **urutan relatif** prediksi, bukan nilai absolutnya. Rank-blend menjamin setiap komponen berkontribusi secara setara tanpa masalah skala -- model dengan prediksi 0.01-0.99 dan model dengan prediksi 0.3-0.7 tetap memiliki bobot efektif yang sama.

> [!IMPORTANT]
> **Kesimpulan**: Kamu punya infrastruktur rank-blend tapi tidak menggunakannya. Ini "uang gratis" 0.001-0.003 KS.

---

## Masalah 4: Disiplin Statistik -- Bootstrap Gate

### Apa yang Farhan lakukan

Sebelum menerima blend apapun, Farhan menjalankan **bootstrap 200x**: resample OOF, hitung KS blend vs KS single, dan hanya terima blend jika selisih > 0 pada **95% confidence interval**.

### Apa yang kamu lakukan

Kamu memutuskan blend berdasarkan satu angka OOF KS: "kalau naik, pakai". Ini rentan noise -- selisih 0.001 OOF bisa saja kebetulan pada suatu random fold split.

Akibatnya, kamu sering memasukkan komponen blend yang sebenarnya netral atau negatif (exp055 "jenuh, di bawah single", exp050 "jenuh"), menambah noise ke submission akhir.

---

## Masalah 5: Feature Engineering -- Bukan Kurang, Tapi Salah Arah

Kamu sudah punya 190+ fitur vs ~120 Farhan, tapi banyak fitur kamu yang **redundan** bukan **komplementer**:

| Isu | Detail |
|:---|:---|
| Eco fitur bertumpuk | `eco_spend_30d`, `eco_sum_done_30d`, `eco_groceries_30d` mengukur hal yang sama di window yang sama. Farhan fokus pada **spike ratio** (rasio short/long window) yang mengukur *perubahan perilaku*, bukan level. |
| ec3 port gagal | Temuan 9: port fitur ec3 ke pipeline-mu justru **menurunkan** KS 0.004. Ini karena regularisasi/kapasitas pipeline berbeda -- fitur yang bagus di pipeline A bisa noise di pipeline B jika hyperparameter tidak disesuaikan. |
| Fitur quantity over quality | 190 fitur dengan `num_leaves=21` artinya banyak fitur tidak pernah dipakai oleh tree. Farhan dengan 120 fitur dan regularisasi yang tepat memaksimalkan signal-to-noise ratio. |

---

## Masalah 6: Konfigurasi Early Stopping yang Tidak Konsisten

Temuan 8 di `TEMUAN_EKSPERIMEN.md` sudah mengidentifikasi ini: **notebook Kaggle berhenti di iter ~43 vs repo iter ~477** karena konfigurasi `metric` berbeda. Meski sudah diperbaiki untuk `train_baseline.py`, masalah serupa masih ada di `train_bagging.py` (seperti disebutkan di Masalah 1).

Selain itu, `model.py` default:
- `metric='auc'` -- early stopping berdasarkan AUC, bukan KS
- `learning_rate=0.05`, `num_leaves=63` -- ini cukup agresif untuk dataset 70k baris

Sementara config terbaik kamu (Optuna exp049/exp052) sudah menemukan `lr=0.018, leaves=21, min_child=203` -- tapi ini **tidak dipakai kembali secara konsisten** di semua script downstream (bagging, xgb, catboost, blend).

---

## Diagnosis: Mengapa 71 Eksperimen Hanya Dapat 0.5128

```
                         Farhan (3 entry, 0.5173)
                              |
    +-- Variance reduction ----+---- 0.5161 (5 seed x 5 fold)
    |                          |
    +-- Model diversity -------+---- +0.001 (LR companion)
    |                          |
    +-- Rank blend + bootstrap-+---- +0.001 (rank-space + gate)
    |
    |      Fadhlur (71 entry, 0.5128)
    |           |
    +-- Bagging broken (ES bug)
    +-- Blend sesama tree (korelasi 0.92)
    +-- Blend probabilitas mentah
    +-- Tidak ada bootstrap gate
    +-- Fitur banyak tapi redundan
    +-- Konfigurasi tidak konsisten antar script
```

---

## Apa yang Harus Dilakukan (Urutan Prioritas)

### Prioritas 1: Fix Bagging Script + Run 5-Seed Bagging (estimasi +0.005-0.010)

Perbaiki `train_bagging.py` agar menggunakan `eval_metric` KS kustom yang sama seperti `train_baseline.py`. Lalu jalankan 5-seed (42,43,44,45,46) bagging dengan config Optuna terbaik (exp052).

### Prioritas 2: Jalankan LR Companion + Rank-Blend (estimasi +0.002-0.004)

Kamu sudah punya `train_lr.py`. Jalankan, lalu rank-blend dengan LGBM bagged.
Formula Farhan: `0.7 * rank(lgbm_bagged) + 0.3 * rank(lr_multi_seed)`.

### Prioritas 3: Bootstrap Gate untuk Semua Blend (estimasi: mencegah degradasi)

Implementasi sederhana:
```python
import numpy as np
from scipy.stats import ks_2samp

def bootstrap_gate(y, pred_a, pred_b, n_boot=200, alpha=0.05):
    diffs = []
    for _ in range(n_boot):
        idx = np.random.choice(len(y), len(y), replace=True)
        ks_a = ks_2samp(pred_a[idx][y[idx]==1], pred_a[idx][y[idx]==0]).statistic
        ks_b = ks_2samp(pred_b[idx][y[idx]==1], pred_b[idx][y[idx]==0]).statistic
        diffs.append(ks_b - ks_a)
    return np.percentile(diffs, 100 * alpha) > 0  # True = blend is better
```

### Prioritas 4: Prune Fitur Redundan (estimasi +0.001-0.003)

Kurangi dari 190 ke ~130 fitur via ablasi grup (bukan gain). Drop fitur yang ablasi-nya menunjukkan delta < 0.0005. Ini mengurangi noise dan membuat tree capacity lebih fokus.

### Prioritas 5: Farhan ec3-style Features Native (estimasi +0.002)

Daripada port ec3 mentah, implementasi *konsep*-nya di pipeline kamu:
- Spike ratio (short/long window): sudah ada sebagian (`eco_vel_30_180`)
- HHI konsentrasi kategori: belum ada
- Payment method ratio 90d: belum ada
- Cancel value ratio: ada (`eco_cancel_rate`) tapi bukan value-weighted

---

## Pertanyaan untuk Kamu

Sebelum saya actionkan apapun, ada beberapa hal yang perlu saya tahu:

1. **Apakah kamu sudah pernah menjalankan `train_lr.py`?** Tidak ada entry di ledger, tapi mungkin dijalankan tapi tidak dicatat?

2. **Di exp051 (bagging 5 seed) yang negatif, apakah `--early-stopping` di-set?** Kalau ya, bug ES di bagging script mungkin penyebab utamanya.

3. **Berapa kuota submission harian yang tersisa?** Ini menentukan apakah kita bisa langsung test atau harus menunggu.
