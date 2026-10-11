# Redesain Strategi 01: Segmentasi (Bahasa Indonesia)

Penulis: Fadhlur, 2026-10-11.
`STRATEGI_AWAL.md` tetap frozen sebagai v1; file ini iterasi terpisah
agar tidak bentrok dengan anggota lain.

## Bukti (OOF exp003, 70k train)

| Segmen | n | OOF KS |
|:---|---:|:---|
| Berhistori repayment | 46.312 | 0,4162 |
| Tanpa histori | 23.688 | 0,3441 |
| Gabungan | 70.000 | 0,3930 |

Kesimpulan: plateau 0,39 adalah plafon satu model untuk dua populasi,
bukan plafon fitur. Model tunggal dikompromi segmen tanpa histori.

## Desain baru

1. **Segment-aware (prioritas 1).** Dua model LightGBM per segmen
   (berhistori: fitur repayment penuh; tanpa histori: bureau/eco/profil
   saja) atau satu model + interaksi segmen eksplisit.
   Target: segmen lemah 0,344 -> 0,38+, gabungan -> 0,43+.
2. **Trajektori (prioritas 2).** Slope DPD 3 cicilan terakhir, flag
   memburuk (recent > lifetime), bulan sejak delinkuensi pertama,
   tren drawdown saldo, tren skor bureau antar laporan.
3. **Long-train + early stopping (prioritas 3).** lr 0,01-0,02,
   n hingga 2000, ganti n=200 fixed.
4. **Ecommerce recency turun prioritas** (sinyal lemah, noise tinggi).

## Antrian eksekusi

| ID | Isi | Target OOF KS |
|:---|:---|:---:|
| fadhlur/exp005 | Fitur trajektori | 0,41-0,43 |
| fadhlur/exp006 | Validasi time-split (tetap dari v1) | selisih < 0,03 |
| fadhlur/exp007 | Model per segmen | gabungan 0,43+ |
| fadhlur/exp008 | Long-train early stopping | +0,005-0,01 |

Gate submit tetap: OOF > 0,45 + fold stabil + audit lolos.
