# Anti-Leakage Tabular (Bahasa Indonesia)

Cutoff informasi: `decision_datetime` pada aplikasi yang dinilai.
Hanya info yang tersedia sebelum cutoff boleh jadi fitur.

## Aturan per tabel

- `customer_profiles`: atribut saat keputusan (boleh, tanpa future).
- `previous_applications`: hanya `decision_datetime < cutoff` saat ini.
- `repayment_history`: hanya cicilan historis sebelum cutoff; kolom
  `last_payment_datetime` yang blank = belum bayar (bukan 0).
- `account_behavior`: hanya `month < bulan cutoff`.
- `bureau_observations`: hanya jika `available_datetime <= decision_datetime`
  (bukan `report_datetime`).
- `ecommerce_transactions`: hanya `transaction_datetime < decision_datetime`,
  hanya `order_status==completed` sebagai spending; `cancelled` bukan spending.
  Dedup `order_total` per `order_id` sebelum agregasi.

## Larangan

- `application_id`, `customer_id`, `loan_id` hanya join key.
- Dilarang memakai `target` test, future DPD, atau agregat full-history tanpa filter cutoff.
- Setiap PR fitur wajib sebutkan kolom waktu yang dipakai untuk filter.
