# Competition Overview: DSC LOGIKA UI 2026 (Knowledge)

Source: Kaggle `data-science-competition-logika-ui-2026` Overview tab.
Scraped 2026-10-11. Canonical spec; do not edit except to sync with Kaggle.

## Task

Binary credit-risk prediction on Indonesian consumer lending.
Predict probability of serious repayment difficulty:
loan end-of-day DPD reaches 90+ within 12 calendar months after origination.

- Target `1` = reaches 90+ DPD within 12 months, else `0`.
- Train positive rate 13.58% (9,507 / 70,000). Imbalanced.
- All amounts in whole IDR. Timestamps Asia/Jakarta local time.

## Metric: Kolmogorov-Smirnov (KS) Statistic

Maximum difference between score CDFs of the two classes:

```
KS = max_s | F1(s) - F0(s) |
```

Official evaluation code (`scipy.stats.ks_2samp` on positives vs negatives).
Higher is better. Ranking metric, no threshold tuning needed.

Verified implementation: `src/evaluate_ks.py`.

## Submission Format

Columns: `application_id,default_probability`.
30,000 rows, every test ID exactly once, probs finite in [0, 1].
Row order does not matter.

## Data Tables (verified local row counts)

| File | Rows | Key columns |
|:---|---:|:---|
| `train_applications.csv` | 70,000 | application_id, customer_id, decision_datetime, loan terms, target |
| `test_applications.csv` | 30,000 | same minus target |
| `customer_profiles.csv` | 100,000 | customer_id, age, region, employment, income, dependents, housing |
| `previous_applications.csv` | 201,984 | history per customer, loan_id blank unless originated |
| `repayment_history.csv` | 1,361,341 | loan_id installments, due_date, paid, days_past_due |
| `account_behavior.csv` | 1,084,418 | customer_id monthly balances, flows, low_balance_days |
| `bureau_observations.csv` | 797,563 | 3 providers, score 300-850, report_datetime vs available_datetime |
| `ecommerce_transactions.csv` | 4,359,817 | items, dedup order_total per order_id, completed vs cancelled |
| `sample_submission.csv` | 30,000 | format reference |

Column definitions: `data/DATA_DICTIONARY.md`.

## Leakage Cutoff

`decision_datetime` is the information cutoff per application.
Bureau usability key is `available_datetime` (not `report_datetime`).
Full per-table rules: `docs/ANTI_LEAKAGE.md`.

## Elimination Scoring (from Rules tab)

- Public LB 45% + Private LB 45% + Notebook reproducibility 10%.
- Top 8 teams advance to offline final at Universitas Indonesia, Depok.
- Notebook deadline 30 Oct 2026 23:59 WIB, late penalty 1 point/minute.
