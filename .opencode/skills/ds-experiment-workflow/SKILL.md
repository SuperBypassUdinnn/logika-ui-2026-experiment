---
name: ds-experiment-workflow
description: Tabular loan-default experiment workflow for Logika UI 2026. Use for EDA, leakage-safe features, LightGBM training, stratified 5-fold OOF KS tracking, ledger updates, and thin-notebook discipline.
---

## Trigger Keywords
`experiment`, `train`, `eda`, `feature`, `leakage`, `lightgbm`, `oof`, `ks`, `fold`, `ledger`, `kaggle sync`, `current/`

## Workflow (7-Step Lifecycle)

1. **Hypothesis & Rationale**: State delta, motivation, expected OOF KS impact, risk. Record in `docs/LEDGER_EKSPERIMEN.md`.
2. **Feature building in `src/`**: Add leakage-safe builders in `src/features.py` using only rows with event time strictly before `decision_datetime`. Bureau uses `available_datetime`, account uses month < decision month, ecommerce uses completed only with order dedup.
3. **Local training**: Full-local on laptop (31GB RAM). Run stratified 5-fold via `src/model.py` with seed 42. Never train on test target.
4. **Artifact save**: One folder `outputs/experiments/expXXX/` with `oof.csv`, `test_pred.npy`, `submission.csv`, `log.md`.
5. **Audit**: Run `python scripts/audit_submission.py outputs/experiments/expXXX/submission.csv --oof outputs/experiments/expXXX/oof.csv`.
6. **Ledger**: Append one row to `docs/LEDGER_EKSPERIMEN.md` in the same PR as artifacts.
7. **Sync**: Update `current/<member>/` via `python scripts/kaggle_sync.py --member <name> --pull` and merge via daily PR.

## Standards

- Seed 42, stratified 5-fold default (`scripts/make_folds.py`). Time-split is robustness check only.
- OOF KS from `src/evaluate_ks.py` is truth. Do not tune prior or threshold without OOF evidence.
- Thin notebooks calling `src/`. No 500-line builders, no root duplicate artifacts, no `__pycache__`.
- IDs (`application_id`, `customer_id`, `loan_id`) are join keys only.
