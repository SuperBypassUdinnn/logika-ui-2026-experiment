# Reproducibility

1. Seed 42 everywhere.
2. Stratified 5-fold default via `scripts/make_folds.py`.
3. OOF KS is truth. Time-split is robustness check.
4. Canonical artifacts only under `outputs/`. No root duplicates.
5. Thin notebooks calling `src/`. No mega builders.
