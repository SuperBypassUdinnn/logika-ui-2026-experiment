# Leakage Prevention

1. Filter every auxiliary table by `decision_datetime` cutoff.
2. Bureau uses `available_datetime`, not `report_datetime`.
3. Account uses month < decision month. E-commerce uses completed only.
4. IDs are join keys only. Document time columns in every feature PR.
