"""Prediction helpers: validate submission format."""

import pandas as pd

REQUIRED_COLS = ["application_id", "default_probability"]


def validate_submission(sub: pd.DataFrame, test_ids: pd.Series) -> list[str]:
    errors = []
    if list(sub.columns) != REQUIRED_COLS:
        errors.append(f"header harus {REQUIRED_COLS}, dapat {list(sub.columns)}")
    if len(sub) != len(test_ids):
        errors.append(f"jumlah baris {len(sub)} != test {len(test_ids)}")
    if sub["application_id"].duplicated().any():
        errors.append("application_id duplikat")
    missing = set(test_ids) - set(sub["application_id"])
    if missing:
        errors.append(f"{len(missing)} application_id test hilang")
    extra = set(sub["application_id"]) - set(test_ids)
    if extra:
        errors.append(f"{len(extra)} application_id di luar test")
    probs = sub["default_probability"]
    if probs.isna().any() or (~probs.between(0, 1)).any():
        errors.append("default_probability harus finite 0-1")
    return errors
