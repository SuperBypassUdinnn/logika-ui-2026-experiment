"""Shared cutoff helpers for feature builders."""

from __future__ import annotations

import pandas as pd

CUTOFF = "decision_datetime"


def with_cutoff(aux: pd.DataFrame, apps: pd.DataFrame) -> pd.DataFrame:
    """Attach per-application cutoff to event rows (no filtering)."""
    cut = apps[["customer_id", "application_id", CUTOFF]]
    return aux.merge(cut, on="customer_id", how="inner")
