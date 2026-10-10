"""Leakage-safe data loading.

Rule: only rows with event timestamp strictly before the application's
decision_datetime may be used as features.
"""

from __future__ import annotations

import pandas as pd

CUTOFF_COL = "decision_datetime"
ID_COLS = ("application_id", "customer_id", "loan_id")


def parse_times(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def filter_before_cutoff(
    events: pd.DataFrame,
    apps: pd.DataFrame,
    event_time: str,
    app_id: str = "application_id",
) -> pd.DataFrame:
    """Join events to apps and keep only events before cutoff."""
    merged = events.merge(
        apps[[app_id, "customer_id", CUTOFF_COL]], on=["customer_id"], how="inner"
    )
    return merged[merged[event_time] < merged[CUTOFF_COL]].copy()
