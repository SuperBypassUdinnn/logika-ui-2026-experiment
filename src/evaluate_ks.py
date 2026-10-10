"""KS metric: single source of truth for this competition."""

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def ks_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    preds_bad = y_pred[y_true == 1]
    preds_good = y_pred[y_true == 0]
    if len(preds_bad) == 0 or len(preds_good) == 0:
        return 0.0
    return float(ks_2samp(preds_bad, preds_good).statistic)


def score_dataframes(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    row_id_column_name: str = "application_id",
    target_col: str = "default_probability",
) -> float:
    sol = solution.sort_values(by=row_id_column_name).reset_index(drop=True)
    sub = submission.sort_values(by=row_id_column_name).reset_index(drop=True)
    y_true = sol[target_col].to_numpy()
    y_pred = sub[target_col].to_numpy()
    return ks_score(y_true, y_pred)


def fold_ks(y_true: np.ndarray, y_pred: np.ndarray, folds: np.ndarray) -> dict:
    out = {}
    for f in sorted(np.unique(folds)):
        mask = folds == f
        out[int(f)] = ks_score(y_true[mask], y_pred[mask])
    out["oof"] = ks_score(y_true, y_pred)
    return out
