"""Tabular model factory: LightGBM baseline with seed control."""

from __future__ import annotations


def make_lgbm(seed: int = 42, **overrides):
    import lightgbm as lgb

    params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "verbose": -1,
        "seed": seed,
        "deterministic": True,
        "num_threads": 2,
    }
    params.update(overrides)
    return lgb.LGBMClassifier(**params)
