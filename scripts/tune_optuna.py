"""Optuna search for exp049: maximize OOF KS with early stopping.

Usage:
    python scripts/tune_optuna.py --member fadhlur --exp exp049 --trials 40
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import lightgbm as lgb
import optuna

from evaluate_ks import fold_ks
from model import make_lgbm
from scripts.train_baseline import ID_COL, TARGET_COL, load_data
from utils import ensure_dir, seed_all


def run_params(train, test, features, params, seed):
    y = train[TARGET_COL].to_numpy()
    oof = np.zeros(len(train))
    test_pred = np.zeros(len(test))
    for fold in sorted(train["fold"].unique()):
        tr = train["fold"] != fold
        model = make_lgbm(seed=seed, **params)
        model.fit(
            train.loc[tr, features],
            y[tr],
            eval_set=[(train.loc[~tr, features], y[~tr])],
            callbacks=[lgb.early_stopping(200, verbose=False)],
        )
        oof[~tr] = model.predict_proba(train.loc[~tr, features])[:, 1]
        test_pred += model.predict_proba(test[features])[:, 1] / train["fold"].nunique()
    return oof, test_pred


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--member", default="fadhlur")
    ap.add_argument("--exp", default="exp049")
    ap.add_argument("--train", default="data/train_features.parquet")
    ap.add_argument("--test", default="data/test_features.parquet")
    ap.add_argument("--folds", default="train_folds.csv")
    ap.add_argument("--out", default="outputs/experiments")
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--drop-cols", default="rate_x_*")
    args = ap.parse_args()
    seed_all(args.seed)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    drop_extra = [c for c in args.drop_cols.split(",") if c]
    train, test, features = load_data(args.train, args.test, args.folds, drop_extra)
    y = train[TARGET_COL].to_numpy()
    folds = train["fold"].to_numpy()

    def objective(trial):
        params = {
            "n_estimators": 5000,
            "learning_rate": trial.suggest_float("lr", 0.01, 0.05, log=True),
            "num_leaves": trial.suggest_int("leaves", 15, 127),
            "min_child_samples": trial.suggest_int("min_child", 10, 300),
            "feature_fraction": trial.suggest_float("feat_frac", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bag_frac", 0.6, 1.0),
            "bagging_freq": 1,
            "lambda_l1": trial.suggest_float("l1", 0.0, 10.0),
            "lambda_l2": trial.suggest_float("l2", 0.0, 20.0),
            "scale_pos_weight": 3,
            "num_threads": 2,
        }
        oof, _ = run_params(train, test, features, params, args.seed)
        return fold_ks(y, oof, folds)["oof"]

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=args.seed)
    )
    study.optimize(objective, n_trials=args.trials)
    print("BEST:", round(study.best_value, 5), study.best_params)

    oof, test_pred = run_params(train, test, features, study.best_params, args.seed)
    scores = fold_ks(y, oof, folds)
    outdir = ensure_dir(str(Path(args.out) / args.member / args.exp))
    pd.DataFrame(
        {
            "application_id": train[ID_COL],
            "target": y,
            "pred": oof,
            "fold": train["fold"],
        }
    ).to_csv(Path(outdir) / "oof.csv", index=False)
    np.save(Path(outdir) / "test_pred.npy", test_pred)
    test_ids = pd.read_csv("data/test_applications.csv", usecols=["application_id"])
    sub = test_ids.merge(
        pd.DataFrame(
            {"application_id": test[ID_COL], "default_probability": test_pred}
        ),
        on="application_id",
        how="left",
    )
    sub.to_csv(Path(outdir) / "submission.csv", index=False)
    (Path(outdir) / "log.md").write_text(
        f"# {args.member}/{args.exp} optuna\n\n"
        f"- trials: {args.trials}\n- best: {study.best_params}\n"
        f"- OOF KS: {scores['oof']:.5f}\n"
    )
    print(f"artifacts -> {outdir}")


if __name__ == "__main__":
    main()
