"""LambdaRank training for exp057: optimize ranking directly (KS is rank-based).

Usage:
    python scripts/train_rank.py --member fadhlur --exp exp057
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

from evaluate_ks import fold_ks
from scripts.train_baseline import ID_COL, TARGET_COL, load_data
from utils import ensure_dir, seed_all

def chunk_groups(n: int, size: int = 10000) -> list[int]:
    groups = [size] * (n // size)
    if n % size:
        groups.append(n % size)
    return groups


SEED = 42


def norm01(v: np.ndarray) -> np.ndarray:
    lo, hi = v.min(), v.max()
    if hi <= lo:
        return np.full_like(v, 0.5)
    return (v - lo) / (hi - lo)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--member", default="fadhlur")
    ap.add_argument("--exp", default="exp057")
    ap.add_argument("--train", default="data/train_features.parquet")
    ap.add_argument("--test", default="data/test_features.parquet")
    ap.add_argument("--folds", default="train_folds.csv")
    ap.add_argument("--out", default="outputs/experiments")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    seed_all(args.seed)

    train, test, features = load_data(args.train, args.test, args.folds)
    drop_extra = {"rate_x_loan", "rate_x_term"}
    features = [c for c in features if c not in drop_extra]
    y = train[TARGET_COL].to_numpy()
    oof = np.zeros(len(train))
    test_raw = np.zeros(len(test))

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "eval_at": [100],
        "learning_rate": 0.03,
        "num_leaves": 31,
        "min_data_in_leaf": 200,
        "feature_fraction": 0.75,
        "bagging_fraction": 0.7,
        "bagging_freq": 1,
        "lambda_l2": 15.0,
        "verbose": -1,
        "seed": args.seed,
        "deterministic": True,
        "num_threads": 12,
    }
    for fold in sorted(train["fold"].unique()):
        tr = train["fold"] != fold
        dtr = lgb.Dataset(
            train.loc[tr, features], y[tr], group=chunk_groups(int(tr.sum()))
        )
        dva = lgb.Dataset(
            train.loc[~tr, features],
            y[~tr],
            group=chunk_groups(int((~tr).sum())),
            reference=dtr,
        )
        model = lgb.train(
            params,
            dtr,
            num_boost_round=2000,
            valid_sets=[dva],
            callbacks=[lgb.early_stopping(200, verbose=False)],
        )
        oof[~tr] = model.predict(train.loc[~tr, features])
        test_raw += model.predict(test[features]) / train["fold"].nunique()
        print(f"fold {fold}: done iter={model.best_iteration}", flush=True)

    scores = fold_ks(y, oof, train["fold"].to_numpy())
    print(f"OOF KS: {scores['oof']:.5f}")

    outdir = ensure_dir(str(Path(args.out) / args.member / args.exp))
    pd.DataFrame(
        {
            "application_id": train[ID_COL],
            "target": y,
            "pred": norm01(oof),
            "fold": train["fold"],
        }
    ).to_csv(Path(outdir) / "oof.csv", index=False)
    np.save(Path(outdir) / "test_pred.npy", norm01(test_raw))
    pd.DataFrame(
        {"application_id": test[ID_COL], "default_probability": norm01(test_raw)}
    ).to_csv(Path(outdir) / "submission.csv", index=False)
    (Path(outdir) / "log.md").write_text(
        f"# {args.member}/{args.exp} lambdarank\n\n"
        f"- seed: {args.seed}\n- OOF KS: {scores['oof']:.5f}\n"
        + "".join(f"- fold {k}: {v:.5f}\n" for k, v in scores.items() if k != "oof")
        + "- note: raw rank scores min-max normalized to [0,1] for submission\n"
    )
    print(f"artifacts -> {outdir}")


if __name__ == "__main__":
    main()
