"""Segment-specialist training (no-history apps): full CV, saved artifacts.

Usage:
    python scripts/train_segment_specialist.py --member fadhlur --exp exp075
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold

from evaluate_ks import ks_score
from model import make_lgbm
from scripts.train_baseline import ID_COL, TARGET_COL, load_data
from utils import ensure_dir, seed_all

PARAMS = {
    "n_estimators": 2000,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_child_samples": 50,
    "colsample_bytree": 0.7,
    "reg_lambda": 5.0,
    "scale_pos_weight": 3,
    "num_threads": 2,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--member", default="fadhlur")
    ap.add_argument("--exp", default="exp075")
    ap.add_argument("--train", default="data/train_features.parquet")
    ap.add_argument("--test", default="data/test_features.parquet")
    ap.add_argument("--out", default="outputs/experiments")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    seed_all(args.seed)

    train, test, features = load_data(args.train, args.test, "train_folds.csv")
    seg = (train["repay_n"] == 0).to_numpy()
    X, y = train.loc[seg, features], train.loc[seg, TARGET_COL].to_numpy()
    km = lambda yt, yp: ("ks", float(ks_score(np.asarray(yt), np.asarray(yp))), True)

    t0 = time.time()
    oof = np.zeros(len(X))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    for tr, va in skf.split(X, y):
        model = make_lgbm(seed=args.seed, **PARAMS)
        model.fit(
            X.iloc[tr],
            y[tr],
            eval_set=[(X.iloc[va], y[va])],
            eval_metric=km,
            callbacks=[lgb.early_stopping(200, verbose=False)],
        )
        oof[va] = model.predict_proba(X.iloc[va])[:, 1]
    print(f"trained in {time.time() - t0:.1f}s")
    print(f"SEGMENT OOF KS: {ks_score(y, oof):.5f}")

    ids = train.loc[seg, ID_COL].reset_index(drop=True)
    outdir = ensure_dir(str(Path(args.out) / args.member / args.exp))
    pd.DataFrame({"application_id": ids, "target": y, "pred": oof}).to_csv(
        Path(outdir) / "oof.csv", index=False
    )
    teseg = (test["repay_n"] == 0).to_numpy()
    full = make_lgbm(seed=args.seed, **PARAMS)
    full.fit(X, y)
    tp = np.full(len(test), np.nan)
    tp[teseg] = full.predict_proba(test.loc[teseg, features])[:, 1]
    np.save(Path(outdir) / "test_pred.npy", tp)
    pd.DataFrame({"application_id": test[ID_COL], "pred": tp}).to_csv(
        Path(outdir) / "segment_test_pred.csv", index=False
    )
    (Path(outdir) / "log.md").write_text(
        f"# {args.member}/{args.exp} no-history specialist\n\n"
        f"- segment train n: {len(X)}\n"
        f"- segment OOF KS: {ks_score(y, oof):.5f}\n"
    )
    print(f"artifacts -> {outdir}")


if __name__ == "__main__":
    main()
