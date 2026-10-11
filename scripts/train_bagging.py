"""Multi-seed bagging for exp016: average OOF/test over seeds.

Usage:
    python scripts/train_bagging.py --member fadhlur --exp exp016 \
        --params '{...}' --seeds 42 43 44
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluate_ks import fold_ks
from model import make_lgbm
from scripts.train_baseline import ID_COL, TARGET_COL, load_data
from utils import ensure_dir, seed_all


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--member", default="fadhlur")
    ap.add_argument("--exp", default="exp016")
    ap.add_argument("--train", default="data/train_features.parquet")
    ap.add_argument("--test", default="data/test_features.parquet")
    ap.add_argument("--folds", default="train_folds.csv")
    ap.add_argument("--out", default="outputs/experiments")
    ap.add_argument("--params", default="{}")
    ap.add_argument("--early-stopping", type=int, default=0)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    args = ap.parse_args()
    params = json.loads(args.params)

    train, test, features = load_data(args.train, args.test, args.folds)
    y = train[TARGET_COL].to_numpy()
    oof = np.zeros(len(train))
    test_pred = np.zeros(len(test))

    t0 = time.time()
    for seed in args.seeds:
        seed_all(seed)
        for fold in sorted(train["fold"].unique()):
            tr = train["fold"] != fold
            model = make_lgbm(seed=seed, **params)
            if args.early_stopping > 0:
                model.fit(
                    train.loc[tr, features],
                    y[tr],
                    eval_set=[(train.loc[~tr, features], y[~tr])],
                    callbacks=[
                        lgb.early_stopping(args.early_stopping, verbose=False)
                    ],
                )
            else:
                model.fit(train.loc[tr, features], y[tr])
            oof[~tr] += model.predict_proba(train.loc[~tr, features])[:, 1]
            test_pred += (
                model.predict_proba(test[features])[:, 1]
                / train["fold"].nunique()
                / len(args.seeds)
            )
        print(f"seed {seed}: done", flush=True)
    oof /= len(args.seeds)
    print(f"trained in {time.time() - t0:.1f}s")

    scores = fold_ks(y, oof, train["fold"].to_numpy())
    print(f"BAGGED OOF KS: {scores['oof']:.5f}")

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
    pd.DataFrame(
        {"application_id": test[ID_COL], "default_probability": test_pred}
    ).to_csv(Path(outdir) / "submission.csv", index=False)
    (Path(outdir) / "log.md").write_text(
        f"# {args.member}/{args.exp} seed bagging\n\n"
        f"- seeds: {args.seeds}\n- params: {params}\n"
        f"- OOF KS: {scores['oof']:.5f}\n"
    )
    print(f"artifacts -> {outdir}")


if __name__ == "__main__":
    main()
