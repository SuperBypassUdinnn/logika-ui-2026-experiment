"""Seed bagging with per-seed fold partitions and KS early stopping.

Each seed gets its own StratifiedKFold split (unlike fixed train_folds.csv),
matching Farhan's 5-seed x 5-fold protocol. ES uses custom KS eval metric.

Usage:
    python scripts/train_bagging.py --member fadhlur --exp exp073 \
        --params '{...}' --seeds 42 43 44 45 46
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold

from evaluate_ks import fold_ks, ks_score
from model import make_lgbm
from scripts.train_baseline import ID_COL, TARGET_COL, load_data
from utils import ensure_dir, seed_all

SEED = 42


def ks_metric(yt, yp):
    return (
        "ks",
        float(ks_score(np.asarray(yt), np.asarray(yp))),
        True,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--member", default="fadhlur")
    ap.add_argument("--exp", default="exp073")
    ap.add_argument("--train", default="data/train_features.parquet")
    ap.add_argument("--test", default="data/test_features.parquet")
    ap.add_argument("--folds", default="train_folds.csv")
    ap.add_argument("--out", default="outputs/experiments")
    ap.add_argument("--params", default="{}")
    ap.add_argument("--drop-cols", default="")
    ap.add_argument("--early-stopping", type=int, default=200)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    args = ap.parse_args()
    params = json.loads(args.params)

    train, test, features = load_data(args.train, args.test, args.folds)
    drop_extra = [c for c in args.drop_cols.split(",") if c]
    drop = set()
    for c in drop_extra:
        if c.endswith("*"):
            drop |= {x for x in train.columns if x.startswith(c[:-1])}
        else:
            drop.add(c)
    features = [c for c in features if c not in drop]
    print(f"features: {len(features)} (dropped {len(drop_extra)} patterns)")
    y = train[TARGET_COL].to_numpy()
    oofs, tests = [], []

    t0 = time.time()
    for seed in args.seeds:
        seed_all(seed)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        oof = np.zeros(len(train))
        test_pred = np.zeros(len(test))
        for _, (tr, va) in enumerate(skf.split(train, y)):
            model = make_lgbm(seed=seed, **params)
            model.fit(
                train.loc[tr, features],
                y[tr],
                eval_set=[(train.loc[va, features], y[va])],
                eval_metric=ks_metric,
                callbacks=[lgb.early_stopping(args.early_stopping, verbose=False)],
            )
            oof[va] = model.predict_proba(train.loc[va, features])[:, 1]
            test_pred += model.predict_proba(test[features])[:, 1] / 5
        oofs.append(oof)
        tests.append(test_pred)
        print(f"seed {seed}: OOF KS {ks_score(y, oof):.5f}", flush=True)
    oof_bag = np.mean(oofs, axis=0)
    test_bag = np.mean(tests, axis=0)
    print(f"trained in {time.time() - t0:.1f}s")

    ref_fold = pd.read_csv(args.folds)
    ref = train[[ID_COL]].merge(ref_fold, on=ID_COL, how="left")["fold"].to_numpy()
    scores = fold_ks(y, oof_bag, ref)
    print(f"BAGGED OOF KS: {scores['oof']:.5f} (folds ref seed 42)")

    outdir = ensure_dir(str(Path(args.out) / args.member / args.exp))
    pd.DataFrame(
        {"application_id": train[ID_COL], "target": y, "pred": oof_bag, "fold": ref}
    ).to_csv(Path(outdir) / "oof.csv", index=False)
    np.save(Path(outdir) / "test_pred.npy", test_bag)
    test_ids = pd.read_csv("data/test_applications.csv", usecols=["application_id"])
    sub = test_ids.merge(
        pd.DataFrame({"application_id": test[ID_COL], "default_probability": test_bag}),
        on="application_id",
        how="left",
    )
    sub.to_csv(Path(outdir) / "submission.csv", index=False)
    (Path(outdir) / "log.md").write_text(
        f"# {args.member}/{args.exp} seed bagging (per-seed folds, KS ES)\n\n"
        f"- seeds: {args.seeds}\n- params: {params}\n"
        f"- OOF KS: {scores['oof']:.5f}\n"
    )
    print(f"artifacts -> {outdir}")


if __name__ == "__main__":
    main()
