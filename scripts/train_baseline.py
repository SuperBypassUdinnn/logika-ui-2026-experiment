"""Stratified 5-fold LightGBM baseline producing ledger-ready artifacts.

Usage:
    python scripts/make_folds.py --input data/train_applications.csv --out train_folds.csv
    python scripts/train_baseline.py --member fadhlur --exp exp001

Outputs to outputs/experiments/<member>/<exp>/:
oof.csv, test_pred.npy, submission.csv, log.md
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluate_ks import fold_ks
from model import make_lgbm
from utils import ensure_dir, seed_all

ID_COL = "application_id"
TARGET_COL = "target"
SEED = 42


def load_matrix(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    for c in df.select_dtypes(include=["object", "string"]).columns:
        df[c] = df[c].astype("category")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--member", default="fadhlur")
    ap.add_argument("--exp", default="exp001")
    ap.add_argument("--train", default="data/train_features.parquet")
    ap.add_argument("--test", default="data/test_features.parquet")
    ap.add_argument("--folds", default="train_folds.csv")
    ap.add_argument("--out", default="outputs/experiments")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    seed_all(args.seed)

    train = load_matrix(Path(args.train))
    test = load_matrix(Path(args.test))
    folds = pd.read_csv(args.folds)
    train = train.merge(folds, on=ID_COL, how="left")
    assert train["fold"].notna().all(), "missing folds, run make_folds.py first"

    drop = {ID_COL, TARGET_COL, "fold", "customer_id"}
    drop |= set(train.select_dtypes(include=["datetime64[ns]"]).columns)
    features = [c for c in train.columns if c not in drop]
    X, y = train[features], train[TARGET_COL].to_numpy()
    oof = np.zeros(len(train))
    test_pred = np.zeros(len(test))

    t0 = time.time()
    for fold in sorted(train["fold"].unique()):
        tr = train["fold"] != fold
        model = make_lgbm(seed=args.seed)
        model.fit(X[tr], y[tr])
        oof[~tr] = model.predict_proba(X[~tr])[:, 1]
        test_pred += model.predict_proba(test[features])[:, 1] / train["fold"].nunique()
        print(f"fold {fold}: done", flush=True)
    print(f"trained in {time.time() - t0:.1f}s")

    scores = fold_ks(y, oof, train["fold"].to_numpy())
    print(f"OOF KS: {scores['oof']:.5f} | folds: {scores}")

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
        f"# {args.member}/{args.exp} LightGBM baseline\n\n"
        f"- seed: {args.seed}\n"
        f"- features: {len(features)}\n"
        f"- OOF KS: {scores['oof']:.5f}\n"
        + "".join(f"- fold {k}: {v:.5f}\n" for k, v in scores.items() if k != "oof")
    )
    print(f"artifacts -> {outdir}")


if __name__ == "__main__":
    main()
