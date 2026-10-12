"""Logistic-regression companion on engineered features (Farhan-style).

QuantileTransformer + OneHot + LR(C), OOF per seed, rank-blend with
bootstrap significance gate vs LGBM reference.

Usage:
    python scripts/train_lr.py --member fadhlur --exp expLR --seeds 42 43
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluate_ks import ks_score
from scripts.train_baseline import ID_COL, TARGET_COL, load_data
from utils import ensure_dir, seed_all


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--member", default="fadhlur")
    ap.add_argument("--exp", default="expLR")
    ap.add_argument("--train", default="data/train_features.parquet")
    ap.add_argument("--test", default="data/test_features.parquet")
    ap.add_argument("--folds", default="train_folds.csv")
    ap.add_argument("--out", default="outputs/experiments")
    ap.add_argument("--seeds", nargs="+", type=int, default=[42])
    ap.add_argument("--c", type=float, default=0.03)
    args = ap.parse_args()
    seed_all(args.seeds[0])

    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, QuantileTransformer

    train, test, features = load_data(args.train, args.test, args.folds)
    cat_cols = list(train[features].select_dtypes("category").columns)
    num_cols = [c for c in features if c not in cat_cols]
    for c in cat_cols:
        train[c] = train[c].astype(str).fillna("MISSING")
        test[c] = test[c].astype(str).fillna("MISSING")

    def make_lr():
        num = make_pipeline(
            QuantileTransformer(
                n_quantiles=200,
                output_distribution="normal",
                subsample=200_000,
                random_state=args.seeds[0],
            ),
            SimpleImputer(strategy="constant", fill_value=0.0),
        )
        prep = ColumnTransformer(
            [
                ("num", num, num_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ]
        )
        return make_pipeline(prep, LogisticRegression(C=args.c, max_iter=2000))

    X = train[features]
    y = train[TARGET_COL].to_numpy()
    oofs = []
    for s in args.seeds:
        cv = StratifiedKFold(5, shuffle=True, random_state=s)
        oofs.append(
            cross_val_predict(make_lr(), X, y, cv=cv, method="predict_proba")[:, 1]
        )
        print(f"seed {s}: done", flush=True)
    oof = np.mean(oofs, axis=0)
    test_pred = make_lr().fit(X, y).predict_proba(test[features])[:, 1]
    print(f"LR OOF KS: {ks_score(y, oof):.5f}")

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
        f"# {args.member}/{args.exp} logistic regression\n\n"
        f"- seeds: {args.seeds}\n- C: {args.c}\n"
        f"- OOF KS: {ks_score(y, oof):.5f}\n"
    )
    print(f"artifacts -> {outdir}")


if __name__ == "__main__":
    main()
