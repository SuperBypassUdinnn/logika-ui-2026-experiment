"""Stacker for exp023: logistic regression on base OOFs + top features.

Usage:
    python scripts/train_stacker.py --member fadhlur --exp exp023 \
        --src fadhlur/exp019 fadhlur/exp020 fadhlur/exp021
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluate_ks import fold_ks
from scripts.train_baseline import ID_COL, load_data
from utils import ensure_dir, seed_all


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--member", default="fadhlur")
    ap.add_argument("--exp", default="exp023")
    ap.add_argument("--src", nargs="+", required=True)
    ap.add_argument("--train", default="data/train_features.parquet")
    ap.add_argument("--test", default="data/test_features.parquet")
    ap.add_argument("--folds", default="train_folds.csv")
    ap.add_argument("--out", default="outputs/experiments")
    ap.add_argument("--model", default="logreg", choices=["logreg", "lgbm"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    seed_all(args.seed)

    from sklearn.linear_model import LogisticRegression

    base = Path("outputs/experiments")
    ref = pd.read_csv(base / args.src[0] / "oof.csv")
    y = ref["target"].to_numpy()
    oof_stack = np.column_stack(
        [pd.read_csv(base / s / "oof.csv")["pred"].to_numpy() for s in args.src]
    )
    test_stack = np.column_stack(
        [np.load(base / s / "test_pred.npy") for s in args.src]
    )

    train, test, _ = load_data(args.train, args.test, args.folds)
    top = [
        "repay_pay_ratio",
        "repay_dpd_max",
        "bureau_score_mean",
        "eco_spend_sum",
        "annual_interest_rate",
    ]
    top = [c for c in top if c in train.columns]
    oof = np.zeros(len(train))
    test_pred = np.zeros(len(test))
    for fold in sorted(ref["fold"].unique()):
        tr = ref["fold"] != fold
        Xtr = np.column_stack([oof_stack[tr], train.loc[tr, top].to_numpy(dtype=float)])
        Xva = np.column_stack(
            [oof_stack[~tr], train.loc[~tr, top].to_numpy(dtype=float)]
        )
        med = np.nanmedian(Xtr, axis=0)
        Xtr = np.where(np.isnan(Xtr), med, Xtr)
        Xva = np.where(np.isnan(Xva), med, Xva)
        Xt = np.column_stack([test_stack, test[top].to_numpy(dtype=float)])
        Xt = np.where(np.isnan(Xt), med, Xt)
        if args.model == "lgbm":
            import lightgbm as lgb

            cols = [f"m{i}" for i in range(Xtr.shape[1])]
            dtr = pd.DataFrame(Xtr, columns=cols)
            dva = pd.DataFrame(Xva, columns=cols)
            dte = pd.DataFrame(Xt, columns=cols)
            clf = lgb.LGBMClassifier(
                n_estimators=5000,
                learning_rate=0.03,
                num_leaves=15,
                min_child_samples=200,
                verbose=-1,
            )
            clf.fit(
                dtr,
                y[tr],
                eval_set=[(dva, y[~tr])],
                callbacks=[lgb.early_stopping(200, verbose=False)],
            )
            oof[~tr] = clf.predict_proba(dva)[:, 1]
            test_pred += clf.predict_proba(dte)[:, 1] / ref["fold"].nunique()
        else:
            from sklearn.linear_model import LogisticRegression

            mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
            clf = LogisticRegression(C=1.0, max_iter=2000)
            clf.fit((Xtr - mu) / sd, y[tr])
            oof[~tr] = clf.predict_proba((Xva - mu) / sd)[:, 1]
            test_pred += clf.predict_proba((Xt - mu) / sd)[:, 1] / ref["fold"].nunique()
        print(f"fold {fold}: done", flush=True)

    scores = fold_ks(y, oof, ref["fold"].to_numpy())
    print(f"STACKER OOF KS: {scores['oof']:.5f}")

    outdir = ensure_dir(str(Path(args.out) / args.member / args.exp))
    pd.DataFrame(
        {
            "application_id": ref["application_id"],
            "target": y,
            "pred": oof,
            "fold": ref["fold"],
        }
    ).to_csv(Path(outdir) / "oof.csv", index=False)
    np.save(Path(outdir) / "test_pred.npy", test_pred)
    pd.DataFrame(
        {"application_id": test[ID_COL], "default_probability": test_pred}
    ).to_csv(Path(outdir) / "submission.csv", index=False)
    (Path(outdir) / "log.md").write_text(
        f"# {args.member}/{args.exp} stacker ({args.model})\n\n"
        f"- sources: {args.src}\n- top features: {top}\n"
        f"- OOF KS: {scores['oof']:.5f}\n"
    )
    print(f"artifacts -> {outdir}")


if __name__ == "__main__":
    main()
