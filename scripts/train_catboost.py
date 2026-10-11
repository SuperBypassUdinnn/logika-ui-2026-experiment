"""CatBoost 5-fold CV with native categorical support (exp017).

Usage:
    python scripts/train_catboost.py --member fadhlur --exp exp017
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

from evaluate_ks import fold_ks
from scripts.train_baseline import ID_COL, TARGET_COL, load_data
from utils import ensure_dir, seed_all

SEED = 42


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--member", default="fadhlur")
    ap.add_argument("--exp", default="exp017")
    ap.add_argument("--train", default="data/train_features.parquet")
    ap.add_argument("--test", default="data/test_features.parquet")
    ap.add_argument("--folds", default="train_folds.csv")
    ap.add_argument("--out", default="outputs/experiments")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    seed_all(args.seed)

    from catboost import CatBoostClassifier, Pool

    train, test, features = load_data(args.train, args.test, args.folds)
    cat_idx = [
        i
        for i, c in enumerate(features)
        if str(train[c].dtype) in ("category", "object", "string")
    ]
    for c in features:
        if str(train[c].dtype) not in ("category", "object", "string"):
            continue
        tr_s = train[c].astype(str).fillna("MISSING")
        te_s = test[c].astype(str).fillna("MISSING")
        cats = pd.unique(pd.concat([tr_s, te_s], ignore_index=True))
        cats = [x for x in cats if pd.notna(x)]
        train[c] = pd.Categorical(tr_s, categories=cats).astype(str)
        test[c] = pd.Categorical(te_s, categories=cats).astype(str)
    y = train[TARGET_COL].to_numpy()
    oof = np.zeros(len(train))
    test_pred = np.zeros(len(test))

    t0 = time.time()
    for fold in sorted(train["fold"].unique()):
        tr = train["fold"] != fold
        model = CatBoostClassifier(
            iterations=800,
            learning_rate=0.03,
            depth=6,
            l2_leaf_reg=10,
            auto_class_weights="Balanced",
            cat_features=cat_idx,
            random_seed=args.seed,
            verbose=False,
            allow_writing_files=False,
        )
        model.fit(
            Pool(train.loc[tr, features], y[tr], cat_features=cat_idx),
            eval_set=Pool(train.loc[~tr, features], y[~tr], cat_features=cat_idx),
            early_stopping_rounds=100,
        )
        oof[~tr] = model.predict_proba(train.loc[~tr, features])[:, 1]
        test_pred += model.predict_proba(test[features])[:, 1] / train["fold"].nunique()
        print(f"fold {fold}: done best_iter={model.best_iteration_}", flush=True)
    print(f"trained in {time.time() - t0:.1f}s")

    scores = fold_ks(y, oof, train["fold"].to_numpy())
    print(f"OOF KS: {scores['oof']:.5f}")

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
        f"# {args.member}/{args.exp} CatBoost\n\n"
        f"- seed: {args.seed}\n- OOF KS: {scores['oof']:.5f}\n"
        + "".join(f"- fold {k}: {v:.5f}\n" for k, v in scores.items() if k != "oof")
    )
    print(f"artifacts -> {outdir}")


if __name__ == "__main__":
    main()
