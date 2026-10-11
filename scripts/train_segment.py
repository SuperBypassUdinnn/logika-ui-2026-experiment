"""Segment-aware training for exp007: separate models with/without history.

Usage:
    python scripts/train_segment.py --member fadhlur --exp exp007 \
        --params '{"n_estimators": 200, "learning_rate": 0.03, "num_leaves": 31, "min_child_samples": 100}'

Split key: repay_n > 0. The no-history model drops repay_* columns.
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

from evaluate_ks import fold_ks, ks_score
from model import make_lgbm
from scripts.train_baseline import ID_COL, TARGET_COL, load_data
from utils import ensure_dir, seed_all


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--member", default="fadhlur")
    ap.add_argument("--exp", default="exp007")
    ap.add_argument("--train", default="data/train_features.parquet")
    ap.add_argument("--test", default="data/test_features.parquet")
    ap.add_argument("--folds", default="train_folds.csv")
    ap.add_argument("--out", default="outputs/experiments")
    ap.add_argument("--params", default="{}")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    seed_all(args.seed)
    params = json.loads(args.params)

    train, test, features = load_data(args.train, args.test, args.folds)
    y = train[TARGET_COL].to_numpy()
    has_hist = train["repay_n"] > 0
    nohist_feats = [c for c in features if not c.startswith("repay_")]
    print(f"segments: history={int(has_hist.sum())} nohist={int((~has_hist).sum())}")
    print(f"features: history={len(features)} nohist={len(nohist_feats)}")

    oof = np.zeros(len(train))
    test_pred = np.zeros(len(test))
    seg_oof = {}
    t0 = time.time()
    for name, mask, feats in [
        ("hist", has_hist, features),
        ("nohist", ~has_hist, nohist_feats),
    ]:
        tmask = (test["repay_n"] > 0) if name == "hist" else (test["repay_n"] <= 0)
        for fold in sorted(train["fold"].unique()):
            tr = (train["fold"] != fold) & mask
            va = (train["fold"] == fold) & mask
            if tr.sum() == 0 or va.sum() == 0:
                continue
            model = make_lgbm(seed=args.seed, **params)
            model.fit(train.loc[tr, feats], y[tr])
            oof[va] = model.predict_proba(train.loc[va, feats])[:, 1]
            test_pred[tmask] += (
                model.predict_proba(test.loc[tmask, feats])[:, 1]
                / train["fold"].nunique()
            )
        seg = mask.to_numpy()
        seg_oof[name] = ks_score(y[seg], oof[seg])
        print(f"segment {name}: OOF KS {seg_oof[name]:.5f}", flush=True)
    print(f"trained in {time.time() - t0:.1f}s")

    scores = fold_ks(y, oof, train["fold"].to_numpy())
    print(f"OVERALL OOF KS: {scores['oof']:.5f}")

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
        f"# {args.member}/{args.exp} segment models\n\n"
        f"- seed: {args.seed}\n"
        f"- params: {params}\n"
        f"- OOF KS overall: {scores['oof']:.5f}\n"
        + "".join(f"- segment {k}: {v:.5f}\n" for k, v in seg_oof.items())
        + "".join(f"- fold {k}: {v:.5f}\n" for k, v in scores.items() if k != "oof")
    )
    print(f"artifacts -> {outdir}")


if __name__ == "__main__":
    main()
