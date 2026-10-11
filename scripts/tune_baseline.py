"""Grid tuning for exp002: LightGBM hyperparameters, OOF KS selection.

Usage:
    python scripts/tune_baseline.py --member fadhlur --exp exp002

Trains every config with full 5-fold CV, keeps the best OOF artifacts.
"""

from __future__ import annotations

import argparse
import itertools
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scripts.train_baseline import load_data, run_experiment

GRID = {
    "learning_rate": [0.03, 0.05],
    "num_leaves": [31, 63, 127],
    "min_child_samples": [20, 100],
}
FIXED = {"n_estimators": 200}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--member", default="fadhlur")
    ap.add_argument("--exp", default="exp002")
    ap.add_argument("--train", default="data/train_features.parquet")
    ap.add_argument("--test", default="data/test_features.parquet")
    ap.add_argument("--folds", default="train_folds.csv")
    ap.add_argument("--out", default="outputs/experiments")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    train, test, features = load_data(args.train, args.test, args.folds)
    keys = list(GRID)
    results = []
    for vals in itertools.product(*(GRID[k] for k in keys)):
        params = dict(FIXED)
        params.update(dict(zip(keys, vals)))
        with tempfile.TemporaryDirectory() as tmp:
            scores = run_experiment(
                train,
                test,
                features,
                params,
                args.seed,
                tmp,
                f"{args.member}/{args.exp} tuning trial",
            )
        results.append((scores["oof"], params, scores))
        print(f"trial {params} -> OOF KS {scores['oof']:.5f}", flush=True)

    results.sort(key=lambda r: -r[0])
    print("\n=== ranking ===")
    for oof, params, scores in results:
        folds = {k: round(v, 4) for k, v in scores.items() if k != "oof"}
        print(f"{oof:.5f} {params} {folds}")

    _, best_params, _ = results[0]
    outdir = Path(args.out) / args.member / args.exp
    scores = run_experiment(
        train,
        test,
        features,
        best_params,
        args.seed,
        outdir,
        f"{args.member}/{args.exp} tuned LightGBM",
    )
    with open(Path(outdir) / "log.md", "a") as f:
        f.write("\n## Tuning grid\n")
        for oof, params, _ in results:
            f.write(f"- OOF {oof:.5f} {params}\n")
    print(f"BEST OOF KS {scores['oof']:.5f} {best_params}")


if __name__ == "__main__":
    main()
